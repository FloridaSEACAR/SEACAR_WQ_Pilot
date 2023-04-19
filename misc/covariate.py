# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:31:48 2023

@author: qiangy
"""

import geopandas as gpd
import rasterio as rio
from scipy.stats import pearsonr
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

import warnings


def preprocess2(df,col_ls,para_ls):
    
    # ------- Select columns and parameters ------
    df = df[col_ls]
    df = df.loc[df["ParameterName"].isin(para_ls)]
    df["timestamp"]=  pd.to_datetime(df['SampleDate'])
    #---------- remove outliers -----------------
    # Remove total nitrogen outliers (>100)
    df.drop(df[(df['ParameterName'] == 'Total Nitrogen') & 
                     (df['ResultValue'] > 100)].index,inplace=True)

    # Remove a single measurement in 1996-07-22 (RowID: 1582917)
    df.drop(df[df['RowID'] == 1582917].index, inplace=True)

    # Remove turbidity outliers (>25)
    df.drop(df[(df['ParameterName'] == 'Turbidity') & 
                     (df['ResultValue'] > 25)].index, inplace=True)

    # Remove Secchi Depth before 1995 (117 records)
    df.drop(df[(df['ParameterName'] == 'Secchi Depth') & 
                     (df['ResultValue'] < 1995)].index, inplace=True)
    
    # Remove salinity outliers (>25)
    df.drop(df[(df['ParameterName'] == 'Salinity') & 
                     (df['ResultValue'] > 100)].index, inplace=True)

    # Remove Dissolved Oxygen outliers
    df.drop(df[(df['ParameterName'] == 'Dissolved Oxygen') & 
                     (df['ResultValue'] < 100)].index, inplace=True)
    
    return df

# time_period is a list of tuples, where each tuple represents a start and end date for a specific time period.
def combine_dis_con(df_dis, df_con, area, parameter, time_periods):
    df_mean_list = []
    for start, end in time_periods:
        # Select discrete data in time frame and managed areas
        df_dis_filtered = df_dis[(df_dis["ParameterName"] == parameter) & (df_dis["ManagedAreaName"] == area)]
        df_dis_filtered = df_dis_filtered[(df_dis_filtered['timestamp'] > start) & (df_dis_filtered['timestamp'] < end)]
        df_dis_mean = df_dis_filtered.groupby(['Latitude_DD', 'Longitude_DD', "ParameterName", "ManagedAreaName"])["ResultValue"].agg("mean").reset_index()

        # Select continuous data in time frame and managed areas
        df_con_filtered = df_con[(df_con["ParameterName"] == parameter) & (df_con["ManagedAreaName"] == area)]
        df_con_filtered = df_con_filtered[(df_con_filtered['timestamp'] > start) & (df_con_filtered['timestamp'] < end)]
        df_con_mean = df_con_filtered.groupby(['Latitude_DD', 'Longitude_DD', "ParameterName", "ManagedAreaName"])["ResultValue"].agg("mean").reset_index()

        # Concatenate dry and wet dataframes
        df_mean = pd.concat([df_dis_mean, df_con_mean], ignore_index=True)
        df_mean_list.append(df_mean)

    # Concatenate mean dataframes for all time periods
    df_mean = pd.concat(df_mean_list, ignore_index=True)
    gdf = gpd.GeoDataFrame(df_mean, geometry=gpd.points_from_xy(df_mean.Longitude_DD, df_mean.Latitude_DD), crs="EPSG:4326")

    return df_mean, gdf

# Function to generate shapefiles of interested areas and parameters
def create_shapefiles(dfDis, dfCon, areas, params, time_periods_dry, time_periods_wet, WQ_path, SpatialRef):
    
    # Write the filename with shortened area names and parameter names
    area_shortnames = {'Gasparilla Sound-Charlotte Harbor Aquatic Preserve': 'Charlotte_Harbor',
                       'Big Bend Seagrasses Aquatic Preserve': 'Big_Bend',
                       'Estero Bay Aquatic Preserve': 'Estero_Bay'}
    param_shortnames = {'Salinity': 'SA',
                        'Total Nitrogen': 'TN',
                        'Dissolved Oxygen': 'DO'}
    
    # Loop through areas, parameters, and time periods to create shapefiles
    for area in areas:
        area_shortname = area_shortnames.get(area, area)
        
        for parameter in params:
            param_shortname = param_shortnames.get(parameter, parameter)
            
            # Loop through dry season time periods and create shapefiles
            for i, time_period in enumerate(time_periods_dry):
                dry_start, dry_end = time_period
                
            # Combine discrete and continuous data and calculate mean
                dfDryMean, gdfDry = combine_dis_con(dfDis, dfCon, area, parameter, [(dry_start, dry_end)])
        
            # Create shapefile if there is data, print message otherwise
                if not dfDryMean.empty:
                    filename = f"{param_shortname}_Dry_2016_2018_{area_shortname}.shp"
                    gdfDry.to_crs(int(SpatialRef)).to_file(WQ_path+filename, driver='ESRI Shapefile', crs="EPSG:"+SpatialRef)
                else:
                    print(f"No data available for {parameter} in {area} during dry season period.")
            
            # Loop through wet season time periods and create shapefiles        
            for i, time_period in enumerate(time_periods_wet):
                wet_start, wet_end = time_period
                dfWetMean, gdfWet = combine_dis_con(dfDis, dfCon, area, parameter, [(wet_start, wet_end)])
                if not dfWetMean.empty:
                    filename = f"{param_shortname}_Wet_2016_2018_{area_shortname}.shp"
                    gdfWet.to_crs(int(SpatialRef)).to_file(WQ_path+filename, driver='ESRI Shapefile', crs="EPSG:"+SpatialRef)
                else:
                    print(f"No data available for {parameter} in {area} during wet season period.")

# Funtion to extract raster values at points            
def extract_raster_values(shapefile_path, raster_path, column_name):
    gdf = gpd.read_file(shapefile_path)
    # Get the parameter name from the GeoDataFrame
    parameter_name = gdf.iloc[0]['ParameterN']
    with rio.open(raster_path) as src:
        gdf = gdf.to_crs(src.crs)
        raster_data = src.read(1)
        transform = src.transform
        row_index, col_index = rio.transform.rowcol(transform, gdf['geometry'].x, gdf['geometry'].y)
        gdf[column_name] = raster_data[row_index, col_index]
        # Rename the column with the parameter name
        gdf = gdf.rename(columns={"ResultValu": parameter_name})
    return gdf

# Function to catch specific exceptions of file name
def parse_file_name(file_name):
    # Define regular expression pattern of the file name
    # For example,DO_Dry_2016_2018_Charlotte_Harbor
    pattern = r"(\w+)_(\w+)_(\d{4})_(\d{4})_(\w+)"
    try:
        parameter, season, year_from, year_to, area = re.search(pattern, file_name).groups()
        return f"{year_from}-{year_to}", season
    except AttributeError:
        return None, None
    
# Function to add year and season to csv file
def add_year_season_to_gdf(gdf, shapefile_path):
    year, season = parse_file_name(Path(shapefile_path).stem)
    if year is not None and season is not None:
        gdf["Year"] = f"{year}"
        gdf["Season"] = season
    return gdf

# Function to ensure no csv files exists in the folder
def remove_csv_files(dir_name):
    for item in os.listdir(dir_name):
        if item.endswith(".csv"):
            os.remove(os.path.join(dir_name, item))
            
def remove_result_files(dir_name):
    for item in os.listdir(dir_name):
        if item.startswith("regression_result"):
            os.remove(os.path.join(dir_name, item))

# Function to generate all csv files with water depth values and LDI.
def generate_csv_files(shapefile_folder, raster_path, column_name):
    # Iterate over all files in the specified folder
    for file in os.listdir(shapefile_folder):
        # Only process files with .shp extension
        if file.endswith(".shp"):
            shapefile_path = os.path.join(shapefile_folder, file)
            # Build the full file path for the shapefile and the corresponding output CSV file
            output_csv_path = shapefile_path.replace(".shp", ".csv")
            # Check if the column already exists
            if os.path.exists(output_csv_path):
                gdf_existing = pd.read_csv(output_csv_path)
                if column_name in gdf_existing.columns:
                    print(f"Column '{column_name}' already exists in {output_csv_path}")
                    continue
            # If the column does not exist, extract the raster values from the shapefile and append them to the existing CSV file
                gdf = extract_raster_values(shapefile_path, raster_path, column_name)
                gdf_existing[column_name] = gdf[column_name]
                gdf_existing.to_csv(output_csv_path, index=False)
                print(f"Appended {column_name} to {output_csv_path}")
            else:
            # If the output CSV file does not exist, extract the raster values from the shapefile and create a new CSV file
                gdf = extract_raster_values(shapefile_path, raster_path, column_name)
                add_year_season_to_gdf(gdf, shapefile_path)
                gdf.to_csv(output_csv_path, index=False)
                print(f"Created {output_csv_path}")

# Funtion to calculate linear regression
def ols_regression(csv_path, variable_name):
    df = pd.read_csv(csv_path)
    area = find_between_r(csv_path, '2016_2018_','.csv')
    paraName = df.columns[4]  # the specific parameter name is in the 5rd column
    df.drop(df[df.loc[:, paraName] >1000].index,inplace=True) #remove outliers
    column1 = df.loc[:, variable_name]
    column2 = df.loc[:, paraName]
    X = sm.add_constant(column1)
    model = sm.OLS(column2, X).fit()
    # Only get the coefficient, p-values, and r square
    return model.params[1], model.pvalues[1], model.rsquared, column1, column2, area

# Function to find a string between two strings.
def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

# Define a funtion to calculate Pearson correlation
def calculate_pearson_correlation(csv_path, variable_name):
    df = pd.read_csv(csv_path)
    paraName = df.columns[4]  # the specific parameter name is in the 5rd column
    column1 = df.loc[:, variable_name] # dependent variable
    column2 = df.loc[:, paraName] # independent variable
    corr, p_value = pearsonr(column1, column2)

    return corr, p_value, column2, column1

# Function to check the p-values
def check_p_value(p_value):
    if float(p_value) < 0.01:
        return "{:.3f} **".format(float(p_value))
    elif float(p_value) < 0.05:
        return "{:.3f} *".format(float(p_value))
    else:
        return "{:.3f}".format(float(p_value))

# Function to organize coefficient, p-values, and R-square from linear regression results to dataframe
def print_regression_result(csv_folder, variable_name, outpath):
    results = []
    outputs = []
    for file in sorted(os.listdir(csv_folder)):
        if file.endswith(".csv"):
            csv_path = os.path.join(csv_folder, file)
            coeff, p_value, adj_rsq, ind_var, d_var, area = ols_regression(csv_path,variable_name)
            df = pd.read_csv(csv_path)
            dependent = df.columns[4] # the specific parameter name is in the 5rd column
            independent = df[variable_name].name
            year = df.iloc[0, 7]
            season = df.iloc[0, 8]
            p_value_str = check_p_value(p_value)
            results.append((dependent, independent, area, year, season, coeff, adj_rsq, p_value_str))
            outputs.append((dependent, independent, area, year, season, coeff, adj_rsq, p_value_str, ind_var, d_var))
    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Dependent', 'Independent', 'Area', 'Year', 'Season', 'Coefficient', 'R-squared', 'p-value'])
    outputs_df = pd.DataFrame(outputs, columns=['Dependent', 'Independent', 'Area', 'Year', 'Season', 'Coefficient', 'R-squared', 'p-value','ind_values','d_values'])
    # Write the results to an Excel file
    results_df.to_excel(outpath, index=False)
    print(f"Regression results written to {outpath}")
    return results_df, outputs_df

# Function to orgarnize the Pearson correlation results to dataframe
def print_pearson_result(csv_folder, variable_name, output_path):
    results = []
    all_output_ls = []
    for file in sorted(os.listdir(csv_folder)):
        if file.endswith(".csv"):
            csv_path = os.path.join(csv_folder, file)
            corr, p_value, d_var, ind_var = calculate_pearson_correlation(csv_path,variable_name)
            df = pd.read_csv(csv_path)
            dependent = df.columns[4] # the specific parameter name is in the 5rd column
            independent = df[variable_name].name
            year = df.iloc[0, 7]
            season = df.iloc[0, 8]
            p_value_str = check_p_value(p_value)
            results.append((dependent, independent, year, season, corr, p_value_str))
            all_output_ls.append([dependent, independent, year, season, corr, p_value_str,list(ind_var),list(d_var)])
            # Create plots with regression line

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Dependent', 'Independent', 'Year', 'Season', 'Pearson correlation coefficient', 'p-value'])
    all_output_df = pd.DataFrame(all_output_ls, columns=['Dependent', 'Independent', 'Year', 'Season', 'Pearson correlation coefficient', 'p-value','ind_values','d_values'])
    # Generate output path
    # Write the results to an Excel file
    results_df.to_excel(output_path, index=False)
    print(f"Pearson results written to {output_path}")
    return results_df, all_output_df

# Function to plot scatter plots and regression line
def plot_regression(output_df):
    ncol  = 4
    nrow = int(output_df.shape[0] / ncol) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(28, 28))
    n = 0
    for index, row in output_df.iterrows():
        r, c  = int(n/ncol), n % ncol

        ind_var, d_var = np.array(row['ind_values']), np.array(row['d_values'])
        axes[r,c].plot(ind_var, d_var, 'o')
        m, b = np.polyfit(ind_var, d_var, 1)
        axes[r,c].plot(ind_var, m*ind_var+b, color = 'red')
        axes[r,c].set_title(' {} {} season \n R2 = {}, p = {}'.format(row['Year'], row['Season'], round(float(row['R-squared']),3), row['p-value']))
        axes[r,c].set_xlabel(row['Dependent'])
        axes[r,c].set_ylabel(row['Independent'])
        
        plt.subplots_adjust(hspace=0.3)
        n = n+1