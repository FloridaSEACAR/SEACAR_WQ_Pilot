# -*- coding: utf-8 -*-
"""
@author: cong

"""
import pandas as pd
import geopandas as gpd
import os
from shapely.geometry import Point
import matplotlib.pyplot as plt
import glob
import contextily as cx
import warnings
warnings.filterwarnings('ignore')
import math
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import shutil
import numpy.ma as ma
from rasterio.plot import show, plotting_extent
from collections import defaultdict
import matplotlib.gridspec as gridspec



# Predefined abbreviation of managed area names and parameter names
area_shortnames = {
    'Gasparilla Sound-Charlotte Harbor Aquatic Preserve': 'GSCHAP',
    'Estero Bay Aquatic Preserve': 'EBAP',
    'Big Bend Seagrasses Aquatic Preserve': 'BBSAP',
    'Biscayne Bay Aquatic Preserve': 'BBAP',
    'Guana Tolomato Matanzas National Estuarine Research Reserve':'GTMNERR'
}

param_shortnames = {
    'Salinity': 'Sal',
    'Total Nitrogen': 'TN',
    'Dissolved Oxygen': 'DO',
    'Turbidity':'Turb',
    'Secchi Depth':'Secchi'
}

# Function to create shapefiles, preparing for ArcPy Kernel Density method
def create_shp(df, managed_area_names, parameter_names, output_folder):
    for area in managed_area_names:
        if area not in area_shortnames:
            print(f"No managed area found with name: {area}")
            continue

        area_params = []
        for param in parameter_names:
            if param not in param_shortnames:
                print(f"No parameter found with name: {param}")
                continue

            # Filter data for specific area and parameter
            df_filtered = df[(df['ManagedAreaName'] == area) & (df['ParameterName'] == param)]

            # Check if there is data for the specific area and parameter
            if df_filtered.empty:
                print(f"No data found for area: {area} and parameter: {param}")
                continue
            
            # Convert DataFrame to GeoDataFrame with correct initial CRS
            geometry = [Point(xy) for xy in zip(df_filtered['Longitude_DD'], df_filtered['Latitude_DD'])]
            gdf = gpd.GeoDataFrame(df_filtered, crs='EPSG:4326', geometry=geometry)
            
            # Reproject to desired CRS
            gdf = gdf.to_crs('EPSG:3086')
            
            # Generate file name and save GeoDataFrame as Shapefile
            filename = os.path.join(output_folder, 'SHP_' + area_shortnames[area] + '_' + param_shortnames[param] + '.shp')
            gdf.to_file(filename)
            
            print(f"Shapefile for {area_shortnames[area]}: {param_shortnames[param]} has been saved as {os.path.basename(filename)}")
            
            # Print number of points in the Shapefile
            print(f"The Shapefile contains {len(gdf)} points.")
            
            # Add the parameter to the area_params list
            area_params.append(param_shortnames[param])
        
        if area_params:
            params_str = ', '.join(area_params)
            print(f"***{area}***: {params_str}")



# Function to empty folder
def delete_all_files(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

            

# Function to place files in separate folders by area name
def classify_files_by_study_area(src_folder):
    parent_folder = os.path.dirname(src_folder)  
    new_parent_folder = os.path.join(parent_folder, "Gap_SHP_All_byAreas")

    # If the new parent folder exists, remove it and then create it again
    if os.path.exists(new_parent_folder):
        shutil.rmtree(new_parent_folder)
    os.makedirs(new_parent_folder)

    # Iterate over all files in the source directory
    for filename in os.listdir(src_folder):
        if '_' in filename:  # Check if the file name matches the expected pattern
            study_area = filename.split('_')[1]  # Extract the study area from the file name
            
            # Generate the path for the new folder
            destination_folder = os.path.join(new_parent_folder, study_area + "_SHP_All")  

            # Only if the destination folder does not exist, create it
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            # Copy the file to the new folder, leaving the original file unchanged
            shutil.copy(os.path.join(src_folder, filename), destination_folder)



# Function to rescale the standard error prediction
def rescale_tif_files(folder_path, subfolder, year_start, year_end, output_folder_path):
    # find subfolder
    specific_path = os.path.join(folder_path, subfolder)
    
    # create output folder if not exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # find tif
    tif_files = glob.glob(os.path.join(specific_path, '*.tif'))
    
    # generate list of years
    year_list = list(range(year_start, year_end + 1))
    
    # filter time period
    tif_files_filtered = [f for f in tif_files if int(re.search(r'\d{4}', os.path.basename(f)).group()) in year_list]

    # sort files by year
    tif_files_filtered.sort(key=lambda f: int(re.search(r'\d{4}', os.path.basename(f)).group()))

    # process filtered tif
    for file in tif_files_filtered:
        with rasterio.open(file) as src:
            # read tif
            data = src.read(1).astype(np.float64)

            # mask nodata
            masked_data = np.ma.masked_equal(data, src.nodata)

            # calculate min and max of valid data
            valid_min = masked_data.min()
            valid_max = masked_data.max()

            # avoid division by zero
            if valid_max != valid_min:
                rescaled_data = (masked_data - valid_min) / (valid_max - valid_min)
            else:
                rescaled_data = np.zeros_like(masked_data)

            # save rescaled tif file
            new_file_name = f'rescaled_{os.path.basename(file)}'
            new_file_path = os.path.join(output_folder_path, new_file_name)
            with rasterio.open(new_file_path, 'w', driver='GTiff',
                               height=rescaled_data.shape[0], 
                               width=rescaled_data.shape[1], 
                               count=1, dtype='float32',
                               crs=src.crs, 
                               transform=src.transform,
                               nodata=src.nodata) as new_dataset:
                new_dataset.write(rescaled_data.filled(src.nodata).astype('float32'), 1)
        print(f"Rescaled file: {os.path.basename(file)}")



# Function to calculate the average of tif files
def calculate_average_tif(folder_path, output_folder_path):
    # Get all files in the directory
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.tif')]
    
    # Get unique parameters and study areas
    parameters = set([os.path.splitext(os.path.basename(f).split('_')[2])[0] for f in all_files])
    study_areas = set([os.path.splitext(os.path.basename(f).split('_')[1])[0] for f in all_files])
    
    for param in parameters:
        for area in study_areas:
            # Get files for this parameter and study area
            param_files = [f for f in all_files if os.path.splitext(os.path.basename(f).split('_')[2])[0] == param and os.path.splitext(os.path.basename(f).split('_')[1])[0] == area]
            num_files = len(param_files)
            print(f"Found {num_files} files for parameter {param} in {area}")
            
            if not param_files:
                print(f"No files found for parameter {param} in {area}. Skipping...")
                continue
            
            # Read first file to get metadata
            with rasterio.open(param_files[0]) as src0:
                meta = src0.meta
                data_shape = src0.shape
                nodata = src0.nodata
            
            # Initialize sum data and count
            sum_data = ma.zeros(data_shape, dtype=np.float64)
            count = ma.zeros(data_shape, dtype=np.uint16)
            
            # Read and accumulate data
            for file in param_files:
                with rasterio.open(file) as src:
                    # Read data and adjust size if necessary
                    data = src.read(1).astype(np.float64)

                    # If data shape is not consistent with the first file, pad or crop the array
                    if data.shape != data_shape:
                        diff = (data_shape[0] - data.shape[0], data_shape[1] - data.shape[1])
                        if diff[0] >= 0 and diff[1] >= 0:
                            # pad
                            data = np.pad(data, ((0, diff[0]), (0, diff[1])), mode='constant', constant_values=nodata)
                        else:
                            # crop
                            data = data[:data_shape[0], :data_shape[1]]

                    # Mask nodata values
                    if nodata is not None:
                        data = ma.masked_values(data, nodata)

                    # Accumulate data and count valid values
                    sum_data += data.filled(0)
                    count += ~data.mask
            
            # Calculate average
            avg_data = ma.array(sum_data / count, mask=count==0)
            avg_data[count==0] = nodata
            
            # Update metadata for the output file
            meta.update(count=1, dtype='float64', nodata=nodata)
            
            # Write to file
            output_file = os.path.join(output_folder_path, f"SEP_{area}_{param}.tif")
            with rasterio.open(output_file, 'w', **meta) as dst:
                dst.write(avg_data.data, 1)




# Function to merge all KDE and rescaled average SEP tif files into one folder to plot together
def merge_folders(folders, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_count = 0  # Initialize a counter for the files

    for folder in folders:
        # Check if the folder exists
        if os.path.exists(folder):
            # Walk through all files in the folder
            for path, dirs, files in os.walk(folder):
                for filename in files:
                    # Increment the file counter
                    file_count += 1
                    # Construct the file's full path
                    filepath = os.path.join(path, filename)
                    # Construct the destination path
                    destpath = os.path.join(output_folder, filepath.replace(folder + os.sep, ''))
                    # Ensure the destination directory exists
                    os.makedirs(os.path.dirname(destpath), exist_ok=True)
                    # Copy the file
                    shutil.copy2(filepath, destpath)
        else:
            print(f"Folder {folder} does not exist and will be skipped.")
    
    print("All files have been merged into one folder.")
    print(f"A total of {file_count} files have been copied.")  # Print the total file count



# Function to plot KDE and SEP maps in pairs
def plot_pairs(tif_folder, area_name, boundary, width, height):
    area_name_abbrev = area_shortnames[area_name]
    # Get the list of files starting with a specific prefix and ending with .tif, and store them in a dictionary (with the parameter as the key)
    kde_files = {f.split('_')[2]: f for f in os.listdir(tif_folder) if f.startswith('KDE_' + area_name_abbrev + '_') and f.endswith('.tif')}
    sep_files = {f.split('_')[2]: f for f in os.listdir(tif_folder) if f.startswith('SEP_' + area_name_abbrev + '_') and f.endswith('.tif')}

    # Get the sorted list of all parameters
    all_params = sorted(set(list(kde_files.keys()) + list(sep_files.keys())))

    total_plots = len(all_params)
    ncols = len(all_params)
    nrows = 2

    fig, axs = plt.subplots(nrows, ncols, figsize=(width, nrows * height), sharex=True, sharey=True)

    # Iterate over all parameters and plot the image pairs
    for i, param in enumerate(all_params):
        # Plot KDE image pairs
        if param in kde_files:
            plot_file(fig, tif_folder, kde_files[param], 'KDE', boundary, area_name, axs[0, i])
        else:
            axs[0, i].axis('off')
        # Plot SEP image pairs
        if param in sep_files:
            plot_file(fig, tif_folder, sep_files[param], 'SEP', boundary, area_name, axs[1, i])
        else:
            axs[1, i].axis('off')

    plt.tight_layout(h_pad=0.5, w_pad=10)
    plt.show()

    

# Define a function to plot a single image pair
def plot_file(fig, tif_folder, file_name, file_type, boundary, area_name, ax):
    area_name_abbrev = area_shortnames[area_name]
    # Parse the file name to extract the prefix, area, and parameter information
    prefix, area, param = file_name.split('_')
    param = param.replace('.tif', '')
    full_file_path = os.path.join(tif_folder, file_name)

    with rasterio.open(full_file_path) as src:
        data = src.read(1)
        extent = (src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        cx.add_basemap(ax=ax, source=cx.providers.Stamen.TonerLite, crs=src.crs)

        boundary_plot = boundary[boundary['LONG_NAME'] == area_name]
        boundary_plot = boundary_plot.to_crs(src.crs)
        boundary_plot.plot(ax=ax, color='none', edgecolor='blue')

        nodata = src.nodata
        data[data == nodata] = np.nan

        data = np.ma.masked_invalid(data)
        cmap = 'YlOrRd' if file_type == 'KDE' else 'RdYlGn_r'
        im = ax.imshow(data, cmap=cmap, extent=extent, origin="upper", alpha=1)
        
        # Add colorbar and set its title
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cax.tick_params(labelsize=55)
        #cbar.set_label(file_type, size=20)
        
        # Set title
        ax.set_title(f"{file_type} - {area_name_abbrev} - {param}", fontdict={'fontsize': 60, 'fontweight': 'bold'}, pad=25)



# Function to assign season for water quality data
def assign_season(dfAll_orig, season_table, output_file):
    dfAll_orig = dfAll_orig.copy()
    season_table = season_table.copy()

    # Convert 'timestamp' in dfAll_orig and 's_start' and 's_end' in season_table to datetime format
    dfAll_orig['timestamp'] = pd.to_datetime(dfAll_orig['timestamp']).dt.date
    season_table['s_start'] = pd.to_datetime(season_table['s_start'], format='%m/%d/%Y').dt.date
    season_table['s_end'] = pd.to_datetime(season_table['s_end'], format='%m/%d/%Y').dt.date

    # Create a new column in dfAll_orig to store the season information, initialize it to None
    dfAll_orig['season'] = None

    # Counters for match and no match found
    match_counter = 0
    no_match_counter = 0

    # Traverse each row in dfAll_orig
    for index, row in dfAll_orig.iterrows():
        # Get the corresponding year and study area for this row
        year = row['Year']
        area = row['ManagedAreaName']
        timestamp = row['timestamp']

        # Find the corresponding records in season_table
        matching_rows = season_table[(season_table['ma'] == area) & 
                                     ((season_table['st_Year'] == year) | 
                                      (season_table['st_Year'] == year + 1) | 
                                      (season_table['st_Year'] == year - 1))]

        # Traverse these records and find the interval that includes the timestamp
        for _, match in matching_rows.iterrows():
            if match['s_start'] <= timestamp <= match['s_end']:
                # Assign the matched season to dfAll_orig
                dfAll_orig.at[index, 'season'] = match['season']
                match_counter += 1
                break
        else:
            # If no match found
            if no_match_counter < 10:
                print(f"No matching season found for index {index}, year {year}, area {area}, timestamp {timestamp}")
                ma_in_season_table = area in season_table['ma'].unique()
                year_in_season_table = year in season_table['st_Year'].unique()
                print(f"The ManagedAreaName {area} exists in season table: {ma_in_season_table}")
                print(f"The Year {year} exists in season table: {year_in_season_table}")
                no_match_counter += 1

    print(f"Total matched rows: {match_counter}")
    print(f"Total unmatched rows: {no_match_counter}")
    
    # Save the DataFrame to a CSV file
    dfAll_orig.to_csv(output_file, index=False)



# Function to create shapefiles, preparing for ArcPy Kernel Density method
def create_shp_season(df, managed_area_names, parameter_names, output_folder):
    for area in managed_area_names:
        if area not in area_shortnames:
            print(f"No managed area found with name: {area}")
            continue

        area_params = []
        for param in parameter_names:
            if param not in param_shortnames:
                print(f"No parameter found with name: {param}")
                continue

            # Filter data for specific area and parameter
            df_filtered = df[(df['ManagedAreaName'] == area) & (df['ParameterName'] == param)]

            # Check if there is data for the specific area and parameter
            if df_filtered.empty:
                print(f"No data found for area: {area} and parameter: {param}")
                continue
            
            # Convert DataFrame to GeoDataFrame with correct initial CRS
            geometry = [Point(xy) for xy in zip(df_filtered['Longitude_DD'], df_filtered['Latitude_DD'])]
            gdf = gpd.GeoDataFrame(df_filtered, crs='EPSG:4326', geometry=geometry)
            
            # Reproject to desired CRS
            gdf = gdf.to_crs('EPSG:3086')
            
            # Convert 'SampleDate' from datetime to string
            gdf['SampleDate'] = gdf['SampleDate'].astype(str)
            gdf['timestamp'] = gdf['timestamp'].astype(str)

            
            # Generate file name and save GeoDataFrame as Shapefile
            filename = os.path.join(output_folder, 'SHP_' + area_shortnames[area] + '_' + param_shortnames[param] + '.shp')
            gdf.to_file(filename)
            
            print(f"Shapefile for {area_shortnames[area]}: {param_shortnames[param]} has been saved as {os.path.basename(filename)}")
            
            # Print number of points in the Shapefile
            print(f"The Shapefile contains {len(gdf)} points.")
            
            # Add the parameter to the area_params list
            area_params.append(param_shortnames[param])
        
        if area_params:
            params_str = ', '.join(area_params)
            print(f"***{area}***: {params_str}")



# Function to place files in separate folders by area name
def classify_files_by_study_area(src_folder):
    parent_folder = os.path.dirname(src_folder)  

    # Extract the season from the source folder name
    season = os.path.basename(src_folder).split('_')[2] if len(os.path.basename(src_folder).split('_')) >= 3 else None
    new_parent_folder = os.path.join(parent_folder, "Gap_SHP_All_Seasons_byAreas")
    
    # Ensure the new parent folder exists
    if not os.path.exists(new_parent_folder):
        os.makedirs(new_parent_folder)

    # Iterate over all files in the source directory
    for filename in os.listdir(src_folder):
        if '_' in filename:  # Check if the file name matches the expected pattern
            study_area = filename.split('_')[1]  # Extract the study area from the file name
            new_folder_name = f"Gap_SHP_{study_area}_{season}"  # Generate the new folder name
            destination_folder = os.path.join(new_parent_folder, new_folder_name)  # Generate the path for the new folder

            if not os.path.exists(destination_folder):  # If the destination folder does not exist, create it
                os.makedirs(destination_folder)

            # Copy the file to the new folder, leaving the original file unchanged
            shutil.copy(os.path.join(src_folder, filename), destination_folder)


# Function to calculate the average of tif files by season            
def calculate_average_tif_season(folder_path, output_folder_path):
    # Get all files in the directory
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.tif')]

    # Get unique parameters and seasons
    parameters_seasons_files = defaultdict(list)
    study_area_seasons_years = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))  # A nested dictionary to store years for each season and study area
    for f in all_files:
        basename_parts = os.path.basename(f).split('_')
        study_area = basename_parts[1]
        param = basename_parts[2]
        unit = basename_parts[3]
        year = basename_parts[4]
        season = basename_parts[5]

        study_area_seasons_years[study_area][param][season].add(year)
        parameters_seasons_files[(study_area, param, season)].append(f)

    image_count = 0  # Initialize a counter for the generated images

    for (study_area, param, season), param_files in parameters_seasons_files.items():
        num_files = len(param_files)

        # Read first file to get metadata
        with rasterio.open(param_files[0]) as src0:
            meta = src0.meta
            data_shape = src0.shape
            nodata = src0.nodata

        # Initialize sum data and count
        sum_data = ma.zeros(data_shape, dtype=np.float64)
        count = ma.zeros(data_shape, dtype=np.uint16)

        # Read and accumulate data
        for file in param_files:
                with rasterio.open(file) as src:
                    # Read data and adjust size if necessary
                    data = src.read(1).astype(np.float64)

                    # If data shape is not consistent with the first file, pad or crop the array
                    if data.shape != data_shape:
                        diff = (data_shape[0] - data.shape[0], data_shape[1] - data.shape[1])
                        if diff[0] >= 0 and diff[1] >= 0:
                            # pad
                            data = np.pad(data, ((0, diff[0]), (0, diff[1])), mode='constant', constant_values=nodata)
                        else:
                            # crop
                            data = data[:data_shape[0], :data_shape[1]]

                # Mask nodata values
                if nodata is not None:
                    data = ma.masked_values(data, nodata)

                # Accumulate data and count valid values
                sum_data += data.filled(0)
                count += ~data.mask

        # Calculate average
        avg_data = ma.array(sum_data / count, mask=count==0)
        avg_data[count==0] = nodata

        # Update metadata for the output file
        meta.update(count=1, dtype='float64', nodata=nodata)

        # Write to file
        output_file = os.path.join(output_folder_path, f"SEP_{study_area}_{param}_{season}.tif")
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(avg_data.data, 1)

        image_count += 1  # Increase the counter each time an image is generated

    # Print all study areas, parameters, seasons and their corresponding years
    for study_area, param_seasons in study_area_seasons_years.items():
        for param, seasons in param_seasons.items():
            for season, years in seasons.items():
                num_files_for_this_case = len(parameters_seasons_files[(study_area, param, season)])
                print(f"Area: {study_area}, Parameter: {param}, Season: {season} - Found {num_files_for_this_case} files, Years: {sorted(list(years))}")

    print(f"Total generated images: {image_count}")  # Print the total number of generated images



# Function to plot all parameters for all seasons in a study area at once
def plot_pairs_season(tif_folder, area_name, boundary, width, height):
    area_name_abbrev = area_shortnames[area_name]
    
    seasons = ['Winter','Spring', 'Summer', 'Fall']

    kde_files = {(f.split('_')[2], f.split('_')[3].replace('.tif', '')): f for f in os.listdir(tif_folder) if f.startswith('KDE_' + area_name_abbrev + '_') and f.endswith('.tif')}
    sep_files = {(f.split('_')[2], f.split('_')[3].replace('.tif', '')): f for f in os.listdir(tif_folder) if f.startswith('SEP_' + area_name_abbrev + '_') and f.endswith('.tif')}

    all_params = sorted(set([key[0] for key in list(kde_files.keys()) + list(sep_files.keys())]))

    # Sort and group files by parameter
    from itertools import groupby
    groups = groupby(all_params, key=lambda param: param)  # Group by parameter
 
    for parameter, group in groups:
        
        reversed_dict = {value: key for key, value in param_shortnames.items()}
        param_fullname = reversed_dict.get(parameter, parameter)
        print(f"———————— {param_fullname} ————————")  # Print the current parameter
        group_list = list(group)

        # Create subplots for this group
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(width, height))  # change figsize to fit your needs
        ax = ax.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.1)

        for j, season in enumerate(seasons):
            # Plot KDE image pairs
            if (parameter, season) in kde_files:
                plot_file_season(fig, tif_folder, kde_files[(parameter, season)], 'KDE', boundary, area_name, ax[j])
            else:
                ax[j].axis('off')
            
            # Plot SEP image pairs
            if (parameter, season) in sep_files:
                plot_file_season(fig, tif_folder, sep_files[(parameter, season)], 'SEP', boundary, area_name, ax[j + 4])
            else:
                ax[j + 4].axis('off')

        #plt.tight_layout()
        plt.show()

def plot_file_season(fig, tif_folder, file_name, file_type, boundary, area_name, ax):
    area_name_abbrev = area_shortnames[area_name]
    
    # Parse the file name to extract the prefix, area, parameter and season information
    prefix, area, param, season = file_name.split('_')
    season = season.replace('.tif', '')
    full_file_path = os.path.join(tif_folder, file_name)

    with rasterio.open(full_file_path) as src:
        data = src.read(1)
        extent = (src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        cx.add_basemap(ax=ax, source=cx.providers.Stamen.TonerLite, crs=src.crs)

        boundary_plot = boundary[boundary['LONG_NAME'] == area_name]
        boundary_plot = boundary_plot.to_crs(src.crs)
        boundary_plot.plot(ax=ax, color='none', edgecolor='blue')

        nodata = src.nodata
        data[data == nodata] = np.nan

        data = np.ma.masked_invalid(data)
        cmap = 'YlOrRd' if file_type == 'KDE' else 'RdYlGn_r'
        im = ax.imshow(data, cmap=cmap, extent=extent, origin="upper", alpha=1)
        
        # Add colorbar and set its title
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cax.tick_params(labelsize=25)
        
        # Set title, including the season information
        ax.set_title(f"{file_type} - {area_name_abbrev} - {param} - {season}", fontdict={'fontsize': 30, 'fontweight': 'bold'}, pad=20)


