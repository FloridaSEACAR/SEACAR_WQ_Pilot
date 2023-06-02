# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:35:25 2023

@author: qiangy
"""

import numpy as np
import pandas as pd
import geopandas as gpd

def preprocessDis(df,col_ls,para_ls):
    
    # ------- Select columns and parameters ------
    df = df[col_ls]
    df = df.loc[df["ParameterName"].isin(para_ls)]
    df["timestamp"]=  pd.to_datetime(df['SampleDate'])
    #---------- remove outliers -----------------
    # Remove total nitrogen outliers (>100)
    df.drop(df[(df['ParameterName'] == 'Total Nitrogen') & 
                     (df['ResultValue'] > 10)].index,inplace=True)

    # Remove a single measurement in 1996-07-22 (RowID: 1582917)
    df.drop(df[df['RowID'] == 1582917].index, inplace=True)

    # Remove turbidity outliers (>25)
    df.drop(df[(df['ParameterName'] == 'Turbidity') & 
                     (df['ResultValue'] > 25)].index, inplace=True)

    # Remove Secchi Depth before 1995 (117 records)
    df.drop(df[(df['ParameterName'] == 'Secchi Depth') & 
                     (df['Year'] < 1995)].index, inplace=True)
    
    return df

def preprocessCon(df,col_ls,para_ls):
    
    # ------- Select columns and parameters ------
    df = df[col_ls]
    df = df.loc[df["ParameterName"].isin(para_ls)]
    df["timestamp"]=  pd.to_datetime(df['SampleDate'])
    # ------- Select data during daytime ------
    df["Hour"]     = df.apply(lambda x:x["timestamp"].strftime("%H"), axis=1)
    df["Hour"]     = df["Hour"].astype(int)
    df             = df[(df["Hour"]>=8) & (df["Hour"]<=18)]
    #---------- remove outliers -----------------
    # Remove total nitrogen outliers (>100)
    df.drop(df[(df['ParameterName'] == 'Total Nitrogen') & 
                     (df['ResultValue'] > 10)].index,inplace=True)

    # Remove a single measurement in 1996-07-22 (RowID: 1582917)
    df.drop(df[df['RowID'] == 1582917].index, inplace=True)

    # Remove turbidity outliers (>25)
    df.drop(df[(df['ParameterName'] == 'Turbidity') & 
                     (df['ResultValue'] > 25)].index, inplace=True)

    # Remove Secchi Depth before 1995 (117 records)
    df.drop(df[(df['ParameterName'] == 'Secchi Depth') & 
                     (df['Year'] < 1995)].index, inplace=True)
    
    # Remove salinity outlier
    df.drop(df[(df['ParameterName'] == 'Salinity') & 
                     (df['ResultValue'] > 100)].index, inplace=True)
    
    # Remove total nitrogen outlier
    df.drop(df[(df['ParameterName'] == 'Dissolved Oxygen') & 
                     (df['ResultValue'] > 100)].index, inplace=True)
    
    return df

# Preprocess dis and con data and concatinate dis and con data
# The process include select daytime data points from con data
def preprocess(dis, con1, con2):
    # read csv files
    dfDis_orig = pd.read_csv(dis)
    dfCon1_orig = pd.read_csv(con1)
    dfCon2_orig = pd.read_csv(con2)
    
    # preset function parameters
    col_ls = ['RowID','ParameterName','ParameterUnits','ProgramLocationID','ActivityType','ManagedAreaName',
                   'SampleDate','Year','Month','ResultValue','ValueQualifier','Latitude_DD','Longitude_DD']
    para_ls = ["Salinity","Total Nitrogen","Dissolved Oxygen","Turbidity","Secchi Depth"]
        
    # preprocess and concatenate dataframes
    dfCon1 = preprocessCon(dfCon1_orig, col_ls, para_ls)
    dfCon2 = preprocessCon(dfCon2_orig, col_ls, para_ls)
    dfDis  = preprocessDis(dfDis_orig, col_ls, para_ls)
    dfCon  = pd.concat([dfCon1,dfCon2],ignore_index=True)
    return dfDis, dfCon


# Select data in specific area, time period, parameter, and season
# s_date,e_date needs to be in format mm/dd/yyyy format
def select_aggr_area_season(df_all, s_date, e_date, area, para):
    
    df_all = df_all[(df_all['ParameterName']==para)&
          (df_all['SampleDate']>pd.Timestamp(s_date).date())&
           (df_all['SampleDate']<pd.Timestamp(e_date).date())&
            (df_all['ManagedAreaName']==area)]
    
    df_mean = df_all.groupby(['Latitude_DD','Longitude_DD'])["ResultValue"].agg("mean").reset_index()    
    gdf = gpd.GeoDataFrame(df_mean, geometry = gpd.points_from_xy(df_mean.Longitude_DD, df_mean.Latitude_DD), crs="EPSG:4326")
    
    return df_mean, gdf


def combine_dis_con_dry(df_dis,df_con, year):
    
    year_start = str(int(year)-1)
    year_end   = str(year)
    dry_start,dry_end = ('11/01/'+year_start),('04/30/'+year_end)
    
    df_dis["timestamp"]=  pd.to_datetime(df_dis['SampleDate'])
    df_con["timestamp"]=  pd.to_datetime(df_con['SampleDate'])
    
    df_dis = df_dis[(df_dis['timestamp'] > dry_start)&(df_dis['timestamp'] < dry_end)]
    df_dis_mean = df_dis.groupby(['Latitude_DD','Longitude_DD',"ParameterName","ManagedAreaName"])["ResultValue"].agg("mean").reset_index()

    df_con = df_con[(df_con['timestamp'] > dry_start)&(df_con['timestamp'] < dry_end)]
    df_con_mean = df_con.groupby(['Latitude_DD','Longitude_DD',"ParameterName","ManagedAreaName"])["ResultValue"].agg("mean").reset_index()
   
    # Concatenate dry and wet dataframes
    df_mean = pd.concat([df_dis_mean,df_con_mean],ignore_index=True)
    gdf = gpd.GeoDataFrame(df_mean, geometry = gpd.points_from_xy(df_mean.Longitude_DD, df_mean.Latitude_DD), crs="EPSG:4326")
    
    return df_mean, gdf

def combine_dis_con_wet(df_dis,df_con, year):
    
    year_start = str(int(year)-1)
    year_end   = str(year)
    wet_start,wet_end = ('05/01/'+year_end),('10/31/'+year_end)
    
    df_dis["timestamp"]=  pd.to_datetime(df_dis['SampleDate'])
    df_con["timestamp"]=  pd.to_datetime(df_con['SampleDate'])

    df_dis = df_dis[(df_dis['timestamp'] > wet_start)&(df_dis['timestamp'] < wet_end)]
    df_dis_mean = df_dis.groupby(['Latitude_DD','Longitude_DD',"ParameterName","ManagedAreaName"])["ResultValue"].agg("mean").reset_index()

    df_con = df_con[(df_con['timestamp'] > wet_start)&(df_con['timestamp'] < wet_end)]
    df_con_mean = df_con.groupby(['Latitude_DD','Longitude_DD',"ParameterName","ManagedAreaName"])["ResultValue"].agg("mean").reset_index()
   
    # Concatenate dry and wet dataframes
    df_mean = pd.concat([df_dis_mean,df_con_mean],ignore_index=True)
    gdf = gpd.GeoDataFrame(df_mean, geometry = gpd.points_from_xy(df_mean.Longitude_DD, df_mean.Latitude_DD), crs="EPSG:4326")
    
    return df_mean, gdf
