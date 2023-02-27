# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:45:30 2022

@author: qiangy
"""

import numpy as np
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    
    #---------- Select managed areas, and columns -----------------
    # df2 = df.loc[df["ManagedAreaName"].isin(["Guana Tolomato Matanzas National Estuarine Research Reserve",
    #                             "Biscayne Bay Aquatic Preserve",
    #                             "Estero Bay Aquatic Preserve",
    #                             "Gasparilla Sound-Charlotte Harbor Aquatic Preserve",
    #                             "Big Bend Seagrasses Aquatic Preserve"])]
    df2 = df
    df2 = df2[['RowID','ParameterName','ParameterUnits','ProgramLocationID','ActivityType','ManagedAreaName',
               'SampleDate','Year','Month','ResultValue','ValueQualifier','Latitude_DD','Longitude_DD']]
    df2 = df2.loc[df2["ParameterName"].isin(["Salinity","Total Nitrogen","Dissolved Oxygen","Turbidity","Secchi Depth"])]
    df2["timestamp"]=  pd.to_datetime(df2['SampleDate'])

    #---------- remove outliers -----------------
    # Remove total nitrogen outliers (>100)
    df2.drop(df2[(df2['ParameterName'] == 'Total Nitrogen') & 
        (df2['ResultValue'] > 10)].index,inplace=True)
    
    # Remove a single measurement in 1996-07-22 (RowID: 1582917)
    df2.drop(df2[df2['RowID'] == 1582917].index, inplace=True)
    
    # Remove turbidity outliers (>25)
    df2.drop(df2[(df2['ParameterName'] == 'Turbidity') & 
        (df2['ResultValue'] > 25)].index, inplace=True)
    
    # Remove Secchi Depth before 1995 (117 records)
    df2.drop(df2[(df2['ParameterName'] == 'Secchi Depth') & 
        (df2['Year'] < 1995)].index, inplace=True)
    
    #---------- create list of parameters, areas, values and dictionaries -----------------
    listPara = ["Salinity","Total Nitrogen","Dissolved Oxygen","Turbidity","Secchi Depth"] 
    #listArea = ["Guana Tolomato Matanzas National Estuarine Research Reserve",
    #                             "Biscayne Bay Aquatic Preserve",
    #                             "Estero Bay Aquatic Preserve",
    #                             "Gasparilla Sound-Charlotte Harbor Aquatic Preserve",
    #                             "Big Bend Seagrasses Aquatic Preserve" ]
    listArea = df["ManagedAreaName"].unique()
    listValue = ["count", "mean","max","min","std"]
    dictUnits = {"Salinity":"ppt","Total Nitrogen": "mg/L","Dissolved Oxygen": "mg/L","Turbidity": "NTU", "Secchi Depth": "m"}
    dictMonth = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",
                 9:"September",10:"October",11:"November",12:"December"}
    dictArea = {'Gasparilla Sound-Charlotte Harbor Aquatic Preserve':'Charlotte Harbor','Big Bend Seagrasses Aquatic Preserve':'Big Bend',
                'Guana Tolomato Matanzas National Estuarine Research Reserve':'GTM Reserve','Estero Bay Aquatic Preserve':'Estero Bay',
                'Biscayne Bay Aquatic Preserve':'Biscayne Bay','Matlacha Pass Aquatic Preserve':'Matlacha Pass AP',
                'Lemon Bay Aquatic Preserve':'Lemon Bay','Cape Haze Aquatic Preserve':'Cape Haze AP','Pine Island Sound Aquatic Preserve':'Pine Island'}
    
    return df2, listPara, listArea, listValue, dictUnits, dictMonth, dictArea