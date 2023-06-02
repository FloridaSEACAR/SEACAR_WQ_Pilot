import numpy as np
import pandas as pd
import geopandas as gpd
import arcpy
import time,sys
import arcgisscripting
from arcpy.sa import *
arcpy.env.overwriteOutput = True


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

def extract_val_result(inLayer, index):
    cvResult = arcpy.CrossValidation_ga(inLayer)
    Stat = pd.DataFrame(
                {
                  "meanError": round(float(cvResult.meanError),4),
                  "meanStandardizedError": round(float(cvResult.meanStandardized),4),
                  "rootMeanSquareError": round(float(cvResult.rootMeanSquare),4)
                                              },index=[index])
    return Stat    

def interpolation(method, input_point, out_raster, 
                         z_field, out_ga_layer, extent, mask, ga_to_raster,
                         in_explanatory_rasters = None):
    
    start_time = time.time()
    print("Start the interpolation with the {} method".format(method.upper()))
    smooth_r, spatialref, c_size, parProFactor = 10000, 3086, 30, "80%"

# ---------------------------- IDW ---------------------------------
    if   method == "idw":
        with arcpy.EnvManager(extent = extent, mask = mask,outputCoordinateSystem = arcpy.SpatialReference(spatialref), cellSize = c_size, parallelProcessingFactor = parProFactor):

                 arcpy.ga.IDW(in_features = input_point, 
                 z_field = z_field, 
#                This layer is not generated  
                 out_ga_layer = out_ga_layer,
                 out_raster   = out_raster
                )
        
        ValStat = extract_val_result(out_ga_layer, method.upper())
        
        print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
        
        return out_raster, ValStat
                
# ---------------------- Ordinary Kriging ---------------------------
    elif method == "ok":
        out_cv_table = out_raster.replace('.tif','_table')
        with arcpy.EnvManager(extent = extent, mask = mask,outputCoordinateSystem = arcpy.SpatialReference(spatialref), cellSize = c_size, parallelProcessingFactor = parProFactor):
            ok_out = arcpy.ga.ExploratoryInterpolation(in_features = input_point, value_field = z_field, 
                                                   out_cv_table = out_cv_table, out_geostat_layer = out_ga_layer, 
                                                   interp_methods = ['ORDINARY_KRIGING'], comparison_method = 'SINGLE', 
                                                   criterion = 'ACCURACY')
            arcpy.conversion.ExportTable(out_cv_table, out_cv_table + '.csv')
            ValStat = pd.read_csv(out_cv_table + '.csv')
            ValStat = ValStat[ValStat['DESCR'] == 'Ordinary Kriging – Default'][['DESCR','ME','ME_STD','RMSE']].rename(columns = {"RMSE": "rootMeanSquareError", "ME": "meanError",'ME_STD':'meanStandardizedError'})
            os.remove(out_cv_table + '.csv'+'.xml')
            os.remove(out_cv_table + '.csv')
            
            ValStat['DESCR'] = method.upper()
            ValStat = ValStat.set_index('DESCR')
            ValStat.index.name= None
            
            arcpy.GALayerToRasters_ga(in_geostat_layer = out_ga_layer, out_raster = out_raster, output_type = "PREDICTION", cell_size = c_size)
        
            return out_raster, ValStat

        
# ---------------------- Empirical Bayesian Kriging ---------------------------
    elif method == "ebk":
        start_time = time.time()

        with arcpy.EnvManager(extent = extent, mask = mask,outputCoordinateSystem = arcpy.SpatialReference(spatialref), cellSize = c_size, parallelProcessingFactor = parProFactor):
            arcpy.ga.EmpiricalBayesianKriging(in_features = input_point, 
                                      z_field = z_field, 
                                    # This layer is not generated  
                                      out_ga_layer = out_ga_layer,
                                      out_raster   = out_raster,
                                     # transformation_type = 'EMPIRICAL',
                                    search_neighborhood = arcpy.SearchNeighborhoodSmoothCircular(smooth_r,0.5))
            arcpy.GALayerToRasters_ga(out_ga_layer, ga_to_raster,"PREDICTION_STANDARD_ERROR", None, c_size, 1, 1, "")
        ValStat = extract_val_result(out_ga_layer, method.upper())
        print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
        return out_raster, ValStat
            
# ---------------------- Regression Kriging ---------------------------
    elif method == "rk":
        start_time = time.time()
        with arcpy.EnvManager(extent = extent, mask = mask,outputCoordinateSystem = arcpy.SpatialReference(spatialref), cellSize = c_size, parallelProcessingFactor = parProFactor):
            out_surface_raster = arcpy.EBKRegressionPrediction_ga(in_features = input_point, 
                                                                   dependent_field = z_field, 
                                                                  out_ga_layer = out_ga_layer,
                                                                    out_raster = out_raster,
                                                                  in_explanatory_rasters = in_explanatory_rasters,
                                                                   transformation_type = 'EMPIRICAL',
                                                                  search_neighborhood = arcpy.SearchNeighborhoodSmoothCircular(smooth_r,0.5))
            # Convert GA layer to standard error of prediction raster
            arcpy.GALayerToRasters_ga(out_ga_layer, ga_to_raster,"PREDICTION_STANDARD_ERROR", None, c_size, 1, 1, "")
        ValStat = extract_val_result(out_ga_layer, method.upper())
        print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
        return out_raster, ValStat

    

    
def interpolation_auto(method,gis_path,dataframe,managed_area,Year,Season,start_date,end_date,parameter,covariates):
    
    col_ls = ['RowID','ParameterName','ParameterUnits','ProgramLocationID','ActivityType','ManagedAreaName',
                   'SampleDate','Year','Month','ResultValue','ValueQualifier','Latitude_DD','Longitude_DD']
    para_ls = ["Salinity","Total Nitrogen","Dissolved Oxygen","Turbidity","Secchi Depth"]
    para_ls_ab = ["S","TN","DO","T","SD"]
    # Convert full MA names to short names
    dictArea    = {'Gasparilla Sound-Charlotte Harbor Aquatic Preserve':'Charlotte Harbor','Big Bend Seagrasses Aquatic Preserve':'Big Bend',
                    'Guana Tolomato Matanzas National Estuarine Research Reserve':'GTM Reserve','Estero Bay Aquatic Preserve':'Estero Bay',
                    'Biscayne Bay Aquatic Preserve':'Biscayne Bay','Matlacha Pass Aquatic Preserve':'Matlacha Pass AP',
                    'Lemon Bay Aquatic Preserve':'Lemon Bay','Cape Haze Aquatic Preserve':'Cape Haze','Pine Island Sound Aquatic Preserve':'Pine Island'}

    # Convert full MA names to MA name in ORCP_Managed_Areas_Oct2021
    dictArea2    = {'Gasparilla Sound-Charlotte Harbor Aquatic Preserve':'Gasparilla Sound-Charlotte Harbor','Big Bend Seagrasses Aquatic Preserve':'Big Bend Seagrasses',
                    'Guana Tolomato Matanzas National Estuarine Research Reserve':'Guana Tolomato Matanzas NERR','Estero Bay Aquatic Preserve':'Estero Bay',
                    'Biscayne Bay Aquatic Preserve':'Biscayne Bay','Matlacha Pass Aquatic Preserve':'Matlacha Pass',
                    'Lemon Bay Aquatic Preserve':'Lemon Bay','Cape Haze Aquatic Preserve':'Cape Haze','Pine Island Sound Aquatic Preserve':'Pine Island Sound'}
    dictArea3    = {'Gasparilla Sound-Charlotte Harbor Aquatic Preserve':'ch','Big Bend Seagrasses Aquatic Preserve':'bb',
                    'Guana Tolomato Matanzas National Estuarine Research Reserve':'gtm','Estero Bay Aquatic Preserve':'eb',
                    'Biscayne Bay Aquatic Preserve':'bbay','Matlacha Pass Aquatic Preserve':'mp',
                    'Lemon Bay Aquatic Preserve':'lb','Cape Haze Aquatic Preserve':'ch','Pine Island Sound Aquatic Preserve':'pi'}
    #dictArea4 = {'Charlotte Harbor':"ch",'Big Bend':"bb",'GTM Reserve':"gtm",'Estero Bay':"eb",'Biscayne Bay':"bbay"}

    dictPara = {"Salinity":'S','Total Nitrogen':'TN','Dissolved Oxygen':'DO','Turbidity':'T','Secchi Depth':'SD'}
    dictUnits   = {"Salinity":"ppt","Total Nitrogen": "mg/L","Dissolved Oxygen": "mg/L","Turbidity": "NTU", "Secchi Depth": "m"}
    listArea    = ['Guana Tolomato Matanzas National Estuarine Research Reserve',
       'Biscayne Bay Aquatic Preserve',
       'Big Bend Seagrasses Aquatic Preserve',
       'Cape Haze Aquatic Preserve',
       'Gasparilla Sound-Charlotte Harbor Aquatic Preserve',
       'Pine Island Sound Aquatic Preserve',
       'Matlacha Pass Aquatic Preserve', 'Lemon Bay Aquatic Preserve',
       'Estero Bay Aquatic Preserve']
    listPara    = ["Salinity","Total Nitrogen","Dissolved Oxygen","Turbidity","Secchi Depth"]
    SpatialRef = '3086'
    
    method = method
    dataframe = dataframe
    Area   = managed_area
    Year   = Year
    Season = Season
    start_date,end_date = start_date,end_date
    Para   = parameter
    covariates = covariates
    fname = [dictArea[Area],dictArea3[Area],Year,Season,dictPara[Para]]
    
    input_pt = gis_path+"input_point/{}/{}_{}{}_{}.shp".format(*fname)
    
    df,gdf= select_aggr_area_season(dataframe,start_date,end_date, Area, Para)
    
    try:
        gdf   = gdf.to_crs(int(SpatialRef))
        boundary_shp = gis_path+ 'managed_area_boundary/{}.shp'.format(dictArea[Area][0:3])
        gdf.to_file(input_pt,driver='ESRI Shapefile',crs="EPSG:"+SpatialRef)
        MA = gpd.read_file(gis_path + r"managed_area_boundary/ORCP_Managed_Areas_Oct2021.shp")
        boundary = MA[MA['MA_Name']==dictArea2[Area]].to_crs(int(SpatialRef))
        boundary.to_file(boundary_shp , driver='ESRI Shapefile',crs="EPSG:"+SpatialRef)
        extent = str(boundary.geometry.total_bounds).replace('[','').replace(']','')

        if type(covariates) == str:
            in_explanatory_rasters = gis_path + "covariates/{}/{}.tif".format(covariates, dictArea[Area])
        elif type(covariates) == list:
            in_explanatory_rasters = []
            for i in range(len(covariates)):
                in_explanatory_raster = str(gis_path + "covariates/{}/{}.tif".format(covariates[i], dictArea[Area]))
                in_explanatory_rasters.append(in_explanatory_raster)

        in_features = input_pt
        out_raster = gis_path +"output_raster/{}/{}_{}{}_{}.tif".format(*fname)
        value_field = "ResultValu"
        out_ga_layer = gis_path +"ga_layer/{}/{}_{}{}_{}.lyrx".format(*fname)
        ga_to_raster = gis_path + 'standard_error_prediction/{}/{}_{}{}_{}_sep.tif'.format(*fname)
        in_explanatory_rasters = in_explanatory_rasters
        mask = gis_path+ '{}.shp'.format(dictArea3[Area])


        Result,Stat = interpolation(
                        method = method, input_point = in_features, out_raster = out_raster, 
                        z_field = value_field, out_ga_layer = out_ga_layer, extent = extent, 
                        mask = mask, ga_to_raster = ga_to_raster, in_explanatory_rasters = in_explanatory_rasters)
        
        return out_raster,out_ga_layer,ga_to_raster

    except Exception:
            e = sys.exc_info()[1]
            print(Para + " in " + str(Year) + " " + Season + " caused an error:")
            print(e.args[0])
            return np.nan,np.nan,np.nan