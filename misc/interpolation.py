# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:00:32 2023

@author: qiangy
"""
import time
import math  
import sklearn.metrics  
import arcgisscripting
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.mask
import rasterio.plot as rio_pl
import matplotlib.image as mpimg
import os
#import rioxarray as rxr

from rasterio.plot import show
from rasterio.transform import Affine
from rasterio.mask import mask
from rasterio import MemoryFile
from rasterio.profiles import DefaultGTiffProfile
from sklearn.metrics import mean_squared_error
from shapely.geometry import box
from shapely.geometry import Polygon, Point
import contextily as cx
#from pykrige.ok import OrdinaryKriging

# import arcpy and environmental settings
import arcpy
from arcpy.sa import *
arcpy.env.overwriteOutput = True

# Extract validation data from saved GA layers (directly called in Python)
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
                         z_field, out_ga_layer, extent, mask,
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
#       Calculate dry season
        # search_radius = 20000
        # with arcpy.EnvManager(extent = extent, mask = mask,outputCoordinateSystem = arcpy.SpatialReference(3086), 
        #                       cellSize = 30, parallelProcessingFactor = "80%"):
        #     out_surface_raster = arcpy.sa.Kriging(in_point_features = input_point,
        #                                   z_field = z_field,
        #                                   kriging_model = KrigingModelOrdinary("Spherical # # # #"),
        #                                  search_radius = RadiusVariable(20, search_radius))                           

        # out_surface_raster.save(out_raster)
        
        #Generate GA layer of ordinary kriging
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
            
        ValStat = extract_val_result(out_ga_layer, method.upper())
        print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
        return out_raster, ValStat
            
# ---------------------- Empirical Bayesian Kriging ---------------------------
    elif method == "rk":
        with arcpy.EnvManager(extent = extent, mask = mask,outputCoordinateSystem = arcpy.SpatialReference(spatialref), cellSize = c_size, parallelProcessingFactor = parProFactor):
            out_surface_raster = arcpy.EBKRegressionPrediction_ga(in_features = input_point, 
                                                                   dependent_field = z_field, 
                                                                  out_ga_layer = out_ga_layer,
                                                                    out_raster = out_raster,
                                                                  in_explanatory_rasters = in_explanatory_rasters,
                                                                   transformation_type = 'EMPIRICAL',
                                                                  search_neighborhood = arcpy.SearchNeighborhoodSmoothCircular(smooth_r,0.5))
        ValStat = extract_val_result(out_ga_layer, method.upper())
        print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
        return out_raster, ValStat

# Plot interpolated raster
def plot_raster(gdf, extent, raster, title, ax, fig):
    gdf.plot(ax = ax, marker = 'o', color = 'green', markersize = 6)
    extent.plot(ax = ax, color='none', edgecolor='black')
    cx.add_basemap(ax = ax,source=cx.providers.Stamen.TonerLite,crs=gdf.crs)

    #       Raster must added after basemap
    with rasterio.open(raster, "r+") as dryOK:
        band = dryOK.read(1)
        band = np.ma.masked_array(band, mask=(band < 0))
        retted = rio_pl.show(band, transform=dryOK.transform, ax = ax, cmap="RdBu_r")
        ax.set_title(title)
    #           Add legend
        im = retted.get_images()[1]
        fig.colorbar(im, ax=ax,shrink=0.8)


# Plot covariate map        
def plot_covariate(Area,pt_Shp,extentShp,ra_fname,title, ax,fig):
    pt_Shp.plot(ax = ax, marker = 'o', color = 'green', markersize = 6)
    extentShp.plot(ax = ax, color='none', edgecolor='black')
    cx.add_basemap(ax = ax,source=cx.providers.Stamen.TonerLite,crs=pt_Shp.crs)

    #       Raster must added after basemap
    with rasterio.open(ra_fname, "r+") as covar:
        band = covar.read(1)
        band = np.ma.masked_where((band < -200000) | (band > 200000), band)
        retted = rio_pl.show(band, transform=covar.transform, ax = ax, cmap="RdBu_r")
        ax.set_title(title)
        # Add legend
        im = retted.get_images()[1]
        fig.colorbar(im, ax=ax,shrink=0.6)
        
