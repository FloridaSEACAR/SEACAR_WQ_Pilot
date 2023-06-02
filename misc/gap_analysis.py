# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:31:48 2023


@author: qiangy

"""
import os,glob,re,rasterio
import numpy as np
from collections import defaultdict
import numpy.ma as ma

import geopandas as gpd
from shapely.geometry import Point
import math
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from rasterio.mask import mask
from shapely.geometry import mapping
import scipy.stats as stats


# Function to rescale SEP
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
                
# Function to empty folder                
def delete_all_files(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
            
# Function to check the original tif file
def group_tif_by_shapes(folder_path):
    # Get all .tif files in the directory
    tif_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.tif')]
    
    # If no .tif files found, return
    if not tif_files:
        print("No .tif files found in the directory.")
        return
    
    # Initialize a dictionary to group files by their shapes
    shape_dict = defaultdict(list)
    
    # Check the shapes of all files
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            shape_dict[src.shape].append(tif_file)
    
    # Print the results
    for i, (shape, files) in enumerate(shape_dict.items(), 1):
        print(f"Group {i} - Shape: {shape} - Files: {len(files)}")
        for file in files:
            print(f"  {file}")
        print("-" * 40)
        
#  Calculate Average Rescaled SEP 
def calculate_average_tif(folder_path, output_folder_path):
    # Get all files in the directory
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.tif')]
    
    # Get unique parameters
    parameters = set([os.path.splitext(os.path.basename(f).split('_')[2])[0] for f in all_files])  # Remove the .tif extension from param
    
    for param in parameters:
        print(f"Processing parameter: {param}")
        
        # Get files for this parameter
        param_files = [f for f in all_files if os.path.splitext(os.path.basename(f).split('_')[2])[0] == param]  # Remove the .tif extension from param
        num_files = len(param_files)
        print(f"Found {num_files} files for parameter {param}")
        
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
                # Check if data shape is consistent with the first file
                if src.shape != data_shape:
                    print(f"Skipping file {file} due to inconsistent shape.")
                    continue

                # Read data and adjust size if necessary
                data = src.read(1).astype(np.float64)

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
        output_file = os.path.join(output_folder_path, f"avg_{param}.tif")
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(avg_data.data, 1)
            
            
            
# Calculate Average Rescaled SEP by Season 
def calculate_average_tif_season(folder_path, output_folder_path):
    # Get all files in the directory
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.tif')]

    # Get unique parameters and seasons
    parameters_seasons_files = defaultdict(list)
    seasons_years = defaultdict(lambda: defaultdict(set))  # A nested dictionary to store years for each season and parameter
    for f in all_files:
        basename_parts = os.path.basename(f).split('_')
        year_season = basename_parts[1]
        param = os.path.splitext(basename_parts[2])[0]
        season = year_season[4:]  # Extract season from year_season string
        year = year_season[:4]  # Extract year from year_season string
        seasons_years[season][param].add(year)
        parameters_seasons_files[(season, param)].append(f)

    image_count = 0  # Initialize a counter for the generated images

    for (season, param), param_files in parameters_seasons_files.items():
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
                # Check if data shape is consistent with the first file
                if src.shape != data_shape:
                    print(f"Skipping file {file} due to inconsistent shape.")
                    continue

                # Read data and adjust size if necessary
                data = src.read(1).astype(np.float64)

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
        output_file = os.path.join(output_folder_path, f"avg_{season}_{param}.tif")
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(avg_data.data, 1)

        image_count += 1  # Increase the counter each time an image is generated

    # Print all seasons and their corresponding years
    for season, param_years in seasons_years.items():
        for param, years in param_years.items():
            print(f"Season: {season} for parameter: {param} - Found {len(parameters_seasons_files[(season, param)])} files, Years: {sorted(list(years))}")
    print(f"Total number of generated images: {image_count}")  # Print the total number of generated images



area_shortnames = {
    'Gasparilla Sound-Charlotte Harbor Aquatic Preserve': 'Gasparilla Sound-Charlotte Harbor',
    'Estero Bay Aquatic Preserve': 'Estero Bay',
    'Big Bend Seagrasses Aquatic Preserve': 'Big Bend Seagrasses',
    'Biscayne Bay Aquatic Preserve': 'Biscayne Bay',
    'Matlacha Pass Aquatic Preserve': 'Matlacha Pass',
    'Guana Tolomato Matanzas National Estuarine Research Reserve' : 'Guana Tolomato Matanzas NERR'
}

param_shortnames = {
    'Salinity': 'SA',
    'Total Nitrogen': 'TN',
    'Dissolved Oxygen': 'DO',
    'Turbidity':'TB',
    'Secchi Depth':'SD'
}

# Function to plot kernel density map.
def plot_kde_map(df, lat_col, lon_col, area_names, variables,boundary):
    # Preprocess input data
    df_filtered = df[df['ManagedAreaName'].isin(area_names) & df['ParameterName'].isin(variables)]
    
    gdf = gpd.GeoDataFrame(df_filtered,
                        geometry=[Point(x, y) for x, y in zip(df_filtered[lon_col], df_filtered[lat_col])],
                        crs=4326)
    gdf = gdf.to_crs('epsg:3086')

    grouped_gdf = gdf.groupby(['ManagedAreaName', 'ParameterName'])

    # Calculate the total number of subplots
    total_plots = len(grouped_gdf)

    rows = math.ceil(total_plots / 3)
    cols = min(total_plots, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(30, 25), squeeze=False)

    plot_index = 0
    
    for (area_name, variable), group in grouped_gdf:
        row, col = divmod(plot_index, cols)
        ax = axes[row, col]
        kdeplot = sns.kdeplot(x=group.geometry.x, y=group.geometry.y, n_levels=30, fill=True, cmap='YlOrRd', ax=ax, bw_method='scott')
        # Add point locations to the kernel density map
        #group.plot(ax=ax, color='black', markersize=0.1)
        ax.set_title(f"KDE - {area_shortnames[area_name]} - {param_shortnames[variable]}", fontdict={'fontsize': 25, 'fontweight': 'bold'})
        ax.set_box_aspect(1)
        cx.add_basemap(ax=ax, source=cx.providers.Stamen.TonerLite, crs=gdf.crs)
        
        # Create colorbar that matches the height of the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Create a ScalarMappable with the same colormap as the KDE plot
        sm = ScalarMappable(cmap='YlOrRd')
        sm.set_array([])  # not needed for the colorbar but it raises a warning if not present
        plt.colorbar(sm, cax=cax, orientation='vertical')
        
        boundary_plot = boundary[boundary['MA_Name'] == area_shortnames[area_name]]
        boundary_plot.plot(ax=ax, color='none', edgecolor='blue')

        plot_index += 1
        
    for i in range(plot_index, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].remove()

    fig.tight_layout()
    plt.show()
    plt.close()


# Funtion to store kernel density results
def save_kde_tif(df, lat_col, lon_col, area_names, variables, folder_path,boundary, pixel_size=100):
    
    # Remove rows with NaN values
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[lat_col, lon_col], inplace=True)

    df_filtered = df[df['ManagedAreaName'].isin(area_names) & df['ParameterName'].isin(variables)]
    gdf = gpd.GeoDataFrame(df_filtered,
                        geometry=[Point(x, y) for x, y in zip(df_filtered[lon_col], df_filtered[lat_col])],
                        crs=4326)
    gdf = gdf.to_crs('epsg:3086')
    
    grouped = gdf[(gdf['ManagedAreaName'].isin(area_names)) &
                  (gdf['ParameterName'].isin(variables))].groupby(['ManagedAreaName', 'ParameterName'])
    
    tif_paths = []  # Add this line
    for area_name in area_names:
        for variable in variables:
            gdf_filtered = gdf[(gdf['ManagedAreaName'] == area_name) & (gdf['ParameterName'] == variable)]
            if gdf_filtered.empty:
                continue

            x_coords = gdf_filtered.geometry.x
            y_coords = gdf_filtered.geometry.y

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            x_grid, y_grid = np.mgrid[x_min:x_max:pixel_size, y_min:y_max:pixel_size]
            positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

            values = np.vstack([x_coords, y_coords])
            kde = stats.gaussian_kde(values, bw_method='scott')
            kde_result = np.swapaxes(kde(positions).reshape(x_grid.shape), 0, 1)
            kde_result = np.flipud(kde_result)
            
            #print(f'KDE result - min: {np.min(kde_result)}, max: {np.max(kde_result)}, mean: {np.mean(kde_result)}')

            #fig, ax = plt.subplots(figsize=(10, 10))
            #sns.kdeplot(x=gdf_filtered.geometry.x, y=gdf_filtered.geometry.y, n_levels=30, fill=True, cmap='YlOrRd', ax=ax, bw_method='silverman')
            # gdf_filtered.plot(ax=ax, color='black', markersize=0.1,alpha=0.5)
            #cx.add_basemap(ax=ax, source=cx.providers.Stamen.TonerLite, crs=gdf_filtered.crs)
            
            file_name = f"{area_name}_{variable}"
            # Save the KDE result as a GeoTIFF file
            output_tif_path = os.path.join(folder_path, f"{file_name}.tif")
            temp_tif_path = os.path.join(folder_path, f"{file_name}_temp.tif")

            bbox = gdf_filtered.total_bounds
            with rasterio.open(
                temp_tif_path,
                'w',
                driver='GTiff',
                height=kde_result.shape[0],
                width=kde_result.shape[1],
                count=1,
                dtype=kde_result.dtype,
                crs=gdf_filtered.crs,
                transform=rasterio.transform.from_bounds(*bbox, kde_result.shape[1], kde_result.shape[0])
            ) as dst:
                dst.write(kde_result, 1)

            with rasterio.open(temp_tif_path) as src:
                # Clip the raster with the boundary
                boundary_mask = boundary[boundary['MA_Name'] == area_shortnames[area_name]]
                if not boundary_mask.empty:
                    geometry = boundary_mask.geometry
                    out_image, out_transform = mask(src, [mapping(geometry.values[0])], crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({"driver": "GTiff",
                                     "height": out_image.shape[1],
                                     "width": out_image.shape[2],
                                     "transform": out_transform})

                # Save the clipped raster
            with rasterio.open(output_tif_path, "w", **out_meta) as dest:
                dest.write(out_image)

                # Delete the temporary file
            os.remove(temp_tif_path)

            print(f"KDE GeoTIFF saved to: {output_tif_path}")
            tif_paths.append(output_tif_path)

    #return tif_paths

# Funtion to plot kernel density results using "gaussian_kde"
def plot_tifs(tif_folder, area_names, boundary):
    # Calculate the total number of subplots
    files = [f for f in os.listdir(tif_folder) if any(f.startswith(area_name + '_') for area_name in area_names)]
    files.sort()
    total_plots = len(files)

    rows = math.ceil(total_plots / 3)
    cols = min(total_plots, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(30, 25), squeeze=False)
    plot_index = 0

    for area_name in area_names:
        area_files = [f for f in files if f.startswith(area_name + '_')]

        for file in area_files:
            variable = file.split('_')[1].replace('.tif', '')

            # Open the GeoTIFF file
            with rasterio.open(os.path.join(tif_folder, file)) as src:
                # Read the data into an array
                data = src.read(1)

                # Replace zero values with NaN
                data[data == 0] = np.nan

                # Create a masked array where NaN values are masked
                masked_data = np.ma.masked_invalid(data)

                # Get the extent of the data
                extent = rasterio.plot.plotting_extent(src)

                row, col = divmod(plot_index, cols)
                ax = axes[row, col]

                # Set the x-axis range to the extent of the data
                ax.set_xlim(extent[0], extent[1])

                # Set the y-axis range to the extent of the data
                ax.set_ylim(extent[2], extent[3])

                # Add a base map
                cx.add_basemap(ax=ax, source=cx.providers.Stamen.TonerLite, crs=src.crs)

                # Get the boundary for the specific area
                boundary_plot = boundary[boundary['MA_Name'] == area_shortnames[area_name]]
                boundary_plot = boundary_plot.to_crs(src.crs)

                # Plot the boundary on the same axes
                boundary_plot.plot(ax=ax, color='none', edgecolor='blue')

                # Display the image on top
                im = ax.imshow(masked_data, cmap='YlOrRd', extent=extent, origin="upper")

                # Create colorbar that matches the height of the plot
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                fig.colorbar(im, cax=cax, label='KDE')

                # Set title from filename
                ax.set_title(f"KDE - {area_shortnames[area_name]} - {param_shortnames[variable]}", fontdict={'fontsize': 20, 'fontweight': 'bold'})
                ax.set_box_aspect(1)
                plot_index += 1

    for i in range(plot_index, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].remove()

    fig.tight_layout()
    plt.show()
    
def plot_tifs_zoom(tif_folder, area_names, boundary, zoom_to_boundary=False):
    # Calculate the total number of subplots
    files = [f for f in os.listdir(tif_folder) if any(f.startswith(area_name + '_') for area_name in area_names)]
    files.sort()
    total_plots = len(files)

    rows = math.ceil(total_plots / 3)
    cols = min(total_plots, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(30, 25), squeeze=False)
    plot_index = 0

    for area_name in area_names:
        area_files = [f for f in files if f.startswith(area_name + '_')]

        for file in area_files:
            variable = file.split('_')[1].replace('.tif', '')

            # Open the GeoTIFF file
            with rasterio.open(os.path.join(tif_folder, file)) as src:
                # Read the data into an array
                data = src.read(1)

                # Replace zero values with NaN
                data[data == 0] = np.nan

                # Create a masked array where NaN values are masked
                masked_data = np.ma.masked_invalid(data)

                # Get the extent of the data
                extent = rasterio.plot.plotting_extent(src)

                row, col = divmod(plot_index, cols)
                ax = axes[row, col]

                # Add a base map
                cx.add_basemap(ax=ax, source=cx.providers.Stamen.TonerLite, crs=src.crs, zoom=20)

                # Get the boundary for the specific area
                boundary_plot = boundary[boundary['MA_Name'] == area_shortnames[area_name]]
                boundary_plot = boundary_plot.to_crs(src.crs)

                # Plot the boundary on the same axes
                boundary_plot.plot(ax=ax, color='none', edgecolor='blue')

                if zoom_to_boundary:
                    # Get bounds of the boundary
                    bounds = boundary_plot.bounds

                    # Set the x-axis range to the extent of the boundary
                    ax.set_xlim([bounds.minx.min(), bounds.maxx.max()])

                    # Set the y-axis range to the extent of the boundary
                    ax.set_ylim([bounds.miny.min(), bounds.maxy.max()])
                else:
                    # Set the x-axis range to the extent of the data
                    ax.set_xlim(extent[0], extent[1])

                    # Set the y-axis range to the extent of the data
                    ax.set_ylim(extent[2], extent[3])

                im = ax.imshow(masked_data, cmap='YlOrRd', extent=extent, origin="upper")

                # Create colorbar that matches the height of the plot
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                fig.colorbar(im, cax=cax, label='KDE')

                # Set title from filename
                ax.set_title(f"KDE - {area_shortnames[area_name]} - {param_shortnames[variable]}", fontdict={'fontsize': 20, 'fontweight': 'bold'})
                ax.set_box_aspect(1)
                plot_index += 1

    for i in range(plot_index, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].remove()

    fig.tight_layout()
    plt.show()