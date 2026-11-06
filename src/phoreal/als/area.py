#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 09:52:12 2025

@author: ejg2736
"""

import laspy
import numpy as np
# import sys
# import os
# import geopandas as gpd
import pandas as pd
# import shapely.geometry as shp
from shapely.geometry import Point
# from shapely.geometry import Polygon
# import rasterio
# import glob
from pyproj import Transformer, CRS
# from tqdm import tqdm
# import multiprocessing


def combine_dicts(dicts):
  """Combines multiple dictionaries into a single dictionary.

  Args:
    dicts: A list of dictionaries.

  Returns:
    A combined dictionary.
  """

  result = {}
  for d in dicts:
    result.update(d)
  return result


def get_als_from_target(wgs_extent, target_lon, target_lat, radius = 30):
    target_point = Point(target_lon, target_lat)
    a = radius/(111111 * np.cos(np.deg2rad(target_lat)))
    b = radius/111111
    c = np.sqrt(a**2 + b**2)
    distance = c * 2 # Search twice the maximum distance
    point_buffer = target_point.buffer(distance)

    polygons_within_distance = wgs_extent[wgs_extent.intersects(point_buffer)]
    
    if len(polygons_within_distance) == 0:
        print('Warning: No ALS tile identified for intersection')
        return pd.DataFrame()
    elif len(polygons_within_distance) > 1:
        print('Warning: Multiple ALS tiles identified for intersection')
        
    transformer = Transformer.from_crs(4326, 
                                    polygons_within_distance.iloc[0].parse_crs)
    
    target_northing, target_easting = transformer.transform(target_lat, 
                                                            target_lon)
    data_list = []
    for i in range(0,len(polygons_within_distance)):
        las_file = laspy.read(polygons_within_distance.iloc[i].file_name)
        x = las_file.x
        y = las_file.y
        z = las_file.z
        c = las_file.classification
        
        # x = x[c == 2]
        # y = y[c == 2]
        # z = z[c == 2]
        # c = c[c == 2]

        # x, y, z, c = read_and_reproject_las(
        #     polygons_within_distance.iloc[i].file_name)
        
        dist = np.sqrt(np.array(x - target_northing)**2 +\
                       np.array(y - target_easting)**2)
        
        #if len(x[dist < footprint_radius]) == 0:
        #    las_dict = {}
        #    break

        las_dict = {'x':np.array(x[dist < radius]),
                'y':np.array(y[dist < radius]),
                'z':np.array(z[dist < radius]),
                'c':np.array(c[dist < radius])
                }
        
        if len(las_dict['x']) > 0:
            data_list.append(las_dict)
            
    data_dict = combine_dicts(data_list)
        
    df_als = pd.DataFrame(data_dict)
    df_als['parse_crs'] = polygons_within_distance.iloc[0].parse_crs
    
    return df_als



def process_gedi_point(row, extent_gdf):
    """
    Processes a single GEDI point (a row from the GeoDataFrame) to
    calculate the relative height from corresponding ALS data.
    """
    try:
        df_als = get_als_from_target(extent_gdf, row.lon_lowestmode,
                                     row.lat_lowestmode, radius=15)
        if df_als.empty:
            return np.nan

        #asprs: 2, ground; 40, bathy; 42, surface
        # ground_points = df_als[df_als['c'].isin([2,40])]
        ground_points = df_als[df_als['c'].isin([2])]
        df_als = df_als[df_als['c'].isin([0,1,3,4,5])]

        if ground_points.empty or len(df_als) < 5:
            return np.nan

        ground_height = np.median(ground_points.z)
        rel_height = df_als.z - ground_height
        rel_98 = np.percentile(rel_height, 98)
        
        return rel_98
    except Exception as e:
        print(f"An error occurred for point at index {row.name}: {e}")
        return np.nan

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from tqdm import tqdm
    from joblib import Parallel, delayed
    # Using "if __name__ == '__main__':" is good practice for parallel scripts

    # Read las extent
    extent_gpkg = '/home/ejg2736/dev/crossover_analysis/fl_west_Everglades_laz_extent1.gpkg'
    extent_gdf = gpd.read_file(extent_gpkg)

    # Read GEDI files
    gedi_csv = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/workspace/IS2/mangrove_fl/GEDI_Mangrove_Height.csv'
    gedi_csv = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/workspace/IS2/mangrove_fl/GEDI_Mangrove_Height_filtered_only.csv'
    gedi_df = pd.read_csv(gedi_csv)

    # Filter by GEDI Mangrove Heights within ALS Data
    gedi_gdf = gpd.GeoDataFrame(gedi_df, geometry=gpd.points_from_xy(gedi_df.lon_lowestmode, gedi_df.lat_lowestmode), crs="EPSG:4326")

    filtered_gedi_gdf = gpd.sjoin(
        gedi_gdf,
        extent_gdf,
        how="inner",
        predicate='intersects'
    )

    # Compute GEDI in parallel
    print("Processing GEDI points in parallel...")
    rel_height_list = Parallel(n_jobs=7)(
        delayed(process_gedi_point)(row, extent_gdf)
        for _, row in tqdm(filtered_gedi_gdf.iterrows(), total=len(filtered_gedi_gdf))
    )
    
    # Add results to the GeoDataFrame
    filtered_gedi_gdf['als_98'] = rel_height_list
    
    # Save the output
    print("Saving results...")
    filtered_gedi_gdf.to_file('/home/ejg2736/dev/icesat2_topobathy/gedi_als5.gpkg', driver='GPKG')
    print("Done!")


# if __name__ == "__main__":
#     import geopandas as gpd
#     from shapely.geometry import Point, Polygon
#     from tqdm import tqdm
    
#     # Read las extent
#     extent_gpkg = '/home/ejg2736/dev/crossover_analysis/fl_west_Everglades_laz_extent1.gpkg'
#     extent_gdf = gpd.read_file(extent_gpkg)    
    
#     # Read GEDI files
#     gedi_csv = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/workspace/IS2/mangrove_fl/GEDI_Mangrove_Height.csv'
#     gedi_df = pd.read_csv(gedi_csv)
    
#     # Filter by GEDI Mangrove Heights within ALS Data
#     gedi_gdf = gpd.GeoDataFrame(gedi_df, geometry=gpd.points_from_xy(gedi_df.lon_lowestmode, gedi_df.lat_lowestmode), crs="EPSG:4326")


#     filtered_gedi_gdf = gpd.sjoin(
#         gedi_gdf,          # The points you want to filter
#         extent_gdf,        # The polygons to filter with
#         how="inner",         # Only keep points that intersect
#         predicate='intersects' # The spatial relationship
#     )
    
#     # Compute GEDI
#     rel_height_list = []
#     for i in tqdm(range(0,len(filtered_gedi_gdf))):
#         df_als = get_als_from_target(extent_gdf, filtered_gedi_gdf.iloc[i].lon_lowestmode,
#                                      filtered_gedi_gdf.iloc[i].lat_lowestmode, radius = 30)
        
#         df_als = df_als[df_als.c < 3]
#         ground_height = np.median(df_als.z[df_als.c == 2])
#         height_98 = np.percentile(df_als.z,98)    
#         rel_height = height_98 - ground_height
#         rel_height_list.append(rel_height)
#     filtered_gedi_gdf['als_98'] = rel_height_list
    
#     filtered_gedi_gdf.to_file('/home/ejg2736/dev/icesat2_topobathy/gedi_als.gpkg')
    
    