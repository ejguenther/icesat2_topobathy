#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 09:47:49 2025

@author: ejg2736
"""

import numpy as np
import multiprocessing
import laspy
import pandas as pd
import geopandas as gpd
import shapely.geometry as shp
from pyproj import Transformer, CRS
import glob
from tqdm import tqdm

def find_files(directory, extension, recursively=True):
    if type(extension) == str:
        return glob.glob(f"{directory}/**/*.{extension}",recursive=recursively)
    elif type(extension) == list:
        all_files = []
        for ext in extension:
            files = glob.glob(f"{directory}/**/*.{ext}",recursive=recursively)
            all_files.extend(files)
        return all_files

def read_las_header(las_file):
    # Read file header
    las = laspy.open(las_file) #Using laspy.open('f.las') reads only the header
    
    # Grab useful info
    x_min = las.header.x_min
    x_max = las.header.x_max
    y_min = las.header.y_min
    y_max = las.header.y_max
    z_min = las.header.z_min
    z_max = las.header.z_max
    creatation_date = las.header.creation_date
    parse_crs = las.header.parse_crs() 
    # print('2\n')
    # Read into dictionary
    las_dict = {"file_name":las_file,
            "x_min":x_min,
            "x_max":x_max,
            "y_min":y_min,
            "y_max":y_max,
            "z_min":z_min,
            "z_max":z_max,
            "creation_date":creatation_date,
            "parse_crs":parse_crs}
    # print('read_las_header return\n')
    las.close()

    return las_dict

def process_las_file(las_file):
    """Processes a LAS file and returns its header data."""
    try:
        return read_las_header(las_file)
    except Exception as e:
        print(f"\nError processing {las_file}: {e}\n")
        return None

def process_las_files_multiprocessing(las_list, num_processes=None):
    """Processes a list of LAS files using multiprocessing."""
    if num_processes is None:
        num_processes = np.ceil(multiprocessing.cpu_count()/2).astype(int)
    print('Reading header files:')
    pool = multiprocessing.Pool(processes=3)
    results = tqdm(pool.imap_unordered(process_las_file, las_list),
                   total = len(las_list))

    data = [result for result in results if result is not None]
    pool.close()
    return data

def create_gdf_las_extent(las_list, num_processes=None):
    """
    Creates a GeoPandas DataFrame containing geometries for the extent of LAS files.
    
    Args:
        las_list (list): List of paths to LAS files.
        num_processes (int, optional): Number of processes to use for multiprocessing.
            Defaults to None (use single process).
    
    Returns:
        geopandas.GeoDataFrame: GeoPandas DataFrame with geometries and CRS.
    """
    
    # Process LAS files (assuming process_las_files_multiprocessing exists)
    if num_processes != 1:
        data = process_las_files_multiprocessing(las_list,
                                                 num_processes=num_processes)
    else:
        results = []
        for las_file in tqdm(las_list):
            results.append(process_las_file(las_file))
        data = [result for result in results if result is not None]
    df = pd.DataFrame(data)

        
    
    # Handle rows with missing CRS
    missing_crs = df[df['parse_crs'].isnull()]
    if len(missing_crs) > 0:
        print(f"Warning: {len(missing_crs)}/{len(df)} LAS files have missing CRS. Removing them.")
        df = df.dropna(subset=['parse_crs'])
      
    # Check for multiple CRS (if relevant)
    crs_list = df['parse_crs'].unique()
    if len(crs_list) > 1:
        print(f"Warning: There are {len(crs_list)} unique CRS in dataset")
        
    # Initialize empty lat/lon columns with zeros
    df['lat_min'] = np.zeros(len(df))
    df['lat_max'] = np.zeros(len(df))
    df['lon_min'] = np.zeros(len(df))
    df['lon_max'] = np.zeros(len(df))
    
    # Iterate through unique CRS and reproject extents
    for crs in df['parse_crs'].unique():
        transformer = Transformer.from_crs(crs.to_2d(), 4326)
    
        df.loc[df['parse_crs'] == crs, 'lat_min'], df.loc[df['parse_crs'] == crs, 'lon_min'] = transformer.transform(
            df.loc[df['parse_crs'] == crs, 'x_min'], df.loc[df['parse_crs'] == crs, 'y_min'])
    
        df.loc[df['parse_crs'] == crs, 'lat_max'], df.loc[df['parse_crs'] == crs, 'lon_max'] = transformer.transform(
            df.loc[df['parse_crs'] == crs, 'x_max'], df.loc[df['parse_crs'] == crs, 'y_max'])

    # Create geometries from lat/lon extents using Shapely
    df['geometry'] = df.apply(lambda row: shp.box(row['lon_min'], 
                                                  row['lat_min'], 
                                                  row['lon_max'], 
                                                  row['lat_max']), axis=1)
    
    # Create GeoPandas DataFrame
    wgs_extent = gpd.GeoDataFrame(df,geometry='geometry',crs=CRS.from_epsg(4326))
    
    return wgs_extent

if __name__ == "__main__":
    # laz_dir = '/mnt/walker/exports/nfs_share/Data/OpenData/NorthCarolina/phase5'
    # laz_dir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/OpenData/NorthCarolina/phase5'
    # laz_dir = '/home/ejg2736/network_drives/bigtex/exports/vol2/vol2/Data/OpenData/vault/Finland/UTM_WKT'
    # laz_dir = '/home/ejg2736/network_drives/bigtex/exports/vol2/vol2/Data/OpenData/vault/Finland/UTM_WKT'
    # laz_dir = '/home/ejg2736/network_drives/bigtex/exports/vol2/vol2/Data/OpenData/vault/Finland/UTM_WKT'
    # laz_dir = '/home/ejg2736/network_drives/bigtex/exports/vol2/vol2/Data/OpenData/retrievable/CA_UpperSouthAmerica_Eldorado_2019'
    laz_dir = '/home/ejg2736/network_drives/bigtex/exports/vol2/vol2/Data/OpenData/retrievable/CA_UpperSouthAmerica_Eldorado_2019'
    laz_dir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/OpenData/Finland/'

    #laz_dir = '/mnt/walker/exports/nfs_share/Data/OpenData/CA_SanJoaquin'
    # laz_dir = '/home/ejg2736/data/als_test_area'
    # laz_dir = '/mnt/bigtex/vol1/Data/Labeled_PointClouds/Lidar/Alexandria'
    # laz_dir = '/mnt/walker/exports/nfs_share/Data/OpenData/Finland/UTM'
    # laz_dir = '/mnt/walker/exports/nfs_share/Data/OpenData/Finland/unprocessed'
    # laz_dir = '/exports/nfs_share/Data/OpenData/florida_2019_west_everglades'
    # laz_dir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/OpenData/florida_2019_west_everglades'
    # out_file = '/home/ejg2736/dev/crossover_analysis/fl_west_Everglades_laz_extent1.gpkg'
    # out_file = '/home/ejg2736/dev/crossover_analysis/nc_phase5_laz_extent.gpkg'
    out_file = '/home/ejg2736/dev/crossover_analysis/finland_wkt_laz_extent_walker.gpkg'
    # out_file = '/home/ejg2736/dev/crossover_analysis/CA_UpperSouthAmerica_Eldorado_2019.gpkg'

    laz_list = find_files(laz_dir, 'laz')
    # laz_list = laz_list[0:1]
    # extent_gdf = create_gdf_las_extent(laz_list,num_processes=18)
    extent_gdf = create_gdf_las_extent(laz_list,num_processes=8)

    extent_gdf.to_file(out_file, driver='GPKG', layer='layer')     
    