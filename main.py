#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 19:54:03 2025

@author: ejg2736
"""

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import processing, analysis
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def get_atl_at_photon_rate(atl03_file, atl08_file, atl24_file, gt):
    # Read ATL03
    f = h5py.File(atl03_file, 'r') 
    
    # Read photon rate ATL03 data
    lon_ph = np.array(f[gt + '/heights/lon_ph'])
    lat_ph = np.array(f[gt + '/heights/lat_ph'])
    h_ph = np.array(f[gt + '/heights/h_ph'])
    quality_ph = np.array(f[gt + '/heights/quality_ph'])
    
    # Read ATL03 segment rate at ATL03 photon rate
    solar_elevation = processing.get_atl03_segment_to_photon(atl03_file,gt,'/geolocation/solar_elevation')
    alongtrack = processing.get_atl03_segment_to_photon(atl03_file,gt,'/geolocation/segment_dist_x')
    alongtrack = alongtrack + np.array(f[gt + '/heights/dist_ph_along'])
    
    # Read ATL08 signal photon at ATL03 photon rate
    atl08_class_ph = processing.get_atl08_class_to_atl03(atl03_file, atl08_file,gt)
    atl08_norm_h_ph = processing.get_atl08_norm_h_to_atl03(atl03_file, atl08_file,gt)
    
    
    # Read ATL24 photon rate at ATL03 photon rate
    atl24_class_ph = processing.get_atl24_to_atl03(atl03_file, atl24_file, gt)
    atl24_ortho_h_ph = processing.get_atl24_to_atl03(atl03_file, atl24_file, gt,'/ortho_h')
    
    # Combine ATL08 and ATL4 classifications
    combined_class_ph = processing.combine_atl08_and_atl24_classifications(atl08_class_ph,atl24_class_ph)
    
    # Create pandas dataframe
    df_ph = pd.DataFrame(
            {
                "alongtrack": alongtrack,
                "latitude":lat_ph,
                "longitude":lon_ph,
                "h_ph": h_ph,
                "h_norm": atl08_norm_h_ph,
                "ortho_h": atl24_ortho_h_ph,
                "atl08_class":atl08_class_ph,
                "atl24_class":atl24_class_ph,
                "combined_class":combined_class_ph,
                "solar_elevation":solar_elevation,
                "quality_ph":quality_ph
            }
        )
    
    return df_ph

def get_atl_at_seg(df_ph, res = 20):
    key = np.floor((df_ph.alongtrack - np.min(df_ph.alongtrack))/res).astype(int)

    df_ph['key_id'] = key
    df_seg = pd.DataFrame({'key_id':np.unique(key)})
    
    # Calculate median alongtrack
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'alongtrack',
        operation = 'median',
        class_field = 'atl08_class',
        class_id = [1,2,3,40,41],
        outfield = 'alongtrack'
    )    
    
    # Calculate median latitude    
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'latitude',
        operation = 'median',
        class_field = 'atl08_class',
        class_id = [1,2,3,40,41],
        outfield = 'latitude'
    )    

    # Calculate median longitude
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'longitude',
        operation = 'median',
        class_field = 'atl08_class',
        class_id = [1,2,3,40,41],
        outfield = 'longitude'
    )    

    # Calculate te_median, median terrain height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'atl08_class',
        class_id = [1],
        outfield = 'h_te_median'
    )    

    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'h_norm',
        operation = 'get_max98',
        class_field = 'atl08_class',
        class_id = [2,3],
        outfield = 'h_canopy'
    )
    
    # Calculate h_canopy_abs, 98th percentile canopy height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'max98',
        class_field = 'atl08_class',
        class_id = [2,3],
        outfield = 'h_canopy_abs'
    )
    
    # Calculate h_bathy, the median bathymetic height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'atl24_class',
        class_id = [40],
        outfield = 'h_bathy'
    )
    
    # Calculate h_surface, the median sea surface (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'atl24_class',
        class_id = [41],
        outfield = 'h_surface'
    )
    
    # Filter NaN data
    df_seg.loc[df_seg.longitude == 0,"longitude"] = np.nan
    df_seg.loc[df_seg.latitude == 0,"latitude"] = np.nan
    df_seg.dropna(subset =['longitude','latitude'],inplace=True)

    
    # Filter low canopy out
    canopy_check = df_seg.h_canopy_abs - df_seg.h_surface
    df_seg.loc[canopy_check < 1, "h_canopy"] = np.nan
    
    # Identify composition of each segment
    comp_flag = processing.get_measurement_type_string(df_seg, ['h_te_median','h_canopy',
                                                     'h_bathy','h_surface'])
    df_seg['comp_flag'] = comp_flag
    df_seg['comp_flag'] =(
        df_seg['comp_flag']
        .str.replace('h_te_median', 'terrain')
        .str.replace('h_canopy', 'canopy')
        .str.replace('h_surface', 'sea_surface')
        .str.replace('h_bathy', 'bathymetry')
        )
    


    return df_seg
    

if __name__ == "__main__":
    # Define ATL03 File
    atl03_dir = '/data/ATL03'
    atl03_name = 'ATL03_20200212094736_07280607_006_01.h5'
    atl03_file = os.path.join(atl03_dir, atl03_name)
    
    # Define ATL08 File
    atl08_dir = '/data/ATL08'
    atl08_name = 'ATL08_20200212094736_07280607_006_01.h5'
    atl08_file = os.path.join(atl08_dir, atl08_name)
    
    # Define ATL24 file
    atl24_dir = '/data/ATL24'
    atl24_name = 'ATL24_20200212094736_07280607_006_01_001_01.h5'
    atl24_file = os.path.join(atl24_dir, atl24_name)
    
    # Define Granule
    gt = 'gt1r'
    
    # Get photon rate DF
    df_ph = get_atl_at_photon_rate(atl03_file, atl08_file, atl24_file, gt)
    
    # Aggregate photon rate DF to 10 m segment 
    df_seg = get_atl_at_seg(df_ph, res = 10)
    
    # Write out as geopandas dataframe
    geometry_seg = [Point(xy) for xy in zip(df_seg.longitude, df_seg.latitude)]
    gdf_seg = gpd.GeoDataFrame(df_seg, geometry=geometry_seg, crs="EPSG:4326")
    
    # Save geopandas dataframe
    gdf_seg.to_file("is2_topobathy_" + gt + ".gpkg", layer='atl08atl24', driver="GPKG")
    

