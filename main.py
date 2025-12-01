#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored Main Processing Script
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
from pyproj import Transformer


# --- Imports from your refactored Utils ---
# Assuming you moved t2t, get_attribute_info, etc. to utils.helpers
from utils import processing, analysis, readers
from utils.metadata import get_attribute_info, find_corresponding_atl
from utils.geographic_utils import find_utm_zone_epsg, get_geoid_height
from utils.datum_transforms import convert_3d_nad83_to_wgs84
from utils.create_las_swath import create_als_swath

# --- Configuration: ATL08/24 Aggregation Recipe ---
# This replaces the hardcoded get_atl_at_seg function
ATL_AGG_CONFIG = [
    # Geolocation (Median of all valid signal classes)
    {'field': 'latitude', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [1,2,3,40,41], 'outfield': 'latitude'},
    {'field': 'longitude', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [1,2,3,40,41], 'outfield': 'longitude'},
    {'field': 'solar_elevation', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [1,2,3,40,41], 'outfield': 'solar_elevation'},

    # Terrain
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [1], 'outfield': 'h_te_median'},

    # Canopy (Using h_norm)
    {'field': 'h_norm', 'operation': 'get_max98', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy'},
    {'field': 'h_norm', 'operation': 'max', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy_max'},

    # Topobathy (Using combined classes)
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'combined_class', 'class_id': [1,40], 'outfield': 'h_topobathy'},

    # Bathymetry & Surface (From ATL24 classes)
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'atl24_class', 'class_id': [40], 'outfield': 'h_bathy'},
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'atl24_class', 'class_id': [41], 'outfield': 'h_surface'},

    # Counts
    {'field': 'ortho_h', 'operation': 'get_len', 'class_field': 'atl08_class', 'class_id': [1], 'outfield': 'n_terrain'},
    {'field': 'ortho_h', 'operation': 'get_len', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'n_canopy'},
    {'field': 'ortho_h', 'operation': 'get_len', 'class_field': 'atl24_class', 'class_id': [40], 'outfield': 'n_bathy'},
    {'field': 'delta_time', 'operation': 'get_len_unique', 'class_field': 'atl08_class', 'class_id': [0,1,2,3], 'outfield': 'n_shots'},
]

# --- Configuration: ALS Aggregation Recipe ---
# This replaces the hardcoded get_als_at_seg function
ALS_AGG_CONFIG = [
    # Geometry
    {'field': 'x', 'operation': 'median', 'class_field': 'classification', 'class_id': list(range(1,100)), 'outfield': 'x_als'},
    {'field': 'y', 'operation': 'median', 'class_field': 'classification', 'class_id': list(range(1,100)), 'outfield': 'y_als'},

    # Terrain & Bathy
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'classification', 'class_id': [2], 'outfield': 'als_topo_median'},
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'classification', 'class_id': [40], 'outfield': 'als_bathy_median'},
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'classification', 'class_id': [41], 'outfield': 'als_surface_median'},

    # Canopy
    {'field': 'h_norm', 'operation': 'get_max98', 'class_field': 'classification', 'class_id': [3,4,5], 'outfield': 'als_norm_veg_max98'},
]


if __name__ == "__main__":
    
    # 1. Setup (Ideally load this from a config file)
    base_dir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/ICESat-2/REL006/florida_aoi'
    atl03_dir = os.path.join(base_dir, 'atl03')
    atl08_dir = os.path.join(base_dir, 'atl08')
    atl24_dir = os.path.join(base_dir, 'atl24')
    
    extent_gpkg = '/home/ejg2736/dev/crossover_analysis/fl_west_Everglades_laz_extent1.gpkg'
    geoid_file = '/home/ejg2736/dev/geoid/BundleAll/egm08_1.gtx'
    als_geoid_file = '/home/ejg2736/dev/geoid/agisoft/us_noaa_g2012b.tif'
    als_outdir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/workspace/IS2/mangrove_fl/als'
    df_outdir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/workspace/IS2/mangrove_fl/'
   
    target_res = 30
    gt_list = ['gt1r','gt1l','gt2r','gt2l','gt3r','gt3l']

    # 2. File Discovery
    atl24_list = os.listdir(atl24_dir)
    
    # 3. Main Loop
    for filename in atl24_list:
        # Find matching files
        atl03_file = find_corresponding_atl(filename, atl03_dir)
        atl08_file = find_corresponding_atl(filename, atl08_dir)
        atl24_file = os.path.join(atl24_dir, filename)
        
        if not (atl03_file and atl08_file):
            print(f"Skipping {filename}: Missing corresponding ATL03 or ATL08")
            continue

        for gt in gt_list:
            try:
                print(f"Processing {filename} - {gt}")
                file_out_name = f"{Path(atl03_file).stem}_{gt}"
    
                # ---------------------------------------------------------
                # STEP A: Read and Align Photon Data
                # ---------------------------------------------------------
                # Uses the new flexible reader
                df_ph = readers.read_photon_dataframe(atl03_file, gt, atl08_file, atl24_file)
                
                # Apply specific main.py logic (confidence filtering)
                if 'atl24_conf' in df_ph.columns:
                    df_ph.loc[(df_ph['atl24_class'] == 40) & (df_ph['atl24_conf'] > 0.6), 'atl24_class'] = 0
    
                # ---------------------------------------------------------
                # STEP B: Spatial Filtering
                # ---------------------------------------------------------
                extent_gdf = gpd.read_file(extent_gpkg)
                df_ph = processing.filter_df_by_extent(df_ph, extent_gdf.total_bounds)
                
                if df_ph.empty:
                    print('No data in extent')
                    continue
    
                # ---------------------------------------------------------
                # STEP C: Normalization & Geoid
                # ---------------------------------------------------------
                # Apply EGM2008
                geoid_offset = get_geoid_height(np.array(df_ph.longitude), np.array(df_ph.latitude), geoid_file)
                df_ph['ortho_h'] = df_ph.h_ph - geoid_offset
    
                # Calculate specific normalized heights needed for this study
                df_ph['h_topobathy_norm'] = analysis.normalize_heights(
                    df_ph, class_field='combined_class', ground_class=[1,40], target_height='h_ph'
                )
                df_ph['h_te_norm'] = analysis.normalize_heights(
                    df_ph, class_field='atl08_class', ground_class=[1], target_height='h_ph'
                )
    
                # ---------------------------------------------------------
                # STEP D: Aggregate Photons (using CONFIG)
                # ---------------------------------------------------------
                df_seg = analysis.aggregate_by_segment(df_ph, ATL_AGG_CONFIG, res=target_res)
    
                # ---------------------------------------------------------
                # STEP E: Process ALS Data
                # ---------------------------------------------------------
                # Find best UTM zone
                utm_epsg = find_utm_zone_epsg(extent_gdf.iloc[0].lat_min, extent_gdf.iloc[0].lon_min)
                extent_gdf = extent_gdf.to_crs(utm_epsg)
    
                # Load swath if available
                als_outfile = os.path.join(als_outdir, f'als_{file_out_name}.pqt')                
                if os.path.exists(als_outfile):
                    als_swath = pd.read_parquet(als_outfile)
                    als_swath = als_swath.drop('key_id', axis=1)
                else:
            
                    als_swath = create_als_swath(extent_gdf, df_seg)
                    
                    if len(als_swath) == 0:
                        print('No data ALS in extent')
                        continue
    
                    
                    # Calculate norm height for topobathy
                    als_swath['h_topobathy_norm'] = analysis.normalize_heights( 
                        als_swath, class_field = 'classification', ground_class = [2,40], target_height = 'z')
                    
                    # Calculate norm height for topobathy
                    als_swath['h_norm'] = analysis.normalize_heights(
                        als_swath, class_field = 'classification',ground_class = [2, 41], target_height = 'z')
                                        
                    # Calculate Ellipsoid Height
                    
                    transformer = Transformer.from_crs(int(utm_epsg[5:]),4326)
                    als_lat, als_lon  =transformer.transform(als_swath.x, als_swath.y)
                    geoid_offset = get_geoid_height(als_lon + 360, als_lat, als_geoid_file)
                    als_swath['ellip_h'] = als_swath.z + geoid_offset
                    als_swath['latitude'] = als_lat
                    als_swath['longitude'] = als_lon
                    
                    _,_,als_swath.ellip_h = convert_3d_nad83_to_wgs84(als_lon, als_lat, als_swath.ellip_h)
                    # test = test - 1.618700000000004
                    
                    
                    # Apply EGM2008
                    geoid_offset = get_geoid_height(als_lon, als_lat, geoid_file)
                    als_swath['ortho_h'] = als_swath.ellip_h - geoid_offset
                
                
                # als_swath.loc[(als_swath['classification'] == 1)  & (als_swath['h_norm'] > 0.1),'classification'] = 3
                als_swath.loc[(als_swath['classification'] == 1),'classification'] = 3
                
                
                als_seg = analysis.aggregate_by_segment(als_swath, ALS_AGG_CONFIG, res=target_res)
                merged_df = pd.merge(df_seg, als_seg, on='key_id', how='left')            
    
                # ---------------------------------------------------------
                # STEP F: Save Results
                # ---------------------------------------------------------
                # Add Metadata
                atl_info = get_attribute_info(atl03_file, gt)
                for key, val in atl_info.items():
                    merged_df[key] = val
    
                # Save (using your existing paths logic)
        
                df_outfile = os.path.join(df_outdir, f'merged_{target_res}m_{file_out_name}.pqt"')                
    
                merged_df.to_parquet(df_outfile)
                print(f"Saved: {file_out_name}")

            except Exception as e:
                print(f"Failed on {gt}: {e}")