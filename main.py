#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 19:54:03 2025

@author: ejg2736
"""

import h5py
import os
import numpy as np
from utils import processing, analysis
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from utils.geographic_utils import find_utm_zone_epsg, get_geoid_height
from utils.datum_transforms import convert_wgs84_to_nad83_manual
from utils.create_las_swath import create_als_swath

from pyproj import Transformer


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

def get_atl_at_seg(df_ph, res = 20, min_at = None):
    if not min_at:
        min_at = np.min(df_ph.alongtrack)
    key = np.floor((df_ph.alongtrack - min_at)/res).astype(int)

    df_ph['key_id'] = key
    df_seg = pd.DataFrame({'key_id':np.unique(key)})
    
    # Calculate median alongtrack
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'alongtrack',
        operation = 'median',
        class_field = 'atl08_class',
        class_id = [1,2,3,40,41],
        outfield = 'alongtrack_als'
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
    
    # Calculate h_bathy, the median bathymetic height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(df_ph, df_seg, 
        key_field = 'key_id',
        field = 'ortho_h',
        operation = 'median',
        class_field = 'combined_class',
        class_id = [1,40],
        outfield = 'h_topobathy'
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


def get_als_at_seg(als_swath, res = 20, min_at = None, height = 'ellip_h'):
    if not min_at:
        min_at = np.min(als_swath.alongtrack)
    key = np.floor((als_swath.alongtrack - min_at)/res).astype(int)

    als_swath['key_id'] = key
    df_seg = pd.DataFrame({'key_id':np.unique(key)})
    
    # Calculate median alongtrack
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'alongtrack',
        operation = 'median',
        class_field = 'classification',
        class_id = [ 1,  2,  7, 40, 41, 45],
        outfield = 'alongtrack'
    )    

    # Calculate te_median, median terrain height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'z',
        operation = 'median',
        class_field = 'classification',
        class_id = [2],
        outfield = 'als_topo'
    )    
    
    # Calculate te_median, median terrain height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'z',
        operation = 'median',
        class_field = 'classification',
        class_id = [40],
        outfield = 'als_bathy'
    )    

    
    # Calculate te_median, median terrain height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'z',
        operation = 'median',
        class_field = 'classification',
        class_id = [2,40],
        outfield = 'als_topobathy'
    )    
    
    
    # Calculate te_median, median terrain height (orthometric)
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'z',
        operation = 'median',
        class_field = 'classification',
        class_id = [41],
        outfield = 'als_surface'
    )    


    # Calculate h_canopy, 98th percentile canopy height relative to ground
    df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
        key_field = 'key_id',
        field = 'z',
        operation = 'get_max98',
        class_field = 'classification',
        class_id = [1],
        outfield = 'als_unclassed'
    )
    
    # # Calculate h_canopy_abs, 98th percentile canopy height (orthometric)
    # df_seg = analysis.aggregate_segment_metrics(als_swath, df_seg, 
    #     key_field = 'key_id',
    #     field = 'z',
    #     operation = 'max98',
    #     class_field = 'classification',
    #     class_id = [2,3],
    #     outfield = 'h_canopy_abs'
    # )
    
    # # Filter NaN data
    # df_seg.loc[df_seg.longitude == 0,"longitude"] = np.nan
    # df_seg.loc[df_seg.latitude == 0,"latitude"] = np.nan
    # df_seg.dropna(subset =['longitude','latitude'],inplace=True)

    
    # # Filter low canopy out
    # canopy_check = df_seg.h_canopy_abs - df_seg.h_surface
    # df_seg.loc[canopy_check < 1, "h_canopy"] = np.nan
    
    # Identify composition of each segment
    comp_flag = processing.get_measurement_type_string(df_seg, ['als_topo','als_bathy',
                                                     'als_surface','als_unclassed'])
    df_seg['als_comp_flag'] = comp_flag
    df_seg['als_comp_flag'] =(
        df_seg['als_comp_flag']
        .str.replace('als_topo', 'terrain')
        .str.replace('als_unclassed', 'unclassed')
        .str.replace('als_surface', 'sea_surface')
        .str.replace('als_bathy', 'bathymetry')
        )
    


    return df_seg



def filter_df_by_extent(df, extent):
    minx, miny, maxx, maxy = extent
    filtered_df = df[
    (df['longitude'] >= minx) & (df['longitude'] <= maxx) &
    (df['latitude'] >= miny) & (df['latitude'] <= maxy)
    ]
    
    return filtered_df
    
def compute_metrics(ref, measure):
    err = ref - measure
    err = err.dropna()
    n = len(err)
    mean_err = np.mean(err)
    mae = np.mean(np.abs(err))
    squared_diff = (err)**2
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    return n, mean_err, mae, mse, rmse
    
    


if __name__ == "__main__":
    # Define ATL03 File
    atl03_dir = '/Data/ICESat-2/REL006/florida_aoi/atl03'

    atl03_name = 'ATL03_20200212094736_07280607_006_01.h5'
    atl03_file = os.path.join(atl03_dir, atl03_name)
    
    # Define ATL08 File
    atl08_dir = 'Data/ICESat-2/REL006/florida_aoi/atl08'
    atl08_name = 'ATL08_20200212094736_07280607_006_01.h5'
    atl08_file = os.path.join(atl08_dir, atl08_name)
    
    # Define ATL24 file
    atl24_dir = '/Data/ICESat-2/REL006/florida_aoi/atl24'
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
    gdf_seg.to_file("is2_topobathy_" + gt + "_test.gpkg", layer='atl08atl24', driver="GPKG")
    
    
    # Read las extent
    extent_gpkg = 'fl_west_Everglades_laz_extent1.gpkg'
    extent_gdf = gpd.read_file(extent_gpkg)
    
    # Trim Data
    df_ph = filter_df_by_extent(df_ph, extent_gdf.total_bounds)
    
    # Aggregate photon rate DF to 10 m segment 
    df_seg = get_atl_at_seg(df_ph, res = 10)
    
    # Trim
    
    # Find best UTM zone
    utm_epsg = find_utm_zone_epsg(extent_gdf.iloc[0].lat_min,extent_gdf.iloc[0].lon_min)
    extent_gdf = extent_gdf.to_crs(utm_epsg) # Convert extent to UTM
    
    # Read ALS tile
    als_swath = create_als_swath(extent_gdf, df_seg)
    
    # Calculate Ellipsoid Height
    geoid_file = 'us_noaa_g2012b.tif'

    transformer = Transformer.from_crs(int(utm_epsg[5:]),4326)
    als_lat, als_lon  =transformer.transform(als_swath.x, als_swath.y)
    geoid_offset = get_geoid_height(als_lon + 360, als_lat, geoid_file)
    als_swath['ellip_h'] = als_swath.z + geoid_offset
    _,_,als_swath.ellip_h = convert_wgs84_to_nad83_manual(als_lon, als_lat, als_swath.ellip_h)
    # test = test - 1.618700000000004
    
    
    als_seg = get_als_at_seg(als_swath, res = 10, min_at = np.min(df_ph.alongtrack))
    
    
    merged_df = pd.merge(df_seg, als_seg, on='key_id', how='left')
    
    compute_metrics(merged_df.als_topobathy, merged_df.h_topobathy)
    compute_metrics(merged_df.als_topo, merged_df.h_te_median)
    compute_metrics(merged_df.als_bathy, merged_df.h_bathy)
    compute_metrics(merged_df.als_unclassed, merged_df.h_canopy_abs)
    
    
    
    import matplotlib.pyplot as plt
    
    df_als = als_swath
    df_als['lat'] = als_lat
    title = 'test'
    plt.figure()
    plt.plot(df_als.lat[::10],df_als.ellip_h[::10],'.',label='ALS')
    plt.plot(df_ph.latitude[::10],df_ph.h_ph[::10],'.',label='ICESat-2')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Ellipsoid Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()
    
    plt.figure()
    plt.plot(df_als.alongtrack[::10],df_als.ellip_h[::10],'.',label='ALS')
    plt.plot(df_ph.alongtrack,df_ph.h_ph,'.',label='ICESat-2')
    plt.xlabel('Alongtrack (m)')
    plt.ylabel('Ellipsoid Height (m)')
    plt.legend(markerscale=3)
    plt.title(title)
    plt.show()
