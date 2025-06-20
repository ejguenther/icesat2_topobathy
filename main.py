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
# from utils import readers, processing, analysis
from utils import processing, analysis

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Union




    
def old_code():
    ## ATL03
    atl03_dir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/ICESat-2/REL006/florida_aoi/atl03'
    atl03_name = 'ATL03_20200212094736_07280607_006_01.h5'
    atl03_file = os.path.join(atl03_dir, atl03_name)
    f = h5py.File(atl03_file, 'r') 
    
    segment_id = np.array(f['gt1r/geolocation/segment_id'])
    ph_index_beg = np.array(f['gt1r/geolocation/ph_index_beg'])
    delta_time = np.array(f['gt1r/heights/delta_time'])
    lon_ph = np.array(f['gt1r/heights/lon_ph'])
    dist_ph_across = np.array(f['gt1r/heights/dist_ph_across'])
    dist_ph_along = np.array(f['gt1r/heights/dist_ph_along'])
    lat_ph = np.array(f['gt1r/heights/lat_ph'])
    h_ph = np.array(f['gt1r/heights/h_ph'])
    quality_ph = np.array(f['gt1r/heights/quality_ph'])
    signal_conf_ph = np.array(f['gt1r/heights/signal_conf_ph'])
    weight_ph = np.array(f['gt1r/heights/weight_ph'])
    
    alongtrack = processing.get_atl03_segment_to_photon(atl03_file,'gt1r','/geolocation/segment_dist_x')
    alongtrack = alongtrack + np.array(f['gt1r/heights/dist_ph_along'])
    
    solar_elevation = processing.get_atl03_segment_to_photon(atl03_file,'gt1r','/geolocation/solar_elevation')
    
    
    
    # ## ATL08
    atl08_dir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/ICESat-2/REL006/florida_aoi/atl08'
    atl08_name = 'ATL08_20200212094736_07280607_006_01.h5'
    atl08_file = os.path.join(atl08_dir, atl08_name)
    # f = h5py.File(atl08_file, 'r') 
    
    
    atl08_class_ph = processing.get_atl08_class_to_atl03(atl03_file, atl08_file,'gt1r')
    atl08_norm_h_ph = processing.get_atl08_norm_h_to_atl03(atl03_file, atl08_file,'gt1r')
    
    
    # ## ATL24
    atl24_dir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/ICESat-2/REL006/florida_aoi/atl24'
    atl24_name = 'ATL24_20200212094736_07280607_006_01_001_01.h5'
    atl24_file = os.path.join(atl24_dir, atl24_name)
        
    
    atl24_class_ph = processing.get_atl24_to_atl03(atl03_file, atl24_file, 'gt1r')
    atl24_confidence_ph = processing.get_atl24_to_atl03(atl03_file, atl24_file, 'gt1r','/confidence')
    atl24_ortho_h_ph = processing.get_atl24_to_atl03(atl03_file, atl24_file, 'gt1r','/ortho_h')
    
    
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
    
    res = 30
    key = np.floor((alongtrack - np.min(alongtrack))/res).astype(int)
    
    df_ph['key_id'] = key


def get_atl_at_photon_rate(atl03_file, atl08_file, atl24_file, gt):
    # Read ATL03
    f = h5py.File(atl03_file, 'r') 
    
    # Read photon rate ATL03 data
    # segment_id = np.array(f['gt1r/geolocation/segment_id'])
    # ph_index_beg = np.array(f['gt1r/geolocation/ph_index_beg'])
    # delta_time = np.array(f['gt1r/heights/delta_time'])
    lon_ph = np.array(f['gt1r/heights/lon_ph'])
    # dist_ph_across = np.array(f['gt1r/heights/dist_ph_across'])
    # dist_ph_along = np.array(f['gt1r/heights/dist_ph_along'])
    lat_ph = np.array(f['gt1r/heights/lat_ph'])
    h_ph = np.array(f['gt1r/heights/h_ph'])
    quality_ph = np.array(f['gt1r/heights/quality_ph'])
    # signal_conf_ph = np.array(f['gt1r/heights/signal_conf_ph'])
    # weight_ph = np.array(f['gt1r/heights/weight_ph'])
    
    # Read ATL03 segment rate at ATL03 photon rate
    solar_elevation = processing.get_atl03_segment_to_photon(atl03_file,'gt1r','/geolocation/solar_elevation')
    alongtrack = processing.get_atl03_segment_to_photon(atl03_file,'gt1r','/geolocation/segment_dist_x')
    alongtrack = alongtrack + np.array(f['gt1r/heights/dist_ph_along'])
    
    # Read ATL08 signal photon at ATL03 photon rate
    atl08_class_ph = processing.get_atl08_class_to_atl03(atl03_file, atl08_file,'gt1r')
    atl08_norm_h_ph = processing.get_atl08_norm_h_to_atl03(atl03_file, atl08_file,'gt1r')
    
    
    # Read ATL24 photon rate at ATL03 photon rate
    atl24_class_ph = processing.get_atl24_to_atl03(atl03_file, atl24_file, 'gt1r')
    # atl24_confidence_ph = processing.get_atl24_to_atl03(atl03_file, atl24_file, 'gt1r','/confidence')
    atl24_ortho_h_ph = processing.get_atl24_to_atl03(atl03_file, atl24_file, 'gt1r','/ortho_h')
    
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

def get_measurement_type_string(df: pd.DataFrame, columns: list) -> pd.Series:
    """
    Identifies which measurements are present in a row and returns a 
    single descriptive string.
    """
    # Create a boolean DataFrame where True means the value is not NaN
    bool_df = df[columns].notna()
    
    # Use a flexible apply function to join the column names for each row
    def get_names(row):
        # Get names of columns where the row value is True
        present_names = list(row.index[row])
        if present_names:
            return "-".join(present_names)
        return "none"

    return bool_df.apply(get_names, axis=1)
    
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
    

    # df_seg['comp_flag'] =(
    #     df_seg['comp_flag']
    #     .str.replace('h_te_median', 'terrain')
    #     .str.replace('h_canopy', 'canopy')
    #     .str.replace('h_surface', 'sea_surface')
    #     .str.replace('h_bathy', 'bathymetry')
    # )


    return df_seg



def plot_pie(sizes, labels, title = "Distribution of ATL08/ATL24 Classifications"):
    
    fig, ax = plt.subplots()
    wedges, texts = ax.pie(sizes, startangle=90)

    # Add annotations with leader lines
    for i, wedge in enumerate(wedges):
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "arc3,rad=.2" if np.sign(x) == -1 else "arc3,rad=-.2"
    
        ax.annotate(labels[i], xy=(x, y), xytext=(1.35*x, 1.4*y),
                    horizontalalignment=horizontalalignment,
                    arrowprops=dict(arrowstyle="wedge,tail_width=0.5",
                                    fc="0.6", ec="none", alpha=0.7,
                                    connectionstyle=connectionstyle))
    
    ax.set_title(title)
    plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


def plot_pie_with_other(counts: pd.Series, threshold: float = 0.05, title: str = ''):
    """
    Plots a pie chart, grouping small slices into an 'Other' category.

    Args:
        counts: A pandas Series of value counts.
        threshold: The proportion below which a category is considered "small".
        title: The title for the chart.
    """
    # Calculate percentages
    percentages = counts / counts.sum()
    
    # Identify small slices
    small_slices = percentages < threshold
    
    # Create a new series for plotting
    plot_series = counts.copy()
    if small_slices.any():
        # Sum the small slices and add them as 'Other'
        plot_series['Other'] = plot_series[small_slices].sum()
        # Remove the individual small slices
        plot_series = plot_series[~small_slices]

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use autopct to display percentages, using a lambda to avoid printing for small values
    wedges, texts, autotexts = ax.pie(
        plot_series,
        autopct=lambda p: f'{p:.1f}%' if p > threshold*100 else '',
        startangle=90,
        pctdistance=0.85, # Distance of percentage text from center
        explode=[0.05] * len(plot_series) # Slightly explode all slices
    )

    # Styling
    plt.setp(autotexts, size=12, weight="bold", color="white")
    ax.set_title(title, size=18, weight="bold")
    
    # Create a clean legend
    ax.legend(
        wedges,
        plot_series.index,
        title="Measurement Types",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=12
    )
    
    # Ensure the pie is a circle
    ax.axis('equal')
    
    plt.savefig("pie_chart_with_other.png")
    plt.show()
    

if __name__ == "__main__":
    atl03_dir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/ICESat-2/REL006/florida_aoi/atl03'
    atl03_name = 'ATL03_20200212094736_07280607_006_01.h5'
    atl03_file = os.path.join(atl03_dir, atl03_name)
    
    # ## ATL08
    atl08_dir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/ICESat-2/REL006/florida_aoi/atl08'
    atl08_name = 'ATL08_20200212094736_07280607_006_01.h5'
    atl08_file = os.path.join(atl08_dir, atl08_name)
    
    atl24_dir = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/ICESat-2/REL006/florida_aoi/atl24'
    atl24_name = 'ATL24_20200212094736_07280607_006_01_001_01.h5'
    atl24_file = os.path.join(atl24_dir, atl24_name)
    
    gt = 'gt1r'
    
    df_ph = get_atl_at_photon_rate(atl03_file, atl08_file, atl24_file, gt)
    
    df_seg = get_atl_at_seg(df_ph, res = 10)
    
    
    canopy_check = df_seg.h_canopy_abs - df_seg.h_surface
    df_seg.loc[canopy_check < 1, "h_canopy"] = np.nan
    
    # Identify composition within each segment
    comp_flag = get_measurement_type_string(df_seg, ['h_te_median','h_canopy',
                                                     'h_bathy','h_surface'])
    df_seg['comp_flag'] = comp_flag
    
    df_seg.loc[df_seg.longitude == 0,"longitude"] = np.nan
    df_seg.loc[df_seg.latitude == 0,"latitude"] = np.nan
    
    df_seg.dropna(subset =['longitude','latitude'],inplace=True)
    
    
    geometry_seg = [Point(xy) for xy in zip(df_seg.longitude, df_seg.latitude)]
    
    
    gdf_seg = gpd.GeoDataFrame(df_seg, geometry=geometry_seg, crs="EPSG:4326")
    
    gdf_seg.to_file("mangrove_test5_10m.gpkg", layer='atl08atl24', driver="GPKG")
    
    
    
    

# Create point geometry
# geometry_ph = [Point(xy) for xy in zip(lon_ph, lat_ph)]

# # Create GeoDataFrame
# gdf_ph = gpd.GeoDataFrame(df_ph, geometry=geometry_ph, crs="EPSG:4326")

# df_seg = pd.DataFrame({'key_id':np.unique(key)})

# df_seg_out = aggregate_segment_metrics(df_ph, df_seg, field = 'h_ph', 
#                               class_field = 'combined_class', 
#                               class_id = [2,3], key_field = 'key_id',
#                               operation = 'mean', outfield = '')

# def aggregate_segment_metrics(df_ph, df_seg, field = 'h_ph', 
#                               class_field = 'combined_class', 
#                               class_id = [2,3], key_field = 'key_id',
#                               operation = 'mean', outfield = None):
#     if outfield == '':
#         outfield = field + '_' + operation
#     operation = 'mean'
#     key_field = 'key_id'
#     field = 'h_ph'
#     class_field = "combined_class"
#     class_id = [2,3] # An int or a list of ints
#     df_filter = df_ph[df_ph[class_field].isin(class_id)]
#     df_filter = df_filter[[key_field, field]]
#     zgroup = df_filter.groupby(key_field)
#     zout = zgroup.aggregate(operation)
#     zout[outfield] = zout[field]
#     zout = zout.filter([outfield,key_field])
#     df_seg = df_seg.merge(zout, on=key_field,how='left')  
#     return df_seg


# def calculate_seg_meteric(df_in, classification, field, 
#                outfield, key_field = 'bin_id', classfield = 'classification'):
#     df_filter = df_in[df_in[classfield].isin(classification)]
#     df_filter.drop(columns=df_filter.columns.difference([key_field,field]),
#                    inplace=True)
#     zgroup = df_filter.groupby(key_field)
#     zout = zgroup.aggregate(operation)
#     zout[key_field] = zout.index
#     zout = zout.reset_index(drop = True)
#     # zout['segment_id_beg'] = zout['seg_id']
#     zout[outfield] = zout[field]
#     zout = zout.filter([outfield,key_field])
#     # df_out = df_out.merge(zout, on=key_field,how='left')  
#     return df_out


# def aggregate_segment_metrics(
#     df_ph: pd.DataFrame, 
#     df_seg: pd.DataFrame, 
#     *, 
#     key_field: str,
#     field: str,
#     operation: str,
#     class_field: str,
#     class_id: Union[int, List[int]],
#     outfield: str = None
# ) -> pd.DataFrame:
#     """
#     Filters, groups, and aggregates photon data, then merges it into a segment DataFrame.

#     This function uses keyword-only arguments for clarity and safety.

#     Args:
#         df_ph: DataFrame containing photon-level data.
#         df_seg: DataFrame containing segment-level data.
#         *: Denotes that all subsequent arguments must be specified by keyword.
#         key_field: (Required) The column name used to group photons and merge results.
#         field: (Required) The numeric field in df_ph to aggregate (e.g., 'h_ph').
#         operation: (Required) The aggregation function (e.g., 'mean', 'median', 'std').
#         class_field: (Required) The field in df_ph used for filtering.
#         class_id: (Required) A class integer or list of integers to include.
#         outfield: (Optional) The name for the new aggregated column. If None,
#                   a descriptive name is generated (e.g., 'h_ph_mean').

#     Returns:
#         The df_seg DataFrame with the new aggregated column merged in.
#     """
#     # 1. Handle default output field name
#     if outfield is None:
#         outfield = f"{field}_{operation}"

#     # 2. Make class_id robust: ensure it's a list for .isin()
#     if isinstance(class_id, int):
#         class_id = [class_id]

#     # 3. Chain pandas operations for clarity and efficiency
#     #    - Filter rows based on class_id
#     #    - Group by the segment key
#     #    - Aggregate the desired field, renaming the output column directly
#     aggregated_data = (
#         df_ph[df_ph[class_field].isin(class_id)]
#         .groupby(key_field)
#         .agg(
#              **{outfield: pd.NamedAgg(column=field, aggfunc=operation)}
#         )
#     )
    
#     # 4. Merge the aggregated results back into the segment DataFrame
#     #    The result of a groupby is a Series or DataFrame with `key_field` as the index,
#     #    so we merge on the index of the right DataFrame.
#     df_seg_out = df_seg.merge(
#         aggregated_data, 
#         on=key_field, 
#         how='left'
#     )

#     return df_seg_out


# def aggregate_segment_metrics(df_ph, df_seg, field = 'h_ph', 
#                               class_field = 'combined_class', 
#                               class_id = [2,3], key_field = 'key_id',
#                               operation = 'mean', outfield = None):
#     if outfield == '':
#         outfield = field + '_' + operation
#     operation = 'mean'
#     key_field = 'key_id'
#     field = 'h_ph'
#     class_field = "combined_class"
#     class_id = [2,3] # An int or a list of ints
#     df_filter = df_ph[df_ph[class_field].isin(class_id)]
#     df_filter = df_filter[[key_field, field]]
#     zgroup = df_filter.groupby(key_field)
#     zout = zgroup.aggregate(operation)
#     zout[outfield] = zout[field]
#     zout = zout.filter([outfield,key_field])
#     df_seg = df_seg.merge(zout, on=key_field,how='left')  
#     return df_seg


# def aggregate_segment_metrics(df_ph, df_seg, field = 'h_ph', 
#                               class_field = 'combined_class', 
#                               class_id = [2,3], key_field = 'key_id',
#                               operation = 'mean', outfield = None):
#     if outfield == '':
#         outfield = field + '_' + operation
#     operation = 'mean'
#     key_field = 'key_id'
#     field = 'h_ph'
#     class_field = "combined_class"
#     class_id = [2,3] # An int or a list of ints
#     df_filter = df_ph[df_ph[class_field].isin(class_id)]
#     df_filter = df_filter[[key_field, field]]
#     zgroup = df_filter.groupby(key_field)
#     zout = zgroup.aggregate(operation)
#     zout[outfield] = zout[field]
#     zout = zout.filter([outfield,key_field])
#     df_seg = df_seg.merge(zout, on=key_field,how='left')  
#     return df_seg



# Plot

# plt.figure()
# plt.plot(alongtrack[combined_class_ph == 0][::100],h_ph[combined_class_ph == 0][::100],'.',color=[0.8,0.8,0.8],label='Unclassified')
# plt.plot(alongtrack[combined_class_ph == 3],h_ph[combined_class_ph == 3],'.',color=[0.20392157, 0.70196078, 0.20392157],label='Top of Canopy (3)')
# plt.plot(alongtrack[combined_class_ph == 2],h_ph[combined_class_ph == 2],'.',color=[0.12156863, 0.41960784, 0.12156863],label='Canopy (2)')
# plt.plot(alongtrack[combined_class_ph == 1],h_ph[combined_class_ph == 1],'.',color=[0.69803922, 0.44313725, 0.23921569],label='Terrain (1)')
# plt.plot(alongtrack[combined_class_ph == 40],h_ph[combined_class_ph == 40],'.',color=[0.96078431, 0.81960784, 0.59215686],label='Subaqueous Terrain (40)')
# plt.plot(alongtrack[combined_class_ph == 41],h_ph[combined_class_ph == 41],'.',color=[0.        , 0.61568627, 0.76862745],label='Water Surface (41)')
# plt.xlabel('Alongtrack (m)')
# plt.ylabel('Ellipsoid Height (m)')
# plt.legend()
# plt.title('ATL08 and ATL24\nATL24_20200212094736_07280607_006_01_001_01 (gt1r)')



# x_atc = np.array(f['gt1r/x_atc'])
# lat_ph24 = np.array(f['gt1r/lat_ph'])

# index_ph24 = np.array(f['gt1r/index_ph'])
# index_seg24 = np.array(f['gt1r/index_seg'])
# class_ph24 = np.array(f['gt1r/class_ph'])


# class_ph = np.array(f['gt1r/class_ph'])
# ellipse_h = np.array(f['gt1r/ellipse_h'])


# x_atc_0 = x_atc[class_ph == 0]
# ellipse_h_0 = ellipse_h[class_ph == 0]

# x_atc_40 = x_atc[class_ph == 40]
# ellipse_h_40 = ellipse_h[class_ph == 40]

# x_atc_41 = x_atc[class_ph == 41]
# ellipse_h_41 = ellipse_h[class_ph == 41]

# plt.figure()
# plt.plot(lat_ph03[::1000],ellipse_h03[::1000],'.',color=[0.8,0.8,0.8],label='ATL03')
# plt.plot(lat_ph24[::1000],ellipse_h[::1000],'.',color=[0.3,0.3,0.3],label='ATL24')


# plt.figure()
# plt.plot(x_atc_0,ellipse_h_0,'.',color=[0.8,0.8,0.8],label='Class 0')
# plt.plot(x_atc_40,ellipse_h_40,'.',color=[0.96078431, 0.81960784, 0.59215686],label='Class 40')
# plt.plot(x_atc_41,ellipse_h_41,'.',color=[0.        , 0.61568627, 0.76862745],label='Class 41')


