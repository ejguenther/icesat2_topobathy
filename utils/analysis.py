#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:43:38 2025

@author: ejg2736
"""

import pandas as pd
import numpy as np
from typing import List, Union

def get_max98(series):
    try:
        max98 = np.percentile(series, 98)
    except:
        max98 = np.nan
    return max98

def get_len(series):
    try:
        length = len(series)
    except:
        length = 0
    return length

def get_len_unique(series):
    try:
        length = len(np.unique(series))
    except:
        length = 0
    return length

def get_mode(series):
    try:
        mode = series.mode()[0]
    except:
        mode = np.nan
    return mode

def get_skew(series):
    try:
        skew = series.skew()
    except:
        skew = np.nan
    return skew

def _percentile_post(arr, percent):
    pos = (len(arr) - 1) * (percent/100)

    if pos.is_integer():
        out = arr[int(pos)]
    else:
        out = ((arr[int(np.ceil(pos))] - arr[int(np.floor(pos))]) *\
               (pos % 1)) + arr[int(np.floor(pos))]
    return out

def percentile_rh(arr):
    arr = np.array(arr)
    arr.sort()
    out_list = []    
    try:    
        out_list.append(_percentile_post(arr, 10))
        out_list.append(_percentile_post(arr, 15))
        out_list.append(_percentile_post(arr, 20))
        out_list.append(_percentile_post(arr, 25))
        out_list.append(_percentile_post(arr, 30))
        out_list.append(_percentile_post(arr, 35))
        out_list.append(_percentile_post(arr, 40))
        out_list.append(_percentile_post(arr, 45))
        out_list.append(_percentile_post(arr, 50))
        out_list.append(_percentile_post(arr, 55))
        out_list.append(_percentile_post(arr, 60))
        out_list.append(_percentile_post(arr, 65))
        out_list.append(_percentile_post(arr, 70))
        out_list.append(_percentile_post(arr, 75))
        out_list.append(_percentile_post(arr, 80))
        out_list.append(_percentile_post(arr, 85))
        out_list.append(_percentile_post(arr, 90))
        out_list.append(_percentile_post(arr, 95))
    except:
        out_list = [np.nan]*18

    return out_list

def get_pai(series, height_threshold=1.5, k=0.5):
    try:
        total_returns = len(series)
        if total_returns == 0:
            return np.nan
        ground_returns = (series < height_threshold).sum()
        if ground_returns == 0:
            gap_fraction = 1e-4 
        else:
            gap_fraction = ground_returns / total_returns
        pai = -np.log(gap_fraction) / k
        return pai
    except:
        return np.nan

def get_canopy_cover(series, height_threshold=2.0):
    try:
        total_returns = len(series)
        if total_returns == 0:
            return 0.0
        canopy_returns = (series >= height_threshold).sum()
        canopy_cover_pct = (canopy_returns / total_returns) * 100.0
        return canopy_cover_pct
    except:
        return np.nan

def prepare_segment_bins(df_photons, bin_height_m=0.1, background_rate_hz=1e6):
    """
    PART 1: Vertical profile time-binning.
    Organizes mapped ATL03/ATL08 photons chronologically by time-of-arrival.
    """
    # 1. Determine instrument channel array scale
    M = 4 

    # Calculate the number of unique shots
    laser_shots_N = len(np.unique(df_photons.delta_time))
    
    # 2. Extract vertical bounds of the segment window
    max_h = df_photons['h_ph'].max()
    min_h = df_photons['h_ph'].min()
    
    # Create bin edges from HIGHEST to LOWEST elevation (Chronological order)
    bin_edges = np.arange(max_h, min_h - bin_height_m, -bin_height_m)
    num_bins = len(bin_edges) - 1
    
    # Initialize empty arrays for bin aggregation
    E_v = np.zeros(num_bins)   # Raw vegetation events
    E_g = np.zeros(num_bins)   # Raw ground events
    E_sn = np.zeros(num_bins)  # Total observed events
    N_n = np.zeros(num_bins)   # Target background noise counts per bin
    
    # 3. Calculate expected background noise photons per bin across the segment
    delta_t = (2 * bin_height_m) / 3e8  
    bin_noise_photons = background_rate_hz * delta_t * laser_shots_N * M
    
    # 4. Chronological aggregation loop
    for i in range(num_bins):
        upper_bound = bin_edges[i]
        lower_bound = bin_edges[i+1]
        
        bin_subset = df_photons[(df_photons['h_ph'] <= upper_bound) & 
                                (df_photons['h_ph'] > lower_bound)]
        
        E_v[i] = np.sum((bin_subset['atl08_class'] == 2) | (bin_subset['atl08_class'] == 2))
        E_g[i] = np.sum(bin_subset['atl08_class'] == 1)
        E_sn[i] = len(bin_subset)
        
        N_n[i] = bin_noise_photons
        
    return E_v, E_g, E_sn, N_n, M, laser_shots_N

def aggregate_segment_metrics(
    df_ph: pd.DataFrame, 
    df_seg: pd.DataFrame, 
    *, 
    key_field: str,
    field: str,
    operation: str,
    class_field: str,
    class_id: Union[int, List[int]],
    outfield: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Filters, groups, and aggregates photon data, then merges it into a segment DataFrame.

    This function uses keyword-only arguments for clarity and safety.

    Args:
        df_ph: DataFrame containing photon-level data.
        df_seg: DataFrame containing segment-level data.
        *: Denotes that all subsequent arguments must be specified by keyword.
        key_field: (Required) The column name used to group photons and merge results.
        field: (Required) The numeric field in df_ph to aggregate (e.g., 'h_ph').
        operation: (Required) The aggregation function (e.g., 'mean', 'median', 'std').
        class_field: (Required) The field in df_ph used for filtering.
        class_id: (Required) A class integer or list of integers to include.
        outfield: (Optional) The name for the new aggregated column. If None,
                  a descriptive name is generated (e.g., 'h_ph_mean').

    Returns:
        The df_seg DataFrame with the new aggregated column merged in.
    """
    # 1. Handle default output field name
    if outfield is None:
        outfield = f"{field}_{operation}"

    # 2. Make class_id robust: ensure it's a list for .isin()
    if isinstance(class_id, int):
        class_id = [class_id]

    # 3. Chain pandas operations for clarity and efficiency
    #    - Filter rows based on class_id
    #    - Group by the segment key
    #    - Aggregate the desired field, renaming the output column directly
    if operation == 'get_max98' or operation == 'max98':
        aggregated_data = (
            df_ph[df_ph[class_field].isin(class_id)]
            .groupby(key_field)
            .agg(
                 **{outfield: pd.NamedAgg(column=field, aggfunc=get_max98)}
            )
        )
    elif operation == 'get_len':
        aggregated_data = (
            df_ph[df_ph[class_field].isin(class_id)]
            .groupby(key_field)
            .agg(
                 **{outfield: pd.NamedAgg(column=field, aggfunc=get_len)}
            )
        )
        
    elif operation == 'get_len_unique':
        aggregated_data = (
            df_ph[df_ph[class_field].isin(class_id)]
            .groupby(key_field)
            .agg(
                 **{outfield: pd.NamedAgg(column=field, aggfunc=get_len)}
            )
        )

    elif operation == 'get_skew':
        aggregated_data = (
            df_ph[df_ph[class_field].isin(class_id)]
            .groupby(key_field)
            .agg(
                 **{outfield: pd.NamedAgg(column=field, aggfunc=get_skew)}
            )
        )

    elif operation == 'get_mode':
        aggregated_data = (
            df_ph[df_ph[class_field].isin(class_id)]
            .groupby(key_field)
            .agg(
                 **{outfield: pd.NamedAgg(column=field, aggfunc=get_mode)}
            )
        )

    elif operation == 'percentile_rh':
        aggregated_data = (
            df_ph[df_ph[class_field].isin(class_id)]
            .groupby(key_field)
            .agg(
                 **{outfield: pd.NamedAgg(column=field, aggfunc=percentile_rh)}
            )
        )
    
    elif operation == 'get_pai':
        height_threshold = kwargs.get('height_threshold', 1.5)
        k = kwargs.get('k', 0.5)
        pai_func = lambda x: get_pai(x, height_threshold=height_threshold, k=k)
        aggregated_data = (
            df_ph[df_ph[class_field].isin(class_id)]
            .groupby(key_field)
            .agg(
                 **{outfield: pd.NamedAgg(column=field, aggfunc=pai_func)}
            )
        )
    
    elif operation == 'get_canopy_cover':
        height_threshold = kwargs.get('height_threshold', 2.0)
        cc_func = lambda x: get_canopy_cover(x, height_threshold=height_threshold)
        aggregated_data = (
            df_ph[df_ph[class_field].isin(class_id)]
            .groupby(key_field)
            .agg(
                 **{outfield: pd.NamedAgg(column=field, aggfunc=cc_func)}
            )
        )
    
    elif operation == 'prepare_segment_bins':
        bin_height_m = kwargs.get('bin_height_m', 0.1)
        background_rate_hz = kwargs.get('background_rate_hz', 1e6)
        
        def apply_prep(df_group):
            E_v, E_g, E_sn, N_n, M, laser_shots_N = prepare_segment_bins(
                df_group, bin_height_m, background_rate_hz
            )
            return pd.Series(
                [E_v, E_g, E_sn, N_n, M, laser_shots_N],
                index=['E_v', 'E_g', 'E_sn', 'N_n', 'M', 'laser_shots_N']
            )
        
        aggregated_data = (
            df_ph[df_ph[class_field].isin(class_id)]
            .groupby(key_field)
            .apply(apply_prep)
        )
        
        # If outfield prefix is provided, prepend it to the new column names
        if outfield:
            aggregated_data.columns = [f"{outfield}_{c}" for c in aggregated_data.columns]
    
    else:
        aggregated_data = (
            df_ph[df_ph[class_field].isin(class_id)]
            .groupby(key_field)
            .agg(
                 **{outfield: pd.NamedAgg(column=field, aggfunc=operation)}
            )
        )
    
    # 4. Merge the aggregated results back into the segment DataFrame
    #    The result of a groupby is a Series or DataFrame with `key_field` as the index,
    #    so we merge on the index of the right DataFrame.
    df_seg_out = df_seg.merge(
        aggregated_data, 
        on=key_field, 
        how='left'
    )

    return df_seg_out


def normalize_heights(df, class_field = 'classification', 
                      ground_class = [2], 
                      ground_res = 1, 
                      target_height = 'z',
                      out_field = 'norm_h'):
    # Statis variables
    t_ind = np.int32(np.floor((df.alongtrack - np.min(df.alongtrack)) / ground_res))
    df['t_ind'] = t_ind
    
    
    df_g = df[df[class_field].isin(ground_class)]
    df_g = df_g[['t_ind','alongtrack',class_field,target_height]]
    zgroup = df_g.groupby('t_ind')
    zout = zgroup.aggregate("median")
    zout = zout.reindex(list(range(0,np.max(t_ind) + 1)))
    zout = zout.interpolate(method='linear', axis=0).ffill().bfill()
    ground = zout[target_height][df.t_ind]
    norm_height = np.array(df[target_height]) - np.array(ground)
    
    return norm_height

def aggregate_by_segment(df, config_list, res=20, min_at=None):
    """
    Generic function to aggregate a DataFrame into segments based on a configuration list.
    """
    # 1. Calculate Segment Keys (Common to all)
    if min_at is None:
        min_at = np.min(df.alongtrack)
        
    # Create segment ID based on resolution
    key = np.floor((df.alongtrack - min_at) / res).astype(int)
    df['key_id'] = key
    
    # Initialize Segment DataFrame
    df_seg = pd.DataFrame({'key_id': np.unique(key)})
    df_seg['alongtrack'] = ((np.unique(key) * res) + min_at) + (res / 2)

    # 2. Iterate through the configuration list
    for cfg in config_list:
        # Optional: Skip if the source column doesn't exist (e.g., h_topobathy_norm)
        if 'field' in cfg and cfg['field'] not in df.columns:
            continue
            
        # Extract kwargs if present
        op_kwargs = cfg.get('kwargs', {})
            
        # Call your existing single-metric aggregator
        df_seg = aggregate_segment_metrics(
            df, 
            df_seg, 
            key_field='key_id',
            field=cfg['field'],
            operation=cfg['operation'],
            class_field=cfg['class_field'],
            class_id=cfg['class_id'],
            outfield=cfg.get('outfield'),
            **op_kwargs
        )

    # 3. Standard Cleanup (can be customized if needed)
    if 'longitude' in df_seg.columns:
        df_seg.dropna(subset=['longitude', 'latitude'], inplace=True)

    return df_seg