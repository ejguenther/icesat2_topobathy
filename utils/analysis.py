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
        out_list.append(_percentile_post(arr, 20))
        out_list.append(_percentile_post(arr, 25))
        out_list.append(_percentile_post(arr, 30))
        out_list.append(_percentile_post(arr, 40))
        out_list.append(_percentile_post(arr, 50))
        out_list.append(_percentile_post(arr, 60))
        out_list.append(_percentile_post(arr, 70))
        out_list.append(_percentile_post(arr, 75))
        out_list.append(_percentile_post(arr, 80))
        out_list.append(_percentile_post(arr, 90))
        out_list.append(_percentile_post(arr, 98))
        out_list.append(_percentile_post(arr, 100))
    except:
        out_list = [np.nan, np.nan,np.nan,np.nan,np.nan,np.nan,
                                np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                                np.nan]

    return out_list

def aggregate_segment_metrics(
    df_ph: pd.DataFrame, 
    df_seg: pd.DataFrame, 
    *, 
    key_field: str,
    field: str,
    operation: str,
    class_field: str,
    class_id: Union[int, List[int]],
    outfield: str = None
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