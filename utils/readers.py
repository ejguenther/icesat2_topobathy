#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:33:35 2025

@author: ejg2736
"""

import os
import numpy as np
import h5py
from . import processing
import pandas as pd
from geographic_utils import get_df_distance

def __appendGlobalList(name):
    if name:
        global_list.append(name)

def read_h5(in_file03, fieldName, label):

    # Initialize output
    dataOut = []
    
    if not os.path.isfile(in_file03):
      print('.h5 file does not exist')
    try:
      with h5py.File(in_file03, 'r') as f:
          dsname=''.join([label, fieldName])
          if dsname in f:
              dataOut = np.array(f[dsname])
              if('signal_conf_ph' in fieldName.lower()):
                  dataOut = dataOut[:,0]
          else:
              dataOut = []
    except Exception as e:
        print('Python message: %s\n' % e)
    return dataOut

def get_h5_keys(h5_file,group = None):
    global global_list
    global_list = []
    try:
        h = h5py.File(h5_file, 'r')
    except:
        print("Could not find file or file was not proper H5 file")
    if group:
        group = str(group)
        h[group].visit(__appendGlobalList)
        
    return global_list

def get_h5_keys_info(atl08filepath,gt):
    keys = get_h5_keys(atl08filepath, gt)
    h = h5py.File(atl08filepath, 'r')
    key_name = []
    key_type = []
    key_len = []
    for key in keys:
        try:
            data = h[gt + '/' + key]
            kname = str(key)
            ktype = str(data.dtype)
            klen = int(len(data))
            key_name.append(kname)
            key_type.append(ktype)
            key_len.append(klen)
        except:
            kname = str(key)
            ktype = 'Group'
            klen = 0
            key_name.append(kname)
            key_type.append(ktype)
            key_len.append(klen)
    key_info = [list(a) for a in zip(key_name, key_type, key_len)]
    return key_info

def get_precise_segment_dist(row, geoseg_df):
    """
    Calculates the alongtrack distance
    between the 'id_start' and 'id_end' of the current row in df2 (inclusive).
    """
    start = row['segment_id_beg']
    end = row['segment_id_end']
    
    # 1. Filter df1: Select rows where df1['id'] is between start and end
    # The .between() method is useful for this
    filtered_df = geoseg_df[geoseg_df['segment_id'].between(start, end, inclusive='both')]
    # 2. Sum the 'segment_length' column of the filtered result.
    # Use .sum() on the 'apples' column. If no rows match, .sum() will return 0.
    total_segment_length = filtered_df['segment_length'].sum()
    
    return total_segment_length

# Read ATL03 Heights, put in Pandas DF
def read_atl08_land_segments(atl08filepath, gt):
    # Iterate through keys for "Land Segments"
    keys = get_h5_keys(atl08filepath,gt + '/land_segments')
    key_info = get_h5_keys_info(atl08filepath,gt + '/land_segments')
    # Read each key, put it in pandas df
    for idx, key in enumerate(keys):
        data = read_h5(atl08filepath, '/land_segments/' + key, gt)
        if key_info[idx][1] != 'Group':
            if idx == 0:
                df = pd.DataFrame(data,columns=[key.split('/')[-1]])
            else:
                if len(data.shape) == 2:
                    cols = data.shape[1]
                    for idx2 in range(0,cols):
                        df = pd.concat([df,pd.DataFrame(data[:,idx2],columns=\
                                                        [key.split('/')[-1] +\
                                                         '_' + str(idx2)])],
                                       axis=1)                        
                else:
                    df = pd.concat([df,pd.DataFrame(data,columns=\
                                                    [key.split('/')[-1]])],
                                   axis=1)
    return df

def read_atl08(atl08_file, gt, atl03_file = None):
    precise_segment_at = True
    # Read all land segments
    df_atl08 = read_atl08_land_segments(atl08_file, gt)
    
    # Read geolocation ATL03 data to calculate alongtrack
    if atl03_file:
        f = h5py.File(atl03_file, 'r') 
        segment_id = np.array(f[gt + '/geolocation/segment_id'])
        segment_dist_x = np.array(f[gt + '/geolocation/segment_dist_x'])
        segment_length = np.array(f[gt + '/geolocation/segment_length'])
        
        data = {
        'segment_id': segment_id,
        'segment_dist_x': segment_dist_x,
        'segment_length': segment_length,
        }
        
        df_geoseg = pd.DataFrame(data)
        
        if precise_segment_at:
            '''
            Precise implementation to calculate ATL08 alongtrack
            '''
            df_atl08['total_segment_length'] = df_atl08.apply(lambda row: get_precise_segment_dist(row, df_geoseg),axis=1)
            df_atl08 = pd.merge(df_atl08,df_geoseg,left_on = 'segment_id_beg',right_on = 'segment_id',how='left')
            df_atl08['alongtrack'] = df_atl08['segment_dist_x'] + (df_atl08['total_segment_length'] / 2)
            df_atl08 = df_atl08.drop(columns=['segment_id','segment_dist_x','segment_length','total_segment_length'])  
        else:
            '''
            Fast implementation to calculate ATL08 but makes assumptions
            '''
            df_atl08.loc[df_atl08.subset_can_flag_0 == -127,'subset_te_flag_0'] = -1
            df_atl08.loc[df_atl08.subset_can_flag_1 == -127,'subset_te_flag_1'] = -1
            df_atl08.loc[df_atl08.subset_can_flag_2 == -127,'subset_te_flag_2'] = -1
            df_atl08.loc[df_atl08.subset_can_flag_3 == -127,'subset_te_flag_3'] = -1
            df_atl08.loc[df_atl08.subset_can_flag_4 == -127,'subset_te_flag_4'] = -1
                    
            conditions = [
                (df_atl08['subset_te_flag_0'] > -1),
                (df_atl08['subset_te_flag_1'] > -1),
                (df_atl08['subset_te_flag_2'] > -1),
                (df_atl08['subset_te_flag_3'] > -1),
                (df_atl08['subset_te_flag_4'] > -1)
                ]
        
            # Create a list of the values to assign for each condition
            choices = [50, 30, 10, -10, -30]
            
            # Use np.select to create the new column
            df_atl08['alongtrack'] = np.select(conditions, choices, default=0)
            
            df_atl08 = pd.merge(df_atl08,df_geoseg,left_on = 'segment_id_beg',right_on = 'segment_id',how='left')
            df_atl08['alongtrack'] = df_atl08['segment_dist_x'] + df_atl08['alongtrack'] 
            df_atl08 = df_atl08.drop(columns=['segment_id','segment_dist_x','segment_length'])  
    else:
        '''
        Estimates alongtrack for ATL08 is not ATL03 is given
        This method does not require an ATL03 but will slightly underestimate 
        the alongtrack distance because it calculates the closest distance 
        between the two centroids, and not the orbital path. Every ~100 m 
        segment will underestimate just a little, but will compound as the 
        track moves along.  In testing it underestimated alongtrack bins at
        approximtely relative alongtrack of 936 km by 60 m.
        
        '''
        print('Warning: Estimating ATL08 Alongtrack without ATL03 Segments')
        df_atl08 = get_df_distance(df_atl08)
        df_atl08['distance_meters'] = df_atl08['distance_meters'].cumsum()
        df_atl08 = df_atl08.rename(columns={'distance_meters':'alongtrack'})
    return df_atl08

def expand_atl08_20m_df(df):
    # --- 1. Add a unique ID for each 100m row ---
    df_with_id = df.reset_index().rename(columns={'index': 'segment_100m_index'})

    # --- 2. Define the 'stubnames' (the prefixes) ---
    stub_prefixes = ['latitude_20m', 'longitude_20m','h_te_best_fit_20m', 
                     'h_canopy_20m', 'subset_te_flag','subset_can_flag']
    
    
    # --- 3. Perform the wide-to-long transformation ---
    try:
        df_long = pd.wide_to_long(
            df_with_id,
            stubnames=stub_prefixes,
            i='segment_100m_index',     # The column that IDs each 100m segment
            j='sub_segment_index',   # The name for the new column (0, 1, 2, 3, 4)
            sep='_'                  # The separator between prefix and number
        )
    except ValueError as e:
        print(f"Error during transformation: {e}")
        df_long = pd.DataFrame() # Create empty for rest of script to run
    
    if not df_long.empty:
        # --- 5. Clean up the resulting DataFrame ---
        df_long = df_long.reset_index()
    
        # Sort for readability (optional, but recommended)
        df_long = df_long.sort_values(by=['segment_100m_index', 'sub_segment_index'])
        df_long = df_long.reset_index()
        
    return df_long

def read_atl08_20m(atl08_file, gt):
    df_atl08 = read_atl08_land_segments(atl08_file, gt)
    df_atl08_20m = expand_atl08_20m_df(df_atl08)
    return df_atl08_20m


def read_atl03_data_mapping(filepath: str, beam_label: str):
    """
    Reads specified datasets from an ATL03 HDF5 file for a given beam.

    Args:
        filepath: Path to the ATL03 .h5 file.
        beam_label: The beam label (e.g., 'gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r').
        return_delta_time: If True, also returns the delta_time dataset.

    Returns:
        A tuple containing:
        - ph_index_beg (np.array): Photon index beginning.
        - segment_id (np.array): Segment ID.
        Returns empty arrays for all expected outputs if the file or datasets
        cannot be read.

    Raises:
        FileNotFoundError: If the input file does not exist.
        IOError: If there's an issue opening or reading the HDF5 file.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")

    ph_index_beg = np.array([])
    segment_id = np.array([])

    try:
        with h5py.File(filepath, 'r') as h5_file:
            # Helper function to read a dataset if it exists
            def get_dataset(name):
                full_path = f"{beam_label}/{name}"
                if full_path in h5_file:
                    return np.array(h5_file[full_path])
                print(f"Warning: Dataset '{full_path}' not found in {filepath}.")
                return np.array([])

            segment_id = get_dataset('geolocation/segment_id')
            ph_index_beg = get_dataset('geolocation/ph_index_beg')

    except Exception as e:
        print(f"Error reading HDF5 file {filepath}: {e}")
        pass # Errors within get_dataset are handled there.


    return ph_index_beg, segment_id


def read_atl08_data_mapping(filepath: str, beam_label: str):
    """
    Reads specified datasets from an ATL08 HDF5 file for a given beam.

    Args:
        filepath: Path to the ATL08 .h5 file.
        beam_label: The beam label (e.g., 'gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r').

    Returns:
        A tuple containing:
        - classed_pc_indx (np.array): Classified photon index.
        - classed_pc_flag (np.array): Classified photon flag.
        - classed_index_seg (np.array): ATL08 segment ID.
        Returns empty arrays for all expected outputs if the file or datasets
        cannot be read.

    Raises:
        FileNotFoundError: If the input file does not exist.
        IOError: If there's an issue opening or reading the HDF5 file.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")

    classed_pc_indx = np.array([])
    classed_pc_flag = np.array([])
    classed_index_seg = np.array([])

    try:
        with h5py.File(filepath, 'r') as h5_file:
            # Helper function to read a dataset if it exists
            def get_dataset(name):
                full_path = f"{beam_label}/{name}"
                if full_path in h5_file:
                    return np.array(h5_file[full_path])
                print(f"Warning: Dataset '{full_path}' not found in {filepath}.")
                return np.array([])

            # Datasets for ATL08
            # Note: Original code had '/signal_photons/ph_segment_id' for classed_index_seg.
            # Assuming this path is correct.
            classed_pc_indx = get_dataset('signal_photons/classed_pc_indx')
            classed_pc_flag = get_dataset('signal_photons/classed_pc_flag')
            classed_index_seg = get_dataset('signal_photons/ph_segment_id')
            ph_h = get_dataset('signal_photons/ph_h')


    except Exception as e:
        print(f"Error reading HDF5 file {filepath}: {e}")
        # To match original behavior of returning empty lists on any error after file check:
        return np.array([]), np.array([]), np.array([])


    return classed_pc_indx, classed_pc_flag, classed_index_seg, ph_h

def read_photon_dataframe(atl03_file, gt, atl08_file=None, atl24_file=None):
    """
    Reads ATL03 and optionally merges ATL08 and ATL24 data.
    Returns a single aligned DataFrame.
    """
    # 1. Base: Read ATL03 (This is common to ALL workflows)
    with h5py.File(atl03_file, 'r') as f:
        # Read ATL03
        lon_ph = np.array(f[gt + '/heights/lon_ph'])
        lat_ph = np.array(f[gt + '/heights/lat_ph'])
        h_ph = np.array(f[gt + '/heights/h_ph'])
        
        # Read photon rate ATL03 data
        lon_ph = np.array(f[gt + '/heights/lon_ph'])
        lat_ph = np.array(f[gt + '/heights/lat_ph'])
        h_ph = np.array(f[gt + '/heights/h_ph'])
        quality_ph = np.array(f[gt + '/heights/quality_ph'])
        delta_time = np.array(f[gt + '/heights/delta_time'])
        
        # Read ATL03 segment rate at ATL03 photon rate
        solar_elevation = processing.get_atl03_segment_to_photon(atl03_file,gt,'/geolocation/solar_elevation')
        alongtrack = processing.get_atl03_segment_to_photon(atl03_file,gt,'/geolocation/segment_dist_x')
        alongtrack = alongtrack + np.array(f[gt + '/heights/dist_ph_along'])

    # Initialize Dictionary
    data_dict = {
        "latitude": lat_ph,
        "longitude": lon_ph,
        "h_ph": h_ph,
        "quality_ph": quality_ph,
        "delta_time": delta_time,
        "alongtrack": alongtrack,
        "solar_elevation": solar_elevation,
        "alongtrack": alongtrack,
    }

    # Optional: Add ATL08
    if atl08_file:
        atl08_class = processing.get_atl08_class_to_atl03(atl03_file, atl08_file, gt)
        atl08_norm = processing.get_atl08_norm_h_to_atl03(atl03_file, atl08_file, gt)
        data_dict["atl08_class"] = atl08_class
        data_dict["h_norm"] = atl08_norm

    # Optional: Add ATL24
    if atl24_file:
        atl24_class = processing.get_atl24_to_atl03(atl03_file, atl24_file, gt)
        atl24_ortho = processing.get_atl24_to_atl03(atl03_file, atl24_file, gt, '/ortho_h')
        
        data_dict["atl24_class"] = atl24_class
        data_dict["ortho_h"] = atl24_ortho

        # Handle logic that requires both ATL08 and ATL24
        if atl08_file:
            combined = processing.combine_atl08_and_atl24_classifications(data_dict["atl08_class"], atl24_class)
            contested = processing.identify_contested_photons(data_dict["atl08_class"], atl24_class)
            data_dict["combined_class"] = combined
            data_dict["contested_class"] = contested

    return pd.DataFrame(data_dict)