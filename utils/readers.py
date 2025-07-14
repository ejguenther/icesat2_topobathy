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

def get_atl_at_photon_rate(atl03_file, atl08_file, atl24_file, gt):
    """
    Reads and aligns data from ATL03, ATL08, and ATL24 files to the ATL03 photon rate.

    This function orchestrates the reading of various datasets from three different
    ICESat-2 data products. It aligns all data to the native resolution of the
    ATL03 photon events for a specified ground track (gt) and compiles them
    into a single, unified pandas DataFrame.

    Args:
        atl03_file (str): The file path for the ATL03 data file.
        atl08_file (str): The file path for the corresponding ATL08 data file.
        atl24_file (str): The file path for the corresponding ATL24 data file.
        gt (str): The ground track identifier to process (e.g., 'gt1l', 'gt1r', 'gt2l', etc.).

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a single ATL03
                      photon event. The columns include aligned data from all
                      three input products.
    
    Raises:
        FileNotFoundError: If any of the input file paths do not exist.
        KeyError: If a required HDF5 dataset is not found in a file.
    """
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