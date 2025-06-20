#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:33:35 2025

@author: ejg2736
"""

import os
import numpy as np
import h5py
from typing import List

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