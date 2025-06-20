#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 08:37:49 2025

@author: ejg2736
"""

import numpy as np
import h5py
from typing import List
import pandas as pd
from typing import List, Union
from . import readers

def is_member(elements_to_check, reference_elements, comparison_mode='normal'):
    """
    Determines if elements of one array are present in another, similar to MATLAB's ismember function.

    This function supports two modes:
    1. 'normal': Checks for element-wise membership.
    2. 'rows': Treats each row as a unique entity and checks for row-wise membership.
              In this mode, multi-column arrays are converted into 1-D arrays of
              comma-separated strings to ensure unique row comparison.

    Args:
        elements_to_check (np.ndarray): The array whose elements (or rows) are to be checked
                                        for membership in `reference_elements`.
        reference_elements (np.ndarray): The array against which membership is checked.
        comparison_mode (str, optional): The mode of comparison. Can be 'normal' or 'rows'.
                                         Defaults to 'normal'.

    Returns:
        tuple: A tuple containing:
            - is_member_boolean (np.ndarray): A boolean array of the same shape as `elements_to_check`
                                              (or its 1-D string representation in 'rows' mode),
                                              where True indicates that the corresponding element (or row)
                                              is found in `reference_elements`.
            - matched_original_indices (np.ndarray): An array of indices from `reference_elements`
                                                     indicating the first occurrence of each
                                                     matching element from `elements_to_check`.
                                                     If an element in `elements_to_check` is not found,
                                                     its corresponding index will be undefined (e.g., -1 or 0,
                                                     depending on numpy's internal handling for non-matches).
    """

    # If the comparison mode is 'rows', convert multi-column arrays into 1-D arrays of strings.
    # This step ensures that `np.isin` correctly identifies unique rows by treating them
    # as single string entities.
    if comparison_mode.lower() == 'rows':
        # Convert numerical arrays to string arrays to allow for concatenation.
        elements_to_check_str = elements_to_check.astype('str')
        reference_elements_str = reference_elements.astype('str')

        # Initialize the 1-D arrays with the first column's string representation.
        # This avoids conditional assignment inside the loop.
        processed_elements_to_check = np.char.array(elements_to_check_str[:, 0])
        processed_reference_elements = np.char.array(reference_elements_str[:, 0])

        # Concatenate subsequent columns with commas to form a unique string for each row.
        # This effectively creates a unique identifier for each row.
        for i in range(1, np.shape(elements_to_check_str)[1]):
            processed_elements_to_check = processed_elements_to_check + ',' + np.char.array(elements_to_check_str[:, i])
            processed_reference_elements = processed_reference_elements + ',' + np.char.array(reference_elements_str[:, i])

        # Update the input arrays to their string-converted, 1-D representations for `np.isin`.
        elements_to_check = processed_elements_to_check
        reference_elements = processed_reference_elements

    # Determine which elements from `elements_to_check` are present in `reference_elements`.
    # This returns a boolean array indicating membership.
    is_member_boolean = np.isin(elements_to_check, reference_elements)

    # Extract the elements from `elements_to_check` that were found in `reference_elements`.
    matching_elements = elements_to_check[is_member_boolean]

    # Find unique matching elements and their inverse indices.
    unique_matching_elements, unique_matching_elements_inverse_indices = np.unique(matching_elements, return_inverse=True)

    # Find unique elements in `reference_elements` and their first occurrence indices.
    unique_reference_elements, unique_reference_indices = np.unique(reference_elements, return_index=True)

    # Map unique matching elements to their first occurrence indices in `reference_elements`.
    matching_elements_in_reference_indices = unique_reference_indices[np.isin(unique_reference_elements, unique_matching_elements, assume_unique=True)]

    # Map these indices back to the original `elements_to_check` order.
    matched_original_indices = matching_elements_in_reference_indices[unique_matching_elements_inverse_indices]

    return is_member_boolean, matched_original_indices

def get_atl03_segment_to_photon(atl03_filepath: str, ground_track: str, geolocation_field: str = '/geolocation/segment_id'):
    """
    Maps ATL03 segment IDs to individual photon records.

    Reads ATL03 HDF5 data to expand segment-level metadata (like segment_id)
    down to the photon level based on photon counts within each segment.

    Args:
        atl03_filepath (str): Path to the ATL03 HDF5 file.
        ground_track (str): The ground track identifier (e.g., 'gt1r', 'gt2l').
        geolocation_field (str): The subfield, options include 
            '/geolocation/segment_id'
            '/geolocation/segment_dist_x'
            '/geolocation/solar_elevation'

    Returns:
        np.ndarray: An array of segment IDs, where each entry corresponds to a single
                    ATL03 photon.
    """
    with h5py.File(atl03_filepath, 'r') as f:
        # Load necessary ATL03 datasets
        photon_heights = np.asarray(f[ground_track + '/heights/h_ph'])
        segment_photon_count = np.array(f[ground_track + '/geolocation/segment_ph_cnt'])
        atl03_photon_index_beginning = np.array(f[ground_track + '/geolocation/ph_index_beg'])
        atl03_segment_ids = np.array(f[ground_track + geolocation_field])

    # Initialize array to store segment ID for each photon
    photon_segment_ids = np.zeros(len(photon_heights), dtype=atl03_segment_ids.dtype)

    # Loop through segments to assign segment ID to each photon within it
    for i in range(len(atl03_segment_ids)):
        start_index = atl03_photon_index_beginning[i]
        # Only process if segment has valid photon indices (non-zero)
        if start_index > 0:
            # Adjust to 0-based indexing for numpy
            start_index -= 1
            end_index = start_index + segment_photon_count[i]
            # Assign the current segment ID to all photons in this segment
            photon_segment_ids[start_index:end_index] = atl03_segment_ids[i]

    return photon_segment_ids


def get_atl08_mapping(ph_index_beg, segment_id,
                      classed_pc_indx, classed_pc_flag,
                      classed_index_seg):
    """
    Maps ATL08 classified photons back to the ATL03 photon data structure.

    This function takes ATL03 and ATL08 data arrays and generates a comprehensive
    array that classifies all ATL03 photons based on ATL08 classifications where available.

    Args:
        ph_index_beg (np.ndarray): Array of starting photon indices for each
                                                  ATL03 segment.
        segment_id (np.ndarray): Array of segment IDs for ATL03 data.
        atl08_classed_photon_index (np.ndarray): Array of relative photon indices within
                                                 ATL08 segments for classified photons.
        classed_pc_flag (np.ndarray): Array of classification flags (e.g., ground, canopy)
                                                for ATL08 classified photons.
        classed_index_seg (np.ndarray): Array of segment IDs for ATL08 data.

    Returns:
        np.ndarray: An array, `all_photons_classified`, where each element corresponds to an
                    ATL03 photon and contains its ATL08 classification flag (-1 if not classified).
                    The size of this array corresponds to the maximum photon index + 1.
    """

    # Filter out zero-indexed entries from ATL03 data, as they typically indicate invalid or
    # unpopulated entries.
    non_zero_indices = ph_index_beg != 0
    ph_index_beg = ph_index_beg[non_zero_indices]
    segment_id = segment_id[non_zero_indices]

    # Use the `is_member` function to find ATL08 segments that have corresponding ATL03 segments.
    # This identifies which ATL08 segments can be mapped to ATL03 data.
    atl03_segments_in_08_boolean, atl03_segments_in_08_indices = is_member(
        classed_index_seg, segment_id
    )

    # Extract ATL08 classified photon indices and their corresponding classification values
    # for only those ATL08 segments that were found to have a match in ATL03.
    atl08_matched_indices = classed_pc_indx[atl03_segments_in_08_boolean]
    atl08_matched_values = classed_pc_flag[atl03_segments_in_08_boolean]

    # Determine the new mapping of ATL08 classified photons into the ATL03 photon index space.
    # `atl03_photon_beginning_indices` holds the indices within the filtered `ph_index_beg`.
    atl03_photon_beginning_indices = atl03_segments_in_08_indices
    # `atl03_photon_beginning_values` gets the actual starting photon index values from ATL03.
    atl03_photon_beginning_values = ph_index_beg[atl03_photon_beginning_indices]
    # `new_mapping_indices` calculates the absolute index for each classified ATL08 photon
    # within the overall ATL03 photon array. The '-2' adjustment is common in array indexing
    # when converting between 0-based and 1-based systems or adjusting for specific data formats.
    new_mapping_indices = atl08_matched_indices + atl03_photon_beginning_values - 2

    # Determine the maximum required size for the output array, which is the last calculated
    # mapping index plus one (to account for 0-based indexing).
    max_output_size = new_mapping_indices[-1]

    # Initialize an array to hold all ATL03 photon classifications.
    # It's pre-populated with -1 to indicate unclassified photons.
    all_photons_classified = (np.zeros(max_output_size + 1, dtype=int))

    # Populate the `all_photons_classified` array with the ATL08 classifications
    # at the calculated `new_mapping_indices`.
    all_photons_classified[new_mapping_indices] = atl08_matched_values

    # Return the array containing classifications for all ATL03 photons.
    return all_photons_classified


def get_atl08_class_to_atl03(atl03_file, atl08_file, beam_label):
    ph_index_beg, segment_id = readers.read_atl03_data_mapping(atl03_file, beam_label)
    classed_pc_indx, classed_pc_flag, classed_index_seg, _ = readers.read_atl08_data_mapping(atl08_file, beam_label)
    
    all_photons_classified = get_atl08_mapping(ph_index_beg, segment_id,
                          classed_pc_indx, classed_pc_flag,
                          classed_index_seg)
    
    with h5py.File(atl03_file, 'r') as f:
        atl03_h_ph = np.array(f[beam_label + '/heights/h_ph'])

    if len(all_photons_classified) < len(atl03_h_ph):
        n_zeros = len(atl03_h_ph) - len(all_photons_classified)
        zeros = np.zeros(n_zeros)
        all_photons_classified = np.append(all_photons_classified, zeros)
    return all_photons_classified

def get_atl08_norm_h_to_atl03(atl03_file, atl08_file, beam_label):
    ph_index_beg, segment_id = readers.read_atl03_data_mapping(atl03_file, beam_label)
    classed_pc_indx, _, classed_index_seg, ph_h = readers.read_atl08_data_mapping(atl08_file, beam_label)
    
    all_photons_norm_h = get_atl08_mapping(ph_index_beg, segment_id,
                          classed_pc_indx, ph_h,
                          classed_index_seg)
    
    with h5py.File(atl03_file, 'r') as f:
        atl03_h_ph = np.array(f[beam_label + '/heights/h_ph'])

    if len(all_photons_norm_h) < len(atl03_h_ph):
        n_zeros = len(atl03_h_ph) - len(all_photons_norm_h)
        zeros = np.zeros(n_zeros)
        all_photons_norm_h = np.append(all_photons_norm_h, zeros)
    return all_photons_norm_h



def get_atl24_to_atl03(
    atl03_input_source: Union[str, int, np.ndarray],
    atl24_filepath: str,
    ground_track: str,
    atl24_field_path: str = '/class_ph'
):
    """
    Maps specified ATL24 data to ATL03 photon indices.

    This function initializes an array corresponding to ATL03 photons (either by
    reading an ATL03 file, using a given length, or using a provided array).
    It then populates this array with data from an ATL24 file at specified
    photon indices.

    Args:
        atl03_input_source (Union[str, int, np.ndarray]):
            Source for determining the base ATL03 photon array.
            - If str: Path to the ATL03 HDF5 file. The length of the
                      '/heights/h_ph' dataset under the given ground_track
                      will be used to initialize the output array.
            - If int: The desired length of the output array (e.g., total
                      number of photons in ATL03).
            - If np.ndarray: A pre-initialized NumPy array to be used as the
                             base. Its dtype should be compatible with the
                             data from atl24_field_path.
        atl24_filepath (str): Path to the ATL24 HDF5 file.
        ground_track (str): The ground track identifier (e.g., 'gt1r', 'gt2l').
                            This is used to access data in both ATL03 (if path
                            is provided) and ATL24 files.
        atl24_field_path (str): HDF5 path to the dataset within the ATL24 file's
                                ground_track group that contains the values to be
                                mapped. Examples: '/class_ph', '/geolocation/segment_id'.
                                This path is appended to the ground_track path.

    Returns:
        np.ndarray: An array corresponding to ATL03 photons, populated with
                    values from the specified ATL24 field at the relevant
                    photon indices.
    """

    # Read necessary data from ATL24 file
    # This includes the photon indices (relative to ATL03) and the actual field data
    try:
        with h5py.File(atl24_filepath, 'r') as f_atl24:
            # Construct the full HDF5 path for ATL24 data
            # atl24_field_path is expected to start with '/', e.g., '/class_ph'
            # So, ground_track + atl24_field_path becomes e.g., 'gt1r/class_ph'
            full_atl24_data_path = ground_track + atl24_field_path
            full_atl24_index_path = ground_track + '/index_ph'

            if full_atl24_index_path not in f_atl24:
                raise ValueError(f"ATL24 photon index field not found: {full_atl24_index_path} in {atl24_filepath}")
            if full_atl24_data_path not in f_atl24:
                raise ValueError(f"ATL24 data field not found: {full_atl24_data_path} in {atl24_filepath}")

            atl24_photon_index = np.array(f_atl24[full_atl24_index_path])
            atl24_data_to_map = np.array(f_atl24[full_atl24_data_path])
    except Exception as e:
        raise IOError(f"Error reading ATL24 file {atl24_filepath}: {e}")

    # Initialize the output array (atl24_new_field) based on atl03_input_source
    if isinstance(atl03_input_source, np.ndarray):
        # User provided a pre-initialized array
        atl24_new_field = atl03_input_source
        # Basic check: ensure the provided array is large enough for max index
        if len(atl24_photon_index) > 0 and atl24_photon_index.max() >= len(atl24_new_field):
            raise ValueError(
                f"Provided 'atl03_input_source' array (length {len(atl24_new_field)}) "
                f"is too small for max ATL24 photon index ({atl24_photon_index.max()})."
            )
    elif isinstance(atl03_input_source, int):
        # User provided a length
        array_length = atl03_input_source
        if array_length < 0:
            raise ValueError("Provided length for 'atl03_input_source' cannot be negative.")
        atl24_new_field = np.zeros(array_length, dtype=atl24_data_to_map.dtype)
    elif isinstance(atl03_input_source, str):
        # User provided an ATL03 filepath
        atl03_filepath = atl03_input_source
        try:
            with h5py.File(atl03_filepath, 'r') as f_atl03:
                # Field used to determine the length of the ATL03 photon list
                # This path is constructed similarly to ATL24 paths
                atl03_length_field_path = ground_track + '/heights/h_ph'
                if atl03_length_field_path not in f_atl03:
                    raise ValueError(
                        f"ATL03 field for length determination ('{atl03_length_field_path}') "
                        f"not found in {atl03_filepath}."
                    )
                # Using .shape[0] for length to handle potentially empty datasets gracefully
                array_length = f_atl03[atl03_length_field_path].shape[0]
                atl24_new_field = np.zeros(array_length, dtype=atl24_data_to_map.dtype)
        except Exception as e:
            raise IOError(f"Error reading ATL03 file {atl03_filepath} for length: {e}")
    else:
        raise TypeError(
            "atl03_input_source must be a file path (str), an integer (length), "
            "or a NumPy array."
        )

    # Perform the mapping: place ATL24 data into the new field at specified indices
    # Ensure atl24_new_field is not empty and indices are within bounds if atl24_photon_index is not empty
    if len(atl24_photon_index) > 0:
        if len(atl24_new_field) == 0:
             # This case might occur if atl03_input_source led to a zero-length array
             # but atl24_photon_index is not empty.
            print("Warning: ATL24 photon indices exist, but target array 'atl24_new_field' is empty. No mapping performed.")
        elif atl24_photon_index.max() >= len(atl24_new_field):
            raise IndexError(
                f"Max index in 'atl24_photon_index' ({atl24_photon_index.max()}) "
                f"is out of bounds for 'atl24_new_field' (length {len(atl24_new_field)})."
            )
        else:
            try:
                atl24_new_field[atl24_photon_index] = atl24_data_to_map
            except IndexError as e:
                # This might happen if atl24_photon_index contains negative values or other issues
                raise IndexError(f"Error assigning ATL24 data to new field using indices: {e}. "
                                 f"Max index: {atl24_photon_index.max()}, "
                                 f"Min index: {atl24_photon_index.min()}, "
                                 f"Target array length: {len(atl24_new_field)}")
            except ValueError as e:
                # This might happen due to dtype mismatches if atl03_input_source was a user-provided array
                # with an incompatible dtype that couldn't be caught earlier, or shape mismatch.
                raise ValueError(f"Error assigning ATL24 data, possibly due to dtype or shape mismatch: {e}. "
                                 f"Target array dtype: {atl24_new_field.dtype}, "
                                 f"Source data dtype: {atl24_data_to_map.dtype}.")
    elif len(atl24_data_to_map) > 0 :
        # atl24_photon_index is empty, but there's data in atl24_data_to_map.
        # This implies no mapping can be done.
        print(f"Warning: 'atl24_photon_index' is empty, but 'atl24_data_to_map' (length {len(atl24_data_to_map)}) is not. No data mapped.")


    return atl24_new_field

def combine_atl08_and_atl24_classifications(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
    """
    Combines two NumPy arrays based on a specific set of rules.

    The rules are applied in the following order of priority:
    1. If a value in `array_b` is 40 or 41, it overrules any other value.
    2. If a value in one array is 0 and the other is non-zero, the non-zero value is used.
    3. Otherwise, the value from `array_a` is used as the default.

    Args:
        array_a (np.ndarray): The base NumPy array.
        array_b (np.ndarray): The array with overriding values.

    Returns:
        np.ndarray: The new array resulting from the combination rules.

    Raises:
        ValueError: If the input arrays are not of the same length.
    """
    # Ensure arrays are of the same length for the operations to make sense
    if array_a.shape != array_b.shape:
        raise ValueError("Input arrays must be of the same length and shape.")

    # 1. Initialize the result array as a copy of array_a (this sets the baseline)
    result_array = np.copy(array_a)

    # 2. Apply the "Zero vs. Non-zero" rule where a is 0 and b is not.
    # The case where b is 0 and a is not is already handled by initializing with a.
    condition_a_zero_b_nonzero = (array_a == 0) & (array_b != 0)
    result_array[condition_a_zero_b_nonzero] = array_b[condition_a_zero_b_nonzero]

    # 3. Apply the priority rule: if b is 40 or 41, it overrules.
    # This is applied last to ensure it takes precedence over the other rules.
    condition_b_priority = (array_b == 40) | (array_b == 41)
    result_array[condition_b_priority] = array_b[condition_b_priority]

    return result_array

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
    
