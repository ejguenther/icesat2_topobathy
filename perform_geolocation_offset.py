"""
Script to calculate the geolocation offset of ICESat-2 (IS2) data relative to Airborne Lidar Scanning (ALS) data.
"""

import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np
from scipy.optimize import minimize, brute, fmin

# Import functions from main and utils
from main import ATL_AGG_CONFIG, ALS_AGG_CONFIG, ATL08_AGG_CONFIG
from utils import readers, analysis, processing, plotter
from utils.create_las_swath import create_als_swath, transform_als_swath
from utils.datum_transforms import convert_3d_nad83_to_wgs84, get_geoid_height
from utils.geolocation import create_interpolator, calculate_mae_cost
from utils.create_las_extent import create_gdf_las_extent

# --------------------------------------------------------------------------- #
# CONFIGURATION - Set these variables before running
# --------------------------------------------------------------------------- #

# IS2 File Configuration
IS2_DIR = '/home/ejg2736/network_drives/walker/exports/nfs_share/Users/ajm7578/CornerCube/version6_test'
IS2_FILENAME = 'processed_ATL03_20190928175636_00280506_006_02.h5'
IS2_FILEPATH = os.path.join(IS2_DIR, IS2_FILENAME)

# Define the ground track of interest (e.g., 'gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r')
GT = 'gt2r'

# Extent file (GeoPackage containing processing area)
EXTENT_FILE = '/home/ejg2736/dev/icesat2_topobathy/data/wsmr_rel006.gpkg'

# ALS configuration
ALS_DIR = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/WhiteSands_Analysis/WSMR_Lidar/ir_1'
ALS_GEOID_FILE = '/home/ejg2736/dev/geoid/agisoft/us_noaa_g2018u0.tif'
ALS_EPSG = 'EPSG:32613' # UTM zone of the ALS data
ALS_OUTDIR = '/home/ejg2736/network_drives/walker/exports/nfs_share/Data/workspace/IS2/footprint_exp/als_swaths'

# Search Grid Configuration for Offset Optimization (start, stop, step in meters)
SEARCH_GRID = (slice(-10, 10, 0.5), slice(-10, 10, 0.5))

# --------------------------------------------------------------------------- #
# MAIN PROCESSING
# --------------------------------------------------------------------------- #

def main():
    print(f"--- Starting Geolocation Offset Calculation for {IS2_FILENAME} ({GT}) ---")
    
    # 1. Read the Extent File
    if not os.path.exists(EXTENT_FILE):
        print(f"Warning: Extent file not found at {EXTENT_FILE}.")
        extent_gdf = create_gdf_las_extent(ALS_DIR, num_processes=None)
        extent_gdf.to_file(EXTENT_FILE, driver='GPKG')
    
    print(f"Loading extent from {EXTENT_FILE}...")
    extent_gdf = gpd.read_file(EXTENT_FILE)
    
    # 2. Read ICESat-2 Data
    print(f"Reading IS2 data...")
    df_ph = readers.read_photon_dataframe(IS2_FILEPATH, GT)
    df_ph['crosstrack'] = 0 
    
    # 3. Load or Create ALS Swath
    file_out_name = f"{os.path.basename(IS2_FILENAME).split('.h5')[0]}_{GT}"
    als_outfile = os.path.join(ALS_OUTDIR, f'als_{file_out_name}.pqt')                
    
    if os.path.exists(als_outfile):
        print(f"Loading existing ALS swath from {als_outfile}...")
        als_swath = pd.read_parquet(als_outfile)
    else:
        print("Creating new ALS swath from points. This may take a while...")
        os.makedirs(ALS_OUTDIR, exist_ok=True)
        als_swath = create_als_swath(extent_gdf, df_ph, num_workers = 1)
        
        print("Transforming ALS swath datums/projections...")
        als_swath = transform_als_swath(
            als_swath, 
            ALS_EPSG, 
            source_geoid_file=ALS_GEOID_FILE,
            target_geoid_file=None, 
            input_units='feet', 
            source_datum='nad83'
        )
        print(f"Saving ALS swath to {als_outfile}...")
        als_swath.to_parquet(als_outfile)

    # 4. Create Surface Interpolator
    print("Creating ALS surface interpolator (resolution: 1m)...")
    als_surface_interpolator = create_interpolator(als_swath, grid_resolution=1, ground_only=True, ground_class = 9)

    # 5. Calculate Geolocation Offset
    print("Calculating optimal geolocation offset using brute force search...")
    
    # Filter for valid photons (class 1 often denotes ground/canopy photons of interest, 
    # adjust if atl08_class definitions differ for your specific dataset)
    valid_photons = df_ph[df_ph.signal_conf_ph1 > 3]
    
    if len(valid_photons) == 0:
        print("Error: No valid photons (atl08_class == 1) found in the dataset.")
        return
        
    result = brute(
        calculate_mae_cost, 
        ranges=SEARCH_GRID, 
        args=(
            valid_photons.alongtrack, 
            valid_photons.crosstrack, 
            valid_photons.h_ph, 
            als_surface_interpolator
        ),
        finish=fmin, 
        full_output=True 
    )

    optimal_shift_at, optimal_shift_xt = result[0]
    best_mae = result[1]

    # 6. Display Results
    print("\n" + "="*40)
    print("Optimization Results:")
    print("="*40)
    print(f"Along-Track (AT) shift: {optimal_shift_at:.2f} m")
    print(f"Cross-Track (XT) shift: {optimal_shift_xt:.2f} m")
    print(f"Best Mean Absolute Error (MAE): {best_mae:.2f} m")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
