import os
import argparse
from pathlib import Path
import sys
import json
from scipy.optimize import brute, fmin

print("Loading pandas...", flush=True)
import pandas as pd
print("Loading numpy...", flush=True)
import numpy as np
print("Loading geopandas...", flush=True)
import geopandas as gpd
print("Loading shapely...", flush=True)
from shapely.geometry import Point
import traceback

print("Loading custom utils...", flush=True)
from utils import readers, analysis, processing
from utils.create_las_swath import create_als_swath, transform_als_swath, prepare_icesat2_track
from utils.geolocation import calculate_mae_cost, create_interpolator
from utils.building_processing import (
    find_intersected_buildings, 
    convert_buildings_to_atxt, 
    filter_grazing_hits,
    clip_als_to_buffered_building,
    extract_building_edges_2d,
    calculate_orthogonal_distance,
    classify_photons,
)


print("All imports complete.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Batch process footprint edge analysis")
    parser.add_argument('--data-dir', type=str, default='/home/ejg2736/network_drives/walker/exports/nfs_share/Data/workspace/IS2/footprint_exp/austin_data', help="Directory containing ATL03 and ATL08 data")
    parser.add_argument('--extent-file', type=str, default='/home/ejg2736/dev/icesat2_topobathy/data/austin_laz_bigtex.gpkg', help="Extent GeoPackage")
    parser.add_argument('--geoid-file', type=str, default='/home/ejg2736/dev/geoid/agisoft/us_noaa_g2018u0.tif', help="ALS Geoid File")
    parser.add_argument('--buildings-file', type=str, default='/home/ejg2736/network_drives/walker/exports/nfs_share/Data/workspace/footprint/austin_bldgs.gpkg', help="Austin Buildings File")
    parser.add_argument('--out-dir', type=str, default='/home/ejg2736/network_drives/walker/exports/nfs_share/Data/workspace/IS2/footprint_exp/outputs2', help="Output Directory")
    parser.add_argument('--als-swath-dir', type=str, default='/home/ejg2736/network_drives/walker/exports/nfs_share/Data/workspace/IS2/footprint_exp/als_swaths', help="Directory to cache ALS swaths")
    parser.add_argument('--test-mode', action='store_true', help="Run in test mode")
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.als_swath_dir, exist_ok=True)
    
    # 1. Load Extent and Buildings once
    print(f"Loading extent file: {args.extent_file}", flush=True)
    extent_gdf = gpd.read_file(args.extent_file)
    print(f"Loading buildings file: {args.buildings_file}", flush=True)
    austin_buildings = gpd.read_file(args.buildings_file)
    print(f"Reprojecting buildings file.", flush=True)
    gdf_buildings_utm = austin_buildings.to_crs('EPSG:32614')
    
    print(f"Globbing data dir: {args.data_dir}", flush=True)
    atl03_files = list(Path(args.data_dir).glob("ATL03_*.h5"))
    
    if args.test_mode:
        atl03_files = [f for f in atl03_files if 'ATL03_20181215070448_11860106_007_01.h5' in f.name]
        gts_to_process = ['gt2l']
    else:
        gts_to_process = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']
        
    print(f"Found {len(atl03_files)} ATL03 files to process.", flush=True)
    
    for atl03_file in atl03_files:
        atl03_basename = atl03_file.name.split('.h5')[0]
        atl08_name = atl03_file.name.replace('ATL03_', 'ATL08_')
        atl08_file = Path(args.data_dir) / atl08_name
        
        if not atl08_file.exists():
            print(f"Warning: Corresponding ATL08 file not found for {atl03_file.name}, skipping.", flush=True)
            continue
            
        print(f"Processing Pass: {atl03_basename}", flush=True)
        
        for gt in gts_to_process:
            print(f"  Processing Track: {gt}", flush=True)
            
            try:
                # Get photon rate DF
                df_ph = readers.read_photon_dataframe(str(atl03_file), gt, str(atl08_file))
                if getattr(df_ph, 'empty', True):
                    print(f"    Empty dataframe found for {atl03_basename} {gt}. Skipping.", flush=True)
                    continue
                
                df_ph['crosstrack'] = 0
                
                # Filter df_ph by extent
                df_ph = processing.filter_df_by_extent(df_ph, extent_gdf.total_bounds)
                if getattr(df_ph, 'empty', True):
                    print(f"    Empty dataframe after extent filtering for {atl03_basename} {gt}. Skipping.", flush=True)
                    continue
                    
                print(f"    Preparing icesat2 track...", flush=True)
                # Prepare track
                is2_line_utm, line_x, line_y, line_at_dist = prepare_icesat2_track(
                    df_ph, utm_epsg='EPSG:32614', resolution_m=10.0
                )
                
                print(f"    Finding intersected buildings...", flush=True)
                # Find hit buildings
                candidates_utm = find_intersected_buildings(
                    is2_line_utm, gdf_buildings_utm, buffer_meters=5.0, building_filter_size=300.0
                )
                
                if candidates_utm.empty:
                    print(f"    No buildings hit for {atl03_basename} {gt}. Skipping.", flush=True)
                    continue
                    
                print(f"    Converting to along-track...", flush=True)
                buildings_atxt = convert_buildings_to_atxt(
                    candidates_utm, is2_line_utm, line_x, line_y, line_at_dist
                )
                
                hit_buildings = filter_grazing_hits(buildings_atxt)
                
                if len(hit_buildings) == 0:
                    print(f"    No valid hit buildings after grazing filter for {atl03_basename} {gt}. Skipping.", flush=True)
                    continue

                # Process ALS Swath
                file_out_name = f"{atl03_basename}_{gt}"
                als_outfile = os.path.join(args.als_swath_dir, f'als_{file_out_name}.pqt')                
                
                if os.path.exists(als_outfile):
                    print(f"    Loading existing ALS Swath from {als_outfile}", flush=True)
                    als_swath = pd.read_parquet(als_outfile)
                else:
                    print(f"    Generating ALS Swath...", flush=True)
                    als_swath = create_als_swath(extent_gdf, df_ph)
                    als_swath = transform_als_swath(
                        als_swath, 
                        'EPSG:32614', 
                        source_geoid_file=args.geoid_file,
                        target_geoid_file=None, 
                        input_units='feet', 
                        source_datum='nad83'
                    )
                    als_swath.to_parquet(als_outfile)

                # Create interpolator for ALS surface
                als_surface_interpolator = create_interpolator(als_swath, grid_resolution=1)    

                # Perform geolocation alignment on ALS/ICESat-2
                search_grid = (slice(-10, 10, 0.5), slice(-10, 10, 0.5))

                result = brute(
                    calculate_mae_cost, 
                    ranges=search_grid, 
                    args=(df_ph.alongtrack[df_ph.atl08_class == 1], 
                    df_ph.crosstrack[df_ph.atl08_class == 1], 
                    df_ph.h_ph[df_ph.atl08_class == 1], als_surface_interpolator),
                    finish=fmin, 
                    full_output=True 
                )

                optimal_shift_at, optimal_shift_xt = result[0]

                # Apply crosstrack and alongtrack shift to the ALS swath to make the math easier
                als_swath['alongtrack'] = als_swath['alongtrack'] - optimal_shift_at
                als_swath['crosstrack'] = als_swath['crosstrack'] - optimal_shift_xt
                
                # Iterate hit buildings and process edges
                print(f"    Processing {len(hit_buildings)} target buildings...", flush=True)
                
                for target_building_id in hit_buildings.index:
                    target_building = hit_buildings.loc[target_building_id].geometry.buffer(20)
                    target_atxt = buildings_atxt.loc[target_building_id].geometry
                    target_als = clip_als_to_buffered_building(als_swath, target_building, buffer_m=1.0)
                    target_als = target_als[np.abs(target_als.crosstrack) < 6] 
                    
                    if getattr(target_als, 'empty', True):
                        continue
                        
                    edge = extract_building_edges_2d(target_als, target_atxt)

                    edge_out = os.path.join(args.out_dir, f"edges_{atl03_basename}_{gt}_{target_building_id}.json")
                    with open(edge_out, "w") as f:
                        json.dump(edge, f, indent=4)
                    
                    for direction in ['entry', 'exit']:
                        out_name = f"{atl03_basename}_{gt}_{target_building_id}_{direction}"
                        
                        if edge[direction]['valid']:
                            intercept = edge[direction]['intercept']
                            target_ph = df_ph[df_ph.alongtrack.between(intercept - 50, intercept + 50)]
                            
                            if getattr(target_ph, 'empty', True):
                                continue
                                
                            target_ph = target_ph.copy()
                            target_ph['crosstrack'] = 0
                            
                            target_ph = calculate_orthogonal_distance(target_ph, edge[direction], direction, threshold_m=15.0)
                            target_ph = classify_photons(target_ph, edge[direction]['roof_median_h'], z_tolerance=1.0)
                            
                            target_ph_out = os.path.join(args.out_dir, f"{out_name}_target_ph.pqt")
                            
                            target_ph.to_parquet(target_ph_out)
                            print(f"      Saved {direction} edge output for building {target_building_id}", flush=True)

            except Exception as e:
                print(f"    Error processing {atl03_basename} {gt}: {e}", flush=True)
                traceback.print_exc()

if __name__ == "__main__":
    main()
