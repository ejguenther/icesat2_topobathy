#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 12:36:17 2025

@author: ejg2736
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import laspy
import shapely
from shapely.geometry import LineString
from pyproj import Transformer
from matplotlib.path import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# from utils.geographic_utils import find_utm_zone_epsg
from geographic_utils import find_utm_zone_epsg
import os


# Constants for the analysis
DECIMATION = 10         # Downsample lidar points by this factor
ICESAT2BUFFER = 100    # Buffer around ICESat-2 gt to find ALS tiles
CROSSTRACK_LIMIT = 25  # Final crosstrack distance filter in meters
BBOX_BUFFER = 100      # Buffer around the trimmed line for coarse filtering

def prepare_icesat2_track(df_seg, utm_epsg, skip_rate=1):
    """Prepares the ICESat-2 ground track data by creating a projected LineString."""
    # Extract data, skipping points for performance if needed
    lon = df_seg['longitude'][::skip_rate].values
    lat = df_seg['latitude'][::skip_rate].values
    at_dist = df_seg['alongtrack'][::skip_rate].values
    
    # Project latitude and longitude to the specified UTM zone
    transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    is2_x, is2_y = transformer.transform(lon, lat)
    
    is2_line = LineString(zip(is2_x, is2_y))
    return is2_line, is2_x, is2_y, at_dist

def estimate_signed_crosstrack(points_xy, line):
    """Calculates signed cross-track distance for points relative to a line."""
    if line.is_empty or len(line.coords) < 2:
        return np.array([np.nan] * len(points_xy))
        
    unsigned_dist = shapely.distance(shapely.points(points_xy), line)
    
    p1, p2 = np.array(line.coords[0]), np.array(line.coords[-1])
    line_vec = p2 - p1
    point_vecs = points_xy - p1
    
    cross_product_z = line_vec[0] * point_vecs[:, 1] - line_vec[1] * point_vecs[:, 0]
    return np.sign(cross_product_z) * unsigned_dist

def estimate_alongtrack(points_xy, line, line_x, line_y, line_at_dist):
    """Estimates along-track distance by projecting points onto the line."""
    # Project points onto the detailed, vertex-filtered line
    projected_dist_on_line = shapely.line_locate_point(line, shapely.points(points_xy))
    
    # Create the distance array for the vertices of the trimmed line itself
    line_vertex_dist = shapely.line_locate_point(line, shapely.points(line_x, line_y))
    
    # Interpolate to find the original along-track distance
    return np.interp(projected_dist_on_line, line_vertex_dist, line_at_dist)
    
def process_lidar_tile(tile_info, is2_line, is2_x, is2_y, is2_at, utm_epsg):
    """
    Processes a single lidar tile to extract points along the ICESat-2 track.
    This function is designed to be run in parallel.
    """
    file_name, tile_geom = tile_info
    # file_name = tile_info.file_name
    # tile_geom = tile_info.geometry.buffer(100)
    base_name = os.path.basename(file_name)
    print(f"Processing tile: {base_name}")
    
    # 1. Read and project lidar data
    las = laspy.read(file_name)
    transformer = Transformer.from_crs(las.header.parse_crs(), utm_epsg, always_xy=True)
    x_coords, y_coords = transformer.transform(las.x[::DECIMATION], las.y[::DECIMATION])
    points_xy = np.column_stack((x_coords, y_coords))
    
    # Read other las info to close the lasfile
    z = np.array(las.z)
    classification = np.array(las.classification)
    
    # 2. Trim the ICESat-2 line to the buffered tile extent for local analysis
    trimmed_line = is2_line.intersection(tile_geom.buffer(BBOX_BUFFER))
    if trimmed_line.is_empty:
        return pd.DataFrame() # Return empty if no intersection

    # 3. Perform a fast bounding box pre-filter on lidar points
    min_x, min_y, max_x, max_y = trimmed_line.bounds
    bbox_filter = (
        (points_xy[:, 0] >= min_x - BBOX_BUFFER) & (points_xy[:, 0] <= max_x + BBOX_BUFFER) &
        (points_xy[:, 1] >= min_y - BBOX_BUFFER) & (points_xy[:, 1] <= max_y + BBOX_BUFFER)
    )
    candidate_points_xy = points_xy[bbox_filter]
    if len(candidate_points_xy) == 0:
        return pd.DataFrame()

    # 4. Calculate signed cross-track distance and filter
    crosstrack_dist = estimate_signed_crosstrack(candidate_points_xy, trimmed_line)
    crosstrack_mask = np.abs(crosstrack_dist) < CROSSTRACK_LIMIT
    
    final_points_xy = candidate_points_xy[crosstrack_mask]
    if len(final_points_xy) == 0:
        return pd.DataFrame()
        
    # 5. Get vertices of the ICESat-2 line that are within the tile's buffered extent
    polygon_path = Path(np.array(tile_geom.buffer(BBOX_BUFFER).exterior.coords))
    original_vertices = np.vstack((is2_x, is2_y)).T
    is_inside_mask = polygon_path.contains_points(original_vertices)
    
    trimmed_is2_x, trimmed_is2_y = is2_x[is_inside_mask], is2_y[is_inside_mask]
    trimmed_is2_at = is2_at[is_inside_mask]
    trimmed_line_from_vertices = LineString(zip(trimmed_is2_x, trimmed_is2_y))
    
    # 6. Calculate along-track distance for the final points
    alongtrack_dist = estimate_alongtrack(final_points_xy, trimmed_line_from_vertices, 
                                          trimmed_is2_x, trimmed_is2_y, trimmed_is2_at)

    # 7. Create the final DataFrame for this tile
    df_tile = pd.DataFrame({
        'x': final_points_xy[:, 0],
        'y': final_points_xy[:, 1],
        'z': z[::DECIMATION][bbox_filter][crosstrack_mask],
        'classification': classification[::DECIMATION][bbox_filter][crosstrack_mask],
        'crosstrack': crosstrack_dist[crosstrack_mask],
        'alongtrack': alongtrack_dist,
        'file': base_name
    })
    return df_tile

def create_swath_parallel(extent_gdf, df_seg, num_workers = 4):
    """
    Generates a swath of lidar points along an ICESat-2 ground track by
    processing lidar tiles in parallel.
    """
    # Determine the best UTM zone from the first tile's location
    utm_epsg = find_utm_zone_epsg(extent_gdf.iloc[0].lat_min,extent_gdf.iloc[0].lon_min)
    extent_gdf = extent_gdf.to_crs(utm_epsg) # Convert extent to UTM
    
    # Prepare the ICESat-2 track once
    is2_line, is2_x, is2_y, is2_at = prepare_icesat2_track(df_seg, utm_epsg)

    # Spatially select lidar tiles that intersect the ground track
    intersecting_tiles = extent_gdf[extent_gdf.intersects(is2_line.buffer(100))]
    
    # Create a list of tuples (file_name, geometry) to iterate over
    tiles_to_process = list(zip(intersecting_tiles['file_name'], intersecting_tiles['geometry']))

    df_list = []
    if num_workers == 1:
        for i in range(0,len(tiles_to_process)):
            process_lidar_tile(tiles_to_process[i], is2_line, is2_x, is2_y, is2_at, utm_epsg)
        
    else:
        # Use a ProcessPoolExecutor to run the processing in parallel
        # `partial` is used to "pre-fill" the arguments that are the same for every tile
        
        worker_func = partial(process_lidar_tile, 
                              is2_line=is2_line, 
                              is2_x=is2_x, 
                              is2_y=is2_y, 
                              is2_at=is2_at, 
                              utm_epsg=utm_epsg)
    
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # executor.map applies the worker function to each item in the list
            results = executor.map(worker_func, tiles_to_process)
            for df_tile in results:
                if not df_tile.empty:
                    df_list.append(df_tile)

    if not df_list:
        return pd.DataFrame()
        
    # Concatenate results from all processed tiles into a single DataFrame
    return pd.concat(df_list, ignore_index=True)