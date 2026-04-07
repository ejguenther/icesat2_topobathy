import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from utils.create_las_swath import prepare_icesat2_track, estimate_alongtrack, estimate_signed_crosstrack
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor


def find_intersected_buildings(is2_line_utm, gdf_buildings_utm, buffer_meters=5.0, building_filter_size=300.0):
    """Strictly handles finding and filtering the geographic candidate buildings."""
    search_corridor = is2_line_utm.buffer(buffer_meters)
    candidates = gdf_buildings_utm[gdf_buildings_utm.intersects(search_corridor)].copy()
    candidates['area_sqm'] = candidates.geometry.area
    return candidates[candidates['area_sqm'] > building_filter_size]

def remove_complex_intersections(gdf_candidates, is2_line_utm):
    """
    Filters out buildings where the ground track enters and exits multiple times 
    (e.g., U-shaped or courtyard buildings).
    """
    if gdf_candidates.empty:
        return gdf_candidates
        
    # Calculate the exact geometric intersection of the ground track line 
    # against every candidate building polygon.
    intersections = gdf_candidates.geometry.intersection(is2_line_utm)
    
    # Keep only buildings where the intersection is a single continuous line segment.
    # This automatically drops 'MultiLineString' (multiple entries/exits) 
    # and 'Point' (grazing the exact corner).
    simple_crossings_mask = intersections.geom_type == 'LineString'
    
    # Apply the mask and return
    gdf_filtered = gdf_candidates[simple_crossings_mask].copy()
    
    dropped_count = len(gdf_candidates) - len(gdf_filtered)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} buildings due to complex geometry intersections.")
        
    return gdf_filtered

def convert_buildings_to_atxt(gdf_candidates_utm, is2_line_utm, line_x, line_y, line_at_dist):
    """Vectorized transformation of geometries to the AT/XT coordinate space."""
    
    if gdf_candidates_utm.empty:
        return gdf_candidates_utm
    
    # 1. FLATTEN: Gather all coordinates into one list, keeping track of their lengths
    all_coords = []
    poly_lengths = []
    
    for geom in gdf_candidates_utm.geometry:
        coords = np.array(geom.exterior.coords)
        all_coords.append(coords)
        poly_lengths.append(len(coords))  # Remember how many points make up this polygon
        
    # Stack everything into one giant (N, 2) NumPy array
    flat_coords = np.vstack(all_coords)
    
    # 2. COMPUTE: Run your math ONCE on the giant array
    # (NumPy is optimized to do this at C-speed)
    flat_at = estimate_alongtrack(flat_coords, is2_line_utm, line_x, line_y, line_at_dist)
    flat_xt = estimate_signed_crosstrack(flat_coords, is2_line_utm) 
    
    # 3. RECONSTRUCT: Slice the flat arrays back into their respective Polygons
    atxt_geometries = []
    start_idx = 0
    
    for length in poly_lengths:
        end_idx = start_idx + length
        
        # Slice out the specific AT and XT coordinates for this polygon
        poly_at = flat_at[start_idx:end_idx]
        poly_xt = flat_xt[start_idx:end_idx]
        
        atxt_geometries.append(Polygon(zip(poly_at, poly_xt)))
        
        # Move the starting index forward for the next polygon
        start_idx = end_idx
        
    # Build and return the final GeoDataFrame
    gdf_atxt = gdf_candidates_utm.copy()
    gdf_atxt.geometry = atxt_geometries
    gdf_atxt.crs = None 
    
    return gdf_atxt

def filter_grazing_hits(gdf_atxt, footprint_radius_m=7.0):
    """
    Filters out buildings that do not fully encompass the width of the laser footprint.
    In AT/XT space, X is Along-Track and Y is Cross-Track.
    """
    # .bounds returns a dataframe with columns: minx, miny, maxx, maxy
    bounds = gdf_atxt.bounds
    
    # miny is the maximum distance to the "left" (Negative XT)
    # maxy is the maximum distance to the "right" (Positive XT)
    straddles_left = bounds['miny'] < -footprint_radius_m
    straddles_right = bounds['maxy'] > footprint_radius_m
    
    # The building must extend past the footprint on BOTH sides
    full_hit_mask = straddles_left & straddles_right
    
    # Return only the buildings that take a direct hit
    good_hits = gdf_atxt[full_hit_mask].copy()
    
    print(f"Filtered out {len(gdf_atxt) - len(good_hits)} grazing hits.")
    return good_hits


def clip_als_to_buffered_building(df_als_swath, building_atxt_polygon, buffer_m=20.0):
    """
    Clips the ALS DataFrame to a buffered building polygon in AT/XT space.
    """
    # Create the buffered cookie-cutter
    search_area = building_atxt_polygon.buffer(buffer_m)
    
    # Temporarily convert the ALS points to a GeoDataFrame
    gdf_als = gpd.GeoDataFrame(
        df_als_swath, 
        geometry=gpd.points_from_xy(df_als_swath.alongtrack, df_als_swath.crosstrack)
    )
    
    # Filter points that fall within the buffered polygon
    # This is highly optimized under the hood by GeoPandas spatial indexing (GEOS)
    clipped_gdf = gdf_als[gdf_als.geometry.within(search_area)]
    
    # Drop the geometry column to return a standard Pandas DataFrame
    return clipped_gdf.drop(columns=['geometry'])


def extract_building_edges_2d(df_trench, buffer_shape):
    """
    Isolates the target roof and extracts the entry and exit lines.
    Returns a dictionary with the line parameters and validity status.
    """
    # --- PHASE 1: ISOLATE ---
    df_bldgs = df_trench[df_trench['classification'] == 6].copy()
    if df_bldgs.empty:
        return None
        
    # Cluster the building points based on spatial proximity
    clustering = DBSCAN(eps=3.0, min_samples=10).fit(df_bldgs[['alongtrack', 'crosstrack']])
    df_bldgs['cluster'] = clustering.labels_
    
    # Ignore noise (-1) and isolate valid clusters
    valid_clusters = df_bldgs[df_bldgs['cluster'] != -1]
    if valid_clusters.empty:
        return None
        
    # Get the reference center from the input shape
    ref_xt = buffer_shape.centroid.y
    ref_at = buffer_shape.centroid.x
    
    # Calculate distance from EVERY valid photon to the buffer center
    distances = np.sqrt(
        (valid_clusters['crosstrack'] - ref_xt)**2 + 
        (valid_clusters['alongtrack'] - ref_at)**2
    )
    
    # Find the index of the single photon closest to the buffer center
    closest_point_idx = distances.idxmin()
    
    # Grab the cluster ID that this specific photon belongs to
    target_cluster_id = valid_clusters.loc[closest_point_idx, 'cluster']
    
    # Isolate the target roof
    target_roof = valid_clusters[valid_clusters['cluster'] == target_cluster_id].copy()
    
    # --- PHASE 2: FIT LINES & CHECK CORNERS ---
    # Bin by XT to find the front and back edges
    xt_bins = np.arange(-6, 7, 1.0)
    target_roof['xt_bin'] = np.digitize(target_roof['crosstrack'], xt_bins)
    
    # Get the entry (min AT) and exit (max AT) points for each XT bin
    entry_pts = target_roof.groupby('xt_bin')[['crosstrack', 'alongtrack']].min().values
    exit_pts = target_roof.groupby('xt_bin')[['crosstrack', 'alongtrack']].max().values
    
    # --- Process Entry Edge ---
    xt_entry = entry_pts[:, 0].reshape(-1, 1)
    at_entry = entry_pts[:, 1]
    
    # Force RANSAC to only consider points within 0.5m of the line as "inliers"
    ransac_entry = RANSACRegressor(residual_threshold=1.5).fit(xt_entry, at_entry)
    entry_valid, entry_reason, entry_metrics = is_valid_wall(xt_entry, at_entry, ransac_entry)
    
    if entry_valid:
        # Pass the full df_trench so it has access to Vegetation classes
        entry_valid, entry_reason, entry_metrics = check_local_edge_conditions(
            df_trench, ransac_entry, is_entry=True
        )

    if not entry_valid:
        print(f"Rejecting Entry Edge: {entry_reason}")
        
    # --- Process Exit Edge ---
    xt_exit = exit_pts[:, 0].reshape(-1, 1)
    at_exit = exit_pts[:, 1]
    
    ransac_exit = RANSACRegressor(residual_threshold=1.5).fit(xt_exit, at_exit)
    exit_valid, exit_reason, exit_metrics = is_valid_wall(xt_exit, at_exit, ransac_exit)
    
    if exit_valid:
        # Pass the full df_trench so it has access to Vegetation classes
        exit_valid, exit_reason, exit_metrics = check_local_edge_conditions(
            df_trench, ransac_exit, is_entry=False
        )
    
    if not exit_valid:
        print(f"Rejecting Exit Edge: {exit_reason}")
    
    # Build the final dictionary
    edges = {
        'entry': {
            'slope': ransac_entry.estimator_.coef_[0], 
            'intercept': ransac_entry.estimator_.intercept_,
            'valid': entry_valid,
            'reason': entry_reason,
            'roof_median_h': entry_metrics['median_height'],
            'roof_iqr': entry_metrics['iqr']
        },
        'exit': {
            'slope': ransac_exit.estimator_.coef_[0], 
            'intercept': ransac_exit.estimator_.intercept_,
            'valid': exit_valid,
            'reason': exit_reason,
            'roof_median_h': exit_metrics['median_height'],
            'roof_iqr': exit_metrics['iqr']
        }
    }
    
    return edges



def create_line(slope, intercept, x_start, x_stop):
    # Calculate the y-coordinate for the starting x
    y_start = (slope * x_start) + intercept
    
    # Calculate the y-coordinate for the stopping x
    y_stop = (slope * x_stop) + intercept
    
    return (x_start, x_stop), (y_start, y_stop)


def is_valid_wall(xt_pts, at_pts, ransac_model, straightness_threshold=0.7, curve_threshold=1.5 ):
    """
    Checks if the RANSAC fit represents a clean, straight wall rather than a corner.
    """
    # 1. The Inlier Ratio Check (Catches "Extent" corners)
    inlier_mask = ransac_model.inlier_mask_
    inlier_ratio = np.sum(inlier_mask) / len(inlier_mask)
    
    if inlier_ratio < straightness_threshold:
        return False, "Failed Inlier Ratio (Likely an Extent Corner)", {'iqr': np.nan, 'median_height': np.nan}
        
    # 2. The Slice Check (Catches "Slice" corners)
    # Get the predicted AT values for all XT points
    predicted_at = ransac_model.predict(xt_pts)
    
    # Calculate the signed residuals (actual - predicted)
    residuals = at_pts - predicted_at.flatten()
    
    # If it's a 'V' shape (Slice), the edges will have high positive error 
    # and the center will have high negative error (or vice versa).
    # We split the wall into Left, Center, and Right thirds.
    thirds = np.array_split(residuals, 3)
    
    left_mean = np.mean(thirds[0])
    center_mean = np.mean(thirds[1])
    right_mean = np.mean(thirds[2])
    
    # If the center bends more than 1 meter away from the line connecting the edges
    if abs(center_mean - (left_mean + right_mean)/2) > curve_threshold:
        return False, "Failed Residual Curve (Likely a Slice Corner)", {'iqr': np.nan, 'median_height': np.nan}
        
    return True, "Valid Straight Wall", {'iqr': np.nan, 'median_height': np.nan}

def check_local_edge_conditions(df_trench, ransac_model, is_entry=True, clearance_m=8.0, roof_depth_m=3.0):
    """
    Checks if the immediate vicinity of the edge is flat and clear of obstructions.
    """
    # Predict the exact AT coordinate of the wall for every point in the trench
    predicted_wall_at = ransac_model.predict(df_trench['crosstrack'].values.reshape(-1, 1)).flatten()
    
    # Calculate the relative distance of every point to the wall
    # Negative means before the wall, Positive means after the wall
    relative_dist = df_trench['alongtrack'] - predicted_wall_at

    # 1. Define the Masks based on whether this is the entry or exit wall
    if is_entry:
        clearance_mask = (relative_dist >= -clearance_m) & (relative_dist < -0.75)
        roof_mask = (relative_dist >= 0) & (relative_dist <= roof_depth_m)
    else:
        clearance_mask = (relative_dist > 0.75) & (relative_dist <= clearance_m)
        roof_mask = (relative_dist >= -roof_depth_m) & (relative_dist <= 0)
        

        
    # --- CHECK 1: THE ROOF FLATNESS ---
    df_roof = df_trench[roof_mask]
    
    # Isolate just the building points on the roof lip
    df_roof_bldg = df_roof[df_roof['classification'] == 6]
    
    if len(df_roof_bldg) < 10:
        return False, "Failed Flatness: Not enough building points near the edge", {'iqr': np.nan, 'median_height': np.nan}
        
    # Use the Interquartile Range (IQR) to check for flatness, ignoring stray noise
    median_height = np.median(df_roof_bldg['h_norm'])
    q25, q75 = np.percentile(df_roof_bldg['h_norm'], [25, 75])
    iqr = q75 - q25

    metrics = {'iqr': iqr, 'median_height': median_height}
    
    if iqr > 0.5: # 0.5 meters of roughness tolerance
        return False, f"Failed Flatness: Roof edge is too rough or sloped (IQR: {iqr:.2f}m)", metrics

    # --- CHECK 2: THE CLEARANCE ZONE ---
    df_clearance = df_trench[clearance_mask]
    
    # Look for points within +/- 1 m of roof height

    obstructions = df_clearance[
        (df_clearance['h_norm'] > q25 - 1) & 
        (df_clearance['h_norm'] < q75 + 1)
    ]
    
    if not obstructions.empty:
        return False, "Failed Clearance: Tall vegetation or building in approach path", metrics
        

    # --- CHECK 3: THE ROOF HEIGHT ---
    # Check minimum height of the roof edge
    median_height = df_roof_bldg['h_norm'].median()
    if median_height < 3.0:
        return False, f"Failed Height: Roof edge is too short ({median_height:.1f}m)", metrics

    return True, "Valid Edge", metrics


import numpy as np

def calculate_orthogonal_distance(df_ph, edge_params, edge_type, threshold_m=10.0):
    """
    Subsets photons near a building edge and calculates their signed 
    orthogonal distance to the wall line.
    
    Parameters:
    -----------
    df_ph : DataFrame
        The ICESat-2 ATL03 photons (must have 'alongtrack' and 'crosstrack').
    edge_params : dict
        Contains 'slope' (m) and 'intercept' (b) of the wall line.
    edge_type : str
        'entry' or 'exit'.
    threshold_m : float
        The +/- distance from the wall to keep (e.g., 10 meters).
    """
    m = edge_params['slope']
    b = edge_params['intercept']
    
    # 1. Broad Spatial Filter (Performance Optimization)
    # The centerline (XT=0) hits the wall exactly at AT = b.
    # We filter the DataFrame to a broad bounding box first so we aren't 
    # doing complex math on millions of photons. We add a 5m buffer to the 
    # threshold to account for highly diagonal walls.
    at_center = b
    broad_mask = (df_ph['alongtrack'] >= at_center - (threshold_m + 5.0)) & \
                 (df_ph['alongtrack'] <= at_center + (threshold_m + 5.0))
    
    df_edge = df_ph[broad_mask].copy()
    
    # 2 & 3. Calculate Orthogonal Distance
    # Vectorized calculation for every photon in the subset
    at_ph = df_edge['alongtrack'].values
    xt_ph = df_edge['crosstrack'].values
    
    wall_at = (m * xt_ph) + b
    
    # Positive means AT_ph > wall_at. Negative means AT_ph < wall_at.
    ortho_dist = (at_ph - wall_at) / np.sqrt(m**2 + 1)
    
    # 4. Flip the sign for Exit edges
    # For an 'entry' edge, moving forward (+AT) puts you ON the roof (+ distance).
    # For an 'exit' edge, moving forward (+AT) puts you OFF the roof. 
    # We multiply by -1 so that for BOTH edges, Negative = Ground, Positive = Roof.
    if edge_type == 'exit':
        ortho_dist = ortho_dist * -1.0
        
    df_edge['dist_to_wall'] = ortho_dist
    
    # 5. Strict Threshold Filter
    # Now that we have exact orthogonal distances, we trim it strictly to +/- 10m
    strict_mask = np.abs(df_edge['dist_to_wall']) <= threshold_m
    df_final = df_edge[strict_mask].copy()
    
    return df_final


def classify_photons(df_local_ph, roof_median, z_tolerance=2.0):
    """
    Tags photons as 'roof', 'ground', or 'noise'.
    """
    # Ground is assumed to be roughly 0 in h_norm space
    is_ground = (df_local_ph['h_norm'] >= -z_tolerance) & (df_local_ph['h_norm'] <= z_tolerance)
    
    # Roof is bounded around the ALS-derived median
    is_roof = (df_local_ph['h_norm'] >= roof_median - z_tolerance) & \
              (df_local_ph['h_norm'] <= roof_median + z_tolerance)
              
    df_local_ph['target_class'] = 'noise'
    df_local_ph.loc[is_ground, 'target_class'] = 'ground'
    df_local_ph.loc[is_roof, 'target_class'] = 'roof'
    
    return df_local_ph


def compute_esf(df_ph, min_dist=-10.0, max_dist=10.0, bin_size=0.5):
    """
    Computes the Edge Spread Function (ESF) ratio from classified photons.
    """
    # 1. Filter out noise. We only want ground and building returns.
    # (Checking for both 'building' and 'roof' to be safe with naming)
    valid_classes = ['ground', 'building', 'roof']
    df_clean = df_ph[df_ph['target_class'].isin(valid_classes)].copy()
    
    # Normalize naming just in case
    df_clean['target_class'] = df_clean['target_class'].replace('roof', 'building')
    
    # 2. Define the spatial bins
    # This creates the bin edges: [-10.0, -9.5, -9.0 ... 10.0]
    bins = np.arange(min_dist, max_dist + bin_size, bin_size)
    
    # Calculate the midpoint of each bin for plotting later (e.g., -9.75)
    bin_centers = bins[:-1] + (bin_size / 2.0)
    
    # 3. Assign each photon to a bin
    df_clean['dist_bin'] = pd.cut(
        df_clean['dist_to_wall'], 
        bins=bins, 
        labels=bin_centers, 
        include_lowest=True
    )
    
    # 4. Count photons per class in each bin
    # unstack() pivots the table so 'ground' and 'building' become columns
    counts = df_clean.groupby(['dist_bin', 'target_class'], observed=False).size().unstack(fill_value=0)
    
    # Safety check: ensure both columns exist even if a class is entirely missing
    if 'building' not in counts.columns: counts['building'] = 0
    if 'ground' not in counts.columns: counts['ground'] = 0
        
    # 5. Calculate the ESF Ratio
    counts['total_photons'] = counts['building'] + counts['ground']
    
    # Calculate ratio, handling empty bins (division by zero) by placing NaN
    counts['esf_ratio'] = np.where(
        counts['total_photons'] > 0, 
        counts['building'] / counts['total_photons'], 
        np.nan
    )
    
    # Clean up the dataframe to return a flat, easy-to-plot table
    df_esf = counts.reset_index()
    df_esf = df_esf.rename(columns={'dist_bin': 'distance_to_wall'})
    
    return df_esf