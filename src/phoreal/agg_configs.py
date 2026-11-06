"""Aggregation recipes for phoreal.analysis (moved from main.py).

Each entry: field, operation, class_field, class_id, outfield, optional kwargs.
"""

# --- Configuration: ATL08/24 Aggregation Recipe ---
ATL_AGG_CONFIG = [
    # Geolocation (Median of all valid signal classes)
    {'field': 'latitude', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [1,2,3,40,41], 'outfield': 'latitude'},
    {'field': 'longitude', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [1,2,3,40,41], 'outfield': 'longitude'},
    {'field': 'solar_elevation', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [1,2,3,40,41], 'outfield': 'solar_elevation'},

    # Terrain
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [1], 'outfield': 'h_te_median'},

    # Canopy (Using h_norm)
    {'field': 'h_norm', 'operation': 'get_max98', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy'},
    {'field': 'h_norm', 'operation': 'max', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy_max'},

    # Topobathy (Using combined classes)
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'combined_class', 'class_id': [1,40], 'outfield': 'h_topobathy'},

    # Bathymetry & Surface (From ATL24 classes)
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'atl24_class', 'class_id': [40], 'outfield': 'h_bathy'},
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'atl24_class', 'class_id': [41], 'outfield': 'h_surface'},

    # Counts
    {'field': 'ortho_h', 'operation': 'get_len', 'class_field': 'atl08_class', 'class_id': [1], 'outfield': 'n_terrain'},
    {'field': 'ortho_h', 'operation': 'get_len', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'n_canopy'},
    {'field': 'ortho_h', 'operation': 'get_len', 'class_field': 'atl24_class', 'class_id': [40], 'outfield': 'n_bathy'},
    {'field': 'delta_time', 'operation': 'get_len_unique', 'class_field': 'atl08_class', 'class_id': [0,1,2,3], 'outfield': 'n_shots'},
]

ATL08_AGG_CONFIG = [
    # Geolocation (Median of all valid signal classes)
    {'field': 'latitude', 'operation': 'mean', 'class_field': 'atl08_class', 'class_id': [1,2,3,40,41], 'outfield': 'latitude'},
    {'field': 'longitude', 'operation': 'mean', 'class_field': 'atl08_class', 'class_id': [1,2,3,40,41], 'outfield': 'longitude'},
    {'field': 'solar_elevation', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [1,2,3,40,41], 'outfield': 'solar_elevation'},

    # Terrain
    {'field': 'h_ph', 'operation': 'mean', 'class_field': 'atl08_class', 'class_id': [1], 'outfield': 'h_te_mean'},
    {'field': 'h_ph', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [1], 'outfield': 'h_te_median'},
    {'field': 'h_ph', 'operation': 'std', 'class_field': 'atl08_class', 'class_id': [1], 'outfield': 'h_te_std'},
    {'field': 'h_ph', 'operation': 'min', 'class_field': 'atl08_class', 'class_id': [1], 'outfield': 'h_te_min'},
    {'field': 'h_ph', 'operation': 'max', 'class_field': 'atl08_class', 'class_id': [1], 'outfield': 'h_te_max'},

    # Canopy (Using h_norm)
    {'field': 'h_norm', 'operation': 'get_max98', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy'},
    {'field': 'h_norm', 'operation': 'max', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy_max'},
    {'field': 'h_norm', 'operation': 'mean', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy_mean'},
    {'field': 'h_norm', 'operation': 'median', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy_median'},
    {'field': 'h_norm', 'operation': 'min', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy_min'},
    {'field': 'h_norm', 'operation': 'max', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy_max'},
    {'field': 'h_norm', 'operation': 'std', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'canopy_openness'},
    {'field': 'h_norm', 'operation': 'std', 'class_field': 'atl08_class', 'class_id': [3], 'outfield': 'toc_roughness'},
    {'field': 'h_norm', 'operation': 'percentile_rh', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'canopy_h_metrics'},

     # Canopy ABS (Using h_ph)
    {'field': 'h_ph', 'operation': 'get_max98', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy_abs'},
    {'field': 'h_ph', 'operation': 'max', 'class_field': 'atl08_class', 'class_id': [2,3], 'outfield': 'h_canopy_max_abs'},

    # Counts
    {'field': 'h_ph', 'operation': 'get_len', 'class_field': 'atl08_class', 'class_id': [1, 2, 3], 'outfield': 'n_seg_ph'},
    {'field': 'h_ph', 'operation': 'get_len', 'class_field': 'atl08_class', 'class_id': [1], 'outfield': 'n_te_photons'},
    {'field': 'h_ph', 'operation': 'get_len', 'class_field': 'atl08_class', 'class_id': [2], 'outfield': 'n_ca_photons'},
    {'field': 'h_ph', 'operation': 'get_len', 'class_field': 'atl08_class', 'class_id': [3], 'outfield': 'n_toc_photons'},


    # Non-stanard Aggregations
    {'field': 'delta_time', 'operation': 'get_len_unique', 'class_field': 'atl08_class', 'class_id': [0,1,2,3], 'outfield': 'n_shots'},
    
    # Complex Metrics
    {'field': 'h_norm', 'operation': 'get_canopy_cover', 'class_field': 'atl08_class', 'class_id': [1,2,3], 'outfield': 'canopy_cover', 'kwargs': {'height_threshold': 2.0}},
    {'field': 'h_norm', 'operation': 'get_pai', 'class_field': 'atl08_class', 'class_id': [1,2,3], 'outfield': 'pai', 'kwargs': {'height_threshold': 2.0, 'k': 0.5}},
    {'field': 'h_ph', 'operation': 'prepare_segment_bins', 'class_field': 'atl08_class', 'class_id': [0,1,2,3,40,41], 'outfield': 'profile', 'kwargs': {'bin_height_m': 0.1, 'background_rate_hz': 1e6}},
]

# --- Configuration: ALS Aggregation Recipe ---
ALS_AGG_CONFIG = [
    # Geometry
    {'field': 'x', 'operation': 'median', 'class_field': 'classification', 'class_id': list(range(1,100)), 'outfield': 'x_als'},
    {'field': 'y', 'operation': 'median', 'class_field': 'classification', 'class_id': list(range(1,100)), 'outfield': 'y_als'},

    # Terrain & Bathy
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'classification', 'class_id': [2], 'outfield': 'als_topo_median'},
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'classification', 'class_id': [40], 'outfield': 'als_bathy_median'},
    {'field': 'ortho_h', 'operation': 'median', 'class_field': 'classification', 'class_id': [41], 'outfield': 'als_surface_median'},

    # Terrain & Bathy
    {'field': 'ellip_h', 'operation': 'median', 'class_field': 'classification', 'class_id': [2], 'outfield': 'als_topo_ellip_median'},
    {'field': 'ellip_h', 'operation': 'median', 'class_field': 'classification', 'class_id': [40], 'outfield': 'als_bathy_ellip_median'},
    {'field': 'ellip_h', 'operation': 'median', 'class_field': 'classification', 'class_id': [41], 'outfield': 'als_surface_ellip_median'},


    # Canopy
    {'field': 'h_norm', 'operation': 'get_max98', 'class_field': 'classification', 'class_id': [3,4,5], 'outfield': 'als_norm_veg_max98'},
    {'field': 'h_norm', 'operation': 'get_canopy_cover', 'class_field': 'classification', 'class_id': list(range(1, 100)), 'outfield': 'als_canopy_cover', 'kwargs': {'height_threshold': 2.0}},
    {'field': 'h_norm', 'operation': 'get_pai', 'class_field': 'classification', 'class_id': list(range(1, 100)), 'outfield': 'als_pai', 'kwargs': {'height_threshold': 2.0, 'k': 0.5}},
]
