import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy import ndimage
import time

def create_interpolator(df_als, grid_resolution=1.0, ground_only=True):
    if ground_only:
        df_als = df_als[df_als['classification'] == 2]
        
    at_coords = df_als['alongtrack'].values
    xt_coords = df_als['crosstrack'].values
    z_values = df_als['ellip_h'].values
    
    at_min, at_max = np.min(at_coords), np.max(at_coords)
    xt_min, xt_max = np.min(xt_coords), np.max(xt_coords)
    
    at_bins = int(np.ceil((at_max - at_min) / grid_resolution))
    xt_bins = int(np.ceil((xt_max - xt_min) / grid_resolution))
    
    # 1. FAST BINNING (Takes milliseconds)
    dtm_z, at_edges, xt_edges, _ = binned_statistic_2d(
        at_coords, xt_coords, z_values, statistic='mean', bins=[at_bins, xt_bins]
    )
    
    # 2. FAST HOLE FILLING (Takes microseconds)
    # This acts like nearest-neighbor but ONLY for the NaN holes. 
    # ndimage.distance_transform_edt is an insanely fast C-optimized function.
    invalid_mask = np.isnan(dtm_z)
    if np.any(invalid_mask):
        indices = ndimage.distance_transform_edt(
            invalid_mask, return_distances=False, return_indices=True
        )
        dtm_z = dtm_z[tuple(indices)]
            
    # 3. FAST SMOOTHING (Takes milliseconds)
    # This blurs the "Minecraft blocks" into a smooth, continuous terrain.
    # It completely fixes the nearest-neighbor offset bias you observed.
    # dtm_z_smooth = ndimage.gaussian_filter(dtm_z, sigma=0.5)
    dtm_z_smooth = dtm_z

    at_centers = (at_edges[:-1] + at_edges[1:]) / 2.0
    xt_centers = (xt_edges[:-1] + xt_edges[1:]) / 2.0
    
    # 4. THE INTERPOLATOR
    # We use 'linear' here so the error surface is smooth, but because 
    # we already filled the NaNs, it runs instantly.
    interpolator = RegularGridInterpolator(
        (at_centers, xt_centers), 
        dtm_z_smooth, 
        method='linear', 
        bounds_error=False, 
        fill_value=np.nan
    )
    
    return interpolator

def calculate_mae_cost(shift, icesat2_at, icesat2_xt, icesat2_z, als_surface_interpolator):
    """
    Objective function to find the optimal along-track and cross-track shift.
    
    Parameters:
    -----------
    shift : tuple or list
        Proposed shift as [delta_AT, delta_XT]
    icesat2_at : ndarray
        Native ATL03 along-track coordinates for the photons
    icesat2_xt : ndarray
        Native ATL03 cross-track coordinates for the photons
    icesat2_z : ndarray
        Photon elevations
    als_surface_interpolator : callable
        A function or SciPy interpolator (e.g., RegularGridInterpolator) 
        that takes (AT, XT) arrays and returns ALS Z elevations.
        
    Returns:
    --------
    float
        The Mean Absolute Error (MAE) of the elevation residuals.
    """
    delta_at, delta_xt = shift
    
    # 1. Apply the trial shift in the AT/XT frame
    shifted_at = icesat2_at + delta_at
    shifted_xt = icesat2_xt + delta_xt
    
    # 2. Sample the ALS surface at these new shifted coordinates
    # Note: Depending on how your ALS swath is structured (e.g., a rasterized 
    # DSM in AT/XT space), this interpolator pulls the coincident ALS elevation.
    als_z = als_surface_interpolator((shifted_at, shifted_xt))
    
    # 3. Filter out NaN values in case the shift pushes points outside the ALS swath
    valid_mask = ~np.isnan(als_z)
    if not np.any(valid_mask):
        return np.inf # Return infinite cost if we shifted completely off the map
        
    # 4. Calculate the residuals
    residuals = icesat2_z[valid_mask] - als_z[valid_mask]
    
    # 5. Compute the cost metric
    # Using MAE (Mean Absolute Error) is often better than RMSE for photon data 
    # because it is less sensitive to noise/outliers (like solar background or clouds).
    mae = np.mean(np.abs(residuals))
    
    return mae


def calculate_z_shift(alongtrack, crosstrack, h_ph, optimal_shift_at, optimal_shift_xt , surface_interpolator):
    """
    Calculates the vertical shift (Z-shift) required to align ICESat-2 photons 
    with the ALS surface, given the optimal AT/XT shift.
    
    Parameters:
    -----------
    alongtrack : ndarray
        ICESat-2 along-track coordinates.
    crosstrack : ndarray
        ICESat-2 cross-track coordinates.
    h_ph : ndarray
        ICESat-2 photon elevations.
    optimal_shift_at : float
        The optimal along-track shift found during geolocation optimization.
    optimal_shift_xt : float
        The optimal cross-track shift found during geolocation optimization.
    surface_interpolator : callable
        A function or SciPy interpolator that takes (AT, XT) arrays and 
        returns the corresponding ALS Z elevations.
        
    Returns:
    --------
    z_shift : float
        The median vertical difference between the photons and the ALS surface 
        at the optimal AT/XT alignment.
    """
    # 1. Apply the optimal AT/XT shift to the photon coordinates
    shifted_at = alongtrack + optimal_shift_at
    shifted_xt = crosstrack + optimal_shift_xt
    
    # 2. Sample the ALS surface at the new shifted coordinates
    als_z = surface_interpolator((shifted_at, shifted_xt))
    
    # 3. Filter out NaN values (where the shift pushed points outside the ALS swath)
    valid_mask = ~np.isnan(als_z)
    
    if not np.any(valid_mask):
        # If all points are outside the swath, we cannot determine a shift.
        # Return 0.0 as a fallback, though this indicates a data issue.
        return 0.0
        
    # 4. Calculate the vertical residuals (the "error" in height)
    residuals = h_ph[valid_mask] - als_z[valid_mask]
    
    # 5. Compute the median vertical shift
    # We use the median instead of the mean to be robust against outliers 
    # (e.g., remaining noise photons or vegetation).
    z_shift = np.median(residuals)
    
    return z_shift  
    