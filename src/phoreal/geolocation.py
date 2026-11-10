import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy import ndimage
import time

def create_interpolator(df_als, grid_resolution=1.0, ground_only=True, ground_class = 2,
                        max_fill_distance=None):
    """
    Grids ALS ``ellip_h`` in (alongtrack, crosstrack) and returns a linear
    RegularGridInterpolator over (at, xt).

    max_fill_distance: empty cells are filled from the nearest non-empty cell.
        If set (metres), cells farther than this from any ALS point stay NaN,
        so gaps (water, buildings, missing tiles) are not extrapolated over.
        None keeps the original behaviour (fill everything).
    """
    if ground_only:
        df_als = df_als[df_als['classification'] == ground_class]
        
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
        distances, indices = ndimage.distance_transform_edt(
            invalid_mask, return_distances=True, return_indices=True
        )
        dtm_z = dtm_z[tuple(indices)]
        if max_fill_distance is not None:
            dtm_z[distances * grid_resolution > max_fill_distance] = np.nan
            
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
    

# --------------------------------------------------------------------------- #
# Offset estimation with diagnostics
# --------------------------------------------------------------------------- #
#
# Conventions
#   alongtrack (at): positive in the direction of travel (segment_dist_x).
#   crosstrack (xt): positive to the LEFT of travel (see
#       phoreal.als.swath.estimate_signed_crosstrack).
#   correction (corr_at, corr_xt): shift ADDED to the reported ICESat-2
#       coordinates to align them with the ALS surface, so
#       reference ~= reported + correction.
#   geolocation error = reported - reference = -correction.
#   heading: Earth-fixed azimuth of travel, degrees clockwise from true north.

from scipy.optimize import brute, fmin


def _valid_fraction(shift, at, xt, interpolator):
    als_z = interpolator((at + shift[0], xt + shift[1]))
    return float(np.mean(~np.isnan(als_z))) if len(als_z) else 0.0


def make_cost_function(at, xt, z, interpolator, fit_dz=False, min_valid_frac=None):
    """
    Returns cost(shift) -> float for the given photons.

    fit_dz=False: MAE of h_ph - ALS (identical to calculate_mae_cost).
    fit_dz=True:  MAE after removing the median residual, i.e. a free vertical
        offset is profiled out at each trial shift.
    min_valid_frac: if set, shifts where fewer than this fraction of photons
        land on ALS data cost inf. This stops the search drifting toward
        partial overlap.
    """
    at = np.asarray(at, dtype=float)
    xt = np.asarray(xt, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(z)

    def cost(shift):
        als_z = interpolator((at + shift[0], xt + shift[1]))
        valid = ~np.isnan(als_z)
        if not np.any(valid):
            return np.inf
        if min_valid_frac is not None and valid.sum() < min_valid_frac * n:
            return np.inf
        r = z[valid] - als_z[valid]
        if fit_dz:
            r = r - np.median(r)
        return float(np.mean(np.abs(r)))

    return cost


def _local_minima(J):
    """Indices of strict-ish local minima of a 2D cost grid (3x3 neighbourhood)."""
    from scipy.ndimage import minimum_filter
    Jf = np.where(np.isfinite(J), J, np.nanmax(J[np.isfinite(J)]) + 1 if np.isfinite(J).any() else 0)
    mins = (Jf == minimum_filter(Jf, size=3, mode='nearest')) & np.isfinite(J)
    return np.argwhere(mins)


def _quadratic_hessian(grid_at, grid_xt, J, i, j, half_width=2):
    """
    Fits J ~ c + b.x + x^T A x around grid cell (i, j). Returns the Hessian
    (2A) as (h_aa, h_ax, h_xx), or NaNs if the fit is not possible.
    """
    sl = (slice(max(i - half_width, 0), i + half_width + 1),
          slice(max(j - half_width, 0), j + half_width + 1))
    xa = grid_at[sl].ravel(); xx = grid_xt[sl].ravel(); y = J[sl].ravel()
    ok = np.isfinite(y)
    if ok.sum() < 6:
        return np.nan, np.nan, np.nan
    xa, xx, y = xa[ok] - grid_at[i, j], xx[ok] - grid_xt[i, j], y[ok]
    A = np.column_stack([np.ones_like(xa), xa, xx, xa**2, xa * xx, xx**2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 2 * coef[3], coef[4], 2 * coef[5]


def bootstrap_offset(at, xt, z, interpolator, center, n_bootstrap=30, block_m=100.0,
                     half_width=4.0, step=0.25, fit_dz=False, min_valid_frac=None,
                     seed=0):
    """
    Block bootstrap of the offset: resample along-track blocks of photons with
    replacement and re-solve on a local grid around ``center``.

    Blocks keep along-track correlation together, so the spread is not
    overstated the way photon-level resampling would be.

    Returns (sigma_at, sigma_xt, corr, frac_on_edge): standard deviations (m),
    their correlation, and the fraction of replicates whose optimum hit the
    local grid edge (if large, the sigmas are lower bounds).
    """
    at = np.asarray(at, float); xt = np.asarray(xt, float); z = np.asarray(z, float)
    blocks = ((at - at.min()) // block_m).astype(int)
    uniq = np.unique(blocks)
    if len(uniq) < 3:
        return np.nan, np.nan, np.nan, np.nan
    members = [np.flatnonzero(blocks == b) for b in uniq]
    offs = np.arange(-half_width, half_width + step / 2, step)
    ga, gx = np.meshgrid(center[0] + offs, center[1] + offs, indexing='ij')
    rng = np.random.default_rng(seed)
    sols, edge = [], 0
    for _ in range(n_bootstrap):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        cost = make_cost_function(at[idx], xt[idx], z[idx], interpolator,
                                  fit_dz=fit_dz, min_valid_frac=min_valid_frac)
        J = np.array([cost((a, x)) for a, x in zip(ga.ravel(), gx.ravel())]).reshape(ga.shape)
        if not np.isfinite(J).any():
            continue
        i, j = np.unravel_index(np.nanargmin(np.where(np.isfinite(J), J, np.nan)), J.shape)
        edge += i in (0, len(offs) - 1) or j in (0, len(offs) - 1)
        sols.append((ga[i, j], gx[i, j]))
    if len(sols) < 3:
        return np.nan, np.nan, np.nan, np.nan
    sols = np.array(sols)
    sa, sx = sols.std(axis=0, ddof=1)
    corr = float(np.corrcoef(sols.T)[0, 1]) if sa > 0 and sx > 0 else np.nan
    return float(sa), float(sx), corr, edge / len(sols)


def estimate_offset(at, xt, z, interpolator, search=((-10, 10, 0.5), (-10, 10, 0.5)),
                    fit_dz=False, min_valid_frac=None, finish=True,
                    return_surface=True, n_bootstrap=0, bootstrap_block_m=100.0):
    """
    Grid search (scipy.optimize.brute) + optional Nelder-Mead finish for the
    horizontal correction aligning photons (at, xt, z) to an ALS surface.

    With fit_dz=False, min_valid_frac=None, finish=True and the default search
    this is the original perform_geolocation_offset.py estimator
    ("baseline_v0").

    Returns a dict:
        corr_at, corr_xt            correction (m). error = -correction.
        cost_zero, cost_opt         objective at (0, 0) and at the optimum
        grid_corr_at/_xt            best grid node before the finish step
        boundary_hit                best grid node on the search edge, or the
                                    finished optimum outside the search box
        valid_frac_zero/_opt        fraction of photons on ALS data
        n_local_minima              local minima in the cost grid
        second_min_cost, second_min_dist   next-best local minimum and its
                                    distance (m) from the best grid node
        second_min_ratio            second_min_cost / grid min cost (~1 = ambiguous)
        hess_aa, hess_ax, hess_xx   curvature of the cost surface at the minimum
        dz_opt                      median residual at the optimum (vertical
                                    offset; profiled out if fit_dz)
        boot_sigma_at/_xt, boot_corr, boot_edge_frac
                                    block-bootstrap uncertainty (if n_bootstrap)
        n_photons
        surface                     (grid_at, grid_xt, J) if return_surface
    """
    at = np.asarray(at, dtype=float)
    xt = np.asarray(xt, dtype=float)
    z = np.asarray(z, dtype=float)
    cost = make_cost_function(at, xt, z, interpolator, fit_dz=fit_dz,
                              min_valid_frac=min_valid_frac)
    ranges = tuple(slice(*r) for r in search)

    x0, fval, grid, J = brute(cost, ranges=ranges, finish=fmin if finish else None,
                              full_output=True, disp=False)
    grid_at, grid_xt = grid[0], grid[1]

    Jfinite = np.where(np.isfinite(J), J, np.inf)
    i, j = np.unravel_index(np.argmin(Jfinite), J.shape)
    grid_best = (grid_at[i, j], grid_xt[i, j])
    on_edge = i in (0, J.shape[0] - 1) or j in (0, J.shape[1] - 1)
    lo_at, hi_at = grid_at.min(), grid_at.max()
    lo_xt, hi_xt = grid_xt.min(), grid_xt.max()
    x0 = np.atleast_1d(x0)
    outside = not (lo_at <= x0[0] <= hi_at and lo_xt <= x0[1] <= hi_xt)

    minima = _local_minima(J)
    others = [m for m in minima if not (m[0] == i and m[1] == j)]
    if others:
        k = min(others, key=lambda m: J[m[0], m[1]])
        second_cost = float(J[k[0], k[1]])
        second_dist = float(np.hypot(grid_at[k[0], k[1]] - grid_best[0],
                                     grid_xt[k[0], k[1]] - grid_best[1]))
    else:
        second_cost, second_dist = np.nan, np.nan

    h_aa, h_ax, h_xx = _quadratic_hessian(grid_at, grid_xt, J, i, j)

    als_z = interpolator((at + x0[0], xt + x0[1]))
    ok = ~np.isnan(als_z)
    dz_opt = float(np.median(z[ok] - als_z[ok])) if ok.any() else np.nan

    out = {
        'corr_at': float(x0[0]),
        'corr_xt': float(x0[1]),
        'cost_zero': float(cost((0.0, 0.0))),
        'cost_opt': float(fval),
        'grid_corr_at': float(grid_best[0]),
        'grid_corr_xt': float(grid_best[1]),
        'boundary_hit': bool(on_edge or outside),
        'valid_frac_zero': _valid_fraction((0.0, 0.0), at, xt, interpolator),
        'valid_frac_opt': _valid_fraction(x0, at, xt, interpolator),
        'n_local_minima': int(len(minima)),
        'second_min_cost': second_cost,
        'second_min_dist': second_dist,
        'second_min_ratio': second_cost / float(Jfinite[i, j]) if np.isfinite(second_cost) and Jfinite[i, j] > 0 else np.nan,
        'hess_aa': float(h_aa), 'hess_ax': float(h_ax), 'hess_xx': float(h_xx),
        'dz_opt': dz_opt,
        'n_photons': int(len(z)),
    }
    if n_bootstrap and np.isfinite(fval):
        sa, sx, rho, edge = bootstrap_offset(at, xt, z, interpolator, x0,
                                             n_bootstrap=n_bootstrap,
                                             block_m=bootstrap_block_m, fit_dz=fit_dz,
                                             min_valid_frac=min_valid_frac)
        out.update(boot_sigma_at=sa, boot_sigma_xt=sx, boot_corr=rho, boot_edge_frac=edge)
    if return_surface:
        out['surface'] = (grid_at, grid_xt, J)
    return out


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def track_heading(lat, lon):
    """
    Earth-fixed azimuth of travel (degrees clockwise from true north, 0-360)
    from the first to the last of time-ordered points, on the WGS84 ellipsoid.
    """
    from pyproj import Geod
    lat = np.asarray(lat); lon = np.asarray(lon)
    az, _, _ = Geod(ellps='WGS84').inv(lon[0], lat[0], lon[-1], lat[-1])
    return float(az % 360.0)


def atxt_to_en(d_at, d_xt, heading_deg):
    """
    Rotates an (along-track, cross-track-left) vector into (east, north).
    """
    a = np.radians(heading_deg)
    east = d_at * np.sin(a) - d_xt * np.cos(a)
    north = d_at * np.cos(a) + d_xt * np.sin(a)
    return east, north


def pointing_vector_enu(ref_elev, ref_azimuth):
    """
    Unit surface-to-spacecraft vector in local ENU from ATL03 ref_elev /
    ref_azimuth (radians; elevation up from the E-N plane, azimuth from north
    toward east).
    """
    ce = np.cos(ref_elev)
    return np.stack([ce * np.sin(ref_azimuth), ce * np.cos(ref_azimuth),
                     np.sin(ref_elev)], axis=-1)


def beam_vertical_angle(ref_elev):
    """Angle (deg) between the line of sight and the local ellipsoid normal."""
    return np.degrees(np.pi / 2 - np.asarray(ref_elev, dtype=float))


def terrain_normal_enu(grad_e, grad_n):
    """Upward unit terrain normal from a surface gradient dz/dE, dz/dN."""
    n = np.stack([-np.asarray(grad_e, float), -np.asarray(grad_n, float),
                  np.ones_like(np.asarray(grad_e, float))], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def terrain_incidence_angle(ref_elev, ref_azimuth, grad_e, grad_n):
    """
    Angle (deg) between the surface-to-spacecraft line of sight and the
    upward terrain normal. 0 = perpendicular incidence.
    """
    l = pointing_vector_enu(ref_elev, ref_azimuth)
    n = terrain_normal_enu(grad_e, grad_n)
    dot = np.clip(np.sum(l * n, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def fit_plane_atxt(at, xt, z):
    """
    Least-squares plane z = a*at + b*xt + c.

    Returns (a, b, c, roughness) where roughness is the standard deviation of
    the residuals (m). NaNs if fewer than 3 points.
    """
    at = np.asarray(at, float); xt = np.asarray(xt, float); z = np.asarray(z, float)
    if len(z) < 3:
        return np.nan, np.nan, np.nan, np.nan
    A = np.column_stack([at - at.mean(), xt - xt.mean(), np.ones_like(at)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    resid = z - A @ coef
    return float(coef[0]), float(coef[1]), float(coef[2]), float(np.std(resid))


def terrain_summary(als_ground, heading_deg):
    """
    Plane-fit terrain metrics for ALS ground points (columns alongtrack,
    crosstrack, ellip_h) in a window.

    Returns slope_deg, aspect_deg (downslope direction, clockwise from north),
    grad_e, grad_n, roughness, and density (points per m^2 of the window's
    at/xt bounding box).
    """
    a, b, _, rough = fit_plane_atxt(als_ground['alongtrack'], als_ground['crosstrack'],
                                    als_ground['ellip_h'])
    if not np.isfinite(a):
        return dict(slope_deg=np.nan, aspect_deg=np.nan, grad_e=np.nan,
                    grad_n=np.nan, roughness=np.nan, als_density=np.nan)
    grad_e, grad_n = atxt_to_en(a, b, heading_deg)
    slope = np.degrees(np.arctan(np.hypot(grad_e, grad_n)))
    aspect = np.degrees(np.arctan2(-grad_e, -grad_n)) % 360.0
    area = (np.ptp(als_ground['alongtrack']) * np.ptp(als_ground['crosstrack']))
    return dict(slope_deg=float(slope), aspect_deg=float(aspect), grad_e=float(grad_e),
                grad_n=float(grad_n), roughness=rough,
                als_density=float(len(als_ground) / area) if area > 0 else np.nan)
