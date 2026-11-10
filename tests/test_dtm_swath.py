"""DTM swath on synthetic GeoTIFF tiles: frame, values and shift recovery."""
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from pyproj import Transformer

from phoreal.als.swath import create_dtm_swaths
from phoreal.geolocation import create_interpolator, estimate_offset

EPSG = 'EPSG:32634'
X0, Y0 = 500000.0, 5800000.0


def terrain(x, y):
    return 200 + 0.02 * (y - Y0) + 4 * np.sin((y - Y0) / 50.0) + 3 * np.sin((x - X0) / 15.0)


@pytest.fixture
def tiles(tmp_path):
    rows = []
    for i in range(3):            # three 500 m tiles stacked northwards
        x0, y0 = X0 - 250, Y0 + 500 * i
        cols, rws = np.meshgrid(np.arange(500) + 0.5, np.arange(500) + 0.5)
        xs, ys = x0 + cols, y0 + 500 - rws
        name = f'tile{i}'
        with rasterio.open(tmp_path / f'{name}_dtm.tif', 'w', driver='GTiff', height=500, width=500,
                           count=1, dtype='float32', crs=EPSG, nodata=-9999,
                           transform=from_origin(x0, y0 + 500, 1, 1)) as r:
            r.write(terrain(xs, ys).astype('float32'), 1)
        rows.append(dict(file_name=str(tmp_path / f'{name}.laz'), geometry=box(x0, y0, x0 + 500, y0 + 500)))
    return gpd.GeoDataFrame(rows, crs=EPSG).to_crs(4326), tmp_path


def photons(at, x_offset=0.0):
    """Photons on a northward track x = X0 + x_offset; alongtrack = northing offset."""
    tr = Transformer.from_crs(EPSG, 4326, always_xy=True)
    lon, lat = tr.transform(np.full(at.size, X0 + x_offset), Y0 + at)
    return pd.DataFrame({'longitude': lon, 'latitude': lat, 'alongtrack': at})


def test_dtm_swath_values_and_shift(tiles):
    gdf, folder = tiles
    at = np.sort(np.random.default_rng(0).uniform(50, 1450, 4000))
    sw = create_dtm_swaths(gdf, {'gt1l': photons(at)}, str(folder), EPSG, crosstrack_limit=20)['gt1l']
    assert len(sw) > 0 and sw['crosstrack'].abs().max() < 20
    # heading north: +crosstrack is left = west
    # pixel centres sit on half-metre offsets; recover them from (at, xt)
    x = X0 - sw['crosstrack'].to_numpy(float)
    y = Y0 + sw['alongtrack'].to_numpy(float)
    assert np.allclose(x - np.floor(x), 0.5, atol=0.05)
    assert np.allclose(sw['ellip_h'], terrain(x, y), atol=0.05)

    # Shift recovery is not tested here: binning a pixel lattice into the
    # at/xt grid aliases (see create_dtm_swaths); create_dtm_grids is used for fits.


@pytest.mark.parametrize("x_offset", [0.0, 0.37])
def test_dtm_grid_shift_recovery(tiles, x_offset):
    from phoreal.als.swath import create_dtm_grids
    gdf, folder = tiles
    at = np.sort(np.random.default_rng(0).uniform(50, 1450, 4000))
    interp, ground = create_dtm_grids(gdf, {'gt1l': photons(at, x_offset)}, str(folder), EPSG,
                                      crosstrack_limit=20)['gt1l']
    # surface at the track itself equals the terrain
    assert np.nanmax(np.abs(interp((at, np.zeros_like(at))) - terrain(X0 + x_offset, Y0 + at))) < 0.05
    true_corr = (2.5, -1.5)
    z = terrain(np.full(at.size, X0 + x_offset), Y0 + at)
    res = estimate_offset(at - true_corr[0], np.zeros_like(at) - true_corr[1], z, interp,
                          min_valid_frac=0.9, return_surface=False)
    assert res['corr_at'] == pytest.approx(true_corr[0], abs=0.2)
    assert res['corr_xt'] == pytest.approx(true_corr[1], abs=0.2)
    assert len(ground) > 0
