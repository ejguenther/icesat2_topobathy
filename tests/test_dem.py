"""Reference-DEM helpers on synthetic rasters: sampling, slope, priority fill."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from phoreal.geo.dem import DemStack, bilinear, plane_fit_gradient

EPSG = 'EPSG:26911'
X0, Y0 = 400000.0, 4100000.0
GX, GY = 0.3, -0.2          # dz/dx (east), dz/dy (north)


def plane(x, y):
    return 1000 + GX * (x - X0) + GY * (y - Y0)


def write(path, nx=300, ny=300, x0=X0, y0=Y0, hole=None, offset=0.0):
    cols, rows = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    a = plane(x0 + cols, y0 + ny - rows).astype('float32') + offset
    if hole is not None:
        a[hole] = np.nan
    with rasterio.open(path, 'w', driver='GTiff', height=ny, width=nx, count=1, dtype='float32',
                       crs=EPSG, nodata=np.nan, transform=from_origin(x0, y0 + ny, 1, 1)) as r:
        r.write(a, 1)
    return path


def test_bilinear_exact_on_plane():
    tr = from_origin(X0, Y0 + 50, 1, 1)
    cols, rows = np.meshgrid(np.arange(50) + 0.5, np.arange(50) + 0.5)
    a = plane(X0 + cols, Y0 + 50 - rows)
    x = X0 + np.array([10.3, 25.0, 40.9])
    y = Y0 + np.array([5.5, 20.2, 44.1])
    np.testing.assert_allclose(bilinear(a, tr, x, y), plane(x, y), atol=1e-9)
    assert np.isnan(bilinear(a, tr, [X0 - 5], [Y0 + 5]))[0]


def test_plane_fit_gradient_and_nodata():
    cols, rows = np.meshgrid(np.arange(100) + 0.5, np.arange(100) + 0.5)
    a = plane(X0 + cols, Y0 + 100 - rows)
    a[50, 50] = np.nan
    gx, gy = plane_fit_gradient(a, 1.0, 40)
    np.testing.assert_allclose(gx[25, 25], GX, atol=1e-9)
    np.testing.assert_allclose(gy[25, 25], GY, atol=1e-9)
    assert np.isnan(gx[50, 60]) and np.isnan(gx[0, 0])     # window touches nodata / edge


def test_stack_slope_aspect_and_priority(tmp_path):
    hole = (slice(100, 200), slice(100, 200))
    first = write(tmp_path / 'a.tif', hole=hole, offset=5.0)
    second = write(tmp_path / 'b.tif')
    st = DemStack([first, second])
    x = X0 + np.array([50.0, 150.0, 1000.0])
    y = Y0 + np.array([50.0, 150.0, 50.0])
    s = st.sample(x, y, slope_window_m=40)
    np.testing.assert_allclose(s['elev'][:2], plane(x[:2], y[:2]) + [5.0, 0.0], atol=1e-3)
    assert list(s['src']) == [0, 1, -1]
    slope = np.degrees(np.arctan(np.hypot(GX, GY)))
    np.testing.assert_allclose(s['slope_deg'][:2], slope, atol=1e-3)
    # downslope points along -gradient = (-0.3, +0.2): azimuth atan2(-0.3, 0.2)
    np.testing.assert_allclose(s['aspect_deg'][:2], np.degrees(np.arctan2(-GX, -GY)) % 360, atol=1e-3)


def test_itrf_to_dem_frame_vertical_shift():
    pytest.importorskip('pyproj')
    from phoreal.geo.dem import itrf_to_dem_frame
    try:
        x, y, h = itrf_to_dem_frame([-118.25], [37.35], [2000.0], 2025.13, 'EPSG:26911')
    except RuntimeError:
        pytest.skip('GEOID18 grid unavailable (offline)')
    # GEOID18 ~ -27 m here, plus ~0.6 m ITRF2014 -> NAD83(2011): NAVD88 h ~ 2026 m
    assert 2020 < h[0] < 2032
    assert 380000 < x[0] < 400000
