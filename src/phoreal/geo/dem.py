"""Reference DEM sampling: datum transform, bilinear values, plane-fit slope.

Written for USGS 3DEP DEMs (NAD83(2011) horizontal, NAVD88 heights) compared
with ICESat-2 (ITRF2014 ellipsoidal heights). Rather than converting the DEM,
the ICESat-2 points are moved into the DEM frame (``itrf_to_dem_frame``), so a
difference is simply ``h_dem_frame - dem``.

Rasters are given as paths in priority order (``DemStack``): a point takes its
value from the first raster with valid data there.
"""
import numpy as np
import pyproj
import rasterio
from pyproj import CRS, Transformer
from scipy import ndimage

NAVD88 = 'EPSG:5703'
ITRF2014_3D = 'EPSG:7912'


def itrf_to_dem_frame(lon, lat, h, epoch, dem_crs, vertical_crs=NAVD88,
                      src_crs=ITRF2014_3D, allow_network=True):
    """
    Transform ICESat-2 points (lon, lat, ellipsoidal h; ITRF2014 by default)
    at decimal-year ``epoch`` into the DEM frame: x, y in ``dem_crs`` and h in
    ``vertical_crs`` (NAVD88 via GEOID18 for 3DEP). ``dem_crs`` may be any
    horizontal CRS (e.g. 'EPSG:6340' or 'EPSG:26911' for UTM 11N).

    The PROJ geoid grid (us_noaa_g2018u0.tif) is fetched from the PROJ CDN when
    ``allow_network`` and it is not installed. Raises if the vertical shift
    could not be applied (PROJ would otherwise return h unchanged).
    """
    if allow_network:
        pyproj.network.set_network_enabled(True)
    horiz = CRS(dem_crs)
    if horiz.name.startswith('NAD83 / UTM zone') and horiz.utm_zone \
            and horiz.utm_zone.endswith('N') and 1 <= int(horiz.utm_zone[:-1]) <= 19:
        # 3DEP tiles carry the generic "NAD83 / UTM" CRS; their realisation is NAD83(2011)
        horiz = CRS(f'EPSG:{6329 + int(horiz.utm_zone[:-1])}')   # 6340 = NAD83(2011) / UTM 11N
    dst = CRS.from_user_input(f'{horiz.to_string()}+{vertical_crs}') \
        if vertical_crs else horiz
    t = Transformer.from_crs(CRS(src_crs), dst, always_xy=True, only_best=True)
    lon, lat, h = (np.asarray(a, dtype=float) for a in (lon, lat, h))
    ep = np.full(lon.shape, float(epoch))
    x, y, hz, _ = t.transform(lon, lat, h, ep, errcheck=False)
    x, y, hz = np.asarray(x), np.asarray(y), np.asarray(hz)
    if vertical_crs and np.all(np.abs(hz - h)[np.isfinite(hz)] < 1e-6):
        raise RuntimeError('vertical datum shift not applied (geoid grid missing?)')
    return x, y, hz


def bilinear(a, transform, x, y):
    """Bilinear sample of array ``a`` (NaN = nodata) at map coords; NaN outside."""
    col, row = ~transform * (np.asarray(x, float), np.asarray(y, float))
    c0 = np.floor(col - 0.5).astype(np.int64)
    r0 = np.floor(row - 0.5).astype(np.int64)
    fc, fr = col - 0.5 - c0, row - 0.5 - r0
    out = np.full(len(c0), np.nan)
    ok = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < a.shape[1]) & (r0 + 1 < a.shape[0])
    c, r, u, v = c0[ok], r0[ok], fc[ok], fr[ok]
    out[ok] = (a[r, c] * (1 - u) * (1 - v) + a[r, c + 1] * u * (1 - v)
               + a[r + 1, c] * (1 - u) * v + a[r + 1, c + 1] * u * v)
    return out


def plane_fit_gradient(a, res, window_m):
    """
    Least-squares plane gradient over a square window (side ~``window_m``)
    centred on every pixel: returns (dz/dx, dz/dy) in map units, y = north.
    Pixels whose window holds any nodata get NaN.
    """
    k = max(1, int(round(window_m / res / 2)))
    u = np.arange(-k, k + 1, dtype=float)
    ones = np.ones_like(u)
    valid = np.isfinite(a)
    z = np.where(valid, a, 0.0)
    su2 = (u ** 2).sum() * len(u)
    gx = ndimage.correlate1d(ndimage.correlate1d(z, u, axis=1, mode='constant'), ones, axis=0, mode='constant')
    gy = ndimage.correlate1d(ndimage.correlate1d(z, u, axis=0, mode='constant'), ones, axis=1, mode='constant')
    n = ndimage.correlate1d(ndimage.correlate1d(valid.astype(float), ones, axis=1, mode='constant'),
                            ones, axis=0, mode='constant')
    full = n > len(u) ** 2 - 0.5
    gx = np.where(full, gx / su2 / res, np.nan)
    gy = np.where(full, -gy / su2 / res, np.nan)    # rows run south
    return gx, gy


class DemStack:
    """
    Rasters in priority order, all in one horizontal CRS (checked). ``sample``
    reads each raster once and fills points that are still NaN.
    """

    def __init__(self, paths):
        self.paths = [str(p) for p in paths]
        crs, self.bounds = set(), []
        for p in self.paths:
            with rasterio.open(p) as r:
                c = CRS(r.crs.to_wkt())
                crs.add(c.to_epsg() or c.to_wkt())
                self.bounds.append(r.bounds)
        if len(crs) > 1:
            raise ValueError(f'DemStack rasters must share one CRS, got {crs}')
        self.crs = CRS(crs.pop()) if crs else None

    def sample(self, x, y, slope_window_m=None):
        """
        Returns dict: elev, src (index into paths, -1 = none) and, with
        ``slope_window_m``, gx, gy (dz/dx, dz/dy), slope_deg and aspect_deg
        (downslope azimuth, degrees clockwise from grid north).
        """
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = len(x)
        out = {'elev': np.full(n, np.nan), 'src': np.full(n, -1, np.int32)}
        if slope_window_m:
            out['gx'] = np.full(n, np.nan)
            out['gy'] = np.full(n, np.nan)
        order = np.argsort(y, kind='stable')     # y-sorted lookup: tracks run north-south
        ys = y[order]
        for i, (p, b) in enumerate(zip(self.paths, self.bounds)):
            lo = np.searchsorted(ys, b.bottom, side='left')
            hi = np.searchsorted(ys, b.top, side='right')
            if hi <= lo:
                continue
            idx = np.sort(order[lo:hi])
            idx = idx[(x[idx] >= b.left) & (x[idx] <= b.right) & np.isnan(out['elev'][idx])]
            if not len(idx):
                continue
            with rasterio.open(p) as r:
                a = r.read(1, masked=True).astype(float).filled(np.nan)
                tr, res = r.transform, r.res[0]
            e = bilinear(a, tr, x[idx], y[idx])
            ok = np.isfinite(e)
            idx, e = idx[ok], e[ok]
            out['elev'][idx] = e
            out['src'][idx] = i
            if slope_window_m and len(idx):
                gx, gy = plane_fit_gradient(a, res, slope_window_m)
                out['gx'][idx] = bilinear(gx, tr, x[idx], y[idx])
                out['gy'][idx] = bilinear(gy, tr, x[idx], y[idx])
        if slope_window_m:
            g = np.hypot(out['gx'], out['gy'])
            out['slope_deg'] = np.degrees(np.arctan(g))
            out['aspect_deg'] = np.degrees(np.arctan2(-out['gx'], -out['gy'])) % 360
        return out
