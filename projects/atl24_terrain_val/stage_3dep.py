"""Stage USGS 3DEP DEM corridors along the granule's beams.

1. Build a corridor (beam centre line +/- corridor_buffer_m) from the photon
   parquets written by build_photons.py, US part only.
2. Ask the TNM Access API for DEM tiles that intersect it.
3. Read only the corridor window of each cloud-optimised tile over HTTP,
   blank pixels outside the corridor and write a small compressed GeoTIFF.
4. Write <dem_dir>/<product>/manifest.csv (tile, project, URL, dates, CRS,
   output file, valid pixels). Re-running skips tiles already staged.

Products: 1m = "Digital Elevation Model (DEM) 1 meter" (default),
          13 = "National Elevation Dataset (NED) 1/3 arc-second" (fallback).
3DEP heights are NAVD88 metres; tiles are NAD83 (UTM for 1 m, geographic for 1/3").

Run from this folder:  uv run python stage_3dep.py [--product 1m] [--workers 8]
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from phoreal.config import load_config

HERE = Path(__file__).resolve().parent
API = 'https://tnmaccess.nationalmap.gov/api/v1/products'
DATASETS = {'1m': 'Digital Elevation Model (DEM) 1 meter',
            '13': 'National Elevation Dataset (NED) 1/3 arc-second'}
US_LAT = (31.3, 49.05)         # CONUS part of this track (Mexico / Canada excluded)
WORK_CRS = 'EPSG:26911'        # metric CRS for buffering (UTM 11N covers this track)
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                GDAL_HTTP_MAX_RETRY='5', GDAL_HTTP_RETRY_DELAY='2', VSI_CACHE='TRUE')


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def beam_lines(photon_dir, beams, gap_m=500.0, step=5):
    """Per-beam centre lines (segment-median positions), split at data gaps."""
    lines = []
    for gt in beams:
        df = pd.read_parquet(photon_dir / f'photons_{gt}.parquet',
                             columns=['latitude', 'longitude', 'segment_id'])
        df = df[(df.latitude > US_LAT[0]) & (df.latitude < US_LAT[1])]
        seg = df.groupby('segment_id')[['longitude', 'latitude']].median()
        pts = gpd.GeoSeries(gpd.points_from_xy(seg.longitude, seg.latitude), crs=4326).to_crs(WORK_CRS)
        xy = np.column_stack([pts.x, pts.y])[::step]
        brk = np.flatnonzero(np.hypot(*np.diff(xy, axis=0).T) > gap_m) + 1
        for part in np.split(xy, brk):
            if len(part) > 1:
                lines.append(LineString(part))
    return lines


def tnm_query(dataset, bbox, max_items=500):
    q = {'datasets': dataset, 'bbox': ','.join(f'{v:.5f}' for v in bbox),
         'max': max_items, 'outputFormat': 'JSON'}
    url = API + '?' + urllib.parse.urlencode(q)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)['items']
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def find_tiles(corridor_ll, dataset, chunk_deg=0.25):
    """Query the API in latitude chunks; keep tiles whose bbox meets the corridor."""
    items = {}
    lat0, lat1 = corridor_ll.bounds[1], corridor_ll.bounds[3]
    for la in np.arange(lat0, lat1, chunk_deg):
        part = corridor_ll.intersection(box(-180, la, 180, la + chunk_deg))
        if part.is_empty:
            continue
        for it in tnm_query(dataset, part.bounds):
            bb = it['boundingBox']
            if box(bb['minX'], bb['minY'], bb['maxX'], bb['maxY']).intersects(part):
                items[it['downloadURL']] = it
    return list(items.values())


def utm_zone_ok(title, lon):
    """1 m tiles that straddle zones are delivered in both; keep the zone of the location."""
    m = re.search(r'1 Meter (\d+) x', title)
    return m is None or int(m.group(1)) == int((lon + 180) // 6) + 1


def stage_tile(it, corridor_work, out_dir):
    url = it['downloadURL']
    out = out_dir / Path(url).name
    rec = {'title': it['title'], 'url': url, 'publication_date': it.get('publicationDate'),
           'project': (re.search(r'/Projects/([^/]+)/', url) or [None, None])[1],
           'out_file': str(out.relative_to(HERE)) if out.is_relative_to(HERE) else str(out)}
    if out.exists():
        with rasterio.open(out) as r:
            rec.update(crs=r.crs.to_string(), status='exists')
        return rec
    with rasterio.Env(**GDAL_ENV), rasterio.open('/vsicurl/' + url) as src:
        corr = gpd.GeoSeries([corridor_work], crs=WORK_CRS).to_crs(src.crs).iloc[0]
        part = corr.intersection(box(*src.bounds))
        rec['crs'] = src.crs.to_string()
        if part.is_empty:
            rec['status'] = 'no_overlap'
            return rec
        win = from_bounds(*part.bounds, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        a = src.read(1, window=win, masked=True).astype('float32').filled(np.nan)
        tr = src.window_transform(win)
        a[geometry_mask([part], out_shape=a.shape, transform=tr)] = np.nan
        rec['valid_px'] = int(np.isfinite(a).sum())
        if rec['valid_px'] == 0:
            rec['status'] = 'empty'
            return rec
        prof = dict(driver='GTiff', height=a.shape[0], width=a.shape[1], count=1, dtype='float32',
                    crs=src.crs, transform=tr, nodata=np.nan, compress='deflate', predictor=3,
                    tiled=True, blockxsize=256, blockysize=256)
    tmp = out.with_suffix('.part.tif')
    with rasterio.open(tmp, 'w', **prof) as dst:
        dst.write(a, 1)
    tmp.rename(out)
    rec['status'] = 'staged'
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=HERE / 'config.yaml')
    ap.add_argument('--product', choices=list(DATASETS), default='1m')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--list-only', action='store_true', help='query tiles, do not download')
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = resolve(cfg['work_dir'])
    out_dir = resolve(cfg['dem_dir']) / args.product
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = beam_lines(work / 'photons', cfg['beams'])
    corridor = unary_union([ln.buffer(cfg['dem']['corridor_buffer_m']) for ln in lines])
    corridor_ll = gpd.GeoSeries([corridor], crs=WORK_CRS).to_crs(4326).iloc[0]
    print(f'corridor: {len(lines)} line parts, {corridor.area / 1e6:.0f} km^2')

    items = find_tiles(corridor_ll, DATASETS[args.product])
    if args.product == '1m':
        items = [it for it in items
                 if utm_zone_ok(it['title'], (it['boundingBox']['minX'] + it['boundingBox']['maxX']) / 2)]
    gb = sum((it.get('sizeInBytes') or 0) for it in items) / 1e9
    print(f'{len(items)} tiles intersect the corridor ({gb:.0f} GB as whole tiles)')
    if args.list_only:
        pd.DataFrame([{k: it.get(k) for k in ('title', 'downloadURL', 'publicationDate', 'sizeInBytes')}
                      for it in items]).to_csv(out_dir / 'tile_list.csv', index=False)
        return

    recs = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(stage_tile, it, corridor, out_dir): it for it in items}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                recs.append(fu.result())
            except Exception as e:
                it = futs[fu]
                recs.append({'title': it['title'], 'url': it['downloadURL'], 'status': f'error: {e}'})
            if i % 20 == 0 or i == len(futs):
                print(f'  {i}/{len(futs)} tiles', flush=True)
    man = pd.DataFrame(recs).sort_values('title')
    man.to_csv(out_dir / 'manifest.csv', index=False)
    print(man['status'].str.split(':').str[0].value_counts().to_string())


if __name__ == '__main__':
    main()
