"""
Builds an index of the Poland ALS tiles from their LAS headers.

Outputs (in <work_dir>/als_inventory/):
    parts/part_NNNNN.parquet   checkpointed header records (resumable)
    als_tiles.gpkg             one polygon per tile (EPSG:4326) with path,
                               survey_id, utm_epsg, point_count, density
    als_coverage.gpkg          dissolved coverage, one row per survey_id plus
                               an 'all' row; used for the CMR granule search

Usage:
    uv run python inventory_als.py [--config config.yaml] [--workers 8]
                                   [--chunk 2000] [--limit N]

Header reads over the network drive take ~1-4 s each, so the first full run
over ~44k tiles takes hours; re-running skips finished chunks.
"""
import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from phoreal.als.las_extent import process_las_files_multiprocessing, process_las_file
from phoreal.config import load_config


def tile_records(header_dicts):
    """Header dicts -> DataFrame with serialisable CRS columns."""
    rows = []
    for d in header_dicts:
        crs = d.pop('parse_crs')
        horiz = crs.to_2d() if crs is not None else None
        d['crs_name'] = crs.name if crs is not None else None
        d['utm_epsg'] = horiz.to_epsg() if horiz is not None else None
        d['creation_date'] = str(d['creation_date'])
        rows.append(d)
    df = pd.DataFrame(rows)
    if len(df):
        df['survey_id'] = df['file_name'].map(lambda p: os.path.basename(p).split('_')[0])
    return df


def build_geometries(df):
    """Native-CRS bbox polygons, reprojected per UTM zone to EPSG:4326."""
    parts = []
    for epsg, grp in df.groupby('utm_epsg'):
        geoms = [box(r.x_min, r.y_min, r.x_max, r.y_max) for r in grp.itertuples()]
        g = gpd.GeoDataFrame(grp.copy(), geometry=geoms, crs=f"EPSG:{int(epsg)}")
        area = (grp['x_max'] - grp['x_min']) * (grp['y_max'] - grp['y_min'])
        g['density'] = grp['point_count'] / area.replace(0, np.nan)
        parts.append(g.to_crs(4326))
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)


def build_coverage(tiles, simplify_deg=0.0005):
    """Dissolves tiles into coverage polygons per survey and overall."""
    cov = tiles.dissolve(by='survey_id', aggfunc={'point_count': 'sum', 'file_name': 'count'})
    cov = cov.rename(columns={'file_name': 'n_tiles'}).reset_index()
    total = gpd.GeoDataFrame(
        {'survey_id': ['all'], 'point_count': [tiles['point_count'].sum()],
         'n_tiles': [len(tiles)]},
        geometry=[shapely.union_all(cov.geometry.values)], crs=tiles.crs)
    cov = pd.concat([cov, total], ignore_index=True)
    cov['geometry'] = cov.geometry.simplify(simplify_deg)
    return gpd.GeoDataFrame(cov, crs=tiles.crs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=Path(__file__).with_name('config.yaml'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--limit', type=int, default=None, help='only the first N tiles (testing)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    laz_dir = Path(cfg['als']['laz_dir'])
    out_dir = Path(cfg['work_dir']) / 'als_inventory'
    part_dir = out_dir / 'parts'
    part_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(str(p) for p in laz_dir.glob('*.laz'))
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} LAZ files in {laz_dir}")

    chunks = [files[i:i + args.chunk] for i in range(0, len(files), args.chunk)]
    for k, chunk in enumerate(chunks):
        part = part_dir / f"part_{k:05d}_{len(chunk)}.parquet"
        if part.exists():
            continue
        print(f"chunk {k + 1}/{len(chunks)}")
        if args.workers > 1:
            headers = process_las_files_multiprocessing(chunk, num_processes=args.workers)
        else:
            headers = [h for h in map(process_las_file, chunk) if h is not None]
        failed = len(chunk) - len(headers)
        if failed:
            print(f"  {failed} headers could not be read")
        tile_records(headers).to_parquet(part)

    df = pd.concat([pd.read_parquet(p) for p in sorted(part_dir.glob('part_*.parquet'))],
                   ignore_index=True)
    no_crs = df['utm_epsg'].isna()
    if no_crs.any():
        print(f"Warning: {no_crs.sum()} tiles without a usable CRS are excluded")
        df = df[~no_crs]

    tiles = build_geometries(df)
    tiles.to_file(out_dir / 'als_tiles.gpkg', driver='GPKG')
    cov = build_coverage(tiles)
    cov.to_file(out_dir / 'als_coverage.gpkg', driver='GPKG')

    print(f"tiles: {len(tiles)}, surveys: {tiles['survey_id'].nunique()}, "
          f"UTM zones: {sorted(tiles['utm_epsg'].unique().tolist())}")
    print(f"median density: {tiles['density'].median():.1f} pts/m^2")
    print(f"wrote {out_dir / 'als_tiles.gpkg'} and {out_dir / 'als_coverage.gpkg'}")


if __name__ == '__main__':
    main()
