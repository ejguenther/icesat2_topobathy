"""
Finds and downloads ICESat-2 ATL03 / ATL08 Release 007 granules over the
Poland ALS coverage.

Steps (run in order; each is resumable):

    search          CMR search over the coverage bbox, keep granules whose
                    polygon intersects the coverage, pair ATL03 with ATL08.
                    -> <work_dir>/granules/granules.parquet
    download-atl08  download paired ATL08 files (small).
    tracks          per-beam ALS overlap from ATL08 land-segment positions.
                    -> <work_dir>/granules/beams.parquet
    estimate        print the ATL03 download size for granules that pass
                    the min_overlap_km filter.
    download-atl03  download each selected ATL03 file, subset it to the
                    coverage (all beams), delete the full file.

Usage:
    uv run python download_granules.py STEP [--config config.yaml] [--limit N]

Earthdata credentials come from ~/.netrc (urs.earthdata.nasa.gov).
"""
import argparse
import datetime as dt
import re
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon

from phoreal.config import load_config

GTS = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
NAME_RE = re.compile(r'(ATL0[38])_(\d{14})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})')


def parse_name(name):
    """ATL03_20181215070448_11860106_007_01.h5 -> dict of parts."""
    m = NAME_RE.search(name)
    if not m:
        return None
    prod, t, rgt, cycle, region, rel, rev = m.groups()
    return dict(product=prod, time=t, rgt=int(rgt), cycle=int(cycle),
                region=int(region), release=rel, revision=int(rev),
                key=f"{t}_{rgt}{cycle}{region}")


# --------------------------------------------------------------------------- #
# Paths / coverage
# --------------------------------------------------------------------------- #

def paths(cfg):
    work = Path(cfg['work_dir'])
    gran = Path(cfg['granule_dir'])
    p = dict(
        coverage=work / 'als_inventory' / 'als_coverage.gpkg',
        out=work / 'granules',
        atl08=gran / 'ATL08',
        atl03=gran / 'ATL03_subset',
    )
    for k in ('out', 'atl08', 'atl03'):
        p[k].mkdir(parents=True, exist_ok=True)
    return p


def load_coverage(path, buffer_deg=0.0):
    cov = gpd.read_file(path)
    geom = cov.loc[cov['survey_id'] == 'all', 'geometry'].iloc[0]
    if buffer_deg:
        geom = geom.buffer(buffer_deg)
    shapely.prepare(geom)
    return geom


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #

def granule_polygon(g):
    """Granule footprint polygon from CMR UMM-G (lon/lat)."""
    geo = g['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']
    polys = []
    for gp in geo.get('GPolygons', []):
        pts = gp['Boundary']['Points']
        polys.append(Polygon([(p['Longitude'], p['Latitude']) for p in pts]))
    return shapely.union_all(polys) if polys else None


def step_search(cfg, p):
    import earthaccess
    earthaccess.login(strategy='netrc')
    s = cfg['search']
    cov = load_coverage(p['coverage'])
    bbox = tuple(cov.bounds)
    end = s.get('end') or dt.date.today().isoformat()

    rows = []
    for short in s['short_names']:
        res = earthaccess.search_data(short_name=short, version=s['version'],
                                      bounding_box=bbox, temporal=(s['start'], end),
                                      count=-1)
        print(f"{short}: {len(res)} granules in bbox")
        for g in res:
            links = g.data_links()
            # GranuleUR can carry a random suffix; the file name is in the URL
            name = links[0].rsplit('/', 1)[-1] if links else g['umm']['GranuleUR']
            parts = parse_name(name)
            if parts is None:
                continue
            poly = granule_polygon(g)
            hits = poly is not None and poly.intersects(cov)
            rows.append(dict(name=name, **parts, size_mb=float(g.size() if callable(g.size) else g.size),
                             url=links[0] if links else None,
                             intersects=hits))
    df = pd.DataFrame(rows)
    # Highest revision per product/key
    df = df.sort_values('revision').groupby(['product', 'key'], as_index=False).last()

    a03 = df[df['product'] == 'ATL03']
    a08 = df[df['product'] == 'ATL08']
    paired = a03.merge(a08, on='key', how='outer', suffixes=('_03', '_08'))
    # Keep a pair if either product's footprint touches the ALS coverage
    hit = paired['intersects_03'].fillna(False).astype(bool) | paired['intersects_08'].fillna(False).astype(bool)
    paired = paired[hit].copy()

    # The bbox search can miss a partner whose CMR footprint differs (seen for
    # ATL08); look missing partners up by granule name.
    for sfx, short in (('_08', 'ATL08'), ('_03', 'ATL03')):
        for i in paired.index[paired['name' + sfx].isna()]:
            found = earthaccess.search_data(short_name=short, version=s['version'],
                                            granule_name=f"{short}_{paired.at[i, 'key']}_*",
                                            count=10)
            cands = []
            for g in found:
                links = g.data_links()
                name = links[0].rsplit('/', 1)[-1] if links else g['umm']['GranuleUR']
                parts = parse_name(name)
                if parts:
                    cands.append((parts['revision'], name, parts, g, links))
            if not cands:
                continue
            _, name, parts, g, links = max(cands, key=lambda c: c[0])
            for k, v in dict(name=name, **parts, size_mb=float(g.size() if callable(g.size) else g.size),
                             url=links[0] if links else None, intersects=False).items():
                paired.at[i, k + sfx] = v

    paired['status'] = np.where(paired['name_03'].isna(), 'no_atl03',
                         np.where(paired['name_08'].isna(), 'no_atl08', 'paired'))
    paired = paired.reset_index(drop=True)
    out = p['out'] / 'granules.parquet'
    paired.to_parquet(out)
    print(paired['status'].value_counts().to_string())
    ok = paired[paired['status'] == 'paired']
    print(f"paired: {len(ok)} granules, ATL03 {ok['size_mb_03'].sum() / 1e3:.1f} GB, "
          f"ATL08 {ok['size_mb_08'].sum() / 1e3:.1f} GB (full files)")
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# download helpers
# --------------------------------------------------------------------------- #

def _download(urls, dest):
    import earthaccess
    earthaccess.login(strategy='netrc')
    return earthaccess.download(urls, local_path=str(dest), threads=4, show_progress=False)


def step_download_atl08(cfg, p, limit=None):
    g = pd.read_parquet(p['out'] / 'granules.parquet')
    g = g[g['status'] == 'paired']
    todo = [u for n, u in zip(g['name_08'], g['url_08']) if not (p['atl08'] / n).exists()]
    if limit:
        todo = todo[:limit]
    print(f"downloading {len(todo)} ATL08 files to {p['atl08']}")
    for i in range(0, len(todo), 20):
        _download(todo[i:i + 20], p['atl08'])


# --------------------------------------------------------------------------- #
# tracks: per-beam overlap from ATL08
# --------------------------------------------------------------------------- #

def atl08_beam_overlap(atl08_file, cov):
    rows = []
    with h5py.File(atl08_file, 'r') as f:
        orient = int(f['orbit_info/sc_orient'][0])
        for gt in GTS:
            if f'{gt}/land_segments/latitude' not in f:
                continue
            lat = f[f'{gt}/land_segments/latitude'][()]
            lon = f[f'{gt}/land_segments/longitude'][()]
            inside = shapely.contains_xy(cov, lon, lat)
            spot = f[gt].attrs.get('atlas_spot_number', b'-1')
            btype = f[gt].attrs.get('atlas_beam_type', b'')
            rows.append(dict(
                gt=gt, n_seg=int(inside.sum()),
                overlap_km=float(inside.sum()) * 0.1,   # ATL08 land segments are 100 m
                spot=int(spot.decode() if isinstance(spot, bytes) else spot),
                beam_type=btype.decode() if isinstance(btype, bytes) else btype,
                sc_orient_first=orient,
            ))
    return rows


def step_tracks(cfg, p):
    g = pd.read_parquet(p['out'] / 'granules.parquet')
    g = g[g['status'] == 'paired']
    cov = load_coverage(p['coverage'])
    rows = []
    for key, name08 in zip(g['key'], g['name_08']):
        f = p['atl08'] / name08
        if not f.exists():
            continue
        try:
            for r in atl08_beam_overlap(f, cov):
                rows.append(dict(key=key, name_08=name08, **r))
        except OSError as e:
            print(f"cannot read {f}: {e}")
    beams = pd.DataFrame(rows)
    beams.to_parquet(p['out'] / 'beams.parquet')
    s = beams.groupby('key')['overlap_km'].max()
    print(f"{len(s)} granules read; beams with ALS overlap: {(beams['overlap_km'] > 0).sum()}")
    print(f"granules with a beam >= {cfg['search']['min_overlap_km']} km: "
          f"{(s >= cfg['search']['min_overlap_km']).sum()}")


def selected_granules(cfg, p):
    g = pd.read_parquet(p['out'] / 'granules.parquet')
    beams = pd.read_parquet(p['out'] / 'beams.parquet')
    best = beams.groupby('key')['overlap_km'].max()
    keep = best[best >= cfg['search']['min_overlap_km']].index
    return g[g['key'].isin(keep) & (g['status'] == 'paired')]


def step_estimate(cfg, p):
    sel = selected_granules(cfg, p)
    done = sum((p['atl03'] / n).exists() for n in sel['name_03'])
    print(f"{len(sel)} ATL03 granules selected ({done} already subset)")
    print(f"full-file download volume: {sel['size_mb_03'].sum() / 1e3:.1f} GB "
          f"(largest {sel['size_mb_03'].max() / 1e3:.1f} GB; only one full file "
          f"is on disk at a time)")


# --------------------------------------------------------------------------- #
# ATL03 subsetting
# --------------------------------------------------------------------------- #

def _copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _write_sliced(src_grp, dst_grp, length, mask):
    """Copy datasets of src_grp; those with first dim == length are masked."""
    for name, obj in src_grp.items():
        if isinstance(obj, h5py.Group):
            sub = dst_grp.create_group(name)
            _copy_attrs(obj, sub)
            _write_sliced(obj, sub, length, mask)
        elif obj.shape and obj.shape[0] == length:
            d = dst_grp.create_dataset(name, data=obj[()][mask], compression='gzip')
            _copy_attrs(obj, d)
        else:
            src_grp.copy(obj, dst_grp, name=name)


def subset_atl03(src_path, dst_path, cov):
    """
    Writes a copy of an ATL03 granule restricted to segments whose reference
    photon lies inside ``cov``. Photon arrays, geophys_corr and bckgrd_atlas are
    cut to match and ph_index_beg is rewritten, so readers and the ATL08 class
    mapping work unchanged. Beams with no segments inside are dropped.
    """
    kept = {}
    with h5py.File(src_path, 'r') as fi, h5py.File(dst_path, 'w') as fo:
        _copy_attrs(fi, fo)
        fo.attrs['phoreal_subset'] = 'segments with reference photon inside ALS coverage'
        for top in fi:
            if top not in GTS:
                fi.copy(fi[top], fo, name=top)
        for gt in GTS:
            if gt not in fi:
                continue
            geo = fi[f'{gt}/geolocation']
            lat = geo['reference_photon_lat'][()]
            lon = geo['reference_photon_lon'][()]
            seg_mask = shapely.contains_xy(cov, lon, lat)
            if not seg_mask.any():
                continue
            # Extend to whole runs of segments so empty (ph_index_beg == 0)
            # segments inside a run are kept too
            idx = np.flatnonzero(seg_mask)
            seg_mask[idx.min():idx.max() + 1] = True  # one contiguous block per beam

            beg = geo['ph_index_beg'][()]
            cnt = geo['segment_ph_cnt'][()]
            n_ph = fi[f'{gt}/heights/h_ph'].shape[0]
            ph_mask = np.zeros(n_ph, dtype=bool)
            for b, c in zip(beg[seg_mask], cnt[seg_mask]):
                if b > 0 and c > 0:
                    ph_mask[b - 1:b - 1 + c] = True

            g_out = fo.create_group(gt)
            _copy_attrs(fi[gt], g_out)
            n_seg = len(seg_mask)
            for sub in fi[gt]:
                src = fi[f'{gt}/{sub}']
                dst = g_out.create_group(sub)
                _copy_attrs(src, dst)
                if sub in ('geolocation', 'geophys_corr'):
                    _write_sliced(src, dst, n_seg, seg_mask)
                elif sub == 'heights':
                    _write_sliced(src, dst, n_ph, ph_mask)
                elif sub == 'bckgrd_atlas':
                    t = src['delta_time'][()]
                    seg_t = geo['delta_time'][()][seg_mask]
                    bg_mask = (t >= seg_t.min() - 1.0) & (t <= seg_t.max() + 1.0)
                    _write_sliced(src, dst, len(t), bg_mask)
                else:
                    for name in src:
                        src.copy(src[name], dst, name=name)

            # Rewrite ph_index_beg (1-based, 0 = empty) for the kept photons
            new_cnt = cnt[seg_mask]
            new_beg = np.where(new_cnt > 0, np.cumsum(new_cnt) - new_cnt + 1, 0)
            g_out['geolocation/ph_index_beg'][...] = new_beg
            kept[gt] = int(ph_mask.sum())
    return kept


def step_download_atl03(cfg, p, limit=None):
    sel = selected_granules(cfg, p)
    cov = load_coverage(p['coverage'], buffer_deg=cfg['search'].get('subset_buffer_deg', 0.02))
    todo = [(n, u) for n, u in zip(sel['name_03'], sel['url_03'])
            if not (p['atl03'] / n).exists()]
    if limit:
        todo = todo[:limit]
    log = p['out'] / 'atl03_subset_log.csv'
    print(f"{len(todo)} ATL03 granules to download and subset")
    for k, (name, url) in enumerate(todo):
        print(f"[{k + 1}/{len(todo)}] {name}")
        with tempfile.TemporaryDirectory(dir=p['atl03']) as tmp:
            files = _download([url], tmp)
            if not files:
                print("  download failed")
                continue
            part = p['atl03'] / (name + '.part')
            kept = subset_atl03(files[0], part, cov)
            part.rename(p['atl03'] / name)
        line = pd.DataFrame([dict(name=name, beams=len(kept), photons=sum(kept.values()))])
        line.to_csv(log, mode='a', header=not log.exists(), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['search', 'download-atl08', 'tracks', 'estimate', 'download-atl03'])
    ap.add_argument('--config', default=Path(__file__).with_name('config.yaml'))
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = paths(cfg)
    {
        'search': lambda: step_search(cfg, p),
        'download-atl08': lambda: step_download_atl08(cfg, p, args.limit),
        'tracks': lambda: step_tracks(cfg, p),
        'estimate': lambda: step_estimate(cfg, p),
        'download-atl03': lambda: step_download_atl03(cfg, p, args.limit),
    }[args.step]()


if __name__ == '__main__':
    main()
