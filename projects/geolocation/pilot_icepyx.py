"""
Pilot: pick N ATL03/ATL08 R007 granule pairs over small ALS AOIs, download
them with icepyx (Harmony spatial subsetting), and lay them out the way
run_offsets.py expects.

Steps (run in order):
    tiles     ALS tile index for the AOIs, from inventory_als.py parts
              -> <work_dir>/als_inventory/{als_tiles,als_coverage}.gpkg
    screen    one icepyx/Harmony ATL08 subset order per AOI (all dates); the
              per-beam ground-segment overlap shows which granules actually
              have data (CMR footprints include cloudy / empty passes)
              -> <granule_dir>/ATL08, <work_dir>/granules/screen.parquet
    search    icepyx ATL03 search per AOI, paired with screened ATL08, then a
              stratified pick of N granules (ascending/descending x day/night,
              spread in time) among those with ground data
              -> <work_dir>/granules/candidates.parquet, granules.parquet
    download  one Harmony ATL03 subset job per selected granule
    beams     per-beam ALS overlap from the subset ATL08
              -> <work_dir>/granules/beams.parquet
    verify    download one full ATL03 granule, subset it locally with the
              verified subsetter, and compare photons, segment metadata and
              ATL08 classes with the Harmony subset

Then: uv run python run_offsets.py --config <pilot config>

Config (``pilot`` section of the YAML):
    pilot:
      inventory_parts: /path/to/als_inventory/parts   # from inventory_als.py
      n_granules: 10
      date_range: ["2018-10-13", "2026-11-10"]
      aois:
        name: [west, south, east, north]
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
from shapely.geometry import box, Polygon

from phoreal.config import load_config
from download_granules import (parse_name, atl08_beam_overlap, subset_atl03,
                               load_coverage, GTS)
from inventory_als import build_geometries, build_coverage


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def dirs_from(cfg):
    work = Path(cfg['work_dir']); gran = Path(cfg['granule_dir'])
    d = dict(work=work, inv=work / 'als_inventory', gr=work / 'granules',
             atl03=gran / 'ATL03_subset', atl08=gran / 'ATL08', tmp=gran / 'tmp')
    for k in ('inv', 'gr', 'atl03', 'atl08', 'tmp'):
        d[k].mkdir(parents=True, exist_ok=True)
    return d


def solar_elevation(when, lat, lon):
    """Approximate solar elevation (deg); good to ~1 deg, enough for day/night."""
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60 + when.second / 3600
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return float(90 - np.degrees(np.arccos(np.clip(cosz, -1, 1))))


def cmr_polygon(entry):
    """icepyx/CMR 'polygons' (lat lon pairs) -> shapely Polygon (lon, lat)."""
    vals = [float(v) for v in entry['polygons'][0][0].split()]
    return Polygon(list(zip(vals[1::2], vals[0::2])))


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #

def step_tiles(cfg, d):
    parts = sorted(Path(cfg['pilot']['inventory_parts']).glob('part_*.parquet'))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    tiles = build_geometries(df[df['utm_epsg'].notna()])
    aoi = shapely.union_all([box(*b) for b in cfg['pilot']['aois'].values()]).buffer(0.01)
    tiles = tiles[tiles.intersects(aoi)].reset_index(drop=True)
    tiles.to_file(d['inv'] / 'als_tiles.gpkg', driver='GPKG')
    build_coverage(tiles).to_file(d['inv'] / 'als_coverage.gpkg', driver='GPKG')
    print(f"{len(tiles)} tiles in AOIs from {len(parts)} inventory parts; surveys: "
          f"{tiles['survey_id'].value_counts().to_dict()}")


def step_screen(cfg, d):
    import icepyx as ipx
    pc = cfg['pilot']
    cov = load_coverage(d['inv'] / 'als_coverage.gpkg')
    rows = []
    for name, bbox in pc['aois'].items():
        q = ipx.Query('ATL08', list(bbox), list(pc['date_range']), version='007')
        print(f"{name}: ordering {len(q.avail_granules(ids=True)[0])} ATL08 subsets")
        with tempfile.TemporaryDirectory(dir=d['tmp']) as tmp:
            files = q.download_granules(tmp, subset=True) or []
            for f in files:
                f = Path(f)
                if f.suffix != '.h5':
                    continue
                shutil.move(str(f), d['atl08'] / f.name)
                parts = parse_name(f.name)
                try:
                    beams = atl08_beam_overlap(d['atl08'] / f.name, cov)
                except (OSError, KeyError) as e:
                    print(f"   cannot read {f.name}: {e}")
                    beams = []
                strong = [b for b in beams if b['beam_type'] == 'strong']
                rows.append(dict(aoi=name, key=parts['key'], file_08=f.name,
                                 n_beams=len(beams),
                                 n_strong_ge_min=sum(b['overlap_km'] >= pc.get('min_beam_km', 2.0)
                                                     for b in strong),
                                 ground_km_total=sum(b['overlap_km'] for b in beams),
                                 sc_orient=beams[0]['sc_orient_first'] if beams else -1))
    s = pd.DataFrame(rows)
    s.to_parquet(d['gr'] / 'screen.parquet')
    print(s.groupby('aoi')['n_strong_ge_min'].apply(lambda x: (x >= 2).sum()).rename(
        'granules with >= 2 strong beams on ALS').to_string())


def search_aoi(name, bbox, date_range):
    import icepyx as ipx
    rows = []
    for prod in ('ATL03', 'ATL08'):
        q = ipx.Query(prod, list(bbox), list(date_range), version='007')
        q.avail_granules()
        for e in q.granules.avail:
            gname = e['producer_granule_id']
            parts = parse_name(gname)
            if parts is None:
                continue
            poly = cmr_polygon(e)
            rows.append(dict(aoi=name, name=gname, **parts,
                             size_mb=float(e.get('granule_size', np.nan)),
                             time_start=e['time_start'],
                             overlap_frac=poly.intersection(box(*bbox)).area / box(*bbox).area))
    df = pd.DataFrame(rows)
    df = df.sort_values('revision').groupby(['product', 'key'], as_index=False).last()
    a03, a08 = df[df['product'] == 'ATL03'], df[df['product'] == 'ATL08']
    pairs = a03.merge(a08[['key', 'name', 'revision']], on='key', suffixes=('_03', '_08'))
    return pairs.rename(columns={'name_03': 'name_03', 'name_08': 'name_08'})


def stratified_pick(c, n, seed=0):
    """
    Picks n granules spread evenly in time over the whole search period,
    cycling through the (ascending/descending x day/night) strata: pick j
    takes the granule in stratum j % 4 closest to the j-th time target.
    """
    c = c.sort_values('time').copy()
    t = c['t'].astype('int64')
    t0, t1 = t.min(), t.max()
    targets = t0 + (np.arange(n) + 0.5) / n * (t1 - t0)
    keys = sorted(c.groupby(['ascending', 'day']).groups)
    keys = keys[seed % len(keys):] + keys[:seed % len(keys)]
    remaining = set(c.index)
    picks = []
    for j, target in enumerate(targets):
        stratum = keys[j % len(keys)]
        pool = [i for i in remaining if (c.at[i, 'ascending'], c.at[i, 'day']) == stratum]
        if not pool:
            pool = list(remaining)
        if not pool:
            break
        best = min(pool, key=lambda i: abs(t[i] - target))
        picks.append(best)
        remaining.discard(best)
    return c.loc[picks]


def step_search(cfg, d):
    pc = cfg['pilot']
    allc = []
    for name, bbox in pc['aois'].items():
        c = search_aoi(name, bbox, pc['date_range'])
        lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        c['t'] = pd.to_datetime(c['time_start'])
        c['ascending'] = c['region'] <= 4          # N hemisphere: regions 1-4 ascending
        c['solar_elev_approx'] = [solar_elevation(t.to_pydatetime(), lat, lon) for t in c['t']]
        c['day'] = c['solar_elev_approx'] > 0
        allc.append(c)
        print(f"{name}: {len(c)} ATL03/ATL08 pairs")
    cand = pd.concat(allc, ignore_index=True)
    screen = pd.read_parquet(d['gr'] / 'screen.parquet')
    cand = cand.merge(screen, on=['aoi', 'key'], how='inner')
    cand = cand[cand['n_strong_ge_min'] >= 2]
    print(f"{len(cand)} pairs with >= 2 strong beams of ground data on the ALS")
    cand.drop(columns='t').to_parquet(d['gr'] / 'candidates.parquet')

    n = pc['n_granules']
    per = {name: n // len(pc['aois']) + (i < n % len(pc['aois']))
           for i, name in enumerate(pc['aois'])}
    sel = pd.concat([stratified_pick(cand[cand['aoi'] == a], k, seed=i)
                     for i, (a, k) in enumerate(per.items())])
    sel = sel.drop(columns='t').assign(status='selected')
    sel.to_parquet(d['gr'] / 'granules.parquet')
    print(sel[['aoi', 'name_03', 'ascending', 'day', 'solar_elev_approx', 'sc_orient',
               'ground_km_total']]
          .to_string(index=False))


def harmony_download(prod, row, bbox, dest, tmp_root):
    """One icepyx/Harmony subset job for a single granule (rgt + cycle + day)."""
    import icepyx as ipx
    day = f"{row['time'][:4]}-{row['time'][4:6]}-{row['time'][6:8]}"
    q = ipx.Query(prod, list(bbox), [day, day], version='007',
                  tracks=[f"{row['rgt']:04d}"], cycles=[f"{row['cycle']:02d}"])
    ids = q.avail_granules(ids=True)[0]
    if len(ids) != 1:
        raise RuntimeError(f"{prod} {row['key']}: expected 1 granule, query found {ids}")
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        files = q.download_granules(tmp, subset=True) or []
        files = [Path(f) for f in files if str(f).endswith(('.h5', '.nc4'))]
        if len(files) != 1:
            raise RuntimeError(f"{prod} {row['key']}: Harmony returned {files}")
        out = dest / files[0].name
        shutil.move(str(files[0]), out)
    return out.name


def step_download(cfg, d):
    g = pd.read_parquet(d['gr'] / 'granules.parquet')
    aois = cfg['pilot']['aois']
    if 'file_03' not in g:
        g['file_03'] = None
    g = g.reset_index(drop=True)
    for i, row in g.iterrows():
        for prod, col, dest in (('ATL03', 'file_03', d['atl03']),):
            if isinstance(row[col], str) and (dest / row[col]).exists():
                continue
            print(f"[{i + 1}/{len(g)}] {prod} {row['key']} ({row['aoi']})")
            try:
                g.at[i, col] = harmony_download(prod, row, aois[row['aoi']], dest, d['tmp'])
            except Exception as e:
                print(f"   failed: {e}")
            g.to_parquet(d['gr'] / 'granules.parquet')
    # run_offsets.py finds files by name_03 / name_08
    g['name_03'] = g['file_03'].fillna(g['name_03'])
    g['name_08'] = g['file_08'].fillna(g['name_08'])
    g['status'] = np.where(g['file_03'].notna() & g['file_08'].notna(), 'paired', 'download_failed')
    g.to_parquet(d['gr'] / 'granules.parquet')
    print(g['status'].value_counts().to_string())


def step_beams(cfg, d):
    g = pd.read_parquet(d['gr'] / 'granules.parquet')
    cov = load_coverage(d['inv'] / 'als_coverage.gpkg')
    rows = []
    for key, n08 in zip(g['key'], g['name_08']):
        f = d['atl08'] / n08
        if f.exists():
            rows += [dict(key=key, name_08=n08, **r) for r in atl08_beam_overlap(f, cov)]
    b = pd.DataFrame(rows)
    b.to_parquet(d['gr'] / 'beams.parquet')
    print(b.pivot_table(index='key', columns='gt', values='overlap_km').round(1).to_string())


# --------------------------------------------------------------------------- #
# verify Harmony subset against a local subset of the full granule
# --------------------------------------------------------------------------- #

def compare_subsets(harmony_03, local_03, atl08):
    from phoreal.io.readers import read_photon_dataframe, read_segment_geolocation
    report = []
    with h5py.File(harmony_03, 'r') as fh:
        gts = [gt for gt in GTS if f'{gt}/heights/h_ph' in fh]
        missing = [p for p in ('orbit_info/sc_orient', 'orbit_info/sc_orient_time',
                               'ancillary_data/release',
                               'ancillary_data/atlas_engineering/ph_uncorrelated_error')
                   if p not in fh]
    report.append(f"groups missing from Harmony file: {missing or 'none'}")
    for gt in gts:
        with h5py.File(harmony_03, 'r') as fh:
            beg = fh[f'{gt}/geolocation/ph_index_beg'][()]
            cnt = fh[f'{gt}/geolocation/segment_ph_cnt'][()]
            n_ph = fh[f'{gt}/heights/h_ph'].shape[0]
        nz = beg > 0
        consistent = (cnt.sum() == n_ph and np.all(np.diff(beg[nz]) == cnt[nz][:-1])
                      and (beg[nz][0] == 1 if nz.any() else True))
        h = read_photon_dataframe(str(harmony_03), gt, atl08_file=str(atl08))
        l = read_photon_dataframe(str(local_03), gt, atl08_file=str(atl08))
        m = h.merge(l, on=['delta_time', 'h_ph'], suffixes=('_h', '_l'))
        same_class = np.mean(m['atl08_class_h'] == m['atl08_class_l']) if len(m) else np.nan
        same_at = np.allclose(m['alongtrack_h'], m['alongtrack_l']) if len(m) else False
        sh = read_segment_geolocation(harmony_03, gt)
        report.append(
            f"{gt}: harmony {len(h)} ph, local {len(l)} ph, matched {len(m)}; "
            f"ph_index_beg self-consistent={consistent}; "
            f"atl08_class agreement={same_class:.4f}; alongtrack equal={same_at}; "
            f"ground ph h/l={(h.atl08_class == 1).sum()}/{(l.atl08_class == 1).sum()}; "
            f"segment fields missing={sh.attrs['missing_fields']}")
    return report


def step_verify(cfg, d):
    import earthaccess
    g = pd.read_parquet(d['gr'] / 'granules.parquet')
    g = g[g['status'] == 'paired']
    row = g.iloc[0]
    bbox = cfg['pilot']['aois'][row['aoi']]
    earthaccess.login(strategy='netrc')
    res = earthaccess.search_data(short_name='ATL03', version='007',
                                  granule_name=re.search(r'ATL03_\d{14}_\d{8}_\d{3}_\d{2}',
                                                         row['name_03']).group(0) + '*')
    url = [u for u in res[0].data_links() if u.endswith('.h5')][0]
    full_dir = d['tmp'] / 'verify'; full_dir.mkdir(exist_ok=True)
    full = Path(earthaccess.download([url], local_path=str(full_dir))[0])
    cov = box(*bbox); shapely.prepare(cov)
    local = full_dir / ('local_' + full.name)
    subset_atl03(full, local, cov)
    lines = [f"verify granule {row['name_03']} (AOI {row['aoi']})"]
    lines += compare_subsets(d['atl03'] / row['name_03'], local, d['atl08'] / row['name_08'])
    (d['gr'] / 'harmony_verify.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    full.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['tiles', 'screen', 'search', 'download', 'beams', 'verify'])
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = dirs_from(cfg)
    {'tiles': step_tiles, 'screen': step_screen, 'search': step_search, 'download': step_download,
     'beams': step_beams, 'verify': step_verify}[args.step](cfg, d)


if __name__ == '__main__':
    main()
