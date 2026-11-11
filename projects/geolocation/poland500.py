"""
National multi-granule acquisition for the Poland geolocation run.

Steps (each resumable):
    search   icepyx ATL03 + ATL08 R007 search over the ALS coverage bbox;
             keep pairs whose CMR footprint touches the coverage
             -> <work>/granules/candidates.parquet
    atl08    whole-granule ATL08 download (icepyx links, earthaccess transfer)
             -> <granule_dir>/<atl08_subdir>/
    screen   per-beam and per-survey ATL08 ground km on the ALS
             -> <work>/granules/screen.parquet, screen_beams.parquet
    select   stratified pick of N granules (orientation x asc/desc x
             day/night x year, spread over surveys) in interleaved order
             -> <work>/granules/granules.parquet, beams.parquet
    atl03    one icepyx/Harmony subset job per selected granule, run
             concurrently; local-subset fallback
             -> <granule_dir>/<atl03_subdir>/

Usage:
    uv run python poland500.py STEP --config local/config_poland500.yaml
"""
import argparse
import json
import re
import shutil
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from phoreal.config import load_config
from download_granules import parse_name, atl08_beam_overlap, subset_atl03, GTS
from pilot_icepyx import cmr_polygon, solar_elevation, harmony_download


def dirs_from(cfg):
    work = Path(cfg['work_dir']); gran = Path(cfg['granule_dir'])
    d = dict(work=work, inv=work / 'als_inventory', gr=work / 'granules', logs=work / 'logs',
             atl03=gran / cfg.get('atl03_subdir', 'atl03'),
             atl08=gran / cfg.get('atl08_subdir', 'atl08'),
             tmp=gran / 'tmp')
    for k in ('gr', 'logs', 'atl03', 'atl08', 'tmp'):
        d[k].mkdir(parents=True, exist_ok=True)
    return d


def coverage(d):
    cov = gpd.read_file(d['inv'] / 'als_coverage.gpkg')
    allg = cov.loc[cov['survey_id'] == 'all', 'geometry'].iloc[0]
    shapely.prepare(allg)
    surveys = cov[cov['survey_id'] != 'all'].reset_index(drop=True)
    return allg, surveys


def granule_name(name):
    m = re.search(r'ATL0[38]_\d{14}_\d{8}_\d{3}_\d{2}', name)
    return m.group(0) if m else None


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #

def step_search(cfg, d):
    import icepyx as ipx
    cov, surveys = coverage(d)
    bbox = [round(v, 3) for v in cov.bounds]
    # cheap stand-in for the coverage when computing each granule's subset bbox
    boxes = shapely.union_all([box(*b) for b in surveys.geometry.bounds.values])
    shapely.prepare(boxes)
    s = cfg['search']
    rows = []
    for prod in ('ATL03', 'ATL08'):
        q = ipx.Query(prod, bbox, [s['start'], s['end']], version=s['version'])
        q.avail_granules()
        for e in q.granules.avail:
            name = e['producer_granule_id']
            parts = parse_name(name)
            if parts is None:
                continue
            poly = cmr_polygon(e)
            hit = poly.intersects(cov)
            inter = poly.intersection(boxes) if hit else shapely.Polygon()
            url = next((l['href'] for l in e['links']
                        if l.get('rel', '').endswith('data#') and l['href'].endswith('.h5')), None)
            rows.append(dict(name=name, **parts, size_mb=float(e.get('granule_size', np.nan)),
                             time_start=e['time_start'], url=url,
                             intersects=bool(hit),
                             sub_bbox=json.dumps([round(v, 4) for v in inter.bounds]) if not inter.is_empty else None))
        print(f"{prod}: {len(q.granules.avail)} in bbox")
    df = pd.DataFrame(rows)
    df = df.sort_values('revision').groupby(['product', 'key'], as_index=False).last()
    a03, a08 = df[df['product'] == 'ATL03'], df[df['product'] == 'ATL08']
    pairs = a03.merge(a08[['key', 'name', 'revision', 'size_mb', 'url', 'intersects']],
                      on='key', how='inner', suffixes=('_03', '_08'))
    pairs = pairs[pairs['intersects_03'] | pairs['intersects_08']]
    pairs = pairs[pairs['sub_bbox'].notna()].reset_index(drop=True)
    pairs.to_parquet(d['gr'] / 'candidates.parquet')
    print(f"{len(pairs)} ATL03/ATL08 pairs touch the coverage; ATL08 {pairs['size_mb_08'].sum() / 1e3:.0f} GB")


# --------------------------------------------------------------------------- #
# atl08 download
# --------------------------------------------------------------------------- #

def step_atl08(cfg, d):
    import earthaccess
    earthaccess.login(strategy='netrc')
    c = pd.read_parquet(d['gr'] / 'candidates.parquet')
    todo = [u for n, u in zip(c['name_08'], c['url_08']) if u and not (d['atl08'] / n).exists()]
    print(f"{len(todo)} ATL08 files to download to {d['atl08']}", flush=True)
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        # earthaccess writes directly to the final name; partial files from an
        # interrupted run are re-fetched because they fail to open in screen
        earthaccess.download(batch, local_path=str(d['atl08']),
                             threads=cfg['run500'].get('atl08_threads', 8), show_progress=False)
        print(f"  {min(i + 40, len(todo))}/{len(todo)}", flush=True)


# --------------------------------------------------------------------------- #
# screen
# --------------------------------------------------------------------------- #

def _screen_one(path, cov_wkb, surveys_wkb, survey_ids):
    cov = shapely.from_wkb(cov_wkb); shapely.prepare(cov)
    survey_geoms = [shapely.from_wkb(w) for w in surveys_wkb]
    tree = shapely.STRtree(survey_geoms)
    try:
        beams = atl08_beam_overlap(path, cov)
        per_survey = {}
        with h5py.File(path, 'r') as f:
            for gt in GTS:
                if f'{gt}/land_segments/latitude' not in f:
                    continue
                if f[gt].attrs.get('atlas_beam_type', b'') not in (b'strong', 'strong'):
                    continue
                lat = f[f'{gt}/land_segments/latitude'][()]
                lon = f[f'{gt}/land_segments/longitude'][()]
                pts = shapely.points(lon, lat)
                idx_pt, idx_poly = tree.query(pts, predicate='within')
                for sid, n in zip(*np.unique(np.asarray(survey_ids)[idx_poly], return_counts=True)):
                    per_survey[sid] = per_survey.get(sid, 0) + int(n) * 0.1
        return dict(ok=True, beams=beams, per_survey=per_survey)
    except Exception as e:
        return dict(ok=False, error=repr(e)[:300])


def step_screen(cfg, d):
    c = pd.read_parquet(d['gr'] / 'candidates.parquet')
    cov, surveys = coverage(d)
    args = (shapely.to_wkb(cov), [shapely.to_wkb(g) for g in surveys.geometry], list(surveys['survey_id']))
    rows, brows = [], []
    todo = [(k, n) for k, n in zip(c['key'], c['name_08']) if (d['atl08'] / n).exists()]
    print(f"screening {len(todo)} ATL08 files", flush=True)
    rb = cfg['run500']
    with ProcessPoolExecutor(20) as ex:
        futs = {ex.submit(_screen_one, str(d['atl08'] / n), *args): (k, n) for k, n in todo}
        for f in as_completed(futs):
            k, n = futs[f]; r = f.result()
            if not r['ok']:
                rows.append(dict(key=k, name_08=n, ok=False, error=r['error'])); continue
            strong = [b for b in r['beams'] if b['beam_type'] == 'strong']
            rows.append(dict(key=k, name_08=n, ok=True,
                             n_strong_ok=sum(b['overlap_km'] >= rb['min_beam_km'] for b in strong),
                             ground_km=sum(b['overlap_km'] for b in r['beams']),
                             sc_orient=r['beams'][0]['sc_orient_first'] if r['beams'] else -1,
                             surveys=json.dumps(r['per_survey'])))
            brows += [dict(key=k, name_08=n, **b) for b in r['beams']]
    s = pd.DataFrame(rows); s.to_parquet(d['gr'] / 'screen.parquet')
    pd.DataFrame(brows).to_parquet(d['gr'] / 'screen_beams.parquet')
    ok = s[s['ok'] & (s['n_strong_ok'] >= rb['min_strong_beams'])]
    print(f"read errors: {(~s['ok']).sum()}; usable (>= {rb['min_strong_beams']} strong beams "
          f">= {rb['min_beam_km']} km): {len(ok)}")


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #

def step_select(cfg, d):
    rb = cfg['run500']
    c = pd.read_parquet(d['gr'] / 'candidates.parquet')
    s = pd.read_parquet(d['gr'] / 'screen.parquet')
    s = s[s['ok'] & (s['n_strong_ok'] >= rb['min_strong_beams'])]
    c = c.merge(s.drop(columns='name_08'), on='key')
    c['t'] = pd.to_datetime(c['time_start'])
    c['year'] = c['t'].dt.year
    c['ascending'] = c['region'] <= 4
    lon_lat = c['sub_bbox'].map(lambda b: ((json.loads(b)[0] + json.loads(b)[2]) / 2,
                                           (json.loads(b)[1] + json.loads(b)[3]) / 2))
    c['solar_elev_approx'] = [solar_elevation(t.to_pydatetime(), lat, lon)
                              for t, (lon, lat) in zip(c['t'], lon_lat)]
    c['day'] = c['solar_elev_approx'] > 0
    c['stratum'] = list(zip(c['sc_orient'], c['ascending'], c['day']))
    c['survey_set'] = c['surveys'].map(lambda j: set(json.loads(j)))

    n = min(rb['n_granules'], len(c))
    strata_cnt, year_cnt, survey_cnt = {}, {}, {}
    chosen, remaining = [], set(c.index)
    km = c['ground_km'] / c['ground_km'].max()
    while len(chosen) < n and remaining:
        idx = np.fromiter(remaining, int)
        sub = c.loc[idx]
        score = (-sub['stratum'].map(lambda k: strata_cnt.get(k, 0)) * 3.0
                 - sub['year'].map(lambda y: year_cnt.get(y, 0)) * 0.5
                 + sub['survey_set'].map(lambda ss: sum(1.0 / (1 + survey_cnt.get(x, 0)) for x in ss)) * 0.3
                 + km.loc[idx] * 0.5)
        best = int(score.idxmax())
        chosen.append(best); remaining.discard(best)
        r = c.loc[best]
        strata_cnt[r['stratum']] = strata_cnt.get(r['stratum'], 0) + 1
        year_cnt[r['year']] = year_cnt.get(r['year'], 0) + 1
        for x in r['survey_set']:
            survey_cnt[x] = survey_cnt.get(x, 0) + 1
    sel = c.loc[chosen].copy()
    sel['order'] = np.arange(len(sel))
    sel = sel.drop(columns=['t', 'survey_set', 'stratum'])
    sel['status'] = 'selected'
    sel.to_parquet(d['gr'] / 'granules.parquet')

    sb = pd.read_parquet(d['gr'] / 'screen_beams.parquet')
    sb[sb['key'].isin(sel['key'])].to_parquet(d['gr'] / 'beams.parquet')

    print(f"selected {len(sel)} of {len(c)} usable granules; surveys covered: "
          f"{len(set().union(*[set(json.loads(j)) for j in sel['surveys']]))}")
    print(pd.crosstab([sel['sc_orient'], sel['ascending']], sel['day']).to_string())
    print(sel['year'].value_counts().sort_index().to_string())


# --------------------------------------------------------------------------- #
# atl03 (Harmony subsets, concurrent)
# --------------------------------------------------------------------------- #

def _fetch_atl03(row, d):
    bbox = json.loads(row['sub_bbox'])
    bbox = [bbox[0] - 0.02, bbox[1] - 0.02, bbox[2] + 0.02, bbox[3] + 0.02]
    last = None
    for attempt in range(3):
        try:
            return harmony_download('ATL03', row, bbox, d['atl03'], d['tmp']), 'harmony'
        except Exception as e:
            last = e
            time.sleep(20)
    # fallback: whole granule + verified local subset
    import earthaccess
    earthaccess.login(strategy='netrc')
    with tempfile.TemporaryDirectory(dir=d['tmp']) as tmp:
        files = earthaccess.download([row['url_03']], local_path=tmp, threads=1, show_progress=False)
        if not files:
            raise RuntimeError(f"harmony failed ({last}); full download failed too")
        cov = box(*bbox); shapely.prepare(cov)
        out = d['atl03'] / ('local_' + Path(files[0]).name)
        part = out.with_suffix('.part')
        subset_atl03(files[0], part, cov)
        part.rename(out)
    return out.name, 'local_fallback'


def existing_atl03(d):
    return {granule_name(p.name): p.name for p in d['atl03'].glob('*.h5') if granule_name(p.name)}


def step_atl03(cfg, d):
    g = pd.read_parquet(d['gr'] / 'granules.parquet').sort_values('order')
    have = existing_atl03(d)
    todo = [r for _, r in g.iterrows() if r['name_03'].replace('.h5', '') not in have]
    log = d['gr'] / 'atl03_download_log.csv'
    print(f"{len(todo)} ATL03 subsets to fetch ({len(g) - len(todo)} present)", flush=True)
    with ThreadPoolExecutor(cfg['run500'].get('harmony_concurrency', 8)) as ex:
        futs = {ex.submit(_fetch_atl03, r, d): r for r in todo}
        for i, f in enumerate(as_completed(futs)):
            r = futs[f]
            try:
                name, how = f.result(); err = ''
            except Exception as e:
                name, how, err = '', 'failed', repr(e)[:300]
            pd.DataFrame([dict(key=r['key'], file=name, how=how, error=err,
                               t=time.strftime('%H:%M:%S'))]).to_csv(
                log, mode='a', header=not log.exists(), index=False)
            print(f"[{i + 1}/{len(todo)}] {r['key']} {how} {err}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['search', 'atl08', 'screen', 'select', 'atl03'])
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = dirs_from(cfg)
    {'search': step_search, 'atl08': step_atl08, 'screen': step_screen,
     'select': step_select, 'atl03': step_atl03}[args.step](cfg, d)


if __name__ == '__main__':
    main()
