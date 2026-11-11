"""
Estimates ICESat-2 horizontal geolocation offsets against the Poland ALS for
every granule with ALS overlap: all beams of a granule in one job, several
along-track window types per beam.

One output record per physical spot x pass x window, including rejected
windows (``accepted`` = False plus ``flags``). Offsets are stored both as the
correction applied to ICESat-2 (``corr_*``) and as the geolocation error
(``err_*`` = -correction), in along/cross-track and east/north.

ALS surface (``offsets.surface``):
    laz  ground points from the LAZ tiles (create_als_swath), per beam
    dtm  1 m DTM GeoTIFFs sampled bilinearly at the at/xt grid nodes
         (create_dtm_grids), tiles read once per granule

Window types (``offsets.window_lengths_m``):
    "segment"  contiguous runs of photons on ALS (split at gaps >
               segment_gap_m, capped at segment_max_m); window_length_m = -1
    <metres>   fixed-length windows
    null       the whole beam as one window (pilot behaviour); also -1

Inputs:
    <work_dir>/als_inventory/als_tiles.gpkg
    <work_dir>/granules/{granules,beams}.parquet
    <granule_dir>/<atl03_subdir>/*.h5, <granule_dir>/<atl08_subdir>/*.h5

Outputs (in <work_dir>/offsets/):
    parts/<granule>.parquet           per-granule records (resume = skip existing)
    cost_surfaces/<granule>_<gt>.npz  brute-force cost grid per window
    refit_cache/<granule>_<gt>.npz    gridded ALS surface (at, xt, z) for refits
    refit_cache/<granule>_<gt>.parquet  ground photons on ALS (at, xt, h_ph, delta_time)
    offsets.parquet                   all parts combined

Usage:
    uv run python run_offsets.py --config CFG [--limit N] [--jobs N]
        [--only SUBSTR] [--collect] [--retry-errors] [--watch]
"""
import argparse
import json
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from phoreal.als.swath import create_als_swath, transform_als_swath, create_dtm_grids
from phoreal.config import load_config
from phoreal.geolocation import (
    create_interpolator, estimate_offset, track_heading, atxt_to_en,
    terrain_summary, beam_vertical_angle, terrain_incidence_angle,
)
from phoreal.io.readers import read_photon_dataframe, read_segment_geolocation

GPS_EPOCH = pd.Timestamp('2018-01-01')   # ATLAS SDP epoch (GPS time)
ORIENT = {0: 'backward', 1: 'forward', 2: 'transition', -1: 'unknown'}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def circ_mean_deg(a):
    a = np.radians(np.asarray(a, float)); a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


def circ_std_deg(a):
    a = np.radians(np.asarray(a, float)); a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    R = np.hypot(np.sin(a).mean(), np.cos(a).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))))


def summarize_segments(seg):
    """Window summary of segment-rate metadata."""
    def q(col, f):
        v = seg[col].to_numpy(float)
        v = v[np.isfinite(v)]
        return float(f(v)) if v.size else np.nan

    orients = seg['sc_orient'].to_numpy()
    out = {
        'n_segments': len(seg),
        'delta_time_start': q('delta_time', np.min),
        'delta_time_end': q('delta_time', np.max),
        'lat_center': q('reference_photon_lat', np.median),
        'lon_center': q('reference_photon_lon', np.median),
        'sc_orient': ORIENT.get(int(np.bincount(orients + 1).argmax()) - 1, 'unknown'),
        'sc_orient_mixed': bool(len(np.unique(orients)) > 1),
        'podppd_nonzero_frac': float(np.mean(seg['podppd_flag'].to_numpy() != 0)),
        'podppd_max': int(np.nanmax(seg['podppd_flag'])) if len(seg) else -1,
        'sigma_horiz_is_default': bool(seg['sigma_horiz_is_default'].all()),
    }
    for col in ['ref_elev', 'beta_angle', 'roll', 'pitch', 'yaw', 'altitude_sc',
                'sigma_along', 'sigma_across', 'sigma_h', 'solar_elevation',
                'bckgrd_rate', 'tx_pulse_energy', 'full_sat_fract', 'near_sat_fract']:
        out[f'{col}_med'] = q(col, np.median)
        out[f'{col}_min'] = q(col, np.min)
        out[f'{col}_max'] = q(col, np.max)
    for col, in_rad in [('ref_azimuth', True), ('orbit_angle', False), ('solar_azimuth', False)]:
        vals = np.degrees(seg[col]) if in_rad else seg[col]
        out[f'{col}_deg_cmean'] = circ_mean_deg(vals)
        out[f'{col}_deg_cstd'] = circ_std_deg(vals)
    return out


def make_windows(at, length):
    """Window ids for along-track positions; length None -> one window."""
    if length is None:
        return np.zeros(len(at), dtype=int)
    return ((at - at.min()) // length).astype(int)


def window_flags(rec, cfg_off):
    flags = []
    if rec['n_photons'] < cfg_off['min_window_photons']:
        flags.append('too_few_photons')
    if rec.get('boundary_hit'):
        flags.append('boundary_hit')
    if np.isfinite(rec.get('valid_frac_opt', np.nan)) and rec['valid_frac_opt'] < cfg_off['min_valid_frac']:
        flags.append('low_overlap')
    if not np.isfinite(rec.get('cost_opt', np.nan)):
        flags.append('no_solution')
    if rec.get('podppd_nonzero_frac', 0) > 0:
        flags.append('podppd_degraded')
    if rec.get('sc_orient_mixed') or rec.get('sc_orient') in ('transition', 'unknown'):
        flags.append('orientation_transition')
    sig = max(rec.get('boot_sigma_at', np.nan), rec.get('boot_sigma_xt', np.nan))
    if np.isfinite(sig) and sig > cfg_off.get('max_sigma_m', np.inf):
        flags.append('weak_constraint')
    if rec.get('boot_edge_frac', 0) > 0.2:
        flags.append('weak_constraint_bootstrap_edge')
    if (np.isfinite(rec.get('second_min_ratio', np.nan)) and rec['second_min_ratio'] < 1.05
            and rec.get('second_min_dist', 0) > 2.0):
        flags.append('ambiguous_minimum')
    return flags


def segment_windows(at, gap_m, max_m):
    """Ids for contiguous runs of along-track positions, split at gaps and capped."""
    order = np.argsort(at)
    a = at[order]
    run = np.concatenate([[0], np.cumsum(np.diff(a) > gap_m)])
    wid = np.zeros(len(at), dtype=int)
    next_id = 0
    for r in np.unique(run):
        m = run == r
        span = a[m].max() - a[m].min()
        n_chunks = max(1, int(np.ceil(span / max_m)))
        chunk = np.minimum(((a[m] - a[m].min()) / (span / n_chunks if span > 0 else 1)).astype(int),
                           n_chunks - 1)
        wid[order[m]] = next_id + chunk
        next_id += n_chunks
    return wid


def windows_for(at, length, off):
    """(window_length_m code, window ids, kind) for one window spec."""
    if length is None:
        return -1, np.zeros(len(at), dtype=int), 'full'
    if length == 'segment':
        return -1, segment_windows(at, off.get('segment_gap_m', 1000), off.get('segment_max_m', 20000)), 'segment'
    return int(length), ((at - at.min()) // length).astype(int), str(int(length))


def bootstrap_block(off, kind):
    """Bootstrap block length for a window kind; None = no bootstrap."""
    b = off.get('bootstrap_block_m', 100.0)
    if isinstance(b, dict):
        return b.get(kind)
    return b


def granule_name(name):
    m = re.search(r'ATL0[38]_\d{14}_\d{8}_\d{3}_\d{2}', str(name))
    return m.group(0) if m else None


class _Skip(Exception):
    pass


# --------------------------------------------------------------------------- #
# Per-beam processing
# --------------------------------------------------------------------------- #

def beam_records(gt, ph, seg, ground, base, off, dirs, tag_beam, cfg, interp=None):
    """
    Windows and fits for one beam, given its photons and ALS ground points.
    ``interp`` (DTM grids) is used as is; otherwise it is built from ``ground``.
    """
    if interp is None:
        interp = create_interpolator(ground, grid_resolution=off['grid_resolution'],
                                     ground_only=False, max_fill_distance=off['max_fill_distance'])
    on_als = ~np.isnan(interp((ph['alongtrack'].to_numpy(), ph['crosstrack'].to_numpy())))
    ph = ph[on_als]
    base = dict(base, n_ground_photons_on_als=len(ph), als_points=len(ground))
    if len(ph) < off['min_window_photons']:
        raise _Skip('too_few_photons_on_als')

    # refit cache: gridded surface + photons
    rc = dirs['refit']
    at_c, xt_c = interp.grid
    zz = np.asarray(interp.values, dtype=np.float32)
    rows = np.flatnonzero(np.isfinite(zz).any(axis=1))   # skip empty rows (gaps between surveys)
    np.savez_compressed(rc / f'{tag_beam}.npz', at=at_c[rows], xt=xt_c, z=zz[rows],
                        at_step=np.float64(at_c[1] - at_c[0]) if len(at_c) > 1 else np.nan)
    ph[['alongtrack', 'crosstrack', 'h_ph', 'delta_time']].to_parquet(rc / f'{tag_beam}.parquet')

    search = tuple(tuple(r) for r in off['search'])
    records, surfaces = [], {}
    ground = ground.sort_values('alongtrack', kind='stable')
    g_at = ground['alongtrack'].to_numpy()
    seg_at = seg['segment_dist_x'].to_numpy()
    at_all = ph['alongtrack'].to_numpy()
    for length in off['window_lengths_m']:
        code, wid, kind = windows_for(at_all, length, off)
        block = bootstrap_block(off, kind)
        for w in np.unique(wid):
            p = ph[wid == w]
            at0, at1 = p['alongtrack'].min(), p['alongtrack'].max()
            rec = dict(base, window_length_m=code, window_kind=kind, window_id=int(w),
                       at_start=float(at0), at_end=float(at1),
                       window_span_m=float(at1 - at0), n_photons=len(p))
            s = seg[(seg_at >= at0 - 20) & (seg_at <= at1)]
            if len(s) == 0:
                continue
            rec.update(summarize_segments(s))
            rec['time_utc'] = str(GPS_EPOCH + pd.to_timedelta(rec['delta_time_start'], unit='s'))
            rec['heading_deg'] = track_heading(s['reference_photon_lat'].to_numpy(),
                                               s['reference_photon_lon'].to_numpy())
            if len(p) >= off['min_window_photons']:
                res = estimate_offset(p['alongtrack'], p['crosstrack'], p['h_ph'], interp,
                                      search=search, fit_dz=off['fit_dz'],
                                      min_valid_frac=off['min_valid_frac'],
                                      n_bootstrap=off.get('n_bootstrap', 0) if block else 0,
                                      bootstrap_block_m=block or 100.0,
                                      bootstrap_method=off.get('bootstrap_method', 'grid'),
                                      max_photons=off.get('max_photons_per_window'))
                surf = res.pop('surface')
                surfaces[f"{kind}_W{w}"] = surf[2]
                surfaces.setdefault('grid_at', surf[0]); surfaces.setdefault('grid_xt', surf[1])
                rec.update(res)
                rec['corr_e'], rec['corr_n'] = atxt_to_en(res['corr_at'], res['corr_xt'], rec['heading_deg'])
                for c in ['at', 'xt', 'e', 'n']:
                    rec[f'err_{c}'] = -rec[f'corr_{c}']
                lo, hi = np.searchsorted(g_at, [at0 + res['corr_at'], at1 + res['corr_at']], side='left')
                g = ground.iloc[lo:hi]
                g = g[np.abs(g['crosstrack'].to_numpy() - res['corr_xt']) <= 10]
                t = terrain_summary(g, rec['heading_deg'])
                rec.update({f'terrain_{k}': v for k, v in t.items()})
                el = np.nanmedian(s['ref_elev']); az = np.radians(rec['ref_azimuth_deg_cmean'])
                rec['beam_vertical_angle_deg'] = float(beam_vertical_angle(el))
                rec['terrain_incidence_deg'] = float(terrain_incidence_angle(
                    el, az, t['grad_e'], t['grad_n'])) if np.isfinite(t['grad_e']) else np.nan
            rec['flags'] = ','.join(window_flags(rec, off))
            rec['accepted'] = rec['flags'] == ''
            records.append(rec)
    np.savez_compressed(dirs['surfaces'] / f'{tag_beam}.npz', **surfaces)
    return records


# --------------------------------------------------------------------------- #
# One job = one granule (all beams with ALS overlap)
# --------------------------------------------------------------------------- #

def process_granule(job, cfg, dirs):
    off = cfg['offsets']
    surface = off.get('surface', 'laz')
    stem = Path(job['file_03']).stem
    t0 = time.time()
    rows = []
    try:
        a03 = dirs['atl03'] / job['file_03']
        a08 = dirs['atl08'] / job['file_08']
        beams, segs, bases = {}, {}, {}
        for gt in job['gts']:
            base = dict(granule_03=job['file_03'], granule_08=job['file_08'], gt=gt,
                        key=job['key'], surface=surface)
            try:
                seg = read_segment_geolocation(a03, gt)
                base.update({k: seg.attrs[k] for k in ['release', 'version', 'rgt', 'cycle',
                                                       'orbit_number', 'atlas_spot_number',
                                                       'atlas_beam_type']})
                base['ph_uncorrelated_error_raw'] = json.dumps(seg.attrs['ph_uncorrelated_error_raw'])
                ph = read_photon_dataframe(str(a03), gt, atl08_file=str(a08))
                n_all = len(ph)
                ph = ph[(ph['atl08_class'] == 1) & (ph['quality_ph'] == 0)]
                base.update(n_photons_beam=n_all, n_ground_photons_beam=len(ph))
                if len(ph) < off['min_window_photons']:
                    raise _Skip('too_few_ground_photons')
                beams[gt], segs[gt] = ph, seg
            except _Skip as e:
                rows.append(dict(base, status='skipped', flags=str(e), accepted=False))
            except Exception as e:
                rows.append(dict(base, status='error', flags=repr(e)[:500], accepted=False,
                                 traceback=traceback.format_exc()[-2000:]))
            bases[gt] = base

        utm_epsg = dirs['utm_epsg']
        if surface == 'dtm' and beams:
            grids = create_dtm_grids(dirs['tiles'], beams, cfg['als']['dtm_dir'], utm_epsg,
                                     crosstrack_limit=off['crosstrack_limit'],
                                     resolution=off['grid_resolution'])
        for gt, ph in beams.items():
            base = dict(bases[gt], utm_epsg=utm_epsg)
            try:
                interp = None
                if surface == 'dtm':
                    if gt not in grids:
                        raise _Skip('no_als')
                    interp, ground = grids[gt]
                else:
                    sw = create_als_swath(dirs['tiles'], ph, num_workers=1, utm_epsg=utm_epsg,
                                          decimation=off['decimation'],
                                          crosstrack_limit=off['crosstrack_limit'],
                                          keep_classes=[cfg['als']['ground_class']])
                    ground = transform_als_swath(sw, utm_epsg, source_datum=None) if len(sw) else sw
                if len(ground) == 0:
                    raise _Skip('no_als')
                recs = beam_records(gt, ph, segs[gt], ground, base, off, dirs, f'{stem}_{gt}', cfg,
                                    interp=interp)
                rows += [dict(r, status='done') for r in recs]
            except _Skip as e:
                rows.append(dict(base, status='skipped', flags=str(e), accepted=False))
            except Exception as e:
                rows.append(dict(base, status='error', flags=repr(e)[:500], accepted=False,
                                 traceback=traceback.format_exc()[-2000:]))
    except Exception as e:
        rows.append(dict(granule_03=job['file_03'], key=job['key'], status='error',
                         flags=repr(e)[:500], accepted=False,
                         traceback=traceback.format_exc()[-2000:]))
    df = pd.DataFrame(rows)
    df['config'] = json.dumps(off)
    df['job_seconds'] = time.time() - t0
    df.to_parquet(dirs['parts'] / f'{stem}.parquet')
    return stem, df['status'].value_counts().to_dict(), time.time() - t0


# --------------------------------------------------------------------------- #
# Jobs, watch loop, main
# --------------------------------------------------------------------------- #

def file_map(folder):
    return {granule_name(p.name): p.name for p in folder.glob('*.h5')
            if granule_name(p.name) and not p.name.endswith('.part')}


def build_jobs(cfg, work, dirs):
    g = pd.read_parquet(work / 'granules' / 'granules.parquet')
    if 'order' in g:
        g = g.sort_values('order')
    beams = pd.read_parquet(work / 'granules' / 'beams.parquet')
    min_km = cfg['search']['min_overlap_km']
    gts = beams[beams['overlap_km'] >= min_km].groupby('key')['gt'].apply(sorted).to_dict()
    f03, f08 = file_map(dirs['atl03']), file_map(dirs['atl08'])
    jobs = []
    for _, r in g.iterrows():
        k03, k08 = granule_name(r['name_03']), granule_name(r['name_08'])
        jobs.append(dict(key=r['key'], file_03=f03.get(k03), file_08=f08.get(k08),
                         gts=gts.get(r['key'], [])))
    return jobs


def collect(dirs, out):
    parts = sorted(dirs['parts'].glob('*.parquet'))
    if not parts:
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(out / 'offsets.parquet')
    done = df[df['status'] == 'done']
    print(f"offsets.parquet: {len(df)} rows from {len(parts)} granules; "
          f"{int(done['accepted'].sum())} accepted windows", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=Path(__file__).with_name('config.yaml'))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--only', default=None, help='substring filter on ATL03 name')
    ap.add_argument('--collect', action='store_true', help='only combine parts')
    ap.add_argument('--retry-errors', action='store_true',
                    help="re-run granules whose part has any status 'error'")
    ap.add_argument('--watch', action='store_true',
                    help='keep running, picking up ATL03 files as they arrive')
    ap.add_argument('--done-flag', default=None,
                    help='with --watch: exit once this file exists and all present granules are done')
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg['work_dir']); gran = Path(cfg['granule_dir'])
    out = work / 'offsets'
    dirs = dict(parts=out / 'parts', surfaces=out / 'cost_surfaces', refit=out / 'refit_cache',
                atl03=gran / cfg.get('atl03_subdir', 'ATL03_subset'),
                atl08=gran / cfg.get('atl08_subdir', 'ATL08'))
    for k in ('parts', 'surfaces', 'refit'):
        dirs[k].mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(dirs, out); return

    tiles = gpd.read_file(work / 'als_inventory' / 'als_tiles.gpkg')
    zones = tiles['utm_epsg'].dropna().astype(int).unique()
    if len(zones) != 1:
        raise SystemExit(f"tiles span several UTM zones {zones}; per-zone handling not implemented")
    dirs['tiles'] = tiles[['file_name', 'geometry']]
    dirs['utm_epsg'] = f"EPSG:{zones[0]}"

    def done_or_queued(j, submitted):
        stem = Path(j['file_03']).stem
        if stem in submitted:
            return True
        part = dirs['parts'] / f'{stem}.parquet'
        if not part.exists():
            return False
        if args.retry_errors and stem not in retried and \
                (pd.read_parquet(part, columns=['status'])['status'] == 'error').any():
            retried.add(stem); part.unlink(); return False
        return True

    retried, submitted = set(), set()
    n_jobs = args.jobs or cfg['offsets'].get('n_jobs', 1)
    t_start = time.time(); n_done = 0
    with ProcessPoolExecutor(n_jobs) as ex:
        running = {}
        while True:
            jobs = build_jobs(cfg, work, dirs)
            if args.only:
                jobs = [j for j in jobs if j['file_03'] and args.only in j['file_03']]
            ready = [j for j in jobs if j['file_03'] and j['file_08'] and j['gts']
                     and not done_or_queued(j, submitted)]
            if args.limit:
                ready = ready[:max(0, args.limit - len(submitted))]
            # keep at most 2x workers queued so new arrivals keep selection order
            for j in ready[:max(0, 2 * n_jobs - len(running))]:
                running[ex.submit(process_granule, j, cfg, dirs)] = j
                submitted.add(Path(j['file_03']).stem)
            if running:
                fin, _ = wait(list(running), timeout=30, return_when=FIRST_COMPLETED)
                for f in fin:
                    j = running.pop(f)
                    try:
                        stem, st, secs = f.result()
                        n_done += 1
                        el = (time.time() - t_start) / 3600
                        print(f"[{time.strftime('%H:%M:%S')}] done {n_done} ({el:.2f} h) "
                              f"{stem} {st} {secs:.0f}s", flush=True)
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] worker failure {j['key']}: {e!r}", flush=True)
                continue
            n_missing = sum(1 for j in jobs if not j['file_03'])
            if not args.watch:
                break
            if args.done_flag and Path(args.done_flag).exists():
                print(f"download finished; {n_missing} selected granules have no ATL03 file", flush=True)
                break
            time.sleep(60)
    collect(dirs, out)


if __name__ == '__main__':
    main()
