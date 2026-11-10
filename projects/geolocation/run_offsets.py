"""
Estimates ICESat-2 horizontal geolocation offsets against the Poland ALS for
every (granule, beam) with ALS overlap, in one or more along-track windows.

One output record per physical spot x pass x window, including rejected
windows (``accepted`` = False plus ``flags``). Offsets are stored both as the
correction applied to ICESat-2 (``corr_*``) and as the geolocation error
(``err_*`` = -correction), in along/cross-track and east/north.

Inputs (from inventory_als.py and download_granules.py):
    <work_dir>/als_inventory/als_tiles.gpkg
    <work_dir>/granules/{granules,beams}.parquet
    <granule_dir>/ATL03_subset/*.h5, <granule_dir>/ATL08/*.h5

Outputs (in <work_dir>/offsets/):
    parts/<granule>_<gt>.parquet      per-job records (resume = skip existing)
    cost_surfaces/<granule>_<gt>.npz  brute-force cost grid per window
    als_swaths/<granule>_<gt>.pqt     cached ALS ground swath
    offsets.parquet                   all parts combined (--collect)

Usage:
    uv run python run_offsets.py [--config config.yaml] [--limit N]
                                 [--jobs N] [--only GRANULE_SUBSTR] [--collect]
"""
import argparse
import json
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from phoreal.als.swath import create_als_swath, transform_als_swath
from phoreal.config import load_config
from phoreal.geo.geographic_utils import find_utm_zone_epsg
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


# --------------------------------------------------------------------------- #
# One job = one granule x beam
# --------------------------------------------------------------------------- #

def process_job(job, cfg, dirs):
    off = cfg['offsets']
    tag = f"{Path(job['name_03']).stem}_{job['gt']}"
    part = dirs['parts'] / f'{tag}.parquet'
    base = dict(granule_03=job['name_03'], granule_08=job['name_08'], gt=job['gt'],
                key=job['key'])
    try:
        a03 = dirs['atl03'] / job['name_03']
        a08 = dirs['atl08'] / job['name_08']

        seg = read_segment_geolocation(a03, job['gt'])
        base.update({k: seg.attrs[k] for k in ['release', 'version', 'rgt', 'cycle',
                                               'orbit_number', 'atlas_spot_number',
                                               'atlas_beam_type']})
        base['ph_uncorrelated_error_raw'] = json.dumps(seg.attrs['ph_uncorrelated_error_raw'])

        ph = read_photon_dataframe(str(a03), job['gt'], atl08_file=str(a08))
        n_all = len(ph)
        ph = ph[(ph['atl08_class'] == 1) & (ph['quality_ph'] == 0)]
        base.update(n_photons_beam=n_all, n_ground_photons_beam=len(ph))
        if len(ph) < off['min_window_photons']:
            raise _Skip('too_few_ground_photons')

        # ALS swath (cached)
        utm_epsg = find_utm_zone_epsg(float(ph['latitude'].median()), float(ph['longitude'].median()))
        swath_file = dirs['swaths'] / f'{tag}.pqt'
        if swath_file.exists():
            swath = pd.read_parquet(swath_file)
        else:
            swath = create_als_swath(dirs['tiles'], ph, num_workers=1, utm_epsg=utm_epsg,
                                     decimation=off['decimation'],
                                     crosstrack_limit=off['crosstrack_limit'],
                                     keep_classes=[cfg['als']['ground_class']])
            if len(swath) == 0:
                raise _Skip('no_als')
            swath = transform_als_swath(swath, utm_epsg, source_datum=None)
            swath.to_parquet(swath_file)
        ground = swath[swath['classification'] == cfg['als']['ground_class']]
        interp = create_interpolator(ground, grid_resolution=off['grid_resolution'],
                                     ground_only=False,
                                     max_fill_distance=off['max_fill_distance'])

        # Keep photons that sit on ALS data before any shift
        on_als = ~np.isnan(interp((ph['alongtrack'].to_numpy(), ph['crosstrack'].to_numpy())))
        ph = ph[on_als]
        base.update(n_ground_photons_on_als=len(ph), utm_epsg=utm_epsg,
                    als_points=len(ground))
        if len(ph) < off['min_window_photons']:
            raise _Skip('too_few_photons_on_als')

        search = tuple(tuple(r) for r in off['search'])
        records, surfaces = [], {}
        seg_at = seg['segment_dist_x'].to_numpy()
        for length in off['window_lengths_m']:
            wid = make_windows(ph['alongtrack'].to_numpy(), length)
            for w in np.unique(wid):
                p = ph[wid == w]
                at0, at1 = p['alongtrack'].min(), p['alongtrack'].max()
                rec = dict(base, window_length_m=length if length is not None else -1,
                           window_id=int(w), at_start=float(at0), at_end=float(at1),
                           window_span_m=float(at1 - at0), n_photons=len(p))
                s = seg[(seg_at >= at0 - 20) & (seg_at <= at1)]
                rec.update(summarize_segments(s))
                rec['time_utc'] = str(GPS_EPOCH + pd.to_timedelta(rec['delta_time_start'], unit='s'))
                rec['heading_deg'] = track_heading(s['reference_photon_lat'].to_numpy(),
                                                   s['reference_photon_lon'].to_numpy())

                if len(p) >= off['min_window_photons']:
                    res = estimate_offset(p['alongtrack'], p['crosstrack'], p['h_ph'], interp,
                                          search=search, fit_dz=off['fit_dz'],
                                          min_valid_frac=off['min_valid_frac'],
                                          n_bootstrap=off.get('n_bootstrap', 0),
                                          bootstrap_block_m=off.get('bootstrap_block_m', 100.0))
                    surf = res.pop('surface')
                    surfaces[f"L{rec['window_length_m']}_W{w}"] = surf[2]
                    surfaces.setdefault('grid_at', surf[0]); surfaces.setdefault('grid_xt', surf[1])
                    rec.update(res)
                    rec['corr_e'], rec['corr_n'] = atxt_to_en(res['corr_at'], res['corr_xt'], rec['heading_deg'])
                    for c in ['at', 'xt', 'e', 'n']:
                        rec[f'err_{c}'] = -rec[f'corr_{c}']

                    # Terrain from ALS ground under the registered photons
                    g = ground[(ground['alongtrack'] >= at0 + res['corr_at'])
                               & (ground['alongtrack'] <= at1 + res['corr_at'])
                               & (np.abs(ground['crosstrack'] - res['corr_xt']) <= 10)]
                    t = terrain_summary(g, rec['heading_deg'])
                    rec.update({f'terrain_{k}': v for k, v in t.items()})
                    el = np.nanmedian(s['ref_elev']); az = np.radians(rec['ref_azimuth_deg_cmean'])
                    rec['beam_vertical_angle_deg'] = float(beam_vertical_angle(el))
                    rec['terrain_incidence_deg'] = float(terrain_incidence_angle(
                        el, az, t['grad_e'], t['grad_n'])) if np.isfinite(t['grad_e']) else np.nan

                rec['flags'] = ','.join(window_flags(rec, off))
                rec['accepted'] = rec['flags'] == ''
                records.append(rec)

        np.savez_compressed(dirs['surfaces'] / f'{tag}.npz', **surfaces)
        df = pd.DataFrame(records)
        df['status'] = 'done'
    except _Skip as e:
        df = pd.DataFrame([dict(base, status='skipped', flags=str(e), accepted=False)])
    except Exception as e:  # keep going; record the failure
        df = pd.DataFrame([dict(base, status='error', flags=repr(e)[:500], accepted=False,
                                traceback=traceback.format_exc()[-2000:])])
    df['config'] = json.dumps(off)
    df.to_parquet(part)
    return tag, df['status'].iloc[0]


class _Skip(Exception):
    pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_jobs(cfg, work, atl03_dir):
    g = pd.read_parquet(work / 'granules' / 'granules.parquet')
    beams = pd.read_parquet(work / 'granules' / 'beams.parquet')
    min_km = cfg['search']['min_overlap_km']
    jobs = beams[beams['overlap_km'] >= min_km].merge(
        g[['key', 'name_03', 'name_08']], on=['key', 'name_08'])
    jobs = jobs[jobs['name_03'].map(lambda n: (atl03_dir / n).exists())]
    return jobs.sort_values(['key', 'gt']).to_dict('records')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=Path(__file__).with_name('config.yaml'))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--only', default=None, help='substring filter on ATL03 name')
    ap.add_argument('--collect', action='store_true', help='only combine parts')
    args = ap.parse_args()

    cfg = load_config(args.config)
    work = Path(cfg['work_dir']); gran = Path(cfg['granule_dir'])
    out = work / 'offsets'
    dirs = dict(parts=out / 'parts', surfaces=out / 'cost_surfaces', swaths=out / 'als_swaths',
                atl03=gran / 'ATL03_subset', atl08=gran / 'ATL08')
    for k in ('parts', 'surfaces', 'swaths'):
        dirs[k].mkdir(parents=True, exist_ok=True)

    if not args.collect:
        jobs = build_jobs(cfg, work, dirs['atl03'])
        if args.only:
            jobs = [j for j in jobs if args.only in j['name_03']]
        todo = [j for j in jobs
                if not (dirs['parts'] / f"{Path(j['name_03']).stem}_{j['gt']}.parquet").exists()]
        if args.limit:
            todo = todo[:args.limit]
        print(f"{len(jobs)} jobs, {len(todo)} to run")
        if todo:
            dirs['tiles'] = gpd.read_file(work / 'als_inventory' / 'als_tiles.gpkg')
            n_jobs = args.jobs or cfg['offsets'].get('n_jobs', 1)
            if n_jobs > 1:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_jobs, verbose=5)(
                    delayed(process_job)(j, cfg, dirs) for j in todo)
            else:
                results = [process_job(j, cfg, dirs) for j in todo]
            print(pd.Series([r[1] for r in results]).value_counts().to_string())

    parts = sorted(dirs['parts'].glob('*.parquet'))
    if parts:
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df.to_parquet(out / 'offsets.parquet')
        done = df[df['status'] == 'done']
        print(f"offsets.parquet: {len(df)} rows from {len(parts)} jobs; "
              f"{int(done['accepted'].sum())} accepted windows")


if __name__ == '__main__':
    main()
