"""
Joint per-pass refit of ICESat-2 horizontal offsets from the refit cache
written by run_offsets.py (no ALS or ATL03 re-read).

Passes are defined in time, so every beam covers the same stretch of ground:
the segment windows of all beams of a granule are merged into contiguous
time stretches, and each stretch is cut into equal chunks of at most
``--chunk-km`` (default 20 km, ~2.8 s at 7.1 km/s). A pass = one chunk; each
beam contributes its cached ground photons with delta_time in the chunk.

For each pass:
    solo     each beam fitted on its own (same photons, same search)
    model 1  one along-/cross-track shift shared by all beams: minimise the
             photon-pooled MAE  sum_b n_b MAE_b(s) / sum_b n_b
    model 2  shared shift plus a fixed per-spot x orientation term c, with
             beam b evaluated at s + c[spot_b, orient_b]; c is the median of
             (solo - model 1) over well-determined beams in passes with >= 4 beams

Search: 2 m brute force over +-10 m, then bounded Nelder-Mead.
Uncertainty: leave-one-beam-out jackknife (passes with >= 3 beams).

Outputs (in <work_dir>/offsets/pass_refit/):
    passes.parquet      one row per pass (model 1 and model 2, E/N and AT/XT)
    beams.parquet       one row per beam in a pass (solo fit, residuals)
    spot_terms.csv      model-2 per-spot terms (correction convention, metres)

Usage:
    uv run python refit_pass.py --config CFG [--jobs N] [--limit N] [--chunk-km 20]
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from phoreal.config import load_config
from phoreal.geolocation import make_cost_function, atxt_to_en

COLS = ['granule_03', 'gt', 'window_kind', 'status', 'n_photons', 'delta_time_start',
        'delta_time_end', 'atlas_spot_number', 'atlas_beam_type', 'sc_orient',
        'heading_deg', 'key', 'solar_elevation_med', 'accepted']
SEARCH = 10.0
GROUND_SPEED = 7100.0     # m/s, measured from the cache


# --------------------------------------------------------------------------- #
# Passes
# --------------------------------------------------------------------------- #

def time_chunks(w, chunk_s, gap_s):
    """Merge a granule's window intervals into stretches, then split into chunks."""
    iv = w[['delta_time_start', 'delta_time_end']].sort_values('delta_time_start').to_numpy()
    stretches = []
    for s, e in iv:
        if stretches and s <= stretches[-1][1] + gap_s:
            stretches[-1][1] = max(stretches[-1][1], e)
        else:
            stretches.append([s, e])
    chunks = []
    for s, e in stretches:
        n = max(1, int(np.ceil((e - s) / chunk_s)))
        edges = np.linspace(s, e, n + 1)
        chunks += list(zip(edges[:-1], edges[1:]))
    return chunks


# --------------------------------------------------------------------------- #
# Cost functions from the cache
# --------------------------------------------------------------------------- #

class BeamCache:
    def __init__(self, refit_dir, stem_gt):
        z = np.load(refit_dir / f'{stem_gt}.npz')
        self.at, self.xt, self.z, self.step = z['at'], z['xt'], z['z'], float(z['at_step'])
        self.ph = pd.read_parquet(refit_dir / f'{stem_gt}.parquet',
                                  columns=['alongtrack', 'crosstrack', 'h_ph', 'delta_time'])
        self.ph = self.ph.sort_values('delta_time', kind='stable')
        self.t = self.ph['delta_time'].to_numpy()

    def cost(self, t0, t1, off, seed=0):
        lo, hi = np.searchsorted(self.t, [t0, t1])
        ph = self.ph.iloc[lo:hi]
        if len(ph) < off['min_window_photons']:
            return None, len(ph)
        cap = off.get('max_photons_per_window')
        if cap and len(ph) > cap:
            ph = ph.iloc[np.sort(np.random.default_rng(seed).choice(len(ph), cap, replace=False))]
        at0, at1 = ph['alongtrack'].min(), ph['alongtrack'].max()
        pad = SEARCH + 2 * self.step
        sel = (self.at >= at0 - pad) & (self.at <= at1 + pad)
        if sel.sum() < 2:
            return None, 0
        a = self.at[sel]
        full = np.arange(a.min(), a.max() + self.step / 2, self.step)   # restore trimmed rows
        grid = np.full((len(full), len(self.xt)), np.nan)
        grid[np.rint((a - full[0]) / self.step).astype(int)] = self.z[sel]
        interp = RegularGridInterpolator((full, self.xt), grid, bounds_error=False, fill_value=np.nan)
        f = make_cost_function(ph['alongtrack'], ph['crosstrack'], ph['h_ph'], interp,
                               fit_dz=off['fit_dz'], min_valid_frac=off['min_valid_frac'])
        return f, len(ph)


def fit(costs, weights, offsets, start=None, simplex=1.0):
    """Minimise the photon-weighted joint cost; beam b is evaluated at s + offsets[b]."""
    w = np.asarray(weights, float) / np.sum(weights)

    def J(s):
        tot = 0.0
        for f, wi, o in zip(costs, w, offsets):
            c = f((s[0] + o[0], s[1] + o[1]))
            if not np.isfinite(c):
                return np.inf
            tot += wi * c
        return tot

    if start is None:
        g = np.arange(-SEARCH, SEARCH + 0.1, 2.0)
        vals = np.array([[J((a, x)) for x in g] for a in g])
        if not np.isfinite(vals).any():
            return np.nan, np.nan, np.inf, J
        i, j = np.unravel_index(np.nanargmin(np.where(np.isfinite(vals), vals, np.nan)), vals.shape)
        start = (g[i], g[j])
    x0 = np.clip(np.asarray(start, float), -SEARCH, SEARCH)
    if not np.isfinite(J(x0)):
        return np.nan, np.nan, np.inf, J
    sim = np.array([x0, x0 + [simplex, 0], x0 + [0, simplex]])
    r = minimize(J, x0, method='Nelder-Mead', bounds=[(-SEARCH, SEARCH)] * 2,
                 options=dict(initial_simplex=sim, xatol=0.02, fatol=1e-5, maxiter=300))
    return float(r.x[0]), float(r.x[1]), float(r.fun), J


def solve(costs, weights, offsets, start=None):
    at, xt, jmin, J = fit(costs, weights, offsets, start=start)
    out = dict(corr_at=at, corr_xt=xt, cost_opt=jmin, cost_zero=J((0.0, 0.0)),
               boundary_hit=bool(np.isfinite(at) and max(abs(at), abs(xt)) > SEARCH - 0.5),
               jk_sigma_at=np.nan, jk_sigma_xt=np.nan)
    n = len(costs)
    if n >= 3 and np.isfinite(at):
        th = np.array([fit([costs[k] for k in range(n) if k != i], [weights[k] for k in range(n) if k != i],
                           [offsets[k] for k in range(n) if k != i], start=(at, xt), simplex=0.3)[:2]
                       for i in range(n)], float)
        if np.isfinite(th).all():
            jk = np.sqrt((n - 1) / n * ((th - th.mean(axis=0)) ** 2).sum(axis=0))
            out['jk_sigma_at'], out['jk_sigma_xt'] = float(jk[0]), float(jk[1])
    return out


# --------------------------------------------------------------------------- #
# One job = one granule
# --------------------------------------------------------------------------- #

def process_granule(gran, w, refit_dir, off, chunk_s, spot_terms, m1=None):
    """Model 1 + solo fits (spot_terms None), or model 2 starting from model 1."""
    stem = Path(gran).stem
    beams = {}
    for gt, r in w.groupby('gt'):
        if (refit_dir / f'{stem}_{gt}.npz').exists():
            beams[gt] = (BeamCache(refit_dir, f'{stem}_{gt}'), r.iloc[0])
    passes, beam_rows = [], []
    for k, (t0, t1) in enumerate(time_chunks(w, chunk_s, gap_s=1000 / GROUND_SPEED)):
        pid = f"{w['key'].iloc[0]}_C{k}"
        costs, weights, meta = [], [], []
        for gt, (bc, r) in beams.items():
            f, n = bc.cost(t0, t1, off)
            if f is not None:
                costs.append(f); weights.append(n); meta.append(r)
        if not costs:
            continue
        base = dict(pass_id=pid, granule_03=gran, key=w['key'].iloc[0], t0=t0, t1=t1,
                    span_km=(t1 - t0) * GROUND_SPEED / 1000,
                    heading_deg=float(np.median([m['heading_deg'] for m in meta])),
                    sc_orient=meta[0]['sc_orient'],
                    solar_elevation=float(np.median([m['solar_elevation_med'] for m in meta])),
                    n_beams=len(costs), n_strong=sum(m['atlas_beam_type'] == 'strong' for m in meta),
                    n_photons=int(sum(weights)),
                    spots=','.join(str(int(m['atlas_spot_number'])) for m in meta))
        if spot_terms is None:
            res = solve(costs, weights, [(0.0, 0.0)] * len(costs))
            passes.append(dict(base, **{f'm1_{k_}': v for k_, v in res.items()}))
            for f, n, m in zip(costs, weights, meta):     # solo fits, same photons
                a, x, jm, J = fit([f], [n], [(0.0, 0.0)])
                beam_rows.append(dict(pass_id=pid, gt=m['gt'], spot=int(m['atlas_spot_number']),
                                      beam_type=m['atlas_beam_type'], sc_orient=m['sc_orient'],
                                      n_photons=n, solo_corr_at=a, solo_corr_xt=x, solo_cost=jm,
                                      solo_cost_zero=J((0.0, 0.0)),
                                      solo_boundary=bool(np.isfinite(a) and max(abs(a), abs(x)) > SEARCH - 0.5)))
        else:
            offs = [spot_terms.get((int(m['atlas_spot_number']), m['sc_orient']), (0.0, 0.0)) for m in meta]
            s0 = m1.get(pid)
            if s0 is not None and np.all(np.isfinite(s0)):
                s0 = (s0[0] - np.mean([o[0] for o in offs]), s0[1] - np.mean([o[1] for o in offs]))
            else:
                s0 = None
            res = solve(costs, weights, offs, start=s0)
            passes.append(dict(pass_id=pid, **{f'm2_{k_}': v for k_, v in res.items()}))
    return passes, beam_rows


def run(by_g, refit_dir, off, chunk_s, jobs, spot_terms=None, m1=None):
    P, B = [], []
    with ProcessPoolExecutor(jobs) as ex:
        futs = {ex.submit(process_granule, g, w, refit_dir, off, chunk_s, spot_terms,
                          None if m1 is None else {k: v for k, v in m1.items() if k.startswith(w['key'].iloc[0])}): g
                for g, w in by_g}
        for i, f in enumerate(as_completed(futs)):
            try:
                p, b = f.result(); P += p; B += b
            except Exception as e:
                print(f"  failed {futs[f]}: {e!r}", flush=True)
            if (i + 1) % 50 == 0 or i + 1 == len(futs):
                print(f"  [{time.strftime('%H:%M:%S')}] {i + 1}/{len(futs)} granules", flush=True)
    return pd.DataFrame(P), pd.DataFrame(B)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--jobs', type=int, default=18)
    ap.add_argument('--limit', type=int, default=None, help='first N granules (testing)')
    ap.add_argument('--chunk-km', type=float, default=20.0)
    ap.add_argument('--out', default='pass_refit')
    args = ap.parse_args()
    cfg = load_config(args.config); off = cfg['offsets']
    out_dir = Path(cfg['work_dir']) / 'offsets'
    refit_dir = out_dir / 'refit_cache'
    res_dir = out_dir / args.out; res_dir.mkdir(exist_ok=True)
    chunk_s = args.chunk_km * 1000 / GROUND_SPEED

    d = pd.read_parquet(out_dir / 'offsets.parquet', columns=COLS)
    w = d[(d['status'] == 'done') & (d['window_kind'] == 'segment')
          & (d['n_photons'] >= off['min_window_photons']) & d['delta_time_start'].notna()]
    grans = w['granule_03'].drop_duplicates()
    if args.limit:
        grans = grans.head(args.limit)
    by_g = [(g, x) for g, x in w[w['granule_03'].isin(grans)].groupby('granule_03')]
    print(f"{len(by_g)} granules, {len(w)} segment windows as input", flush=True)

    t0 = time.time()
    print("model 1 (shared shift) + solo fits", flush=True)
    p1, b = run(by_g, refit_dir, off, chunk_s, args.jobs)
    b = b.merge(p1[['pass_id', 'm1_corr_at', 'm1_corr_xt', 'n_beams']], on='pass_id')
    b['res_at'] = b['solo_corr_at'] - b['m1_corr_at']
    b['res_xt'] = b['solo_corr_xt'] - b['m1_corr_xt']

    good = b[(b['n_beams'] >= 4) & ~b['solo_boundary'] & b['res_at'].notna()]
    terms = good.groupby(['spot', 'sc_orient']).agg(c_at=('res_at', 'median'), c_xt=('res_xt', 'median'),
                                                    n=('res_at', 'size')).reset_index()
    terms.to_csv(res_dir / 'spot_terms.csv', index=False)
    print(terms.round(2).to_string(index=False), flush=True)
    spot_terms = {(int(r.spot), r.sc_orient): (float(r.c_at), float(r.c_xt)) for r in terms.itertuples()}

    print("model 2 (shared shift + per-spot terms)", flush=True)
    m1 = {r.pass_id: (r.m1_corr_at, r.m1_corr_xt) for r in p1.itertuples()}
    p2, _ = run(by_g, refit_dir, off, chunk_s, args.jobs, spot_terms, m1)
    p = p1.merge(p2, on='pass_id', how='left')
    for m in ['m1', 'm2']:
        e, n = atxt_to_en(p[f'{m}_corr_at'].to_numpy(), p[f'{m}_corr_xt'].to_numpy(), p['heading_deg'].to_numpy())
        p[f'{m}_corr_e'], p[f'{m}_corr_n'] = e, n
        for c in ['at', 'xt', 'e', 'n']:
            p[f'{m}_err_{c}'] = -p[f'{m}_corr_{c}']
    p.to_parquet(res_dir / 'passes.parquet')
    b.to_parquet(res_dir / 'beams.parquet')
    print(f"wrote {len(p)} passes, {len(b)} beam rows to {res_dir} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
