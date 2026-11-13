"""Per-segment terrain heights for ATL08 ground, ATL24 terrain and all signal vs 3DEP.

Segments are ATL03 ~20 m geolocation segments (``segment_id``), identical for
both classifiers. Heights are compared in the 3DEP frame: every photon is
moved from ITRF2014 (epoch of the granule) to NAD83(2011) UTM + NAVD88
(GEOID18), so dz = h_navd88 - dem.

Photon selections S:
  a08  ATL08 ground (atl08_class == 1)
  a24  ATL24 terrain (atl24_class == atl24_classes.terrain)
  sig  ATL03 land signal (signal_conf_ph0 >= analysis.signal_conf_min)

Per segment and S: n_S, h_S (median h_navd88), dzc_S = h_S - DEM at the segment
centre, dz_S = median over the photons of (h_navd88 - DEM at the photon),
mad_S, p20_S, p80_S (of photon dz), below_S = share of S photons below the
signal peak (mode of the signal photons' dz, 0.25 m bins).
Coverage: n_both, n_a08_only, n_a24_only (photon counts).
Reference: DEM at the centre, slope / aspect (plane fit, dem.slope_window_m),
track heading and slope orientation relative to the track.
Official ATL08: dz of h_te_best_fit_20m (20 m) and of the 100 m h_te_best_fit,
h_te_median, h_te_interp, plus segment_snowcover / landcover / night / cloud.

Output: <work_dir>/segments.parquet, <work_dir>/photons_dz.parquet (selected photons)
Run from this folder:  uv run python build_segments.py
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from phoreal.config import load_config
from phoreal.geo.dem import DemStack, itrf_to_dem_frame
from phoreal.io.readers import get_atl03_metadata

HERE = Path(__file__).resolve().parent
US_LAT = (31.3, 49.05)
PEAK_BIN = 0.25
SECONDS_PER_YEAR = 365.25 * 86400


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def dem_paths(dem_dir):
    """Staged 1 m tiles first (newest publication first), then 1/3 arc-second if staged."""
    paths = []
    for product in ('1m', '13'):
        man = dem_dir / product / 'manifest.csv'
        if not man.exists():
            continue
        m = pd.read_csv(man)
        m = m[m['status'].isin(['staged', 'exists'])].sort_values('publication_date', ascending=False)
        paths += [resolve(p) for p in m['out_file']]
    return paths


def atl08_segments(atl08_file, gt):
    """ATL08 values per 20 m segment_id (100 m values repeated for their five 20 m parts)."""
    with h5py.File(atl08_file, 'r') as f:
        g = f[gt + '/land_segments']
        beg = g['segment_id_beg'][:]

        def rd(name):
            a = g[name][:].astype(float)
            a[np.abs(a) > 1e30] = np.nan
            return a
        bf20 = rd('terrain/h_te_best_fit_20m')
        hundred = {k: rd('terrain/' + k) for k in ('h_te_best_fit', 'h_te_median', 'h_te_interp')}
        flags = {k: g[k][:] for k in ('segment_snowcover', 'segment_landcover', 'night_flag',
                                      'cloud_flag_atm', 'terrain_flg')}
    rows = {'segment_id': (beg[:, None] + np.arange(5)).ravel(), 'atl08_h_te_best_fit_20m': bf20.ravel()}
    for k, v in {**hundred, **flags}.items():
        rows[('atl08_' + k) if k.startswith('h_te') else k] = np.repeat(v, 5)
    return pd.DataFrame(rows).drop_duplicates('segment_id')


def load_beam(cfg, work, gt):
    ph = pd.read_parquet(work / 'photons' / f'photons_{gt}.parquet',
                         columns=['latitude', 'longitude', 'h_ph', 'delta_time', 'segment_id', 'alongtrack',
                                  'signal_conf_ph0', 'atl08_class', 'atl24_class'])
    ph = ph[(ph.latitude > US_LAT[0]) & (ph.latitude < US_LAT[1])]
    a = cfg['analysis']
    sel = pd.DataFrame({
        'a08': ph['atl08_class'].to_numpy() == 1,
        'a24': ph['atl24_class'].to_numpy() == cfg['atl24_classes']['terrain'],
        'sig': ph['signal_conf_ph0'].to_numpy() >= a.get('signal_conf_min', 4)})
    keep = sel.any(axis=1).to_numpy()
    ph = ph[keep].reset_index(drop=True)
    for c in sel:
        ph[c] = sel[c].to_numpy()[keep]
    ph['gt'] = gt
    return ph


def heading_deg(x, y, half=5):
    """Track heading (grid azimuth) from centre positions ordered along track."""
    n = len(x)
    i0 = np.clip(np.arange(n) - half, 0, n - 1)
    i1 = np.clip(np.arange(n) + half, 0, n - 1)
    return np.degrees(np.arctan2(x[i1] - x[i0], y[i1] - y[i0])) % 360


def segment_stats(ph, sname):
    """Grouped stats of one photon selection."""
    s = ph[ph[sname]]
    g = s.groupby(['gt', 'segment_id'])
    out = pd.DataFrame({
        f'n_{sname}': g.size(),
        f'h_{sname}': g['h_navd'].median(),
        f'dz_{sname}': g['dz'].median(),
        f'p20_{sname}': g['dz'].quantile(0.2),
        f'p80_{sname}': g['dz'].quantile(0.8),
        f'below_{sname}': g['below_peak'].mean(),
    })
    med = s.groupby(['gt', 'segment_id'])['dz'].transform('median')
    out[f'mad_{sname}'] = (s['dz'] - med).abs().groupby([s['gt'], s['segment_id']]).median()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=HERE / 'config.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = resolve(cfg['work_dir'])
    a03, a08 = resolve(cfg['granule']['atl03']), resolve(cfg['granule']['atl08'])

    paths = dem_paths(resolve(cfg['dem_dir']))
    if not paths:
        raise SystemExit('no staged DEM tiles: run stage_3dep.py first')
    stack = DemStack(paths)
    print(f'{len(paths)} DEM rasters, CRS {stack.crs.to_string()}')

    # photons of all beams, moved into the DEM frame
    ph = pd.concat([load_beam(cfg, work, gt) for gt in cfg['beams']], ignore_index=True)
    epoch = 2018.0 + ph['delta_time'].mean() / SECONDS_PER_YEAR
    print(f'{len(ph):,} photons in selections, epoch {epoch:.3f}')
    x, y, hv = itrf_to_dem_frame(ph['longitude'], ph['latitude'], ph['h_ph'], epoch, stack.crs)
    ph['x'], ph['y'], ph['h_navd'] = x, y, hv
    ph['dz'] = hv - stack.sample(x, y)['elev']
    ph = ph.drop(columns=['delta_time', 'signal_conf_ph0', 'atl08_class', 'atl24_class'])

    # signal peak per segment (mode of signal dz) and who sits below it
    s = ph[ph['sig'] & ph['dz'].notna()]
    b = np.floor(s['dz'] / PEAK_BIN).astype(np.int64)
    cnt = s.groupby([s['gt'], s['segment_id'], b]).size().rename('c').reset_index()
    cnt = cnt.sort_values('c', ascending=False).drop_duplicates(['gt', 'segment_id'])
    cnt['peak'] = (cnt['dz'] + 0.5) * PEAK_BIN
    ph = ph.merge(cnt[['gt', 'segment_id', 'peak']], on=['gt', 'segment_id'], how='left')
    ph['below_peak'] = np.where(ph['peak'].notna() & ph['dz'].notna(),
                                (ph['dz'] < ph['peak']).astype(float), np.nan)
    ph[['gt', 'segment_id', 'alongtrack', 'h_navd', 'dz', 'a08', 'a24', 'sig']].to_parquet(
        work / 'photons_dz.parquet', index=False)       # photon residuals, for profile plots

    # segment centres (median position of the selected photons) and DEM there
    g = ph.groupby(['gt', 'segment_id'])
    seg = pd.DataFrame({'x': g['x'].median(), 'y': g['y'].median(),
                        'latitude': g['latitude'].median(), 'longitude': g['longitude'].median(),
                        'offset_navd': (g['h_navd'].median() - g['h_ph'].median()),
                        'sig_peak': g['peak'].first()})
    d = stack.sample(seg['x'], seg['y'], slope_window_m=cfg['dem']['slope_window_m'])
    for k in ('elev', 'slope_deg', 'aspect_deg', 'gx', 'gy', 'src'):
        seg['dem_' + k if k in ('elev', 'src') else k] = d[k]
    for sname in ('a08', 'a24', 'sig'):
        seg = seg.join(segment_stats(ph, sname))
        seg[f'n_{sname}'] = seg[f'n_{sname}'].fillna(0).astype(int)
        seg[f'dzc_{sname}'] = seg[f'h_{sname}'] - seg['dem_elev']
    both = ph['a08'] & ph['a24']
    cov = pd.DataFrame({'n_both': both, 'n_a08_only': ph['a08'] & ~ph['a24'],
                        'n_a24_only': ph['a24'] & ~ph['a08']}).groupby([ph['gt'], ph['segment_id']]).sum()
    seg = seg.join(cov).reset_index()

    # track heading, slope orientation relative to track, beam metadata, ATL08 products
    parts = []
    for gt, sg in seg.groupby('gt'):
        sg = sg.sort_values('segment_id').copy()
        sg['heading_deg'] = heading_deg(sg['x'].to_numpy(), sg['y'].to_numpy())
        meta = get_atl03_metadata(str(a03), gt)
        sg['beam_type'], sg['spot'] = meta['beam_type'], int(meta['spot_number'])
        sg = sg.merge(atl08_segments(a08, gt), on='segment_id', how='left')
        parts.append(sg)
    seg = pd.concat(parts, ignore_index=True)
    rel = np.abs((seg['aspect_deg'] - seg['heading_deg'] + 180) % 360 - 180)
    seg['rel_aspect_deg'] = np.minimum(rel, 180 - rel)     # 0 = slope falls along track, 90 = across
    seg['orientation'] = np.select([seg['rel_aspect_deg'] <= 30, seg['rel_aspect_deg'] >= 60],
                                   ['along', 'across'], 'oblique')
    for k in ('h_te_best_fit_20m', 'h_te_best_fit', 'h_te_median', 'h_te_interp'):
        seg['dz_atl08_' + k] = seg['atl08_' + k] + seg['offset_navd'] - seg['dem_elev']

    seg.attrs['epoch'] = epoch
    out = work / 'segments.parquet'
    seg.to_parquet(out, index=False)
    print(f'{len(seg):,} segments -> {out}; with DEM: {seg["dem_elev"].notna().mean():.1%}')


if __name__ == '__main__':
    main()
