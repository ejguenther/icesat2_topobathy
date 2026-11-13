"""Build one photon parquet per beam with ATL08 and ATL24 labels side by side.

Each row is one ATL03 photon. The ATL24 CSV is joined row for row (checked
against ATL03 time and height in ``read_atl24_csv``). ``segment_id`` is the
ATL03 ~20 m geolocation segment, the common segment key for both classifiers.

Also writes QA tables to <work_dir>/qa/:
  crosstab_<gt>.csv      ATL24 class x ATL08 class photon counts
  prob_order.csv         share of photons whose argmax(atl24_prob_k) maps to class_ph
  terrain_agreement.csv  photon-level terrain agreement per beam

Run from this folder:  uv run python build_photons.py [--beams gt1r ...]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from phoreal.config import load_config
from phoreal.io import processing
from phoreal.io.readers import read_atl24_csv, get_atl03_metadata

HERE = Path(__file__).resolve().parent
ATL24_CODES = [0, 1, 2, 40, 41]     # assumed order of atl24_prob_0..4 (checked in prob_order.csv)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def photon_segment_ids(f, gt, n_ph):
    """ATL03 segment_id and segment_dist_x for every photon (vectorised)."""
    g = f[gt + '/geolocation']
    seg_id = g['segment_id'][:]
    cnt = g['segment_ph_cnt'][:].astype(np.int64)
    beg = g['ph_index_beg'][:].astype(np.int64)
    dist_x = g['segment_dist_x'][:]
    ok = (cnt > 0) & (beg > 0)
    # photons are stored segment after segment, so the repeat lines up with ATL03 order
    if not np.array_equal(beg[ok] - 1, np.concatenate(([0], np.cumsum(cnt[ok])[:-1]))) \
            or cnt[ok].sum() != n_ph:
        raise ValueError(f"{gt}: ATL03 segments do not tile the photon array")
    return np.repeat(seg_id[ok], cnt[ok]), np.repeat(dist_x[ok], cnt[ok])


def build_beam(cfg, gt, work):
    a03, a08 = resolve(cfg['granule']['atl03']), resolve(cfg['granule']['atl08'])
    csv = resolve(cfg['granule']['atl24_csv'].format(gt=gt))
    with h5py.File(a03, 'r') as f:
        h = f[gt + '/heights']
        df = pd.DataFrame({
            'latitude': h['lat_ph'][:], 'longitude': h['lon_ph'][:], 'h_ph': h['h_ph'][:],
            'delta_time': h['delta_time'][:], 'quality_ph': h['quality_ph'][:].astype(np.int8),
            'signal_conf_ph0': h['signal_conf_ph'][:, 0].astype(np.int8),
        })
        seg, dist_x = photon_segment_ids(f, gt, len(df))
        df['segment_id'] = seg.astype(np.int32)
        df['alongtrack'] = dist_x + h['dist_ph_along'][:]
    df['atl08_class'] = processing.get_atl08_class_to_atl03(str(a03), str(a08), gt).astype(np.int8)
    df24 = read_atl24_csv(csv, a03, gt, cache_file=work / 'cache' / f'atl24_{gt}.parquet')
    df['atl24_class'] = df24['atl24_class'].to_numpy()
    for k in range(5):
        df[f'atl24_prob_{k}'] = df24[f'atl24_prob_{k}'].to_numpy()

    meta = get_atl03_metadata(str(a03), gt)
    df.attrs.update({'gt': gt, 'beam_type': meta['beam_type'], 'spot_number': meta['spot_number'],
                     'rgt': meta['rgt'], 'cycle': meta['cycle'], 'sc_orient': meta['orientation']})
    out = work / 'photons' / f'photons_{gt}.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df, meta


def qa(df, gt, meta, terrain_code):
    ct = pd.crosstab(df['atl24_class'], df['atl08_class'])
    probs = df[[f'atl24_prob_{k}' for k in range(5)]].to_numpy()
    argmax_code = np.array(ATL24_CODES)[probs.argmax(axis=1)]
    order = {'gt': gt, 'n': len(df), 'argmax_matches_class': float(np.mean(argmax_code == df['atl24_class']))}
    for c in ATL24_CODES:                       # per class, for diagnosing a different order
        m = df['atl24_class'].to_numpy() == c
        order[f'class{c}_argmax'] = (pd.Series(argmax_code[m]).mode().iloc[0] if m.any() else np.nan)
    t08 = df['atl08_class'].to_numpy() == 1
    t24 = df['atl24_class'].to_numpy() == terrain_code
    agree = {'gt': gt, 'beam_type': meta['beam_type'], 'n_photons': len(df),
             'atl08_ground': int(t08.sum()), 'atl24_terrain': int(t24.sum()),
             'both': int((t08 & t24).sum()), 'atl08_only': int((t08 & ~t24).sum()),
             'atl24_only': int((~t08 & t24).sum())}
    agree['jaccard'] = agree['both'] / max(1, agree['both'] + agree['atl08_only'] + agree['atl24_only'])
    return ct, order, agree


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=HERE / 'config.yaml')
    ap.add_argument('--beams', nargs='*')
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = resolve(cfg['work_dir'])
    terrain = cfg['atl24_classes']['terrain']
    orders, agrees = [], []
    for gt in args.beams or cfg['beams']:
        df, meta = build_beam(cfg, gt, work)
        ct, order, agree = qa(df, gt, meta, terrain)
        (work / 'qa').mkdir(parents=True, exist_ok=True)
        ct.to_csv(work / 'qa' / f'crosstab_{gt}.csv')
        orders.append(order); agrees.append(agree)
        print(f"{gt} ({meta['beam_type']}): {len(df):,} photons, "
              f"prob argmax == class {order['argmax_matches_class']:.3f}, terrain Jaccard {agree['jaccard']:.3f}")
        del df
    pd.DataFrame(orders).to_csv(work / 'qa' / 'prob_order.csv', index=False)
    pd.DataFrame(agrees).to_csv(work / 'qa' / 'terrain_agreement.csv', index=False)


if __name__ == '__main__':
    main()
