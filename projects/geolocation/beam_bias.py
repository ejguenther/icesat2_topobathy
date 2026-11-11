"""Per-beam bias and standard deviation by spacecraft orientation, plus heat maps.

Reads the joint per-pass refit output (``refit_pass.py``) and summarises each
beam's solo fit as error = -correction (AT + along travel, XT + to the left).
Fits that hit the search boundary are excluded.

Outputs (in <work_dir>/offsets/review/):
    beam_bias_by_orientation.csv
    fig_beam_bias_heatmap_<orientation>.png   (one per orientation)

Usage:
    uv run python beam_bias.py --config CFG [--refit pass_refit]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from phoreal.config import load_config

BEAMS = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']


def beam_table(beams):
    """Bias (mean) and SD of per-beam error, by orientation and beam, plus per-orientation totals."""
    b = beams[~beams.solo_boundary & beams.solo_corr_at.notna()].copy()
    b['err_at'] = -b.solo_corr_at
    b['err_xt'] = -b.solo_corr_xt
    rows = []
    for orient in ['forward', 'backward']:
        o = b[b.sc_orient == orient]
        for gt in BEAMS:
            d = o[o['gt'] == gt]
            rows.append({'Orientation': orient.title(), 'Beam': gt, 'Spot': int(d.spot.iloc[0]),
                         'Type': d.beam_type.iloc[0], 'n': len(d),
                         'AT bias': d.err_at.mean(), 'AT SD': d.err_at.std(),
                         'XT bias': d.err_xt.mean(), 'XT SD': d.err_xt.std()})
    for orient in ['forward', 'backward']:
        o = b[b.sc_orient == orient]
        rows.append({'Orientation': f'{orient.title()}, all beams',
                     'AT bias': o.err_at.mean(), 'AT SD': o.err_at.std(),
                     'XT bias': o.err_xt.mean(), 'XT SD': o.err_xt.std()})
    return pd.DataFrame(rows)


def write_csv(tab, path):
    """Write the table as displayed: two decimals, explicit sign on biases."""
    t = tab.copy()
    for c in ['AT bias', 'XT bias']:
        t[c] = t[c].map('{:+.2f}'.format)
    for c in ['AT SD', 'XT SD']:
        t[c] = t[c].map('{:.2f}'.format)
    for c in ['Spot', 'n']:
        t[c] = t[c].map(lambda v: '' if pd.isna(v) else str(int(v)))
    t.to_csv(path, index=False)


def heatmap(tab, orient, path):
    d = tab[tab.Orientation == orient].set_index('Beam').loc[BEAMS]
    bias = d[['AT bias', 'XT bias']].T.values.astype(float)
    sd = d[['AT SD', 'XT SD']].T.values.astype(float)
    xl = [f'{g}\nspot {int(s)}\n{t}' for g, s, t in zip(d.index, d.Spot, d.Type)]
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.2), constrained_layout=True)
    lim = np.ceil(np.abs(bias).max() * 4) / 4
    specs = [(axs[0], bias, plt.get_cmap('RdBu_r'), TwoSlopeNorm(0, -lim, lim), 'Bias (m)', '{:+.2f}'),
             (axs[1], sd, plt.get_cmap('Purples'), plt.Normalize(0, np.ceil(sd.max())),
              'Standard deviation (m)', '{:.2f}')]
    for ax, z, cm, nm, lab, fmt in specs:
        im = ax.imshow(z, cmap=cm, norm=nm, aspect='auto')
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                r, g, b, _ = cm(nm(z[i, j]))
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(j, i, fmt.format(z[i, j]), ha='center', va='center', fontsize=11,
                        color='white' if lum < 0.5 else '#1a1a1a')
        ax.set_xticks(range(len(BEAMS)), xl, fontsize=9)
        ax.set_yticks(range(2), ['Along-track', 'Cross-track'])
        ax.set_xticks(np.arange(-.5, len(BEAMS)), minor=True)
        ax.set_yticks(np.arange(-.5, 2), minor=True)
        ax.grid(which='minor', color='white', lw=2)
        ax.tick_params(which='both', length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, label=lab, shrink=0.9, pad=0.01)
    n = d.n.astype(int)
    fig.suptitle(f'ICESat-2 geolocation error vs ALS by beam — {orient.lower()} orientation\n'
                 f'Per-beam fits over ≤20 km chunks (n = {n.min()}–{n.max()} per beam). '
                 'AT + along travel, XT + to the left', fontsize=10.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--refit', default='pass_refit', help='refit output folder under offsets/')
    args = ap.parse_args()
    off_dir = Path(load_config(args.config)['work_dir']) / 'offsets'
    out = off_dir / 'review'
    out.mkdir(parents=True, exist_ok=True)
    tab = beam_table(pd.read_parquet(off_dir / args.refit / 'beams.parquet'))
    write_csv(tab, out / 'beam_bias_by_orientation.csv')
    for orient in ['Forward', 'Backward']:
        heatmap(tab, orient, out / f'fig_beam_bias_heatmap_{orient.lower()}.png')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
