"""Per-beam bias and standard deviation by spacecraft orientation, plus heat maps.

Reads the joint per-pass refit output (``refit_pass.py``) and summarises each
beam's solo fit as error = -correction (AT + along travel, XT + to the left).
Fits that hit the search boundary are excluded.

Outputs (in <work_dir>/offsets/review/):
    beam_bias_by_orientation.csv
    spot_bias.csv                             (bias/SD per ATLAS spot, orientations pooled)
    fig_beam_bias_heatmap_<orientation>.png   (one per orientation)
    fig_beam_error_density.png                (AT/XT error density per beam, all time)
    fig_beam_error_density_<orientation>.png  (same, one orientation, 3x2 panels)
    bias_by_orientation_period.csv            (bias/SD per yaw-flip period, granule-bootstrap CIs)
    fig_bias_over_time.png                    (that table vs time, coloured by orientation)

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
from matplotlib.patches import Ellipse

from phoreal.config import load_config

BEAMS = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']


def beam_errors(beams):
    """Solo per-beam fits as errors (= -correction), boundary hits removed."""
    b = beams[~beams.solo_boundary & beams.solo_corr_at.notna()].copy()
    b['err_at'] = -b.solo_corr_at
    b['err_xt'] = -b.solo_corr_xt
    return b


def beam_table(beams):
    """Bias (mean) and SD of per-beam error, by orientation and beam, plus per-orientation totals."""
    b = beam_errors(beams)
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


def spot_table(beams):
    """Bias (mean) and SD of per-beam error per ATLAS spot, both orientations pooled."""
    b = beam_errors(beams)
    rows = []
    for spot, d in b.groupby('spot'):
        rows.append({'Spot': int(spot), 'Type': d.beam_type.iloc[0], 'n': len(d),
                     'AT bias': d.err_at.mean(), 'AT SD': d.err_at.std(),
                     'XT bias': d.err_xt.mean(), 'XT SD': d.err_xt.std()})
    rows.append({'Spot': 'All spots', 'n': len(b), 'AT bias': b.err_at.mean(), 'AT SD': b.err_at.std(),
                 'XT bias': b.err_xt.mean(), 'XT SD': b.err_xt.std()})
    return pd.DataFrame(rows)


def write_csv(tab, path):
    """Write the table as displayed: two decimals, explicit sign on biases."""
    t = tab.copy()
    for c in ['AT bias', 'XT bias']:
        t[c] = t[c].map('{:+.2f}'.format)
    for c in ['AT SD', 'XT SD']:
        t[c] = t[c].map('{:.2f}'.format)
    for c in ['Spot', 'n']:
        t[c] = t[c].map(lambda v: '' if pd.isna(v) else str(int(v)) if isinstance(v, (int, float)) else v)
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


def sigma_ellipse(ax, x, y, k, **kw):
    """k-sigma covariance ellipse of (x, y), centred on the mean."""
    w, v = np.linalg.eigh(np.cov(x, y))
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse((x.mean(), y.mean()), 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]),
                         angle=ang, fill=False, **kw))


def _density_setup(beams, lim, bin_m):
    """Errors, bin edges, colormap and per-(orientation, beam) histograms on one shared scale."""
    from pilot_review import BLUES, SURFACE
    b = beam_errors(beams)
    edges = np.arange(-lim, lim + bin_m / 2, bin_m)
    cmap = BLUES.reversed().copy()
    cmap.set_under(SURFACE)
    hists = {}
    for (o, gt), d in b.groupby(['sc_orient', 'gt']):
        h, _, _ = np.histogram2d(d.err_at, d.err_xt, bins=[edges, edges])
        hists[o, gt] = h / len(d) * 100  # percent of the beam's fits per bin
    vmax = max(h.max() for h in hists.values())
    return b, edges, cmap, hists, vmax


def _density_panel(ax, d, h, edges, cmap, vmax, lim):
    from pilot_review import INK, SURFACE
    mesh = ax.pcolormesh(edges, edges, h, cmap=cmap, vmin=1e-9, vmax=vmax)
    for k, ls in [(1, '-'), (2, '--')]:
        sigma_ellipse(ax, d.err_xt.values, d.err_at.values, k, color=INK, lw=1.1, ls=ls)
    ax.plot(d.err_xt.mean(), d.err_at.mean(), marker='+', ms=12, mew=2, color=INK)
    ax.axhline(0, color=INK, lw=0.5, alpha=0.4); ax.axvline(0, color=INK, lw=0.5, alpha=0.4)
    ax.set_title(f"{d['gt'].iloc[0]} (spot {int(d.spot.iloc[0])}, {d.beam_type.iloc[0]})",
                 loc='left', fontsize=9)
    ax.text(0.03, 0.03, f"bias AT {d.err_at.mean():+.2f}, XT {d.err_xt.mean():+.2f}\n"
            f"SD AT {d.err_at.std():.2f}, XT {d.err_xt.std():.2f}\nn = {len(d)}",
            transform=ax.transAxes, fontsize=7.5, color=INK, va='bottom',
            bbox=dict(facecolor=SURFACE, edgecolor='none', alpha=0.8, pad=1.5))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([-5, 0, 5]); ax.set_yticks([-5, 0, 5])
    ax.grid(False); ax.set_aspect('equal')
    return mesh


def _density_title(beams, prefix):
    t0, t1 = beams.pass_id.str[:4].min(), beams.pass_id.str[:4].max()
    return (f'{prefix}, all passes {t0}–{t1}\ndarker = more fits, '
            '+ = bias (mean), ellipses = 1σ (solid) and 2σ (dashed)')


def fig_error_density(beams, path, lim=10.0, bin_m=0.5):
    """Cost-surface-style panels: density of per-beam AT/XT error over all passes,
    one panel per beam (columns) and orientation (rows), with bias and 1/2-sigma ellipses."""
    from pilot_review import INK, style
    style()
    b, edges, cmap, hists, vmax = _density_setup(beams, lim, bin_m)
    fig, axes = plt.subplots(2, len(BEAMS), figsize=(3.1 * len(BEAMS), 6.9), squeeze=False)
    for i, orient in enumerate(['forward', 'backward']):
        for j, gt in enumerate(BEAMS):
            ax = axes[i, j]
            d = b[(b.sc_orient == orient) & (b['gt'] == gt)]
            mesh = _density_panel(ax, d, hists[orient, gt], edges, cmap, vmax, lim)
            if i == 1:
                ax.set_xlabel('XT error (m)')
            if j == 0:
                ax.set_ylabel(f'{orient.title()}\nAT error (m)')
    fig.colorbar(mesh, ax=axes, shrink=0.6, pad=0.01).set_label("% of beam's fits per 0.5 m bin")
    fig.suptitle(_density_title(beams, 'Per-beam geolocation error'), x=0.01, ha='left', color=INK)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def fig_error_density_orient(beams, orient, path, lim=10.0, bin_m=0.5):
    """One orientation, 3 rows (gt1-gt3 pairs) x 2 columns (l, r). Colour scale is shared
    with the other orientation so the two figures compare directly."""
    from pilot_review import INK, style
    style()
    b, edges, cmap, hists, vmax = _density_setup(beams, lim, bin_m)
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 10.6), squeeze=False)
    for k, gt in enumerate(BEAMS):
        i, j = divmod(k, 2)
        ax = axes[i, j]
        d = b[(b.sc_orient == orient) & (b['gt'] == gt)]
        mesh = _density_panel(ax, d, hists[orient, gt], edges, cmap, vmax, lim)
        if i == 2:
            ax.set_xlabel('XT error (m)')
        if j == 0:
            ax.set_ylabel('AT error (m)')
    fig.colorbar(mesh, ax=axes, shrink=0.5, pad=0.02).set_label("% of beam's fits per 0.5 m bin")
    fig.suptitle(_density_title(beams, f'Per-beam geolocation error, {orient} orientation'),
                 x=0.02, ha='left', color=INK, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def orientation_periods(beams, n_boot=1000, seed=0):
    """Bias and SD of all-beam error per spacecraft-orientation period (between yaw flips),
    with 95% CIs from a bootstrap over granules within the period."""
    b = beam_errors(beams)
    b['t'] = pd.to_datetime(b.pass_id.str[:14], format='%Y%m%d%H%M%S')
    b['gran'] = b.pass_id.str.rsplit('_', n=1).str[0]
    g = b.groupby('gran').agg(t=('t', 'first'), o=('sc_orient', 'first')).sort_values('t')
    g['period'] = (g.o != g.o.shift()).cumsum()
    b['period'] = b.gran.map(g.period)
    rng = np.random.default_rng(seed)
    rows = []
    for per, d in b.groupby('period'):
        grans = d.gran.unique()
        idx = d.groupby('gran').indices
        at, xt = d.err_at.values, d.err_xt.values
        boot = []
        for _ in range(n_boot):
            ii = np.concatenate([idx[k] for k in rng.choice(grans, len(grans))])
            boot.append([at[ii].mean(), xt[ii].mean(), at[ii].std(ddof=1), xt[ii].std(ddof=1)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        r = {'period': per, 'sc_orient': d.sc_orient.iloc[0], 't0': d.t.min(), 't1': d.t.max(),
             'n_granules': len(grans), 'n_beams': len(d)}
        for k, (name, v) in enumerate([('bias_at', at.mean()), ('bias_xt', xt.mean()),
                                        ('sd_at', at.std(ddof=1)), ('sd_xt', xt.std(ddof=1))]):
            r[name], r[name + '_lo'], r[name + '_hi'] = v, lo[k], hi[k]
        rows.append(r)
    return pd.DataFrame(rows)


ORIENT_COLOR = {'forward': '#2a78d6', 'backward': '#eb6834'}


def fig_bias_over_time(per, path):
    """Bias and SD per orientation period vs time: each period is a bar spanning its dates,
    with a 95% CI band; dotted lines mark the yaw flips between periods."""
    from pilot_review import INK, INK2, style
    style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), sharex=True)
    panels = [('bias_at', 'Along-track bias (m)'), ('bias_xt', 'Cross-track bias (m)'),
              ('sd_at', 'Along-track SD (m)'), ('sd_xt', 'Cross-track SD (m)')]
    flips = [per.t1.iloc[i] + (per.t0.iloc[i + 1] - per.t1.iloc[i]) / 2 for i in range(len(per) - 1)]
    for ax, (c, lab) in zip(axes.flat, panels):
        for _, r in per.iterrows():
            col = ORIENT_COLOR[r.sc_orient]
            ax.fill_between([r.t0, r.t1], r[c + '_lo'], r[c + '_hi'], color=col, alpha=0.22, lw=0)
            ax.plot([r.t0, r.t1], [r[c], r[c]], color=col, lw=2.5, solid_capstyle='round')
        for o, col in ORIENT_COLOR.items():
            v = per[per.sc_orient == o]
            m = np.average(v[c], weights=v.n_beams)
            ax.axhline(m, color=col, lw=1, ls='--', alpha=0.9)
        for f in flips:
            ax.axvline(f, color=INK2, lw=0.7, ls=':')
        if c.startswith('bias'):
            ax.axhline(0, color=INK, lw=0.6)
        ax.set_ylabel(lab)
        ax.grid(axis='x', visible=False)
    lo = min(per[['sd_at_lo', 'sd_xt_lo']].min()) - 0.2
    hi = max(per[['sd_at_hi', 'sd_xt_hi']].max()) + 0.2
    for ax in axes[1]:
        ax.set_ylim(lo, hi)
    blo = min(per[['bias_at_lo', 'bias_xt_lo']].min()) - 0.2
    bhi = max(per[['bias_at_hi', 'bias_xt_hi']].max()) + 0.2
    for ax in axes[0]:
        ax.set_ylim(blo, bhi)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=col, lw=2.5, label=o.title()) for o, col in ORIENT_COLOR.items()]
    handles.append(Line2D([], [], color=INK2, lw=0.7, ls=':', label='Yaw flip (between sampled granules)'))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.985), ncol=3, fontsize=8)
    fig.suptitle('ICESat-2 geolocation error vs ALS by spacecraft-orientation period, all beams pooled\n'
                 'Bar = period mean over its date span; band = 95% CI (granule bootstrap); '
                 'dashed = mean over all periods of that orientation. AT + along travel, XT + to the left',
                 x=0.01, ha='left', color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--refit', default='pass_refit', help='refit output folder under offsets/')
    args = ap.parse_args()
    off_dir = Path(load_config(args.config)['work_dir']) / 'offsets'
    out = off_dir / 'review'
    out.mkdir(parents=True, exist_ok=True)
    beams = pd.read_parquet(off_dir / args.refit / 'beams.parquet')
    tab = beam_table(beams)
    write_csv(tab, out / 'beam_bias_by_orientation.csv')
    write_csv(spot_table(beams), out / 'spot_bias.csv')
    for orient in ['Forward', 'Backward']:
        heatmap(tab, orient, out / f'fig_beam_bias_heatmap_{orient.lower()}.png')
    fig_error_density(beams, out / 'fig_beam_error_density.png')
    for orient in ['forward', 'backward']:
        fig_error_density_orient(beams, orient, out / f'fig_beam_error_density_{orient}.png')
    per = orientation_periods(beams)
    per.round({c: 3 for c in per.select_dtypes('number')}).to_csv(out / 'bias_by_orientation_period.csv', index=False)
    fig_bias_over_time(per, out / 'fig_bias_over_time.png')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
