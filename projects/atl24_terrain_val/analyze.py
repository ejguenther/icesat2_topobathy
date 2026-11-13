"""Error metrics, coverage and figures from segments.parquet.

Tables (<work_dir>/results/*.csv):
  metrics_overall.csv    bias (median), mean, MAE, RMSE, NMAD per estimator and subset
  metrics_by_slope.csv   the same per slope bin (paired segments)
  metrics_by_group.csv   per beam type and per slope orientation (slope >= 10 deg)
  shean_metric.csv       median dz at 17.5-22.5 deg minus median below 3 deg
  coverage_by_slope.csv  segments where both / only ATL24 / only ATL08 / neither find terrain
  detection_rate.csv     % of valid-terrain segments detected by ATL08 vs ATL24 (overall, beam, slope)
  method_compare.csv     photon-level dz vs segment-centre dz, per estimator
Figures (<work_dir>/results/fig_*.png); fig_dz_vs_slope_no_te20*.png omit the ATL08 20 m fit.

"Paired" = segments where ATL08 and ATL24 both have >= analysis.paired_min
terrain photons, so every estimator is measured on the same segments.
Run from this folder:  uv run python analyze.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phoreal.config import load_config

HERE = Path(__file__).resolve().parent

# estimator -> (column, label, colour, linestyle). Categorical slots 1-3 of the
# validated palette; the official product is a neutral dashed line.
EST = {
    'sig': ('dz_sig', 'All signal photons', '#2a78d6', '-'),
    'a08': ('dz_a08', 'ATL08 ground photons', '#eb6834', '-'),
    'a24': ('dz_a24', 'ATL24 terrain photons', '#1baf7a', '-'),
    'te20': ('dz_atl08_h_te_best_fit_20m', 'ATL08 h_te_best_fit_20m', '#52514e', '--'),
}
EST_100M = {
    'te_bf': 'dz_atl08_h_te_best_fit', 'te_med': 'dz_atl08_h_te_median', 'te_int': 'dz_atl08_h_te_interp'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def stats(dz):
    dz = pd.Series(dz).dropna()
    if dz.empty:
        return pd.Series({'n': 0})
    med = dz.median()
    return pd.Series({'n': len(dz), 'bias_median': med, 'mean': dz.mean(),
                      'mae': dz.abs().mean(), 'rmse': np.sqrt((dz ** 2).mean()),
                      'nmad': 1.4826 * (dz - med).abs().median()})


def table(df, cols, by=None):
    rows = []
    groups = [((), df)] if by is None else df.groupby(by, observed=True)
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for name, col in cols.items():
            r = stats(g[col]).astype(object)
            r['estimator'] = name
            if by is not None:
                for b, k in zip([by] if isinstance(by, str) else by, key):
                    r[b] = k
            rows.append(r)
    return pd.DataFrame(rows)


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def fig_dz_vs_slope(p, bins, centers, out, title, est=EST, vertical=False,
                    groups=('all', 'strong', 'weak')):
    every = {'all': ('All beams', p), 'strong': ('Strong beams', p[p.beam_type == 'strong']),
             'weak': ('Weak beams', p[p.beam_type == 'weak'])}
    panels = [every[g] for g in groups]
    m = len(panels)
    if vertical:
        fig, axes = plt.subplots(m, 1, figsize=(8, 3.7 * m), sharex=True, sharey=True, squeeze=False)
    else:
        fig, axes = plt.subplots(1, m, figsize=(5 * m, 4.8), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, (name, d) in zip(axes, panels):
        sb = pd.cut(d.slope_deg, bins)
        for k, (col, lab, c, ls) in est.items():
            g = d.groupby(sb, observed=False)[col]
            med, q1, q3, n = g.median(), g.quantile(0.25), g.quantile(0.75), g.count()
            ok = (n >= 20).to_numpy()
            ax.plot(centers[ok], med.to_numpy()[ok], ls, color=c, lw=2, marker='o', ms=4, label=lab)
            if k in ('a08', 'a24', 'sig'):
                ax.fill_between(centers[ok], q1.to_numpy()[ok], q3.to_numpy()[ok], color=c, alpha=0.12, lw=0)
            if ok.any():
                i = np.flatnonzero(ok)[-1]
                ax.annotate(lab.split(' ')[0] + (' fit' if k == 'te20' else ''), (centers[i], med.iloc[i]),
                            xytext=(4, 0), textcoords='offset points', fontsize=8, color=INK, va='center')
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(f'{name} (n = {len(d):,})', fontsize=10, color=INK)
        if not vertical or ax is axes[-1]:
            ax.set_xlabel('3DEP slope, 40 m window (deg)')
        if vertical:
            ax.set_ylabel('median height - 3DEP (m)')
        style(ax)
    axes[0].set_ylabel('median height - 3DEP 1 m DTM (m)\nband: 25th-75th percentile')
    axes[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_coverage(cov, out, n_min):
    c = cov[cov.min_photons == n_min].set_index('slope_bin')
    parts = [('both', 'Both find terrain', '#2a78d6'), ('a24_only', 'Only ATL24', '#1baf7a'),
             ('a08_only', 'Only ATL08', '#eb6834'), ('neither', 'Neither', '#c3c2b7')]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(c))
    bottom = np.zeros(len(c))
    for k, lab, col in parts:
        v = c[f'frac_{k}'].to_numpy()
        ax.bar(x, v, bottom=bottom, color=col, edgecolor='white', linewidth=2, label=lab, width=0.8)
        bottom += v
    ax.set_xticks(x, [str(i) for i in c.index], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('share of segments with a surface return')
    ax.set_xlabel('3DEP slope bin (deg)')
    ax.set_title(f'Which classifier finds terrain (>= {n_min} photons) in a 20 m segment', fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.28))
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_below_peak(p, bins, centers, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sb = pd.cut(p.slope_deg, bins)
    for k in ('a08', 'a24'):
        _, lab, c, _ = EST[k]
        g = p.groupby(sb, observed=False)[f'below_{k}']
        med, n = g.mean(), g.count()
        ok = (n >= 20).to_numpy()
        ax.plot(centers[ok], med.to_numpy()[ok], color=c, lw=2, marker='o', ms=4, label=lab)
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.set_xlabel('3DEP slope, 40 m window (deg)')
    ax.set_ylabel('share of photons below the signal peak')
    ax.set_title('Terrain photons below the surface-return peak (0.5 = centred)', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_profiles(seg, work, out_dir, n_min, n_sections=3, length_m=1500):
    """
    Steep strong-beam sections where both classifiers have terrain: photon
    heights with the 3DEP line (context), then photon height - 3DEP for the
    ATL08 ground and ATL24 terrain selections against all signal photons.
    """
    s = seg[(seg.beam_type == 'strong')].sort_values(['gt', 'segment_id']).copy()
    paired = ((s.n_a08 >= n_min) & (s.n_a24 >= n_min)).astype(float)
    s['steep'] = s.groupby('gt')['slope_deg'].transform(lambda v: v.rolling(75, center=True).median())
    s['paired_frac'] = paired.groupby(s['gt']).transform(lambda v: v.rolling(75, center=True).mean())
    cand = s[s.paired_frac >= 0.6].sort_values('steep', ascending=False)
    picks = []
    for _, r in cand.iterrows():
        if len(picks) == n_sections:
            break
        if all(r['gt'] != p['gt'] or abs(r['segment_id'] - p['segment_id']) > 5000 for p in picks):
            picks.append(r)
    pdz = pd.read_parquet(work / 'photons_dz.parquet')
    for i, r in enumerate(picks, 1):
        ids = (r['segment_id'] - length_m // 40, r['segment_id'] + length_m // 40)
        ph = pdz[(pdz['gt'] == r['gt']) & pdz.segment_id.between(*ids)]
        sg = seg[(seg['gt'] == r['gt']) & seg.segment_id.between(*ids)].sort_values('segment_id')
        at0 = ph.alongtrack.min()
        seg_at = ph.groupby('segment_id').alongtrack.median().reindex(sg.segment_id).to_numpy() - at0
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                                 gridspec_kw={'height_ratios': [1.2, 1, 1]})
        sig = ph[ph.sig]
        axes[0].scatter(sig.alongtrack - at0, sig.h_navd, s=1, color='#c3c2b7', rasterized=True,
                        label='signal photons')
        axes[0].plot(seg_at, sg.dem_elev, color=INK, lw=1.2, label='3DEP 1 m DTM')
        axes[0].set_ylabel('height NAVD88 (m)')
        axes[0].legend(fontsize=8, frameon=False, loc='upper right', markerscale=4)
        for ax, (k, lab, c) in zip(axes[1:], [('a08', 'ATL08 ground', '#eb6834'),
                                              ('a24', 'ATL24 terrain', '#1baf7a')]):
            ax.scatter(sig.alongtrack - at0, sig.dz, s=1, color='#c3c2b7', rasterized=True, label='signal photons')
            g = ph[ph[k]]
            ax.scatter(g.alongtrack - at0, g.dz, s=2, color=c, rasterized=True, label=f'{lab} photons')
            ax.plot(seg_at, sg[f'dz_{k}'], color=INK, lw=1.2, label=f'{lab}: segment median')
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_ylim(-8, 8)
            ax.set_ylabel('photon height - 3DEP (m)')
            ax.legend(fontsize=8, frameon=False, loc='upper right', markerscale=4, ncol=3)
        for ax in axes:
            style(ax)
        axes[2].set_xlabel('along-track distance (m)')
        fig.suptitle(f"{r['gt']}, segments {ids[0]}-{ids[1]}, lat {r['latitude']:.3f}, "
                     f"lon {r['longitude']:.3f}, median slope {r['steep']:.0f} deg", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f'fig_profile_{i}.png', dpi=130)
        plt.close(fig)


def detection_table(seg, n):
    """% of valid-terrain segments (>= n signal photons, valid 3DEP) where each classifier
    labels >= n terrain photons; overall, per beam type and per slope bin."""
    d = seg[seg.n_sig >= n]
    f08, f24 = d.n_a08 >= n, d.n_a24 >= n
    flags = pd.DataFrame({'ATL08 %': f08, 'ATL24 %': f24, 'Both %': f08 & f24,
                          'ATL08 only %': f08 & ~f24, 'ATL24 only %': f24 & ~f08,
                          'Neither %': ~f08 & ~f24}, index=d.index) * 100
    parts = [flags.mean().to_frame('All').T.assign(Segments=len(d))]
    for bt, v in flags.groupby(d.beam_type):
        parts.append(v.mean().to_frame(f'{bt.title()} beams').T.assign(Segments=len(v)))
    g = flags.groupby(d.slope_bin, observed=False)
    sb = g.mean().assign(Segments=g.size())
    sb.index = [f'Slope {i.left:g}-{i.right:g} deg' if i.right < 90 else f'Slope > {i.left:g} deg'
                for i in sb.index]
    parts.append(sb)
    t = pd.concat(parts)
    t['ATL24 - ATL08 (pts)'] = t['ATL24 %'] - t['ATL08 %']
    t.index.name = 'Group'
    cols = ['Segments', 'ATL08 %', 'ATL24 %', 'ATL24 - ATL08 (pts)', 'Both %',
            'ATL08 only %', 'ATL24 only %', 'Neither %']
    return t[cols].round(1).reset_index()


def fig_detection_by_slope(seg, n, centers, out):
    """% of valid-terrain segments where ATL08 / ATL24 detect terrain, per slope bin,
    strong vs weak beams. Colour = classifier, line style = beam type."""
    d = seg[seg.n_sig >= n]
    fig, ax = plt.subplots(figsize=(9, 5))
    cls = [('ATL08', 'n_a08', EST['a08'][2]), ('ATL24', 'n_a24', EST['a24'][2])]
    beams = [('strong', '-', 'o'), ('weak', '--', 's')]
    for name, col, c in cls:
        for bt, ls, mk in beams:
            v = d[d.beam_type == bt]
            g = (v[col] >= n).groupby(v.slope_bin, observed=False)
            pct, cnt = g.mean().to_numpy() * 100, g.size().to_numpy()
            ok = cnt >= 20
            ax.plot(centers[ok], pct[ok], ls, color=c, lw=2, marker=mk, ms=5,
                    mfc=c if bt == 'strong' else 'white', mew=1.5, label=f'{name}, {bt} beams')
            i = np.flatnonzero(ok)[-1]
            ax.annotate(f'{name} {bt}', (centers[i], pct[i]), xytext=(5, 0), textcoords='offset points',
                        fontsize=8, color=INK, va='center')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, centers[-1] + 4.5)
    ax.set_xlabel('3DEP slope, 40 m window (deg); last point = slopes > 35 deg')
    ax.set_ylabel(f'segments with terrain detected (%)\n(>= {n} terrain photons)')
    ax.set_title(f'Terrain detection by slope, strong vs weak beams\n'
                 f'share of valid-terrain 20 m segments (>= {n} signal photons, valid 3DEP)',
                 fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc='lower left')
    style(ax)
    ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=HERE / 'config.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    a = cfg['analysis']
    work = resolve(cfg['work_dir'])
    out = work / 'results'
    out.mkdir(parents=True, exist_ok=True)

    seg = pd.read_parquet(work / 'segments.parquet')
    seg = seg[seg.dem_elev.notna() & seg.slope_deg.notna()].copy()
    n_min = a.get('paired_min', 5)
    step = a['slope_bin_deg']
    bins = np.r_[np.arange(0, 35, step), 35, 90]
    centers = np.r_[np.arange(0, 35, step) + step / 2, 37.5]
    seg['slope_bin'] = pd.cut(seg.slope_deg, bins)
    mtn = a['mountain']
    seg['mountain'] = (seg.dem_elev >= mtn['min_elev_m']) & (seg.slope_deg >= mtn['min_slope_deg'])
    seg['snow'] = seg.segment_snowcover == 2          # ATL08 (IMS) snow flag; snow is not removed
    paired = seg[(seg.n_a08 >= n_min) & (seg.n_a24 >= n_min)]
    cols = {k: v[0] for k, v in EST.items()} | EST_100M

    subsets = {'all_paired': paired, 'mountain_paired': paired[paired.mountain],
               'snow_paired': paired[paired.snow], 'snowfree_paired': paired[~paired.snow],
               'a08_own_coverage': seg[seg.n_a08 >= n_min], 'a24_own_coverage': seg[seg.n_a24 >= n_min]}
    overall = pd.concat([table(d, cols).assign(subset=k) for k, d in subsets.items()])
    overall.to_csv(out / 'metrics_overall.csv', index=False)
    table(paired, cols, 'slope_bin').to_csv(out / 'metrics_by_slope.csv', index=False)
    steep = paired[paired.slope_deg >= 10]
    pd.concat([table(paired, cols, 'beam_type').assign(group='beam_type'),
               table(steep, cols, 'orientation').assign(group='orientation, slope>=10'),
               table(steep, cols, ['beam_type', 'orientation']).assign(group='beam x orientation, slope>=10')]
              ).to_csv(out / 'metrics_by_group.csv', index=False)

    # Shean: median dz on 17.5-22.5 deg minus median below 3 deg
    rows = []
    for name, d in {'all_paired': paired, 'strong': paired[paired.beam_type == 'strong'],
                    'weak': paired[paired.beam_type == 'weak'], 'snowfree': paired[~paired.snow],
                    'across': paired[paired.orientation == 'across'],
                    'along': paired[paired.orientation == 'along']}.items():
        flat, s20 = d[d.slope_deg < 3], d[d.slope_deg.between(17.5, 22.5)]
        for k, c in cols.items():
            rows.append({'subset': name, 'estimator': k, 'n_flat': flat[c].notna().sum(),
                         'n_20deg': s20[c].notna().sum(),
                         'dz_20deg_minus_flat': s20[c].median() - flat[c].median()})
    pd.DataFrame(rows).to_csv(out / 'shean_metric.csv', index=False)

    # coverage among segments with a surface return
    cov = []
    for n in a['min_photons']:
        d = seg[seg.n_sig >= n]
        f08, f24 = d.n_a08 >= n, d.n_a24 >= n
        cat = pd.DataFrame({'both': f08 & f24, 'a24_only': f24 & ~f08, 'a08_only': f08 & ~f24,
                            'neither': ~f08 & ~f24})
        g = cat.groupby(d.slope_bin, observed=False)
        c = g.mean().add_prefix('frac_').join(g.size().rename('n_segments'))
        cov.append(c.reset_index().assign(min_photons=n))
    cov = pd.concat(cov)
    cov['slope_bin'] = cov.slope_bin.astype(str)
    cov.to_csv(out / 'coverage_by_slope.csv', index=False)
    detection_table(seg, n_min).to_csv(out / 'detection_rate.csv', index=False)
    fig_detection_by_slope(seg, n_min, centers, out / 'fig_detection_by_slope.png')

    # per-photon dz vs segment-centre dz
    mc = []
    for k in ('a08', 'a24', 'sig'):
        for method, c in (('photon', f'dz_{k}'), ('segment_centre', f'dzc_{k}')):
            r = table(paired, {k: c}, 'slope_bin').assign(method=method)
            mc.append(r)
    pd.concat(mc).to_csv(out / 'method_compare.csv', index=False)

    fig_dz_vs_slope(paired, bins, centers, out / 'fig_dz_vs_slope.png',
                    f'Height - 3DEP vs slope, paired segments (both classifiers >= {n_min} terrain photons)')
    fig_dz_vs_slope(paired, bins, centers, out / 'fig_dz_vs_slope_no_te20.png',
                    f'Height - 3DEP vs slope, paired segments (both classifiers >= {n_min} terrain photons)',
                    est={k: v for k, v in EST.items() if k != 'te20'})
    fig_dz_vs_slope(paired, bins, centers, out / 'fig_dz_vs_slope_no_te20_vertical.png',
                    f'Height - 3DEP vs slope, paired segments\n(both classifiers >= {n_min} terrain photons)',
                    est={k: v for k, v in EST.items() if k != 'te20'}, vertical=True)
    fig_dz_vs_slope(paired, bins, centers, out / 'fig_dz_vs_slope_no_te20_strong_weak.png',
                    f'Height - 3DEP vs slope, paired segments (both classifiers >= {n_min} terrain photons)',
                    est={k: v for k, v in EST.items() if k != 'te20'}, groups=('strong', 'weak'))
    fig_dz_vs_slope(paired[paired.mountain], bins, centers, out / 'fig_dz_vs_slope_mountain.png',
                    f"Mountain subset (DEM >= {mtn['min_elev_m']} m, slope >= {mtn['min_slope_deg']} deg)")
    fig_coverage(cov, out / 'fig_coverage.png', n_min)
    fig_below_peak(paired, bins, centers, out / 'fig_below_peak.png')
    fig_profiles(seg, work, out, n_min)

    pd.set_option('display.width', 160)
    show = overall[overall.subset.isin(['all_paired', 'mountain_paired'])]
    print(show.pivot(index='estimator', columns='subset', values=['bias_median', 'rmse', 'nmad']).round(2))
    print(pd.DataFrame(rows).query("subset == 'all_paired'").round(2).to_string(index=False))


if __name__ == '__main__':
    main()
