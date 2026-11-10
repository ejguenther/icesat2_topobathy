"""
Pilot diagnostics for run_offsets.py output.

Writes to <work_dir>/offsets/review/:
    summary_by_spot_orientation.csv   signed error statistics, full-pass windows
    windows.csv                       every window with key diagnostics
    fig_errors_by_spot.png            along- vs cross-track error, one panel per spot
    fig_pass_consistency.png          error per beam for each pass (beam pairs)
    fig_uncertainty_vs_terrain.png    bootstrap sigma vs terrain slope
    fig_cost_surfaces.png             example cost surfaces

Usage:
    uv run python pilot_review.py --config <config>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from phoreal.config import load_config

# Palette: two validated categorical slots for orientation, one-hue ramp for
# sequential cost surfaces, recessive ink for axes/text.
ORIENT_COLOR = {'forward': '#2a78d6', 'backward': '#eb6834'}
OTHER = '#8a8984'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUES = LinearSegmentedColormap.from_list(
    'blues', ['#0d366b', '#184f95', '#2a78d6', '#5598e7', '#86b6ef', '#b7d3f6', '#cde2fb'])


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.color': INK2,
        'ytick.color': INK2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'legend.fontsize': 8, 'font.size': 9,
    })


def orient_legend(ax, extra=()):
    handles = [Line2D([], [], marker='o', ls='', color=c, markersize=7, label=k)
               for k, c in ORIENT_COLOR.items()]
    ax.legend(handles=list(handles) + list(extra), loc='upper right')


def load(cfg):
    out = Path(cfg['work_dir']) / 'offsets'
    df = pd.read_parquet(out / 'offsets.parquet')
    rev = out / 'review'; rev.mkdir(exist_ok=True)
    return df, out, rev


def summary_table(full):
    acc = full[full['accepted']]
    rows = []
    for (spot, orient), g in acc.groupby(['atlas_spot_number', 'sc_orient']):
        r = dict(spot=int(spot), sc_orient=orient, beam_type=g['atlas_beam_type'].iloc[0],
                 n_windows=len(g), n_passes=g['key'].nunique())
        for c in ('err_at', 'err_xt', 'err_e', 'err_n'):
            r[f'{c}_mean'] = g[c].mean(); r[f'{c}_median'] = g[c].median()
            r[f'{c}_sd'] = g[c].std(ddof=1) if len(g) > 1 else np.nan
        r['rmse_radial'] = float(np.sqrt(np.mean(g['err_at'] ** 2 + g['err_xt'] ** 2)))
        rows.append(r)
    return pd.DataFrame(rows)


def fig_errors_by_spot(full, path):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    lim = 10.5
    for spot, ax in zip(range(1, 7), axes.ravel()):
        g = full[full['atlas_spot_number'] == spot]
        acc, rej = g[g['accepted']], g[~g['accepted'] & g['err_at'].notna()]
        ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
        ax.scatter(rej['err_xt'], rej['err_at'], s=36, facecolors='none',
                   edgecolors=OTHER, lw=1.2, zorder=2)
        for o, c in ORIENT_COLOR.items():
            a = acc[acc['sc_orient'] == o]
            ax.errorbar(a['err_xt'], a['err_at'], xerr=a.get('boot_sigma_xt'),
                        yerr=a.get('boot_sigma_at'), fmt='o', ms=7, color=c,
                        mec=SURFACE, mew=2, elinewidth=1, capsize=0, zorder=3)
        btype = g['atlas_beam_type'].iloc[0] if len(g) else ''
        ax.set_title(f"Spot {spot} ({btype}) · {len(acc)} accepted, {len(rej)} flagged",
                     loc='left')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    for ax in axes[1]:
        ax.set_xlabel('Cross-track error (m, + = reported left of true)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Along-track error (m, + = reported ahead)')
    orient_legend(axes[0, 2], [Line2D([], [], marker='o', ls='', mfc='none', mec=OTHER,
                                      markersize=7, label='flagged')])
    fig.suptitle('Geolocation error per full-pass window, by ATLAS spot (bars: bootstrap 1σ)',
                 x=0.01, ha='left', color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_pass_consistency(full, path):
    g = full[full['err_at'].notna()].copy()
    g['date'] = pd.to_datetime(g['time_utc']).dt.strftime('%Y-%m-%d')
    g['pass'] = g['date'] + ' ' + g['aoi_label']
    passes = sorted(g['pass'].unique())
    xpos = {p: i for i, p in enumerate(passes)}
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.9 * len(passes) + 2), 7), sharex=True)
    for ax, comp, lab in ((axes[0], 'err_at', 'Along-track error (m)'),
                          (axes[1], 'err_xt', 'Cross-track error (m)')):
        ax.axhline(0, color=INK2, lw=0.8)
        for _, r in g.iterrows():
            x = xpos[r['pass']] + (r['atlas_spot_number'] - 3.5) * 0.08
            col = ORIENT_COLOR.get(r['sc_orient'], OTHER) if r['accepted'] else OTHER
            filled = r['atlas_beam_type'] == 'strong'
            ax.errorbar(x, r[comp], yerr=r.get(f"boot_sigma_{comp[-2:]}"), fmt='o', ms=7,
                        color=col, mfc=col if filled else 'none', mec=col, mew=1.5,
                        elinewidth=1, capsize=0)
            ax.annotate(str(int(r['atlas_spot_number'])), (x, r[comp]), xytext=(5, 0),
                        textcoords='offset points', fontsize=7, color=INK2, va='center')
        ax.set_ylabel(lab)
    axes[1].set_xticks(range(len(passes)))
    axes[1].set_xticklabels(passes, rotation=35, ha='right')
    orient_legend(axes[0], [
        Line2D([], [], marker='o', ls='', color=INK2, markersize=7, label='strong'),
        Line2D([], [], marker='o', ls='', mfc='none', mec=INK2, markersize=7, label='weak'),
        Line2D([], [], marker='o', ls='', color=OTHER, markersize=7, label='flagged')])
    fig.suptitle('Full-pass error per beam (labels: spot number). Beams of one pass share most pointing error.',
                 x=0.01, ha='left', color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_uncertainty_vs_terrain(df, path):
    lengths = sorted(df['window_length_m'].dropna().unique())
    fig, axes = plt.subplots(1, len(lengths), figsize=(4 * len(lengths), 3.6), sharey=True,
                             squeeze=False)
    for ax, L in zip(axes[0], lengths):
        g = df[(df['window_length_m'] == L) & df['boot_sigma_at'].notna()]
        sig = np.maximum(g['boot_sigma_at'], g['boot_sigma_xt'])
        strong = g['atlas_beam_type'] == 'strong'
        ax.scatter(g.loc[strong, 'terrain_slope_deg'], sig[strong], s=30, color='#2a78d6',
                   edgecolors=SURFACE, lw=1, label='strong')
        ax.scatter(g.loc[~strong, 'terrain_slope_deg'], sig[~strong], s=30, facecolors='none',
                   edgecolors='#2a78d6', lw=1.2, label='weak')
        ax.set_yscale('log')
        ax.set_title('full pass' if L == -1 else f'{int(L)} m windows', loc='left')
        ax.set_xlabel('Terrain slope (deg)')
    axes[0, 0].set_ylabel('Bootstrap σ, larger axis (m)')
    axes[0, -1].legend(loc='upper right')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_cost_surfaces(full, out, path, n=6):
    g = full[full['err_at'].notna()].sort_values('n_photons', ascending=False).head(n)
    if g.empty:
        return
    fig, axes = plt.subplots(1, len(g), figsize=(3.1 * len(g), 3.3), squeeze=False)
    for ax, (_, r) in zip(axes[0], g.iterrows()):
        f = out / 'cost_surfaces' / f"{Path(r['granule_03']).stem}_{r['gt']}.npz"
        if not f.exists():
            ax.set_axis_off(); continue
        z = np.load(f)
        J = z[f"L{int(r['window_length_m'])}_W{int(r['window_id'])}"]
        ax.pcolormesh(z['grid_xt'], z['grid_at'], J, cmap=BLUES, shading='nearest')
        ax.plot(r['corr_xt'], r['corr_at'], marker='+', ms=12, mew=2, color=INK)
        ax.set_title(f"{r['time_utc'][:10]} {r['gt']} (spot {int(r['atlas_spot_number'])})",
                     loc='left', fontsize=8)
        ax.set_xlabel('XT shift (m)'); ax.grid(False); ax.set_aspect('equal')
    axes[0, 0].set_ylabel('AT shift (m)')
    fig.suptitle('Cost (MAE) surfaces, full-pass windows; darker = lower cost, + = solution',
                 x=0.01, ha='left', color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    style()
    df, out, rev = load(cfg)

    gran = pd.read_parquet(Path(cfg['work_dir']) / 'granules' / 'granules.parquet')
    aoi = dict(zip(gran['key'], gran['aoi'])) if 'aoi' in gran else {}
    df['aoi_label'] = df['key'].map(aoi).fillna('').str.split('_').str[0]

    print(df['status'].value_counts().to_string())
    done = df[df['status'] == 'done']
    full = done[done['window_length_m'] == -1]

    cols = ['granule_03', 'gt', 'atlas_spot_number', 'atlas_beam_type', 'sc_orient',
            'aoi_label', 'time_utc', 'window_length_m', 'window_id', 'n_photons',
            'err_at', 'err_xt', 'err_e', 'err_n', 'boot_sigma_at', 'boot_sigma_xt',
            'cost_zero', 'cost_opt', 'dz_opt', 'valid_frac_opt', 'terrain_slope_deg',
            'terrain_roughness', 'solar_elevation_med', 'bckgrd_rate_med',
            'beam_vertical_angle_deg', 'terrain_incidence_deg', 'flags', 'accepted']
    done[[c for c in cols if c in done]].to_csv(rev / 'windows.csv', index=False)
    summ = summary_table(full)
    summ.to_csv(rev / 'summary_by_spot_orientation.csv', index=False)
    print(summ.round(2).to_string(index=False))

    fig_errors_by_spot(full, rev / 'fig_errors_by_spot.png')
    fig_pass_consistency(full, rev / 'fig_pass_consistency.png')
    fig_uncertainty_vs_terrain(done, rev / 'fig_uncertainty_vs_terrain.png')
    fig_cost_surfaces(full, out, rev / 'fig_cost_surfaces.png')
    print(f"wrote review files to {rev}")


if __name__ == '__main__':
    main()
