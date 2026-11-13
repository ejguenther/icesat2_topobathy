"""Render the summary tables as LaTeX (booktabs) PNGs for slides.

Reads outputs/results/metrics_overall.csv and shean_metric.csv (from analyze.py)
and writes, in outputs/results/:
  table_error_metrics.{tex,png}   bias / MAE / RMSE / NMAD, all vs mountain paired segments
  table_shean_metric.{tex,png}    median dz at 17.5-22.5 deg minus median below 3 deg

Needs latex and dvipng on the PATH.  Run from this folder:
  uv run python make_tables.py [--dpi 400]
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from phoreal.config import load_config

HERE = Path(__file__).resolve().parent
NAMES = {'a08': 'ATL08 ground', 'a24': 'ATL24 terrain', 'sig': 'All signal',
         'te20': r'ATL08 \texttt{h\_te\_best\_fit\_20m}'}

PREAMBLE = r"""\documentclass{article}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{booktabs}
\pagestyle{empty}
\begin{document}
"""


def signed(v):
    """+0.12 / --0.12 (en dash as minus, so the sign matches the digits' width)."""
    return f'{v:+.2f}'.replace('-', '--')


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def error_table(m):
    est = ['a08', 'a24', 'sig']
    subsets = [('all_paired', 'All terrain'), ('mountain_paired', 'Mountains')]
    m = m.set_index(['subset', 'estimator'])
    head = ' & '.join(rf'\multicolumn{{3}}{{c}}{{{lab} (n = {int(m.loc[(s, "a08"), "n"]):,})}}'
                      for s, lab in subsets)
    sub = ' & '.join(NAMES[e].replace(' ground', '').replace(' terrain', '') for e in est)
    rows = []
    for col, lab in (('bias_median', 'Bias (median)'), ('mae', 'MAE'), ('rmse', 'RMSE'), ('nmad', 'NMAD')):
        vals = [signed(m.loc[(s, e), col]) if col == 'bias_median' else f'{m.loc[(s, e), col]:.2f}'
                for s, _ in subsets for e in est]
        rows.append(f'{lab} & ' + ' & '.join(vals) + r' \\')
    return '\n'.join([
        r'\begin{tabular}{l rrr @{\hspace{1.5em}} rrr}', r'\toprule',
        rf'(m) & {head} \\', r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}',
        rf' & {sub} & {sub} \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}'])


def shean_table(s):
    s = s.set_index(['subset', 'estimator'])
    cols = [('all_paired', 'All'), ('snowfree', 'Snow-free')]
    head = ' & '.join(rf'{lab} (n = {int(s.loc[(c, "a08"), "n_20deg"]):,})' for c, lab in cols)
    rows = [f'{NAMES[e]} & ' + ' & '.join(signed(s.loc[(c, e), "dz_20deg_minus_flat"]) for c, _ in cols) + r' \\'
            for e in ('a08', 'te20', 'a24', 'sig')]
    return '\n'.join([
        r'\begin{tabular}{l rr}', r'\toprule',
        rf'Bias at 20$^\circ$ minus flat (m) & {head} \\', r'\midrule',
        *rows, r'\bottomrule', r'\end{tabular}'])


def render(body, out_png, dpi):
    out_png.with_suffix('.tex').write_text(body + '\n')
    with tempfile.TemporaryDirectory() as tmp:
        tex = Path(tmp) / 't.tex'
        tex.write_text(PREAMBLE + body + '\n\\end{document}\n')
        subprocess.run(['latex', '-interaction=nonstopmode', '-halt-on-error', tex.name],
                       cwd=tmp, check=True, capture_output=True)
        subprocess.run(['dvipng', '-D', str(dpi), '-T', 'tight', '-bg', 'White', '-o', 'out.png', 't.dvi'],
                       cwd=tmp, check=True, capture_output=True)
        shutil.copy(Path(tmp) / 'out.png', out_png)
    print(f'-> {out_png}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=HERE / 'config.yaml')
    ap.add_argument('--dpi', type=int, default=400)
    args = ap.parse_args()
    res = resolve(load_config(args.config)['work_dir']) / 'results'
    render(error_table(pd.read_csv(res / 'metrics_overall.csv')), res / 'table_error_metrics.png', args.dpi)
    render(shean_table(pd.read_csv(res / 'shean_metric.csv')), res / 'table_shean_metric.png', args.dpi)


if __name__ == '__main__':
    main()
