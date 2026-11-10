"""
Progress and ETA for the overnight run.

Usage: uv run python status.py --config local/config_poland500.yaml
"""
import argparse
import re
import time
from pathlib import Path

import pandas as pd

from phoreal.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    cfg = load_config(ap.parse_args().config)
    work = Path(cfg['work_dir']); gran = Path(cfg['granule_dir'])
    a03 = gran / cfg.get('atl03_subdir', 'atl03'); a08 = gran / cfg.get('atl08_subdir', 'atl08')
    logs = work / 'logs'
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    if (logs / 'driver.log').exists():
        print('driver:', (logs / 'driver.log').read_text().strip().splitlines()[-1])
    cand = work / 'granules' / 'candidates.parquet'
    if cand.exists():
        print(f"ATL08 files: {len(list(a08.glob('*.h5')))} / {len(pd.read_parquet(cand))} candidates")
    sel = work / 'granules' / 'granules.parquet'
    if sel.exists():
        g = pd.read_parquet(sel)
        print(f"selected: {len(g)}; ATL03 subsets present: {len(list(a03.glob('*.h5')))}")
    dl = work / 'granules' / 'atl03_download_log.csv'
    if dl.exists():
        d = pd.read_csv(dl)
        print('ATL03 downloads:', d['how'].value_counts().to_dict())
    parts = list((work / 'offsets' / 'parts').glob('*.parquet'))
    if parts and sel.exists():
        mt = sorted(p.stat().st_mtime for p in parts)
        st = pd.concat([pd.read_parquet(p, columns=['status']) for p in parts])['status'].value_counts()
        print(f"granules processed: {len(parts)} / {len(g)}; window rows by status: {st.to_dict()}")
        if len(mt) > 5:
            recent = [t for t in mt if t > mt[-1] - 3600] or mt
            rate = (len(recent) - 1) / max(recent[-1] - recent[0], 1) * 3600
            left = len(g) - len(parts)
            if rate > 0:
                eta = time.localtime(time.time() + left / rate * 3600)
                print(f"rate (last hour): {rate:.0f} granules/h; ETA {time.strftime('%a %H:%M', eta)}")
    off_log = logs / 'offsets.log'
    if off_log.exists():
        lines = [l for l in off_log.read_text().splitlines() if l.startswith('[')]
        if lines:
            print('last offset log:', lines[-1][:160])


if __name__ == '__main__':
    main()
