"""ATL24 per-beam CSV reader: row-for-row join to ATL03, checks and cache."""
import h5py
import numpy as np
import pandas as pd
import pytest

from phoreal.io.readers import ATLAS_SDP_GPS_EPOCH, read_atl24_csv

GT = 'gt1r'
N = 50


def _write_pair(tmp_path, h_shift=0.0, n_csv=N):
    rng = np.random.default_rng(0)
    dt = 2.25e8 + np.arange(N) * 1e-4
    h = rng.normal(1500, 5, N).astype(np.float32)
    a03 = tmp_path / 'ATL03_test.h5'
    with h5py.File(a03, 'w') as f:
        f[GT + '/heights/delta_time'] = dt
        f[GT + '/heights/h_ph'] = h
    cls = rng.choice([0, 1, 2, 40, 41], N)
    df = pd.DataFrame({
        'GPS_SECONDS': dt + ATLAS_SDP_GPS_EPOCH, 'lat_ph': 40.0, 'lon_ph': -116.0,
        'h_ph': h.astype(float) + h_shift, 'X_ATC': np.arange(N) * 0.7, 'geoid': -20.0,
        'SPOT': 1, 'quality_ph': 0, 'class_ph': cls,
        **{f'atl24_prob_{i}': rng.random(N) for i in range(5)},
        'ortho_h': h.astype(float) + 20.0,
    }).iloc[:n_csv]
    csv = tmp_path / f'ATL03_test_{GT}_ATL24.csv'
    df.to_csv(csv, index=False, float_format='%.6f')
    return a03, csv, cls


def test_aligned_csv_reads_and_caches(tmp_path):
    a03, csv, cls = _write_pair(tmp_path)
    cache = tmp_path / 'cache' / 'atl24.parquet'
    df = read_atl24_csv(csv, a03, GT, cache_file=cache)
    assert len(df) == N
    np.testing.assert_array_equal(df['atl24_class'].to_numpy(), cls)
    assert df['atl24_class'].dtype == np.int16
    assert {f'atl24_prob_{i}' for i in range(5)} <= set(df.columns)
    assert cache.exists()
    df2 = read_atl24_csv(csv, a03, GT, cache_file=cache)   # from cache
    pd.testing.assert_frame_equal(df, df2)


def test_height_mismatch_raises(tmp_path):
    a03, csv, _ = _write_pair(tmp_path, h_shift=0.5)
    with pytest.raises(ValueError, match='do not line up'):
        read_atl24_csv(csv, a03, GT)


def test_row_count_mismatch_raises(tmp_path):
    a03, csv, _ = _write_pair(tmp_path, n_csv=N - 3)
    with pytest.raises(ValueError, match='rows but ATL03'):
        read_atl24_csv(csv, a03, GT)
