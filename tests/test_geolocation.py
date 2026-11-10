"""Synthetic checks for phoreal.geolocation offset estimation and geometry."""
import numpy as np
import pandas as pd
import pytest

from phoreal.geolocation import (
    create_interpolator, estimate_offset, atxt_to_en, track_heading,
    beam_vertical_angle, terrain_incidence_angle, terrain_summary,
)
from phoreal.io.readers import segment_sc_orient


def surface(at, xt):
    """Terrain with along- and cross-track structure that varies along track."""
    return (0.02 * at + 3.0 * np.sin(at / 40.0)
            + xt * (0.3 * np.sin(at / 60.0) + 0.1) + 0.5 * np.cos(xt / 7.0))


@pytest.fixture
def als_interp():
    at, xt = np.meshgrid(np.arange(-50, 1550, 0.5), np.arange(-40, 40, 0.5), indexing='ij')
    at, xt = at.ravel(), xt.ravel()
    df = pd.DataFrame({'alongtrack': at, 'crosstrack': xt,
                       'ellip_h': surface(at, xt), 'classification': 2})
    return create_interpolator(df, grid_resolution=1.0)


@pytest.mark.parametrize("true_corr", [(3.0, -2.0), (-4.5, 1.5), (0.0, 0.0)])
@pytest.mark.parametrize("fit_dz", [False, True])
def test_estimate_offset_recovers_known_correction(als_interp, true_corr, fit_dz):
    rng = np.random.default_rng(0)
    true_at = np.sort(rng.uniform(100, 1400, 3000))
    true_xt = np.zeros_like(true_at)
    z = surface(true_at, true_xt) + rng.normal(0, 0.1, true_at.size)
    if fit_dz:
        z = z + 0.7  # vertical bias that fit_dz should absorb
    # reported = true - correction
    rep_at, rep_xt = true_at - true_corr[0], true_xt - true_corr[1]

    res = estimate_offset(rep_at, rep_xt, z, als_interp, fit_dz=fit_dz, min_valid_frac=0.9)
    assert res['corr_at'] == pytest.approx(true_corr[0], abs=0.3)
    assert res['corr_xt'] == pytest.approx(true_corr[1], abs=0.3)
    assert not res['boundary_hit']
    assert res['cost_opt'] <= res['cost_zero']
    if fit_dz:
        assert res['dz_opt'] == pytest.approx(0.7, abs=0.15)


def test_atxt_to_en():
    # heading north: +at -> north, +xt (left) -> west
    assert np.allclose(atxt_to_en(1, 0, 0), (0, 1))
    assert np.allclose(atxt_to_en(0, 1, 0), (-1, 0))
    # heading east: left is north
    assert np.allclose(atxt_to_en(1, 0, 90), (1, 0), atol=1e-12)
    assert np.allclose(atxt_to_en(0, 1, 90), (0, 1), atol=1e-12)
    # heading south (descending): left is east
    assert np.allclose(atxt_to_en(0, 1, 180), (1, 0), atol=1e-12)


def test_track_heading():
    assert track_heading([52.0, 52.1], [20.0, 20.0]) == pytest.approx(0.0, abs=1e-6)
    assert track_heading([52.1, 52.0], [20.0, 20.0]) == pytest.approx(180.0, abs=1e-6)


def test_incidence_angles():
    nadir = np.pi / 2
    assert beam_vertical_angle(nadir) == pytest.approx(0.0)
    # vertical beam on flat ground
    assert terrain_incidence_angle(nadir, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    # vertical beam on a 45 deg slope
    assert terrain_incidence_angle(nadir, 0.0, 1.0, 0.0) == pytest.approx(45.0)
    # beam tilted 10 deg toward east, terrain facing east (downslope east)
    el = np.radians(80); az = np.radians(90)
    slope10 = np.tan(np.radians(10))
    assert terrain_incidence_angle(el, az, -slope10, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert terrain_incidence_angle(el, az, slope10, 0.0) == pytest.approx(20.0)


def test_terrain_summary_plane():
    at, xt = np.meshgrid(np.arange(0, 100.0), np.arange(-20, 20.0), indexing='ij')
    at, xt = at.ravel(), xt.ravel()
    # heading north; z rises to the east = -xt direction -> dz/dE = 0.1
    df = pd.DataFrame({'alongtrack': at, 'crosstrack': xt, 'ellip_h': -0.1 * xt})
    t = terrain_summary(df, heading_deg=0.0)
    assert t['grad_e'] == pytest.approx(0.1)
    assert t['grad_n'] == pytest.approx(0.0, abs=1e-9)
    assert t['aspect_deg'] == pytest.approx(270.0)  # faces west (downslope)
    assert t['roughness'] == pytest.approx(0.0, abs=1e-9)


def test_segment_sc_orient():
    t = np.array([5.0, 10.0, 15.0, 25.0])
    out = segment_sc_orient(t, [1, 2, 0], [8.0, 12.0, 20.0])
    assert out.tolist() == [-1, 1, 2, 0]
