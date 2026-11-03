"""Flux recovery and error model against injected sources."""
import numpy as np
import pytest
from conftest import gauss_stamp, trail

from sst_pipeline.config import load_config
from sst_pipeline.photometry.aperture_photometry import (enclosed_circle, enclosed_trail, point_photometry,
                                                         solve_zeropoint, trail_photometry)

PC = load_config()["photometry"]
SIG, FWHM, GAIN = 1.5, 1.5 * 2.3548, 1.5


def sky(rng, n=200, level=500.0, rms=10.0):
    return rng.normal(level, rms, (n, n))


def test_enclosed_fractions():
    assert enclosed_circle(100, FWHM) == pytest.approx(1.0)
    assert enclosed_trail(50, 100, 100, FWHM) == pytest.approx(1.0, abs=1e-6)


def test_point_flux_recovered(rng):
    img = sky(rng)
    gauss_stamp(img, 100.3, 99.7, 40000.0, SIG)
    p = point_photometry(img, [100.3], [99.7], FWHM, PC, GAIN, 1.0)[0]
    assert p.flux == pytest.approx(40000.0, rel=0.02)


def test_trail_flux_recovered(rng):
    img = sky(rng)
    trail(img, 50, 60, 150, 140, 60000.0, SIG)
    L, th = np.hypot(100, 80), np.arctan2(80, 100)
    p = trail_photometry(img, 100, 100, L, th, FWHM, PC, GAIN, 1.0)
    assert p.flux == pytest.approx(60000.0, rel=0.03)


def test_error_model_matches_scatter():
    rng = np.random.default_rng(7)
    fl, er = [], []
    for _ in range(60):
        img = sky(rng, n=80)
        gauss_stamp(img, 40, 40, 5000.0, SIG)
        img = img + rng.normal(0, np.sqrt(np.clip(img - 500, 0, None) / GAIN))  # source shot noise
        p = point_photometry(img, [40], [40], FWHM, PC, GAIN, 1.0)[0]
        fl.append(p.flux), er.append(p.flux_err)
    assert np.std(fl) / np.mean(er) == pytest.approx(1.0, abs=0.3)


def test_zeropoint_with_outlier():
    m_inst = np.array([-10, -11, -12, -9.5, -10.5, -11.5, -8.0])
    m_cat = m_inst + 23.5
    m_cat[-1] += 1.0                          # one variable / blended star
    zp, zperr, rms, n = solve_zeropoint(m_cat, m_inst, np.full(7, 0.01), 5)
    assert zp == pytest.approx(23.5, abs=1e-6) and n == 6 and rms < 1e-6
