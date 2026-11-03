"""Plate verification: a pointing-model WCS 40" off is corrected to <0.05"."""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from sst_pipeline.tools.synthetic import make_wcs

from sst_pipeline.astrometry.wcs_solver import refine_wcs, solve_field_command
from sst_pipeline.config import load_config
from sst_pipeline.utils.coordinate_utils import from_tangent_plane, sep_arcsec

AC = load_config()["astrometry"]


def test_refine_recovers_true_wcs(rng):
    true = make_wcs(150.0, 20.0, 0.3)
    ra_off, dec_off = from_tangent_plane(-35.0, 22.0, 150.0, 20.0)
    initial = make_wcs(float(ra_off), float(dec_off), 0.36)
    xi, eta = rng.uniform(-430, 430, 120), rng.uniform(-430, 430, 120)
    ra, dec = from_tangent_plane(xi, eta, 150.0, 20.0)
    mag = rng.uniform(10, 15, 120)
    x, y = true.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    noise = 0.05
    sol = refine_wcs(initial, "header", x + rng.normal(0, noise, 120), y + rng.normal(0, noise, 120),
                     10 ** (-0.4 * mag), ra, dec, mag, (1024, 1024), AC)
    assert sol.verified and sol.n_matched >= 100
    assert sol.offset_arcsec == pytest.approx(np.hypot(35, 22), abs=1.5)
    # position error vs truth across the field
    px, py = rng.uniform(0, 1023, 50), rng.uniform(0, 1023, 50)
    a, b = true.pixel_to_world(px, py), sol.wcs.pixel_to_world(px, py)
    assert np.max(sep_arcsec(a.ra.deg, a.dec.deg, b.ra.deg, b.dec.deg)) < 0.05
    assert sol.rms_arcsec == pytest.approx(noise * 0.82 * np.sqrt(2), rel=0.3)


def test_unrelated_catalogue_not_verified(rng):
    w = make_wcs(150.0, 20.0, 0.0)
    x, y = rng.uniform(0, 1023, 80), rng.uniform(0, 1023, 80)
    ra, dec = from_tangent_plane(rng.uniform(-430, 430, 80), rng.uniform(-430, 430, 80), 150.0, 20.0)
    sol = refine_wcs(w, "header", x, y, np.ones(80), ra, dec, np.full(80, 12.0), (1024, 1024), AC)
    assert not sol.verified


def test_solve_field_command_has_hints_and_workdir():
    cmd = solve_field_command("/w/f.fits", "/w", 0.82, 0.1, 150.0, 20.0, 2.0, 60)
    s = " ".join(cmd)
    assert "--dir /w" in s and "--scale-low 0.7380" in s and "--ra 150.000000" in s and "--new-fits none" in s
