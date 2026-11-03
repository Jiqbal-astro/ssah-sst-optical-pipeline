"""SGP4 prediction validated against skyfield; identification logic."""
import os

import numpy as np
import pytest
from astropy.time import Time, TimeDelta
import astropy.units as u

from sst_pipeline.astrometry.position_extractor import Detection
from sst_pipeline.config import load_config
from sst_pipeline.tracking.known_object_matcher import identify, match_detection
from sst_pipeline.tracking.linker import Tracklet
from sst_pipeline.tracking.orbit_propagator import Propagator
from sst_pipeline.tracking.tle_manager import parse_tle_file, tle_checksum_ok
from sst_pipeline.utils.coordinate_utils import from_tangent_plane, sep_arcsec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TLE = os.path.join(ROOT, "tests", "data", "sample_tle.txt")
SITE = (25.3573, 55.4033, 50.0)
T_PASS = Time("2024-01-02T13:24:00", scale="utc")    # ISS (sample TLE) at ~43.6 deg altitude
CFG = load_config()["tle_tracking"]


@pytest.fixture(scope="module")
def prop():
    return Propagator(parse_tle_file(TLE), *SITE)


def test_checksums():
    l = open(TLE).read().splitlines()[3]
    assert tle_checksum_ok(l)
    assert not tle_checksum_ok(l[:68] + str((int(l[68]) + 1) % 10))


def test_catalogue_loads(prop):
    assert sorted(prop.ids) == [20580, 25544, 48274]


def test_altitude_sign_and_value(prop):
    # regression: transforming the topocentric vector with its distance puts this pass at -84 deg
    p = prop.predict(T_PASS)
    i = list(prop.ids).index(25544)
    assert p.alt[i] == pytest.approx(43.63, abs=0.05)
    assert 300 < p.range_km[i] < 900 and 1000 < p.rate_arcsec_s[i] < 6000


def test_against_skyfield(prop):
    sf = pytest.importorskip("skyfield.api")
    ts = sf.load.timescale(builtin=True)
    o = prop.objects[list(prop.ids).index(25544)]
    sat = sf.EarthSatellite(o.line1, o.line2, o.name, ts)
    topo = sf.wgs84.latlon(*SITE[:2], elevation_m=SITE[2])
    for dt in (-60, 0, 60):
        t = T_PASS + TimeDelta(dt * u.s)
        g = prop.predict(t, light_time=False, star_referenced=False)
        ra, dec, _ = (sat - topo).at(ts.from_astropy(t)).radec()
        alt, az, dist = (sat - topo).at(ts.from_astropy(t)).altaz()
        i = list(prop.ids).index(25544)
        assert sep_arcsec(g.ra[i], g.dec[i], ra._degrees, dec.degrees) < 3.0
        assert g.alt[i] == pytest.approx(alt.degrees, abs=0.002)
        assert g.range_km[i] == pytest.approx(dist.km, abs=0.05)


def test_star_referenced_correction_is_aberration_sized(prop):
    a, g = prop.predict(T_PASS), prop.predict(T_PASS, star_referenced=False)
    i = list(prop.ids).index(25544)
    assert 1.0 < sep_arcsec(a.ra[i], a.dec[i], g.ra[i], g.dec[i]) < 30.0


def _det(ra, dec, t, **kw):
    return Detection(det_id="f#1", frame_id="f", frame_index=0, epoch_utc=t.isot, mjd=t.mjd, x_px=0, y_px=0,
                     ra_deg=ra, dec_deg=dec, ra_err_arcsec=0.3, dec_err_arcsec=0.3, morphology="TRAILED",
                     snr=50, saturated=False, exptime_s=1.0, airmass=1.2, **kw)


def test_identify_with_along_track_error(prop):
    p = prop.predict(T_PASS)
    i = list(prop.ids).index(25544)
    pa = np.radians(p.pa_deg[i])
    ra, dec = from_tangent_plane(300 * np.sin(pa), 300 * np.cos(pa), p.ra[i], p.dec[i])  # 300" ahead
    d = _det(float(ra), float(dec), T_PASS, trail_rate_arcsec_s=p.rate_arcsec_s[i], trail_pa_deg=p.pa_deg[i] % 180)
    match_detection(d, p, CFG)
    assert d.norad_id == 25544
    assert d.oc_along_arcsec == pytest.approx(300, abs=2) and abs(d.oc_cross_arcsec) < 2


def test_cross_track_offset_rejected(prop):
    p = prop.predict(T_PASS)
    i = list(prop.ids).index(25544)
    pa = np.radians(p.pa_deg[i] + 90)
    ra, dec = from_tangent_plane(400 * np.sin(pa), 400 * np.cos(pa), p.ra[i], p.dec[i])
    d = _det(float(ra), float(dec), T_PASS)
    match_detection(d, p, CFG)
    assert d.norad_id is None


def test_wrong_trail_direction_rejected(prop):
    p = prop.predict(T_PASS)
    i = list(prop.ids).index(25544)
    d = _det(p.ra[i], p.dec[i], T_PASS, trail_rate_arcsec_s=p.rate_arcsec_s[i],
             trail_pa_deg=(p.pa_deg[i] + 60) % 180)
    match_detection(d, p, CFG)
    assert d.norad_id is None


def test_tracklet_time_offset(prop):
    dets = []
    for k, dt in enumerate((-2, 0, 2)):
        t = T_PASS + TimeDelta(dt * u.s)
        p = prop.predict(t)
        i = list(prop.ids).index(25544)
        pa = np.radians(p.pa_deg[i])
        off = p.rate_arcsec_s[i] * 0.050                       # object 50 ms "late" along track
        ra, dec = from_tangent_plane(off * np.sin(pa), off * np.cos(pa), p.ra[i], p.dec[i])
        dets.append(_det(float(ra), float(dec), t))
    tr = Tracklet("T0001", "MULTI_FRAME", dets, rate_arcsec_s=float(p.rate_arcsec_s[i]))
    identify(dets, [tr], prop, CFG)
    assert tr.identified and tr.norad_id == 25544
    assert tr.along_track_time_offset_s == pytest.approx(0.050, abs=0.002)
