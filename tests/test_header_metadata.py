import pytest
from astropy.io import fits

from sst_pipeline.config import load_config
from sst_pipeline.io.header_validator import parse_obs_time, validate_header
from sst_pipeline.io.metadata_extractor import extract_metadata
from sst_pipeline.utils.coordinate_utils import parse_dec_deg, parse_ra_deg

RULES = load_config()["header_requirements"]


def hdr(**kw):
    h = fits.Header()
    base = {"DATE-OBS": "2026-03-15T20:00:00.123456", "EXPTIME": 2.0, "OBJECT": "X",
            "TELESCOP": "T", "INSTRUME": "C"}
    base.update(kw)
    for k, v in base.items():
        if v is not None:
            h[k] = v
    return h


def test_valid_header():
    assert validate_header("f", hdr(), RULES).is_valid


@pytest.mark.parametrize("kw", ["DATE-OBS", "EXPTIME", "OBJECT", "TELESCOP", "INSTRUME"])
def test_missing_required(kw):
    r = validate_header("f", hdr(**{kw: None}), RULES)
    assert not r.is_valid and kw in r.missing_required


def test_sexagesimal_ra_dec_from_ascom():
    # ASCOM / MaxIm / N.I.N.A. write sexagesimal strings
    r = validate_header("f", hdr(OBJCTRA="10 00 00.00", OBJCTDEC="+20 00 00.0"), RULES)
    assert r.resolved["RA"] == pytest.approx(150.0) and r.resolved["DEC"] == pytest.approx(20.0)
    assert parse_ra_deg("10:30:00") == pytest.approx(157.5)
    assert parse_dec_deg("-05 30 00") == pytest.approx(-5.5)
    assert parse_ra_deg(10.5, "hour") == pytest.approx(157.5)


def test_date_plus_time_obs():
    t = parse_obs_time(hdr(**{"DATE-OBS": "2026-03-15", "TIME-OBS": "20:00:01.5"}))
    assert t.isot == "2026-03-15T20:00:01.500"


def test_exptime_range():
    assert not validate_header("f", hdr(EXPTIME=-1.0), RULES).is_valid


def test_pixel_scale_uses_cd_determinant():
    import numpy as np
    h = hdr()
    th, sc = np.radians(30), 0.82 / 3600
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -sc * np.cos(th), sc * np.sin(th), sc * np.sin(th), sc * np.cos(th)
    m = extract_metadata("f", "p", h, validate_header("f", h, RULES))
    assert m.header_pixel_scale_arcsec == pytest.approx(0.82, rel=1e-6)   # CD1_1 alone gives 0.71
