import pytest

from sst_pipeline.config import load_config
from sst_pipeline.io.header_validator import HeaderValidationResult
from sst_pipeline.qc.metrics import FrameMetrics
from sst_pipeline.qc.qc_matrix import apply_astrometric_qc, apply_photometric_qc, evaluate_frame

QC = load_config()["qc"]
OK = HeaderValidationResult("f", True)


def good(**kw):
    m = FrameMetrics(frame_id="f", bkg_level=1500, bkg_rms=20, saturation_fraction=0, n_sources=200,
                     n_reference=180, signal_present=True, max_snr=500, fwhm_px=3.4, fwhm_arcsec=2.8,
                     star_ellipticity=0.05)
    for k, v in kw.items():
        setattr(m, k, v)
    return m


HIST = [good() for _ in range(8)]


def test_good_frame_passes_both():
    r = evaluate_frame(good(), OK, HIST, QC)
    assert r.photometry_pass and r.astrometry_pass


def test_dual_track_saturation():
    r = evaluate_frame(good(saturation_fraction=0.002), OK, HIST, QC)
    assert not r.photometry_pass and r.astrometry_pass


def test_low_snr_tracks():
    r = evaluate_frame(good(max_snr=7), OK, HIST, QC)
    assert not r.photometry_pass and r.astrometry_pass
    r = evaluate_frame(good(max_snr=3), OK, HIST, QC)
    assert not r.astrometry_pass


def test_invalid_header_fails_both():
    r = evaluate_frame(good(), HeaderValidationResult("f", False, missing_required=["DATE-OBS"]), HIST, QC)
    assert not r.photometry_pass and not r.astrometry_pass


def test_background_jump_and_cloud():
    assert not evaluate_frame(good(bkg_level=2500), OK, HIST, QC).photometry_pass
    r = evaluate_frame(good(n_reference=60), OK, HIST, QC)
    assert not r.transparency_ok and not r.photometry_pass and r.astrometry_pass


def test_seeing_and_tracking():
    assert not evaluate_frame(good(fwhm_arcsec=7.0), OK, HIST, QC).photometry_pass
    assert not evaluate_frame(good(fwhm_arcsec=12.0), OK, HIST, QC).astrometry_pass
    r = evaluate_frame(good(star_ellipticity=0.5), OK, HIST, QC)
    assert not r.photometry_pass and r.astrometry_pass


def test_rms_and_verification_updates():
    r = evaluate_frame(good(), OK, HIST, QC)
    apply_astrometric_qc(r, "header+refit", True, 150, 2.5, QC)
    assert not r.astrometry_pass and "RMS" in r.reject_reasons_astrometry
    r = evaluate_frame(good(), OK, HIST, QC)
    apply_astrometric_qc(r, "header", False, 0, float("nan"), QC)
    assert not r.astrometry_pass and "not verified" in r.reject_reasons_astrometry
    r = evaluate_frame(good(), OK, HIST, QC)
    apply_photometric_qc(r, 23.5, 0.2, 50, QC, 5)
    assert not r.photometry_pass and r.photometric_rms_mag == pytest.approx(0.2)
