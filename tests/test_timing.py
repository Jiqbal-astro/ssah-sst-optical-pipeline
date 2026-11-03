"""Epoch assignment: mid-exposure, DATE-OBS convention, bias, rolling shutter."""
import pytest
from astropy.time import Time

from sst_pipeline.utils.timing_calibration import TimingModel, build_timing_model, iso, measurement_epoch

T = Time("2026-03-15T20:00:00.000", scale="utc")


def dt_ms(t):
    return (t - T).sec * 1e3


def test_point_epoch_is_mid_exposure():
    # DATE-OBS marks the exposure start; the epoch is mid-exposure
    assert dt_ms(measurement_epoch(T, 2.0, 500, TimingModel())) == pytest.approx(1000.0, abs=1e-3)


@pytest.mark.parametrize("ref,expected", [("start", 1000.0), ("mid", 0.0), ("end", -1000.0)])
def test_date_obs_reference(ref, expected):
    m = TimingModel(date_obs_reference=ref)
    assert dt_ms(measurement_epoch(T, 2.0, 0, m)) == pytest.approx(expected, abs=1e-3)


def test_time_bias_added():
    m = build_timing_model({"time_bias_ms": 12.5}, {"detector_ny": 100})
    assert dt_ms(measurement_epoch(T, 1.0, 0, m)) == pytest.approx(512.5, abs=1e-3)


def test_global_shutter_rows_share_epoch():
    m = TimingModel(shutter_type="global", readout_time_s=0.03)
    assert measurement_epoch(T, 1, 0, m, 1000) == measurement_epoch(T, 1, 999, m, 1000)


def test_rolling_shutter_row_offset_and_direction():
    m = TimingModel(shutter_type="rolling", readout_time_s=0.040, detector_ny=1000)
    assert dt_ms(measurement_epoch(T, 1.0, 500, m)) == pytest.approx(520.0, abs=1e-3)
    m2 = TimingModel(shutter_type="rolling", readout_time_s=0.040, detector_ny=1000,
                     rolling_direction="decreasing_y")
    assert dt_ms(measurement_epoch(T, 1.0, 999, m2)) == pytest.approx(500.0, abs=1e-3)


def test_iso_has_microseconds():
    from astropy.time import TimeDelta
    import astropy.units as u
    s = iso(T + TimeDelta(12.3 * u.us))
    assert s.endswith(".000012")
