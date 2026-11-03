"""
timing_calibration.py
---------------------
Assigns the observation epoch of every measured position.

The epoch of a measurement is the MID-EXPOSURE time of the pixels it was
measured on:

    t_open  = DATE-OBS  - {0 | exptime/2 | exptime}   (date_obs_reference)
              + time_bias
    t_row   = t_open + row_offset(y)                  (rolling shutter only)
    epoch   = t_row + exptime/2

For a uniformly moving object the flux-weighted centroid of its trail lies
at the mid-exposure position, so the same epoch applies to point sources
and to trail midpoints.

Magnitude of the effect (why this matters):
  * Time bias: an object moving at angular rate w is displaced w*dt.
    LEO at ~1 deg/s (ISS overhead: 7.7 km/s / 400 km) -> 10 ms = 36 arcsec.
    GEO at 15 arcsec/s -> 10 ms = 0.15 arcsec.
  * Stamping DATE-OBS (exposure start) instead of mid-exposure for a 2 s
    frame of a GEO object = 15 arcsec; of a LEO object = ~1 degree.

time_bias_ms is best calibrated on-sky: observe GNSS satellites (precise
orbits, ~cm) and fit the mean along-track O-C / angular rate.
"""

from __future__ import annotations

from dataclasses import dataclass

from astropy import units as u
from astropy.time import Time, TimeDelta

from sst_pipeline.utils.logging_config import get_logger

log = get_logger(__name__)

ISO_PRECISION = 6  # microseconds in all written epochs


@dataclass(frozen=True)
class TimingModel:
    date_obs_reference: str = "start"   # start | mid | end
    time_bias_s: float = 0.0
    shutter_type: str = "global"        # global | rolling
    readout_time_s: float = 0.0
    rolling_direction: str = "increasing_y"
    detector_ny: int = 4096

    @property
    def has_rolling_shutter(self) -> bool:
        return self.shutter_type == "rolling" and self.readout_time_s > 0


def build_timing_model(timing_cfg: dict, instrument_cfg: dict) -> TimingModel:
    direction = timing_cfg.get("rolling_direction", "increasing_y")
    legacy = {"top_to_bottom": "increasing_y", "bottom_to_top": "decreasing_y"}
    if direction in legacy:
        log.warning("rolling_direction '%s' is ambiguous for FITS row order; "
                    "interpreting as '%s'", direction, legacy[direction])
        direction = legacy[direction]
    model = TimingModel(
        date_obs_reference=timing_cfg.get("date_obs_reference", "start"),
        time_bias_s=float(timing_cfg.get("time_bias_ms", 0.0)) / 1000.0,
        shutter_type=timing_cfg.get("shutter_type", "global"),
        readout_time_s=float(timing_cfg.get("readout_time_ms", 0.0)) / 1000.0,
        rolling_direction=direction,
        detector_ny=int(instrument_cfg.get("detector_ny", 4096)),
    )
    log.info("Timing: DATE-OBS marks exposure %s; bias=%.3f ms; shutter=%s%s",
             model.date_obs_reference, model.time_bias_s * 1e3, model.shutter_type,
             f" (readout {model.readout_time_s*1e3:.2f} ms)" if model.has_rolling_shutter else "")
    return model


def shutter_open_time(date_obs: Time, exptime_s: float, model: TimingModel) -> Time:
    shift = {"start": 0.0, "mid": -exptime_s / 2.0, "end": -exptime_s}[model.date_obs_reference]
    return date_obs + TimeDelta((shift + model.time_bias_s) * u.s)


def row_offset_s(y_px: float, model: TimingModel, ny: int | None = None) -> float:
    if not model.has_rolling_shutter:
        return 0.0
    n = ny or model.detector_ny
    frac = y_px / n if model.rolling_direction == "increasing_y" else (n - 1 - y_px) / n
    return min(max(frac, 0.0), 1.0) * model.readout_time_s


def measurement_epoch(date_obs: Time, exptime_s: float, y_px: float,
                      model: TimingModel, ny: int | None = None) -> Time:
    """Mid-exposure epoch of a centroid measured at row y_px."""
    t = shutter_open_time(date_obs, exptime_s, model)
    return t + TimeDelta((row_offset_s(y_px, model, ny) + exptime_s / 2.0) * u.s)


def iso(t: Time) -> str:
    t = Time(t, precision=ISO_PRECISION)
    return t.utc.isot
