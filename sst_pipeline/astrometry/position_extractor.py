"""
position_extractor.py
---------------------
Turns a measured source into a Detection: sky position at the correct
mid-exposure epoch, with uncertainties that include both the centroid
error and the plate-solution RMS.

One observation per object per frame. For a trail this is the midpoint,
which corresponds to mid-exposure. Trail end points are kept only as
attributes (trail rate and axis): their epochs depend on the shutter
model and the direction of motion is ambiguous by 180 deg in one frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from astropy.time import Time

from sst_pipeline.detection.source_classifier import Morph
from sst_pipeline.detection.source_extractor import Source
from sst_pipeline.utils.coordinate_utils import pixel_errors_to_sky, position_angle_deg
from sst_pipeline.utils.timing_calibration import TimingModel, iso, measurement_epoch


@dataclass
class Detection:
    det_id: str
    frame_id: str
    frame_index: int
    epoch_utc: str
    mjd: float
    x_px: float
    y_px: float
    ra_deg: float
    dec_deg: float
    ra_err_arcsec: float        # sigma of RA*cos(Dec)
    dec_err_arcsec: float
    morphology: str
    snr: float
    saturated: bool
    exptime_s: float
    airmass: float
    trail_length_px: float = 0.0
    trail_rate_arcsec_s: float = 0.0
    trail_pa_deg: float = float("nan")   # axis, modulo 180
    flux_adu: float = float("nan")
    mag: Optional[float] = None
    mag_err: Optional[float] = None
    mag_band: str = ""
    zp_source: str = ""
    astrometry_verified: bool = False
    photometry_ok: bool = False
    is_catalog_star: bool = False
    is_static: bool = False
    is_candidate: bool = False
    status: str = ""              # LINKED | SINGLE_STREAK | UNLINKED | STATIC | STAR
    tracklet_id: str = ""
    norad_id: Optional[int] = None
    catalog_name: str = ""
    match_quality: str = ""
    oc_along_arcsec: float = float("nan")
    oc_cross_arcsec: float = float("nan")
    oc_total_arcsec: float = float("nan")
    predicted_rate_arcsec_s: float = float("nan")
    predicted_range_km: float = float("nan")
    tle_age_days: float = float("nan")

    @property
    def is_trailed(self) -> bool:
        return self.morphology == Morph.TRAILED.value

    def to_dict(self) -> dict:
        return asdict(self)


def make_detection(frame_id: str, frame_index: int, src: Source, morph: Morph, wcs,
                   date_obs: Time, exptime: float, timing: TimingModel, ny: int,
                   plate_rms_ra: float, plate_rms_dec: float, airmass: float,
                   verified: bool) -> Detection:
    sky = wcs.pixel_to_world(src.x, src.y)
    ra, dec = float(sky.ra.deg), float(sky.dec.deg)
    era, edec = pixel_errors_to_sky(wcs, src.x_err, src.y_err)
    pr = plate_rms_ra if np.isfinite(plate_rms_ra) else 0.0
    pd_ = plate_rms_dec if np.isfinite(plate_rms_dec) else 0.0
    t = measurement_epoch(date_obs, exptime, src.y, timing, ny)
    det = Detection(det_id=f"{frame_id}#{src.idx}", frame_id=frame_id, frame_index=frame_index,
                    epoch_utc=iso(t), mjd=float(t.mjd), x_px=src.x, y_px=src.y,
                    ra_deg=ra % 360.0, dec_deg=dec,
                    ra_err_arcsec=float(np.hypot(era, pr)), dec_err_arcsec=float(np.hypot(edec, pd_)),
                    morphology=morph.value, snr=src.snr, saturated=src.saturated,
                    exptime_s=exptime, airmass=airmass, astrometry_verified=verified)
    if morph == Morph.TRAILED and src.length > 0:
        ends = wcs.pixel_to_world(np.array([src.x0, src.x1]), np.array([src.y0, src.y1]))
        sep = float(ends[0].separation(ends[1]).arcsec)
        det.trail_length_px = src.length
        det.trail_rate_arcsec_s = sep / exptime if exptime > 0 else float("nan")
        det.trail_pa_deg = float(position_angle_deg(ends[0].ra.deg, ends[0].dec.deg,
                                                    ends[1].ra.deg, ends[1].dec.deg)) % 180.0
    return det
