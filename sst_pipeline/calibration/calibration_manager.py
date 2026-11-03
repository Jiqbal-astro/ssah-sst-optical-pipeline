"""
calibration_manager.py
----------------------
CCD reduction:  (raw - bias - dark_1s * t_exp) / flat,  then optional
L.A.Cosmic cosmic-ray cleaning (van Dokkum 2001) via astroscrappy.

Notes:
  * Cosmic-ray cleaning uses astroscrappy. If it is not installed the step
    is skipped and logged; no substitute filter is applied, because simple
    sigma-clip replacement also clips the cores of stars and satellites.
  * Master frames are checked for shape mismatch before use.
  * The CR mask is returned so QC reports the cosmic-ray fraction measured
    before cleaning.
  * Saturation is evaluated on RAW data by the caller (bias subtraction and
    flat division move pixels away from the ADC ceiling).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from sst_pipeline.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class CalibrationFrames:
    bias: Optional[np.ndarray] = None
    dark_1s: Optional[np.ndarray] = None   # bias-subtracted, per second
    flat: Optional[np.ndarray] = None      # normalised, median ~ 1


def build_master_bias(paths: list[str]) -> np.ndarray:
    if not paths:
        raise ValueError("No bias frames supplied")
    return np.median([fits.getdata(p).astype(np.float64) for p in paths], axis=0)


def build_master_dark(paths: list[str], master_bias: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-second, bias-subtracted master dark."""
    if not paths:
        raise ValueError("No dark frames supplied")
    frames = []
    for p in paths:
        with fits.open(p) as h:
            d = h[0].data.astype(np.float64)
            t = float(h[0].header.get("EXPTIME", 0.0))
        if t <= 0:
            raise ValueError(f"Dark {p} has no valid EXPTIME")
        if master_bias is not None:
            d = d - master_bias
        frames.append(d / t)
    return np.median(frames, axis=0)


def build_master_flat(paths: list[str], master_bias: Optional[np.ndarray] = None,
                      master_dark_1s: Optional[np.ndarray] = None) -> np.ndarray:
    if not paths:
        raise ValueError("No flat frames supplied")
    frames = []
    for p in paths:
        with fits.open(p) as h:
            d = h[0].data.astype(np.float64)
            t = float(h[0].header.get("EXPTIME", 0.0))
        if master_bias is not None:
            d = d - master_bias
        if master_dark_1s is not None:
            d = d - master_dark_1s * t
        _, med, _ = sigma_clipped_stats(d)
        frames.append(d / med)
    flat = np.median(frames, axis=0)
    bad = ~np.isfinite(flat) | (flat < 0.05)
    flat[bad] = 1.0          # dead/vignetted pixels: do not amplify noise
    return flat


def load_calibration_frames(cal_cfg: dict) -> CalibrationFrames:
    def _load(key: str, label: str):
        path = cal_cfg.get(key, "")
        if not path:
            log.warning("No %s configured — step skipped (uncalibrated data)", label)
            return None
        if not os.path.exists(path):
            log.warning("%s not found at '%s' — step skipped", label, path)
            return None
        return fits.getdata(path).astype(np.float64)

    return CalibrationFrames(bias=_load("master_bias_path", "master bias"),
                             dark_1s=_load("master_dark_path", "master dark"),
                             flat=_load("master_flat_path", "master flat"))


def _check_shape(name: str, master: Optional[np.ndarray], shape) -> Optional[np.ndarray]:
    if master is not None and master.shape != shape:
        log.error("Master %s shape %s != frame shape %s — NOT applied", name, master.shape, shape)
        return None
    return master


def calibrate_frame(raw: np.ndarray, exptime_s: float, cal: CalibrationFrames,
                    cr_cfg: Optional[dict] = None, gain: float = 1.0,
                    read_noise_e: float = 10.0, saturation_adu: float = 65535.0,
                    ) -> tuple[np.ndarray, Optional[np.ndarray], list[str]]:
    """Returns (calibrated_data, cosmic_ray_mask_or_None, list_of_steps_applied)."""
    data = raw.astype(np.float64, copy=True)
    steps: list[str] = []
    bias = _check_shape("bias", cal.bias, data.shape)
    dark = _check_shape("dark", cal.dark_1s, data.shape)
    flat = _check_shape("flat", cal.flat, data.shape)
    if bias is not None:
        data -= bias
        steps.append("B")
    if dark is not None:
        data -= dark * exptime_s
        steps.append("D")
    if flat is not None:
        data /= flat
        steps.append("F")

    cr_mask = None
    if cr_cfg and cr_cfg.get("cosmic_ray_rejection", False):
        try:
            import astroscrappy
            sat_cal = saturation_adu - (float(np.median(bias)) if bias is not None else 0.0)
            cr_mask, cleaned = astroscrappy.detect_cosmics(
                data, gain=gain, readnoise=read_noise_e, satlevel=sat_cal,
                sigclip=cr_cfg.get("lacosmic_sigclip", 4.5),
                sigfrac=cr_cfg.get("lacosmic_sigfrac", 0.3),
                objlim=cr_cfg.get("lacosmic_objlim", 5.0),
                niter=int(cr_cfg.get("lacosmic_niter", 4)), verbose=False)
            # Older astroscrappy (<1.1) returned electrons, newer returns input
            # units. Detect empirically on untouched pixels instead of assuming.
            cleaned = np.asarray(cleaned, dtype=np.float64)
            ok = ~cr_mask & (np.abs(data) > 1e-6)
            scale = float(np.median(cleaned[ok] / data[ok])) if ok.any() else 1.0
            data = cleaned / scale if abs(scale - gain) < abs(scale - 1.0) else cleaned
            steps.append("CR")
        except ImportError:
            log.warning("astroscrappy not installed — cosmic-ray cleaning skipped "
                        "(pip install astroscrappy)")
    return data, cr_mask, steps


def save_calibrated_frame(data: np.ndarray, header: fits.Header, out_path: str, steps: list[str]):
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header.copy())
    hdu.header["CALSTAT"] = ("".join(steps) or "NONE", "calibration steps applied")
    hdu.writeto(out_path, overwrite=True)
