"""
metrics.py
----------
Per-frame image-quality numbers. Where each is measured matters:

  * saturation, hot pixels  -> RAW data (calibration moves pixels off the
    ADC/non-linearity ceiling)
  * cosmic rays             -> L.A.Cosmic mask if CR cleaning ran, else a
    sharp-pixel estimate on RAW data (always measured before cleaning)
  * background, gradient, bad columns -> calibrated data
  * SNR, seeing (FWHM), star ellipticity (tracking quality), star count
    (transparency)          -> the same SEP catalogue used for science
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from sst_pipeline.detection.source_classifier import Morph
from sst_pipeline.detection.source_extractor import ExtractionResult


@dataclass
class FrameMetrics:
    frame_id: str
    bkg_level: float = 0.0
    bkg_rms: float = 0.0
    saturation_fraction: float = 0.0
    n_sources: int = 0
    n_reference: int = 0
    signal_present: bool = False
    max_snr: float = 0.0
    fwhm_px: float = float("nan")
    fwhm_arcsec: float = float("nan")
    star_ellipticity: float = float("nan")
    cosmic_ray_fraction: float = 0.0
    cosmic_ray_source: str = ""
    hot_pixel_fraction: float = 0.0
    bad_column_fraction: float = 0.0
    background_gradient_fraction: float = 0.0
    max_trail_length_px: float = 0.0
    error: Optional[str] = None


def _sharp_pixels(raw: np.ndarray, med: float, sig: float) -> np.ndarray:
    """Pixels much sharper than any PSF: high, and far above their 4 neighbours."""
    k = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    nb = ndimage.convolve(raw, k, mode="nearest")
    lap = raw - nb
    return (raw > med + 10 * sig) & (lap > 8 * sig) & (lap > 0.6 * (raw - med))


def compute_frame_metrics(frame_id: str, raw: np.ndarray, cal: np.ndarray,
                          cr_mask: Optional[np.ndarray], ext: ExtractionResult,
                          morphs: list[Morph], reference: Morph,
                          saturation_adu: float, pixel_scale_arcsec: float,
                          art_cfg: dict) -> FrameMetrics:
    m = FrameMetrics(frame_id=frame_id)
    try:
        if raw.size == 0:
            m.error = "Empty data array"
            return m
        m.bkg_level, m.bkg_rms = ext.bkg_level, ext.bkg_rms
        m.saturation_fraction = float(np.mean(raw >= saturation_adu))
        srcs = ext.sources
        m.n_sources = len(srcs)
        m.signal_present = m.n_sources > 0
        unsat = [s.snr for s in srcs if not s.saturated]
        m.max_snr = float(max(unsat)) if unsat else 0.0

        good_ref = [s for s, mo in zip(srcs, morphs)
                    if mo == reference and not s.saturated and s.snr >= 20]
        m.n_reference = sum(1 for mo in morphs if mo == reference)
        if good_ref:
            m.fwhm_px = float(np.median([s.fwhm for s in good_ref]))
            m.fwhm_arcsec = m.fwhm_px * pixel_scale_arcsec
            if reference == Morph.POINT:
                m.star_ellipticity = float(np.median([1 - s.b / s.a for s in good_ref]))
        trails = [s.length for s, mo in zip(srcs, morphs) if mo == Morph.TRAILED]
        m.max_trail_length_px = float(max(trails)) if trails else 0.0

        # artefacts
        med_raw = float(np.median(raw))
        sig = max(ext.bkg_rms, 1e-6)
        sharp = _sharp_pixels(raw, med_raw, sig)
        if cr_mask is not None:
            m.cosmic_ray_fraction, m.cosmic_ray_source = float(np.mean(cr_mask)), "lacosmic"
        else:
            m.cosmic_ray_fraction, m.cosmic_ray_source = float(np.mean(sharp)), "sharp-pixel"
        m.hot_pixel_fraction = float(np.mean(sharp & (raw >= art_cfg.get("hot_pixel_adu_threshold", 60000))))

        col = np.median(cal, axis=0)
        mad = 1.4826 * np.median(np.abs(col - np.median(col)))
        m.bad_column_fraction = float(np.mean(np.abs(col - np.median(col)) > 5 * max(mad, sig / np.sqrt(cal.shape[0]))))

        ny, nx = cal.shape
        q = [np.median(cal[:ny // 2, :nx // 2]), np.median(cal[:ny // 2, nx // 2:]),
             np.median(cal[ny // 2:, :nx // 2]), np.median(cal[ny // 2:, nx // 2:])]
        m.background_gradient_fraction = float((max(q) - min(q)) / max(abs(np.median(q)), sig))
    except Exception as exc:  # noqa: BLE001
        m.error = f"{type(exc).__name__}: {exc}"
    return m
