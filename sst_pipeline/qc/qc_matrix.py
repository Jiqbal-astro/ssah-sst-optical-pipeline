"""
qc_matrix.py
------------
Dual-track QC. Photometry is strict, astrometry lenient: a trailed,
marginal-SNR or partly saturated frame can still give a valid position.

Decision inputs
  both tracks : header validity, signal presence, SNR, saturation, seeing
  photometry  : background stability, transparency (star count vs the
                sequence), artefacts, tracking ellipticity, trail length,
                photometric zero-point scatter
  astrometry  : severe CR contamination, catalogue-verified WCS and its RMS

Background stability and transparency are judged against a rolling window
of the most recent valid frames, using both sky level and sky noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from sst_pipeline.io.header_validator import HeaderValidationResult
from sst_pipeline.qc.metrics import FrameMetrics


@dataclass
class QCRow:
    frame_id: str
    header_valid: bool
    bkg_level: float
    bkg_rms: float
    bkg_stable: bool
    snr: float
    saturation_fraction: float
    n_sources: int
    n_reference_stars: int
    signal_present: bool
    fwhm_arcsec: float
    star_ellipticity: float
    transparency_ok: bool
    cosmic_ray_fraction: float
    hot_pixel_fraction: float
    bad_column_fraction: float
    background_gradient_fraction: float
    artifact_flag: bool
    trail_length_px: float
    wcs_source: str = ""
    wcs_verified: bool = False
    n_astrometric_stars: int = 0
    astrometric_rms_arcsec: float = float("nan")
    n_photometric_stars: int = 0
    zeropoint_mag: float = float("nan")
    photometric_rms_mag: float = float("nan")
    photometry_pass: bool = False
    astrometry_pass: bool = False
    reject_reasons_photometry: str = ""
    reject_reasons_astrometry: str = ""
    load_error: Optional[str] = None
    _photo: list = field(default_factory=list, repr=False)
    _astro: list = field(default_factory=list, repr=False)

    def fail(self, track: str, reason: str):
        lst = self._photo if track == "photometry" else self._astro
        lst.append(reason)
        self._sync()

    def _sync(self):
        self.photometry_pass = not self._photo
        self.astrometry_pass = not self._astro
        self.reject_reasons_photometry = "; ".join(self._photo)
        self.reject_reasons_astrometry = "; ".join(self._astro)


def _robust_outlier(value: float, history: list[float], nsig: float, floor_frac: float = 0.02) -> bool:
    arr = np.asarray(history, float)
    med = np.median(arr)
    mad = 1.4826 * np.median(np.abs(arr - med))
    scale = max(mad, floor_frac * abs(med), 1e-9)
    return abs(value - med) / scale > nsig


def evaluate_frame(m: FrameMetrics, hv: HeaderValidationResult,
                   history: list[FrameMetrics], qc: dict[str, Any]) -> QCRow:
    art = qc["artifacts"]
    bg = qc["background"]
    win = [h for h in history[-int(bg["window_frames"]):] if h.error is None and h.signal_present]

    bkg_stable, transp_ok = True, True
    if len(win) >= int(bg["min_frames"]):
        nsig = float(bg["stability_sigma_max"])
        bkg_stable = not (_robust_outlier(m.bkg_level, [h.bkg_level for h in win], nsig) or
                          _robust_outlier(m.bkg_rms, [h.bkg_rms for h in win], nsig))
        ref_med = np.median([h.n_reference for h in win])
        transp_ok = m.n_reference >= qc["transparency"]["min_star_fraction"] * ref_med

    art_notes = []
    if m.cosmic_ray_fraction > art["max_cosmic_ray_fraction"]:
        art_notes.append(f"cosmic rays {m.cosmic_ray_fraction:.4f}")
    if m.hot_pixel_fraction > art["max_hot_pixel_fraction"]:
        art_notes.append(f"hot pixels {m.hot_pixel_fraction:.5f}")
    if m.bad_column_fraction > art["max_bad_column_fraction"]:
        art_notes.append(f"bad columns {m.bad_column_fraction:.3f}")
    if m.background_gradient_fraction > art["max_background_gradient_fraction"]:
        art_notes.append(f"background gradient {m.background_gradient_fraction:.2f}")

    row = QCRow(
        frame_id=m.frame_id, header_valid=hv.is_valid,
        bkg_level=round(m.bkg_level, 3), bkg_rms=round(m.bkg_rms, 3), bkg_stable=bkg_stable,
        snr=round(m.max_snr, 2), saturation_fraction=round(m.saturation_fraction, 7),
        n_sources=m.n_sources, n_reference_stars=m.n_reference, signal_present=m.signal_present,
        fwhm_arcsec=round(m.fwhm_arcsec, 3), star_ellipticity=round(m.star_ellipticity, 3),
        transparency_ok=bool(transp_ok),
        cosmic_ray_fraction=round(m.cosmic_ray_fraction, 6),
        hot_pixel_fraction=round(m.hot_pixel_fraction, 7),
        bad_column_fraction=round(m.bad_column_fraction, 5),
        background_gradient_fraction=round(m.background_gradient_fraction, 4),
        artifact_flag=bool(art_notes), trail_length_px=round(m.max_trail_length_px, 1),
        load_error=m.error)

    def both(reason):
        row.fail("photometry", reason)
        row.fail("astrometry", reason)

    if m.error:
        both(f"metric error: {m.error}")
    if not hv.is_valid:
        both("header invalid (" + ", ".join(hv.missing_required + hv.invalid_values) + ")")
    if not m.signal_present:
        both("no source detected")

    snr = qc["snr"]
    if m.max_snr < snr["photometry_min"]:
        row.fail("photometry", f"SNR {m.max_snr:.1f} < {snr['photometry_min']}")
    if m.max_snr < snr["astrometry_min"]:
        row.fail("astrometry", f"SNR {m.max_snr:.1f} < {snr['astrometry_min']}")

    sat = qc["saturation"]
    if m.saturation_fraction > sat["photometry_max_fraction"]:
        row.fail("photometry", f"saturation {m.saturation_fraction:.5f} > {sat['photometry_max_fraction']}")
    if m.saturation_fraction > sat["astrometry_max_fraction"]:
        row.fail("astrometry", f"saturation {m.saturation_fraction:.5f} > {sat['astrometry_max_fraction']}")

    see = qc["seeing"]
    if np.isfinite(m.fwhm_arcsec):
        if m.fwhm_arcsec > see["photometry_max_fwhm_arcsec"]:
            row.fail("photometry", f"FWHM {m.fwhm_arcsec:.1f}\" > {see['photometry_max_fwhm_arcsec']}\"")
        if m.fwhm_arcsec > see["astrometry_max_fwhm_arcsec"]:
            row.fail("astrometry", f"FWHM {m.fwhm_arcsec:.1f}\" > {see['astrometry_max_fwhm_arcsec']}\"")
    if np.isfinite(m.star_ellipticity) and m.star_ellipticity > qc["tracking"]["max_star_ellipticity"]:
        row.fail("photometry", f"star ellipticity {m.star_ellipticity:.2f} (tracking/focus)")

    if not bkg_stable:
        row.fail("photometry", "background unstable vs sequence")
    if not transp_ok:
        row.fail("photometry", "star count dropped (cloud / low transparency)")
    if art_notes:
        row.fail("photometry", "artifacts: " + "; ".join(art_notes))
    if m.cosmic_ray_fraction > art["severe_cosmic_ray_fraction"]:
        row.fail("astrometry", f"severe cosmic-ray contamination {m.cosmic_ray_fraction:.4f}")
    if m.max_trail_length_px > qc["trail_tolerance"]["max_trail_length_px"]:
        row.fail("photometry", f"trail {m.max_trail_length_px:.0f}px > {qc['trail_tolerance']['max_trail_length_px']}px")
    row._sync()
    return row


def apply_astrometric_qc(row: QCRow, wcs_source: str, verified: bool, n_stars: int,
                         rms_arcsec: float, qc: dict) -> None:
    row.wcs_source, row.wcs_verified = wcs_source, verified
    row.n_astrometric_stars = n_stars
    row.astrometric_rms_arcsec = round(rms_arcsec, 3) if np.isfinite(rms_arcsec) else float("nan")
    lim = qc["rms_thresholds"]["astrometric_rms_max_arcsec"]
    if np.isfinite(rms_arcsec) and rms_arcsec > lim:
        row.fail("astrometry", f"astrometric RMS {rms_arcsec:.2f}\" > {lim}\"")
    if not verified and qc.get("require_verified_wcs", True):
        row.fail("astrometry", f"WCS not verified against reference stars (source={wcs_source or 'none'})")


def apply_photometric_qc(row: QCRow, zp: float, zp_rms: float, n_stars: int,
                         qc: dict, min_stars: int) -> None:
    row.zeropoint_mag = round(zp, 4) if np.isfinite(zp) else float("nan")
    row.photometric_rms_mag = round(zp_rms, 4) if np.isfinite(zp_rms) else float("nan")
    row.n_photometric_stars = n_stars
    lim = qc["rms_thresholds"]["photometric_rms_max_mag"]
    if n_stars < min_stars:
        row.fail("photometry", f"only {n_stars} zero-point stars (< {min_stars})")
    elif zp_rms > lim:
        row.fail("photometry", f"photometric RMS {zp_rms:.3f} > {lim} mag")


def build_qc_matrix(rows: list[QCRow]) -> pd.DataFrame:
    recs = []
    for r in rows:
        d = asdict(r)
        d.pop("_photo"), d.pop("_astro")
        recs.append(d)
    return pd.DataFrame(recs)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    if n == 0:
        return {"total_frames": 0, "photometry_pass": 0, "photometry_pass_pct": 0.0,
                "astrometry_pass": 0, "astrometry_pass_pct": 0.0}
    return {
        "total_frames": n,
        "photometry_pass": int(df["photometry_pass"].sum()),
        "photometry_pass_pct": round(100 * df["photometry_pass"].mean(), 1),
        "astrometry_pass": int(df["astrometry_pass"].sum()),
        "astrometry_pass_pct": round(100 * df["astrometry_pass"].mean(), 1),
        "header_invalid": int((~df["header_valid"]).sum()),
        "no_signal": int((~df["signal_present"]).sum()),
        "artifact_flagged": int(df["artifact_flag"].sum()),
        "wcs_verified": int(df["wcs_verified"].sum()),
        "median_fwhm_arcsec": float(np.nanmedian(df["fwhm_arcsec"])) if df["fwhm_arcsec"].notna().any() else None,
        "median_astrometric_rms_arcsec": float(np.nanmedian(df["astrometric_rms_arcsec"])) if df["astrometric_rms_arcsec"].notna().any() else None,
    }
