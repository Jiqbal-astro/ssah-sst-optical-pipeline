"""
wcs_solver.py
-------------
Plate solution in two steps:

1. INITIAL WCS
   a. header WCS (written by the mount/TCS pointing model), or
   b. local astrometry.net `solve-field` (offline, only with --solve-wcs),
      run on the calibrated frame in a separate work directory (raw data
      are never written to), with pixel-scale and position hints.

2. VERIFY + REFINE against reference stars
   A pointing-model WCS is typically off by tens of arcsec to arcminutes
   and cannot be used for SST astrometry as-is.
   Detected reference stars are matched to the catalogue (2-D offset vote to
   absorb pointing error, then one-to-one nearest-neighbour), and a TAN
   (+SIP distortion when enough stars) WCS is fitted with
   astropy.wcs.utils.fit_wcs_from_points, with 3-sigma clipping.
   The plate is VERIFIED if >= min_match_stars matched and RMS is finite.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import fit_wcs_from_points, proj_plane_pixel_scales
from scipy.spatial import cKDTree
import warnings

from sst_pipeline.utils.logging_config import get_logger

log = get_logger(__name__)
warnings.simplefilter("ignore", FITSFixedWarning)


@dataclass
class WCSSolution:
    success: bool
    wcs: Optional[WCS] = None
    source: str = "none"            # header | solve-field | header+refit | solve-field+refit
    verified: bool = False
    n_matched: int = 0
    rms_ra_arcsec: float = float("nan")
    rms_dec_arcsec: float = float("nan")
    rms_arcsec: float = float("nan")
    offset_arcsec: float = float("nan")   # correction applied to the initial WCS
    message: str = ""
    matched_det_idx: np.ndarray = field(default_factory=lambda: np.array([], int))
    matched_cat_idx: np.ndarray = field(default_factory=lambda: np.array([], int))

    @property
    def pixel_scale_arcsec(self) -> float:
        return float(np.mean(proj_plane_pixel_scales(self.wcs)) * 3600) if self.wcs is not None else float("nan")


def header_wcs(header: fits.Header, shape) -> Optional[WCS]:
    try:
        w = WCS(header, naxis=2)
        if not w.has_celestial or not w.is_celestial:
            return None
        w.pixel_shape = (shape[1], shape[0])
        return w
    except Exception:  # noqa: BLE001
        return None


def solve_field_available() -> bool:
    return shutil.which("solve-field") is not None


def solve_field_command(image_path: str, out_dir: str, scale_arcsec: float, scale_tol: float,
                        ra: Optional[float], dec: Optional[float], radius_deg: float,
                        timeout_s: int) -> list[str]:
    cmd = ["solve-field", image_path, "--overwrite", "--no-plots", "--new-fits", "none",
           "--dir", out_dir, "--cpulimit", str(int(timeout_s)), "--crpix-center",
           "--scale-units", "arcsecperpix",
           "--scale-low", f"{scale_arcsec * (1 - scale_tol):.4f}",
           "--scale-high", f"{scale_arcsec * (1 + scale_tol):.4f}"]
    if ra is not None and dec is not None:
        cmd += ["--ra", f"{ra:.6f}", "--dec", f"{dec:.6f}", "--radius", f"{radius_deg:.3f}"]
    return cmd


def run_solve_field(frame_id: str, data: np.ndarray, header: fits.Header, work_dir: str,
                    scale_arcsec: float, ra_hint, dec_hint, cfg: dict) -> Optional[WCS]:
    if not solve_field_available():
        log.warning("solve-field not installed — blind solving unavailable")
        return None
    os.makedirs(work_dir, exist_ok=True)
    img = os.path.join(work_dir, f"{frame_id}_solve.fits")
    fits.PrimaryHDU(data.astype(np.float32)).writeto(img, overwrite=True)
    cmd = solve_field_command(img, work_dir, scale_arcsec, cfg.get("solve_field_scale_tolerance", 0.1),
                              ra_hint, dec_hint, cfg.get("solve_field_search_radius_deg", 2.0),
                              cfg.get("solve_field_timeout_s", 120))
    try:
        subprocess.run(cmd, capture_output=True, text=True,
                       timeout=cfg.get("solve_field_timeout_s", 120) + 30)
    except subprocess.TimeoutExpired:
        log.warning("solve-field timed out for %s", frame_id)
        return None
    wcs_path = os.path.join(work_dir, f"{frame_id}_solve.wcs")
    if not os.path.exists(wcs_path):
        return None
    return header_wcs(fits.getheader(wcs_path), data.shape)


def _offset_vote(xd, yd, xc, yc, radius_px, bin_px):
    """Most common (catalogue - detection) pixel offset: robust to pointing error."""
    dx = (xc[None, :] - xd[:, None]).ravel()
    dy = (yc[None, :] - yd[:, None]).ravel()
    keep = (np.abs(dx) <= radius_px) & (np.abs(dy) <= radius_px)
    if keep.sum() < 3:
        return None
    nb = max(int(2 * radius_px / bin_px), 3)
    H, ex, ey = np.histogram2d(dx[keep], dy[keep], bins=nb,
                               range=[[-radius_px, radius_px], [-radius_px, radius_px]])
    i, j = np.unravel_index(np.argmax(H), H.shape)
    if H[i, j] < 3:
        return None
    sel = keep.copy()
    sel[keep] = (np.abs(dx[keep] - 0.5 * (ex[i] + ex[i + 1])) <= bin_px) & \
                (np.abs(dy[keep] - 0.5 * (ey[j] + ey[j + 1])) <= bin_px)
    return float(np.median(dx[sel])), float(np.median(dy[sel]))


def _match(xd, yd, xc, yc, tol_px):
    """One-to-one nearest-neighbour match within tol_px."""
    if len(xc) == 0 or len(xd) == 0:
        return np.array([], int), np.array([], int)
    tree = cKDTree(np.c_[xc, yc])
    d, j = tree.query(np.c_[xd, yd], distance_upper_bound=tol_px)
    ok = np.isfinite(d)
    di, cj, dd = np.nonzero(ok)[0], j[ok], d[ok]
    order = np.argsort(dd)
    used_c, used_d, out_d, out_c = set(), set(), [], []
    for k in order:
        if cj[k] in used_c or di[k] in used_d:
            continue
        used_c.add(cj[k]); used_d.add(di[k]); out_d.append(di[k]); out_c.append(cj[k])
    return np.array(out_d, int), np.array(out_c, int)


def _residuals(w, x, y, ra, dec):
    sky = w.pixel_to_world(x, y)
    dra = ((sky.ra.deg - ra + 180) % 360 - 180) * np.cos(np.radians(dec)) * 3600
    ddec = (sky.dec.deg - dec) * 3600
    return dra, ddec


def refine_wcs(initial: WCS, source: str, xd: np.ndarray, yd: np.ndarray, flux: np.ndarray,
               cat_ra: np.ndarray, cat_dec: np.ndarray, cat_mag: np.ndarray,
               shape, cfg: dict) -> WCSSolution:
    ny, nx = shape
    scale = float(np.mean(proj_plane_pixel_scales(initial)) * 3600)
    min_match = int(cfg.get("min_match_stars", 8))
    sol = WCSSolution(success=True, wcs=initial, source=source)
    if len(xd) < min_match or len(cat_ra) < min_match:
        sol.message = f"too few stars to verify (det={len(xd)}, cat={len(cat_ra)})"
        return sol

    xc, yc = initial.world_to_pixel(SkyCoord(cat_ra * u.deg, cat_dec * u.deg))
    margin = cfg.get("initial_offset_search_arcsec", 120.0) / scale
    inside = (xc > -margin) & (xc < nx + margin) & (yc > -margin) & (yc < ny + margin)
    ci = np.nonzero(inside)[0]
    ci = ci[np.argsort(cat_mag[ci])][:400]
    di = np.argsort(-flux)[:200]
    off = _offset_vote(xd[di], yd[di], xc[ci], yc[ci], margin, max(2.0, 1.5 / scale))
    if off is None:
        sol.message = "no consistent star pattern between frame and catalogue"
        return sol
    ox, oy = off
    tol = cfg.get("match_radius_arcsec", 2.0) / scale

    w = initial
    for it, sip in enumerate((None, "final")):
        if it == 0:
            md, mc = _match(xd + ox, yd + oy, xc[ci], yc[ci], 3 * tol)
            mc = ci[mc]
        else:
            xc2, yc2 = w.world_to_pixel(SkyCoord(cat_ra[ci] * u.deg, cat_dec[ci] * u.deg))
            md, mc = _match(xd, yd, xc2, yc2, tol)
            mc = ci[mc]
        if len(md) < min_match:
            sol.message = f"only {len(md)} catalogue matches (< {min_match})"
            sol.n_matched = len(md)
            return sol
        deg = int(cfg.get("sip_degree", 2)) if (sip and len(md) >= cfg.get("min_stars_for_sip", 15)) else None
        if deg is not None and deg < 2:
            deg = None
        keep = np.ones(len(md), bool)
        for _ in range(3):  # fit with 3-sigma clipping
            w = fit_wcs_from_points((xd[md][keep], yd[md][keep]),
                                    SkyCoord(cat_ra[mc][keep] * u.deg, cat_dec[mc][keep] * u.deg),
                                    proj_point="center", projection="TAN", sip_degree=deg)
            dra, ddec = _residuals(w, xd[md], yd[md], cat_ra[mc], cat_dec[mc])
            r = np.hypot(dra, ddec)
            sig = np.median(r[keep]) / 1.1774 if keep.sum() > 3 else np.inf  # Rayleigh median = 1.1774 sigma
            new_keep = r < max(3 * sig, 0.3)
            if new_keep.sum() < min_match or np.array_equal(new_keep, keep):
                break
            keep = new_keep
        md, mc = md[keep], mc[keep]
    w.pixel_shape = (nx, ny)
    dra, ddec = _residuals(w, xd[md], yd[md], cat_ra[mc], cat_dec[mc])
    sol.wcs, sol.source = w, f"{source}+refit"
    sol.n_matched = len(md)
    sol.rms_ra_arcsec = float(np.sqrt(np.mean(dra ** 2)))
    sol.rms_dec_arcsec = float(np.sqrt(np.mean(ddec ** 2)))
    sol.rms_arcsec = float(np.sqrt(np.mean(dra ** 2 + ddec ** 2)))
    c0 = initial.pixel_to_world(nx / 2, ny / 2)
    sol.offset_arcsec = float(c0.separation(w.pixel_to_world(nx / 2, ny / 2)).arcsec)
    sol.verified = sol.n_matched >= min_match and np.isfinite(sol.rms_arcsec)
    sol.matched_det_idx, sol.matched_cat_idx = md, mc
    sol.message = (f"{sol.n_matched} stars, RMS {sol.rms_arcsec:.3f}\" "
                   f"(initial WCS corrected by {sol.offset_arcsec:.1f}\")")
    return sol
