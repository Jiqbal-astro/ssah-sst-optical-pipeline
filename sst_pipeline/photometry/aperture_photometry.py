"""
aperture_photometry.py
----------------------
Point sources : circular aperture + sigma-clipped annulus sky.
Trails        : rotated rectangular aperture (length + end padding) with a
                rectangular sky annulus (as in trailed-source photometry,
                e.g. Veres et al. 2012).
Both are corrected to TOTAL flux with the analytic Gaussian enclosed-energy
fraction, so stars (used for the zero-point) and trails are on one scale.

Uncertainty: CCD equation (Merline & Howell 1995)
    var = F/g + n_ap*s^2 + n_ap^2*s^2/n_sky        [ADU^2]
with s the empirical per-pixel sky RMS (includes read noise).

A zero-point measured on the frame itself already includes atmospheric
extinction; extinction is applied only with the configured fallback ZP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import SigmaClip
from photutils.aperture import (ApertureStats, CircularAnnulus, CircularAperture,
                                RectangularAnnulus, RectangularAperture, aperture_photometry)
from scipy.special import erf

_SQ2 = np.sqrt(2.0)


def enclosed_circle(r: float, fwhm: float) -> float:
    s = fwhm / 2.3548
    return float(1.0 - np.exp(-r * r / (2 * s * s)))


def enclosed_trail(length: float, halfwidth: float, pad: float, fwhm: float) -> float:
    s = fwhm / 2.3548
    across = erf(halfwidth / (_SQ2 * s))
    if length <= 0:
        return float(across)
    u = np.linspace(-length / 2, length / 2, 401)
    hi = 0.5 * (1 + erf((length / 2 + pad - u) / (_SQ2 * s)))
    lo = 0.5 * (1 + erf((-length / 2 - pad - u) / (_SQ2 * s)))
    return float(across * np.mean(hi - lo))


@dataclass
class PhotResult:
    flux: float          # total (aperture-corrected) ADU
    flux_err: float
    snr: float
    m_inst: float        # -2.5 log10(flux / exptime); NaN if flux <= 0
    m_inst_err: float
    sky_per_px: float
    aperture: str


def _finish(flux, var, apcorr, exptime, sky, kind) -> PhotResult:
    err = np.sqrt(max(var, 0.0))
    f, e = flux / apcorr, err / apcorr
    snr = f / e if e > 0 else 0.0
    if f > 0 and exptime > 0:
        m, me = -2.5 * np.log10(f / exptime), 1.0857 / max(snr, 1e-6)
    else:
        m, me = float("nan"), float("nan")
    return PhotResult(float(f), float(e), float(snr), float(m), float(me), float(sky), kind)


def point_photometry(data, x, y, fwhm, cfg, gain, exptime) -> list[PhotResult]:
    x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
    if len(x) == 0:
        return []
    r = max(cfg.get("aperture_radius_fwhm", 1.5) * fwhm, cfg.get("min_aperture_radius_px", 2.0))
    r_in = max(cfg.get("annulus_inner_fwhm", 4.0) * fwhm, r + 2)
    r_out = max(cfg.get("annulus_outer_fwhm", 7.0) * fwhm, r_in + 4)
    pos = np.c_[x, y]
    ap, ann = CircularAperture(pos, r), CircularAnnulus(pos, r_in, r_out)
    st = ApertureStats(data, ann, sigma_clip=SigmaClip(sigma=3.0, maxiters=5))
    raw = aperture_photometry(data, ap, method="exact")["aperture_sum"].value
    area, n_sky = ap.area, np.atleast_1d(st.sum_aper_area.value)
    sky, s = np.atleast_1d(st.median), np.atleast_1d(st.std)
    apcorr = enclosed_circle(r, fwhm)
    out = []
    for i in range(len(x)):
        f = raw[i] - sky[i] * area
        var = max(f, 0) / gain + area * s[i] ** 2 + area ** 2 * s[i] ** 2 / max(n_sky[i], 1)
        out.append(_finish(f, var, apcorr, exptime, sky[i], f"circle r={r:.1f}"))
    return out


def trail_photometry(data, x, y, length, theta, fwhm, cfg, gain, exptime) -> PhotResult:
    hw = max(cfg.get("trail_halfwidth_fwhm", 1.5) * fwhm, cfg.get("min_aperture_radius_px", 2.0))
    pad = cfg.get("trail_end_padding_fwhm", 1.5) * fwhm
    w = length + 2 * pad
    ap = RectangularAperture((x, y), w=w, h=2 * hw, theta=theta)
    ann = RectangularAnnulus((x, y), w_in=w + 4 * fwhm, w_out=w + 16 * fwhm,
                             h_in=2 * hw + 4 * fwhm, h_out=2 * hw + 16 * fwhm, theta=theta)
    st = ApertureStats(data, ann, sigma_clip=SigmaClip(sigma=3.0, maxiters=5))
    raw = float(aperture_photometry(data, ap, method="exact")["aperture_sum"].value[0])
    area = ap.area
    sky, s, n_sky = float(st.median), float(st.std), float(st.sum_aper_area.value)
    f = raw - sky * area
    var = max(f, 0) / gain + area * s * s + area * area * s * s / max(n_sky, 1)
    return _finish(f, var, enclosed_trail(length, hw, pad, fwhm), exptime, sky,
                   f"trail {w:.1f}x{2*hw:.1f}")


def solve_zeropoint(m_cat: np.ndarray, m_inst: np.ndarray, m_inst_err: np.ndarray,
                    min_stars: int) -> tuple[float, float, float, int]:
    """Returns (ZP, ZP_error_of_mean, scatter_RMS, n_used) with 3-sigma clipping."""
    ok = np.isfinite(m_cat) & np.isfinite(m_inst)
    if ok.sum() < max(min_stars, 2):
        return float("nan"), float("nan"), float("nan"), int(ok.sum())
    d = m_cat[ok] - m_inst[ok]
    keep = np.ones(len(d), bool)
    for _ in range(5):   # MAD-based clipping: robust for small star samples
        med = np.median(d[keep])
        mad_std = 1.4826 * np.median(np.abs(d[keep] - med))
        new = np.abs(d - med) <= max(3 * mad_std, 0.05)
        if new.sum() < 2 or np.array_equal(new, keep):
            break
        keep = new
    n = int(keep.sum())
    rms = float(np.std(d[keep], ddof=1)) if n > 1 else float("nan")
    return float(np.median(d[keep])), rms / np.sqrt(n) if n > 1 else float("nan"), rms, n
