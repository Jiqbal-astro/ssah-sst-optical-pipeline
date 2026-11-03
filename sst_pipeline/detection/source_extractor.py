"""
source_extractor.py
-------------------
One SEP pass per frame (Bertin & Arnouts 1996 algorithm) serving QC, WCS
verification, photometric zero-point and target measurement, so all
stages see the same sources.

Every detection is re-measured from its segmentation pixels with flux-
weighted second moments along / across its principal axis:

    var_along  = L^2 / 12 + sigma_psf^2      (uniform trail convolved with PSF)
    var_across = sigma_psf^2
    =>  L = sqrt(12 (var_along - var_across))

This gives trail length, width, direction and the flux-weighted centroid
(= mid-exposure position for uniform motion) for streaks. Trail geometry
is then refined from the along-track profile (refine_trail). Collinear fragments of one trail (broken by noise or a crossing star) are
merged.

Coordinates are 0-based numpy pixels (x = column, y = row), consistent with
astropy WCS pixel_to_world / world_to_pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sep
from scipy import ndimage

from sst_pipeline.utils.logging_config import get_logger

log = get_logger(__name__)

sep.set_extract_pixstack(10_000_000)
sep.set_sub_object_limit(4096)


@dataclass
class Source:
    idx: int
    x: float
    y: float
    x_err: float
    y_err: float
    flux: float              # isophotal (segment) flux, ADU
    flux_err: float
    snr: float
    a: float                 # flux-weighted RMS along major axis (px)
    b: float                 # flux-weighted RMS along minor axis (px)
    theta: float             # major-axis angle, rad, CCW from +x
    elongation: float
    fwhm: float              # PSF FWHM estimate (px): 2.355*b for trails, 2.355*sqrt(ab) else
    length: float            # trail length with PSF removed (px)
    x0: float = 0.0          # trail end points (direction ambiguous by 180 deg)
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    npix: int = 0
    peak: float = 0.0
    saturated: bool = False
    sep_flag: int = 0
    n_fragments: int = 1
    along_min: float = 0.0   # pixel extent along the major axis, relative to centroid
    along_max: float = 0.0
    seg_id: int = 0          # SEP segment id (single-segment sources)
    profile_flux: float = float("nan")   # trails: robust plateau * length (total flux)


@dataclass
class ExtractionResult:
    frame_id: str
    sources: list[Source] = field(default_factory=list)
    bkg_level: float = 0.0
    bkg_rms: float = 0.0
    error: Optional[str] = None


def _moments(xs, ys, w):
    wsum = w.sum()
    if wsum <= 0:
        return None
    xc, yc = (w * xs).sum() / wsum, (w * ys).sum() / wsum
    dx, dy = xs - xc, ys - yc
    cxx, cyy, cxy = (w * dx * dx).sum() / wsum, (w * dy * dy).sum() / wsum, (w * dx * dy).sum() / wsum
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
    theta = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    return xc, yc, l1, l2, theta


def _measure(idx, xs, ys, vals, raw_vals, bkg_rms, gain, saturation, n_frag=1, flag=0):
    w = np.clip(vals, 0, None)
    m = _moments(xs, ys, w)
    if m is None:
        return None
    xc, yc, var_a, var_b, theta = m
    a, b = np.sqrt(var_a), np.sqrt(var_b)
    flux = float(vals.sum())
    flux_err = float(np.sqrt(len(vals) * bkg_rms ** 2 + max(flux, 0.0) / gain))
    snr = flux / flux_err if flux_err > 0 else 0.0
    length = float(np.sqrt(max(12.0 * (var_a - var_b), 0.0)))
    elong = a / b
    fwhm = 2.3548 * (b if elong >= 2.0 else np.sqrt(a * b))
    ux, uy = np.cos(theta), np.sin(theta)
    proj = (xs - xc) * ux + (ys - yc) * uy
    half = length / 2.0
    # centroid uncertainty: RMS width / SNR per principal axis, rotated to x/y
    s_al, s_ac = a / max(snr, 1e-3), b / max(snr, 1e-3)
    sx = np.sqrt((s_al * ux) ** 2 + (s_ac * uy) ** 2)
    sy = np.sqrt((s_al * uy) ** 2 + (s_ac * ux) ** 2)
    return Source(idx=idx, x=float(xc), y=float(yc), x_err=float(sx), y_err=float(sy),
                  flux=flux, flux_err=flux_err, snr=float(snr), a=float(a), b=float(b),
                  theta=float(theta), elongation=float(elong), fwhm=float(fwhm), length=length,
                  x0=float(xc - half * ux), y0=float(yc - half * uy),
                  x1=float(xc + half * ux), y1=float(yc + half * uy),
                  npix=int(len(vals)), peak=float(vals.max()),
                  saturated=bool(raw_vals.max() >= saturation) if raw_vals is not None else False,
                  sep_flag=int(flag), n_fragments=n_frag,
                  along_min=float(proj.min()), along_max=float(proj.max()))


def refine_trail(s: Source, sub: np.ndarray, bkg_rms: float, psf_sigma: float) -> None:
    """Robust trail geometry from the along-track profile.

    A uniform trail convolved with the PSF has an along-track profile
    plateau * Phi((L/2 - |s|)/sigma): it crosses HALF the plateau exactly at
    the true end points. Using the half-maximum crossings of a median-
    smoothed profile makes length, midpoint and flux insensitive to stars
    the trail crosses and to deblending (which bias flux-weighted moments).
    """
    pad = 4 * psf_sigma + 5
    ca = np.arange(-3.0 * psf_sigma, 3.0 * psf_sigma + 1e-9, 0.5)
    win = 2.5 * psf_sigma

    def sample():
        ux, uy = np.cos(s.theta), np.sin(s.theta)
        sa = np.arange(s.along_min - pad, s.along_max + pad, 0.5)
        S, Cc = np.meshgrid(sa, ca, indexing="ij")
        X, Y = s.x + S * ux - Cc * uy, s.y + S * uy + Cc * ux
        v = ndimage.map_coordinates(sub, [Y.ravel(), X.ravel()], order=1, cval=np.nan).reshape(S.shape)
        return sa, v

    # 1-2 passes: re-fit the trail axis from across-track centroids, because
    # moments of a trail that crosses/merges with stars give a tilted axis
    for _ in range(2):
        sa, v = sample()
        inwin = np.abs(ca) <= win
        P = np.nansum(v[:, inwin], axis=1) * 0.5
        core_mask = (sa > s.along_min) & (sa < s.along_max)
        if core_mask.sum() < 5:
            return
        lvl = np.median(P[core_mask])
        use = core_mask & (P > 0.5 * lvl) & (P < 2.0 * lvl)      # skip stars on the trail
        if use.sum() < 5 or lvl <= 0:
            break
        w = np.clip(np.nan_to_num(v[use]), 0, None)
        cbar = (w * ca[None, :]).sum(1) / np.maximum(w.sum(1), 1e-9)
        coef = np.polyfit(sa[use], cbar, 1)
        resid = cbar - np.polyval(coef, sa[use])
        ok = np.abs(resid) < 3 * 1.4826 * np.median(np.abs(resid)) + 1e-6
        if ok.sum() >= 5:
            coef = np.polyfit(sa[use][ok], cbar[ok], 1)
        ux, uy = np.cos(s.theta), np.sin(s.theta)
        s.x, s.y = s.x - coef[1] * uy, s.y + coef[1] * ux          # shift onto the axis
        s.theta = float(s.theta + np.arctan(coef[0]))
    sa, v = sample()
    ux, uy = np.cos(s.theta), np.sin(s.theta)
    P = np.nansum(v[:, np.abs(ca) <= win], axis=1) * 0.5    # flux per unit length within +-2.5 sigma
    Ps = ndimage.median_filter(P, size=max(int(8 * psf_sigma / 0.5), 5), mode="nearest")
    core = Ps[(sa > s.along_min) & (sa < s.along_max)]
    if core.size < 5:
        return
    plateau = float(np.median(core))
    if plateau <= 0:
        return
    above = Ps >= 0.5 * plateau
    idx = np.nonzero(above)[0]
    if idx.size < 3:
        return
    # longest contiguous run above half-max (ignores isolated stars beyond the ends)
    runs = np.split(idx, np.nonzero(np.diff(idx) > 2 * psf_sigma / 0.5 + 1)[0] + 1)
    run = max(runs, key=len)
    i0, i1 = run[0], run[-1]

    def cross(i, j):  # linear interpolation of the half-max crossing between samples i, j
        y0, y1 = Ps[i] - 0.5 * plateau, Ps[j] - 0.5 * plateau
        return sa[i] + (sa[j] - sa[i]) * (y0 / (y0 - y1)) if y0 != y1 else sa[i]
    lo = cross(i0 - 1, i0) if i0 > 0 else sa[i0]
    hi = cross(i1, i1 + 1) if i1 + 1 < len(sa) else sa[i1]
    L = hi - lo
    if L <= 0:
        return
    mid = 0.5 * (lo + hi)
    s.x, s.y = s.x + mid * ux, s.y + mid * uy
    s.length = float(L)
    s.x0, s.y0 = s.x - L / 2 * ux, s.y - L / 2 * uy
    s.x1, s.y1 = s.x + L / 2 * ux, s.y + L / 2 * uy
    s.along_min, s.along_max = s.along_min - mid, s.along_max - mid
    # end-point error from profile noise and edge slope; midpoint averages two ends
    sig_P = 1.4826 * float(np.median(np.abs(core - plateau))) + 1e-9
    sig_end = sig_P * psf_sigma * np.sqrt(2 * np.pi) / plateau
    s_al = sig_end / np.sqrt(2)
    s_ac = s.b / max(s.snr, 1e-3)
    s.x_err = float(np.hypot(s_al * ux, s_ac * uy))
    s.y_err = float(np.hypot(s_al * uy, s_ac * ux))
    s.profile_flux = float(plateau * L / 0.98758)   # +-2.5 sigma window holds 98.76 %


def _collinear(s1: Source, s2: Source, max_ang, max_off, max_gap) -> bool:
    dang = abs(((np.degrees(s1.theta - s2.theta) + 90) % 180) - 90)
    if dang > max_ang:
        return False
    ux, uy = np.cos(s1.theta), np.sin(s1.theta)
    dx, dy = s2.x - s1.x, s2.y - s1.y
    if abs(-uy * dx + ux * dy) > max_off:
        return False
    along = ux * dx + uy * dy
    if np.cos(s1.theta - s2.theta) >= 0:
        lo2, hi2 = along + s2.along_min, along + s2.along_max
    else:
        lo2, hi2 = along - s2.along_max, along - s2.along_min
    gap = max(lo2 - s1.along_max, s1.along_min - hi2, 0.0)
    return gap <= max_gap


def extract_sources(frame_id: str, data: np.ndarray, raw: Optional[np.ndarray], cfg: dict,
                    gain: float, saturation_adu: float, elong_thresh: float = 2.5) -> ExtractionResult:
    if data.size == 0:
        return ExtractionResult(frame_id, error="Empty data array")
    try:
        img = np.ascontiguousarray(data, dtype=np.float64)
        mesh = int(cfg.get("background_mesh_px", 64))
        bkg = sep.Background(img, bw=mesh, bh=mesh, fw=3, fh=3)
        sub = img - bkg.back()
        rms = float(bkg.globalrms)
        objs, seg = sep.extract(sub, float(cfg.get("detection_sigma", 3.0)), err=rms,
                                minarea=int(cfg.get("min_pixels", 5)),
                                deblend_nthresh=int(cfg.get("deblend_nthresh", 32)),
                                deblend_cont=float(cfg.get("deblend_cont", 0.005)),
                                segmentation_map=True)
        res = ExtractionResult(frame_id, bkg_level=float(bkg.globalback), bkg_rms=rms)
        if len(objs) == 0:
            return res

        slices = ndimage.find_objects(seg)
        pix: dict[int, tuple] = {}
        for k, sl in enumerate(slices):
            if sl is None:
                continue
            m = seg[sl] == (k + 1)
            yy, xx = np.nonzero(m)
            pix[k] = (xx + sl[1].start, yy + sl[0].start)

        def measure_ids(ids, new_idx):
            xs = np.concatenate([pix[i][0] for i in ids])
            ys = np.concatenate([pix[i][1] for i in ids])
            raw_vals = raw[ys, xs] if raw is not None and raw.shape == img.shape else None
            return _measure(new_idx, xs.astype(float), ys.astype(float), sub[ys, xs], raw_vals,
                            rms, gain, saturation_adu, n_frag=len(ids),
                            flag=int(np.bitwise_or.reduce([int(objs["flag"][i]) for i in ids])))

        singles = {k: measure_ids([k], k) for k in pix}
        singles = {k: s for k, s in singles.items() if s is not None}

        groups = {k: [k] for k in singles}
        if cfg.get("merge_streak_fragments", True):
            elong_ids = [k for k, s in singles.items() if s.elongation >= elong_thresh]
            parent = {k: k for k in elong_ids}

            def find(k):
                while parent[k] != k:
                    parent[k] = parent[parent[k]]
                    k = parent[k]
                return k
            for i, ki in enumerate(elong_ids):
                for kj in elong_ids[i + 1:]:
                    if _collinear(singles[ki], singles[kj],
                                  cfg.get("fragment_max_angle_deg", 5.0),
                                  cfg.get("fragment_max_offset_px", 3.0),
                                  cfg.get("fragment_max_gap_px", 30.0)):
                        parent[find(kj)] = find(ki)
            merged: dict[int, list[int]] = {}
            for k in elong_ids:
                merged.setdefault(find(k), []).append(k)
            for root, members in merged.items():
                if len(members) > 1:
                    for m_ in members:
                        groups.pop(m_, None)
                    groups[root] = sorted(members)

        out = []
        for new_idx, (root, members) in enumerate(sorted(groups.items())):
            s = singles[root] if len(members) == 1 else measure_ids(members, root)
            if s is not None:
                if len(members) == 1:
                    # windowed centroid (SExtractor XWIN) for compact sources
                    if s.elongation < elong_thresh:
                        # threshold-free FWHM: Gaussian half-light radius = FWHM/2
                        r50, fr = sep.flux_radius(sub, np.array([s.x]), np.array([s.y]),
                                                  np.array([max(5.0 * s.fwhm, 8.0)]), 0.5,
                                                  segmap=seg, seg_id=np.array([root + 1], dtype=np.int32),
                                                  subpix=5)
                        if fr[0] == 0 and np.isfinite(r50[0]) and r50[0] > 0:
                            s.fwhm = float(2.0 * r50[0])
                        sig = max(s.fwhm, 1.0) / 2.3548
                        xw, yw, fl = sep.winpos(sub, np.array([s.x]), np.array([s.y]), np.array([sig]))
                        if fl[0] == 0 and np.hypot(xw[0] - s.x, yw[0] - s.y) < 2.0:
                            s.x, s.y = float(xw[0]), float(yw[0])
                s.idx = new_idx
                out.append(s)
        compact = [s.fwhm for s in out if s.elongation < elong_thresh and s.snr >= 20 and not s.saturated]
        psf_sigma = (float(np.median(compact)) if compact else 3.0) / 2.3548
        for s in out:
            if s.elongation >= elong_thresh and s.length >= 4 * psf_sigma:
                refine_trail(s, sub, rms, psf_sigma)
        res.sources = out
        n_merged = sum(1 for m in groups.values() if len(m) > 1)
        log.debug("frame %s: %d sources (%d merged trails), bkg=%.1f rms=%.2f",
                  frame_id, len(out), n_merged, res.bkg_level, rms)
        return res
    except Exception as exc:  # noqa: BLE001 — never crash a batch on one frame
        log.error("Source extraction failed for %s: %s", frame_id, exc)
        return ExtractionResult(frame_id, error=f"{type(exc).__name__}: {exc}")
