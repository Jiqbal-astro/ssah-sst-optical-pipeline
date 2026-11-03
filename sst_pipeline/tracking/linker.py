"""
linker.py
---------
Static-sky rejection and kinematic linking of RSO candidates.

 1. STATIC SKY: a detection whose sky position recurs (within
    static_match_radius) in >= static_min_frames frames is fixed on the sky
    (star / galaxy below the catalogue limit) and is not an RSO.
 2. CANDIDATES: not static, not a catalogue star, not an artefact, and a
    morphology consistent with an RSO for the tracking mode. Sidereal-mode
    point sources are candidates only when the static test is possible.
 3. LINKING: seed pairs that imply a plausible rate (and, for trails, agree
    with the trail axis and trail rate) are grown with a constant-rate model
    in the tangent plane; a tracklet needs >= min_positions detections and a
    fit RMS <= max_residual. Unused trails can stand alone (a single trail
    gives position + rate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from sst_pipeline.astrometry.position_extractor import Detection
from sst_pipeline.utils.coordinate_utils import sep_arcsec, to_tangent_plane
from sst_pipeline.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class Tracklet:
    tracklet_id: str
    kind: str                                  # MULTI_FRAME | SINGLE_STREAK
    members: list[Detection] = field(default_factory=list)
    rate_arcsec_s: float = float("nan")
    pa_deg: float = float("nan")               # E of N (mod 180 for SINGLE_STREAK)
    rms_arcsec: float = float("nan")
    quality: str = "OK"
    norad_id: Optional[int] = None
    catalog_name: str = ""
    identified: bool = False
    mean_oc_along_arcsec: float = float("nan")
    mean_oc_cross_arcsec: float = float("nan")
    along_track_time_offset_s: float = float("nan")

    def to_dict(self) -> dict:
        m = self.members
        mags = [d.mag for d in m if d.mag is not None]
        return {
            "tracklet_id": self.tracklet_id, "kind": self.kind, "n_positions": len(m),
            "start_utc": m[0].epoch_utc, "end_utc": m[-1].epoch_utc,
            "arc_length_s": round((m[-1].mjd - m[0].mjd) * 86400, 3),
            "mean_ra_deg": round(float(np.mean([d.ra_deg for d in m])), 6),
            "mean_dec_deg": round(float(np.mean([d.dec_deg for d in m])), 6),
            "rate_arcsec_s": round(self.rate_arcsec_s, 4), "pa_deg": round(self.pa_deg, 2),
            "fit_rms_arcsec": round(self.rms_arcsec, 3),
            "mean_mag": round(float(np.mean(mags)), 3) if mags else None,
            "identified": self.identified, "norad_id": self.norad_id,
            "catalog_name": self.catalog_name,
            "mean_oc_along_arcsec": round(self.mean_oc_along_arcsec, 2),
            "mean_oc_cross_arcsec": round(self.mean_oc_cross_arcsec, 2),
            "along_track_time_offset_s": round(self.along_track_time_offset_s, 4),
            "quality": self.quality,
        }


def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def flag_static(dets: list[Detection], radius_arcsec: float, min_frames: int) -> bool:
    """Marks det.is_static. Returns False if too few frames for the test."""
    frames = {d.frame_index for d in dets}
    if len(frames) < min_frames or not dets:
        return False
    xyz = _unit(np.array([d.ra_deg for d in dets]), np.array([d.dec_deg for d in dets]))
    tree = cKDTree(xyz)
    chord = 2 * np.sin(np.radians(radius_arcsec / 3600) / 2)
    for i, nb in enumerate(tree.query_ball_point(xyz, chord)):
        dets[i].is_static = len({dets[j].frame_index for j in nb}) >= min_frames
    return True


def _fit(members: list[Detection], ra0: float, dec0: float, t0: float):
    t = np.array([(d.mjd - t0) * 86400 for d in members])
    xi, eta = to_tangent_plane([d.ra_deg for d in members], [d.dec_deg for d in members], ra0, dec0)
    A = np.c_[np.ones_like(t), t]
    cx, *_ = np.linalg.lstsq(A, xi, rcond=None)
    ce, *_ = np.linalg.lstsq(A, eta, rcond=None)
    res = np.hypot(xi - A @ cx, eta - A @ ce)
    dof = max(len(t) - 2, 1)
    rms = float(np.sqrt(np.sum(res ** 2) / dof)) if len(t) > 2 else 0.0
    return cx, ce, rms


def _pa_diff_mod180(a, b):
    return abs(((a - b) + 90) % 180 - 90)


def _trail_consistent(d: Detection, rate: float, pa: float, cfg) -> bool:
    if not d.is_trailed or not np.isfinite(d.trail_pa_deg):
        return True
    if _pa_diff_mod180(pa % 180, d.trail_pa_deg) > cfg["streak_pa_tolerance_deg"]:
        return False
    return abs(d.trail_rate_arcsec_s / max(rate, 1e-9) - 1) <= cfg["streak_rate_tolerance"]


def link_tracklets(cands: list[Detection], cfg: dict) -> list[Tracklet]:
    cands = sorted(cands, key=lambda d: d.mjd)
    frames = sorted({d.frame_index for d in cands})
    by_frame: dict[int, list[Detection]] = {f: [] for f in frames}
    ftime = {}
    for d in cands:
        by_frame[d.frame_index].append(d)
        ftime[d.frame_index] = d.mjd
    used: set[str] = set()
    tracklets: list[Tracklet] = []
    gap = cfg["max_frame_gap_sec"]
    tol = cfg["link_tolerance_arcsec"]

    for fi_pos, fi in enumerate(frames):
        for a in by_frame[fi]:
            if a.det_id in used:
                continue
            best = None
            for fj in frames[fi_pos + 1:]:
                if (ftime[fj] - ftime[fi]) * 86400 > gap:
                    break
                for b in by_frame[fj]:
                    if b.det_id in used:
                        continue
                    dt = (b.mjd - a.mjd) * 86400
                    if dt <= 0:
                        continue
                    rate = sep_arcsec(a.ra_deg, a.dec_deg, b.ra_deg, b.dec_deg) / dt
                    if not (cfg["min_rate_arcsec_per_sec"] <= rate <= cfg["max_rate_arcsec_per_sec"]):
                        continue
                    xi_b, eta_b = to_tangent_plane(b.ra_deg, b.dec_deg, a.ra_deg, a.dec_deg)
                    pa = float(np.degrees(np.arctan2(xi_b, eta_b)) % 360)
                    if not (_trail_consistent(a, rate, pa, cfg) and _trail_consistent(b, rate, pa, cfg)):
                        continue
                    chain = [a, b]
                    for fk in frames[fi_pos + 1:]:
                        if fk == fj:
                            continue
                        if (ftime[fk] - max(c.mjd for c in chain)) * 86400 > gap:
                            break
                        cx, ce, _ = _fit(chain, a.ra_deg, a.dec_deg, a.mjd)
                        best_d, best_r = None, tol
                        for c in by_frame[fk]:
                            if c.det_id in used or any(c.det_id == q.det_id for q in chain):
                                continue
                            tk = (c.mjd - a.mjd) * 86400
                            xi, eta = to_tangent_plane(c.ra_deg, c.dec_deg, a.ra_deg, a.dec_deg)
                            r = float(np.hypot(xi - (cx[0] + cx[1] * tk), eta - (ce[0] + ce[1] * tk)))
                            if r < best_r and _trail_consistent(c, rate, pa, cfg):
                                best_d, best_r = c, r
                        if best_d is not None:
                            chain.append(best_d)
                            chain.sort(key=lambda d: d.mjd)
                    _, _, rms = _fit(chain, a.ra_deg, a.dec_deg, a.mjd)
                    key = (len(chain), -rms)
                    if best is None or key > best[0]:
                        best = (key, chain, rms)
            if best and len(best[1]) >= cfg["min_positions"] and best[2] <= cfg["max_residual_arcsec"]:
                chain = best[1]
                cx, ce, rms = _fit(chain, chain[0].ra_deg, chain[0].dec_deg, chain[0].mjd)
                tr = Tracklet(tracklet_id=f"T{len(tracklets)+1:04d}", kind="MULTI_FRAME",
                              members=chain, rate_arcsec_s=float(np.hypot(cx[1], ce[1])),
                              pa_deg=float(np.degrees(np.arctan2(cx[1], ce[1])) % 360), rms_arcsec=rms)
                tracklets.append(tr)
                for d in chain:
                    used.add(d.det_id)
                    d.status, d.tracklet_id = "LINKED", tr.tracklet_id

    if cfg.get("report_single_frame_streaks", True):
        for d in cands:
            if d.det_id not in used and d.is_trailed:
                tr = Tracklet(tracklet_id=f"T{len(tracklets)+1:04d}", kind="SINGLE_STREAK",
                              members=[d], rate_arcsec_s=d.trail_rate_arcsec_s,
                              pa_deg=d.trail_pa_deg, rms_arcsec=float("nan"),
                              quality="SINGLE_FRAME (direction ambiguous by 180 deg)")
                tracklets.append(tr)
                used.add(d.det_id)
                d.status, d.tracklet_id = "SINGLE_STREAK", tr.tracklet_id
    for d in cands:
        if d.det_id not in used:
            d.status = "UNLINKED"
    log.info("Linking: %d candidates -> %d multi-frame tracklets, %d single streaks, %d unlinked",
             len(cands), sum(t.kind == "MULTI_FRAME" for t in tracklets),
             sum(t.kind == "SINGLE_STREAK" for t in tracklets),
             sum(d.status == "UNLINKED" for d in cands))
    return tracklets
