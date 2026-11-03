"""
known_object_matcher.py
-----------------------
Correlates detections and tracklets with SGP4 predictions.

TLE errors are dominated by the ALONG-track component (Flohrer et al. 2008:
km-level along-track vs ~100 m cross-track for LEO, growing with TLE age),
so O-C is decomposed along / across the predicted direction of motion and
tested against separate tolerances: a single circular radius either
rejects true matches of a few-day-old LEO TLE or accepts unrelated objects
crossing near the prediction.

The mean along-track O-C divided by the predicted rate is reported as a
time offset: for objects with precise orbits (GNSS) this measures the
camera time bias directly.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from astropy.time import Time

from sst_pipeline.astrometry.position_extractor import Detection
from sst_pipeline.tracking.linker import Tracklet, _pa_diff_mod180
from sst_pipeline.tracking.orbit_propagator import Predictions, Propagator
from sst_pipeline.utils.coordinate_utils import sep_arcsec, to_tangent_plane
from sst_pipeline.utils.logging_config import get_logger

log = get_logger(__name__)


def _oc(det: Detection, p_ra, p_dec, p_pa):
    """O-C (arcsec) along / across predicted motion, about each prediction."""
    xi, eta = to_tangent_plane(det.ra_deg, det.dec_deg, p_ra, p_dec)
    ue, un = np.sin(np.radians(p_pa)), np.cos(np.radians(p_pa))
    return xi * ue + eta * un, -xi * un + eta * ue


def match_detection(det: Detection, pr: Predictions, cfg: dict) -> None:
    ok = pr.valid & (pr.alt >= cfg.get("min_altitude_deg", 0.0))
    if not ok.any():
        det.match_quality = "UNKNOWN"
        return
    idx = np.nonzero(ok)[0]
    a_tol, c_tol = cfg["along_track_tolerance_arcsec"], cfg["cross_track_tolerance_arcsec"]
    near = sep_arcsec(det.ra_deg, det.dec_deg, pr.ra[idx], pr.dec[idx]) <= max(a_tol, c_tol)
    idx = idx[np.atleast_1d(near)]
    best, best_score = None, np.inf
    for i in idx:
        along, cross = _oc(det, pr.ra[i], pr.dec[i], pr.pa_deg[i])
        if abs(along) > a_tol or abs(cross) > c_tol:
            continue
        if det.is_trailed and np.isfinite(det.trail_pa_deg):
            if _pa_diff_mod180(pr.pa_deg[i] % 180, det.trail_pa_deg) > cfg["pa_tolerance_deg"]:
                continue
            if abs(det.trail_rate_arcsec_s / max(pr.rate_arcsec_s[i], 1e-9) - 1) > cfg["rate_tolerance"]:
                continue
        score = (along / a_tol) ** 2 + (cross / c_tol) ** 2
        if score < best_score:
            best, best_score = (i, along, cross), score
    if best is None:
        det.match_quality = "UNKNOWN"
        return
    i, along, cross = best
    det.norad_id, det.catalog_name = int(pr.norad_id[i]), str(pr.name[i])
    det.oc_along_arcsec, det.oc_cross_arcsec = float(along), float(cross)
    det.oc_total_arcsec = float(np.hypot(along, cross))
    det.predicted_rate_arcsec_s = float(pr.rate_arcsec_s[i])
    det.predicted_range_km = float(pr.range_km[i])
    det.tle_age_days = float(pr.tle_age_days[i])
    good = best_score <= 0.25 and abs(det.tle_age_days) <= cfg["max_tle_age_days"]
    det.match_quality = "GOOD" if good else "MARGINAL"


def identify(dets: list[Detection], tracklets: list[Tracklet], prop: Propagator,
             cfg: dict) -> None:
    if not prop.objects:
        for d in dets:
            d.match_quality = "NO_CATALOGUE"
        return
    cache: dict[str, Predictions] = {}
    for d in dets:
        key = d.epoch_utc[:23]
        if key not in cache:
            cache[key] = prop.predict(Time(d.epoch_utc, scale="utc"))
        match_detection(d, cache[key], cfg)

    for tr in tracklets:
        ids = [d.norad_id for d in tr.members if d.norad_id is not None]
        if not ids:
            continue
        nid, votes = Counter(ids).most_common(1)[0]
        if votes < max(1, (len(tr.members) + 1) // 2):
            tr.quality += "|AMBIGUOUS_ID"
            continue
        mem = [d for d in tr.members if d.norad_id == nid]
        if tr.kind == "MULTI_FRAME":
            prate = np.mean([d.predicted_rate_arcsec_s for d in mem])
            if abs(tr.rate_arcsec_s / max(prate, 1e-9) - 1) > cfg["rate_tolerance"]:
                tr.quality += "|RATE_MISMATCH"
                continue
        tr.identified, tr.norad_id, tr.catalog_name = True, nid, mem[0].catalog_name
        tr.mean_oc_along_arcsec = float(np.mean([d.oc_along_arcsec for d in mem]))
        tr.mean_oc_cross_arcsec = float(np.mean([d.oc_cross_arcsec for d in mem]))
        prate = float(np.mean([d.predicted_rate_arcsec_s for d in mem]))
        tr.along_track_time_offset_s = tr.mean_oc_along_arcsec / prate if prate > 0 else float("nan")
    n_id = sum(t.identified for t in tracklets)
    log.info("Identification: %d/%d tracklets correlated with the TLE catalogue; %d UCTs",
             n_id, len(tracklets), len(tracklets) - n_id)
