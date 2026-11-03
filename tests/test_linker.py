"""Static-sky rejection and kinematic linking."""
import numpy as np
import pytest

from sst_pipeline.astrometry.position_extractor import Detection
from sst_pipeline.config import load_config
from sst_pipeline.tracking.linker import flag_static, link_tracklets
from sst_pipeline.utils.coordinate_utils import from_tangent_plane

TC = load_config()["tracklet"]
MJD0 = 61114.8


def det(fi, xi, eta, morph="POINT", **kw):
    ra, dec = from_tangent_plane(xi, eta, 359.99, 10.0)       # field straddles RA = 0/360
    return Detection(det_id=f"f{fi}#{xi:.1f}{eta:.1f}", frame_id=f"f{fi}", frame_index=fi,
                     epoch_utc="", mjd=MJD0 + fi * 10 / 86400, x_px=0, y_px=0, ra_deg=float(ra),
                     dec_deg=float(dec), ra_err_arcsec=0.2, dec_err_arcsec=0.2, morphology=morph,
                     snr=30, saturated=False, exptime_s=2.0, airmass=1.2, **kw)


def scene(rng, nf=8):
    stars = rng.uniform(-400, 400, (40, 2))
    dets = []
    for f in range(nf):
        dets += [det(f, x + rng.normal(0, .1), y + rng.normal(0, .1)) for x, y in stars]
        dets.append(det(f, -300 + 5.0 * 10 * f, 50 + 2.0 * 10 * f))             # mover 1: 5.4"/s
        dets.append(det(f, 100 - 0.4 * 10 * f, -200 + 0.3 * 10 * f))            # mover 2: 0.5"/s
        dets.append(det(f, *rng.uniform(-400, 400, 2)))                         # noise detection
    return dets


def test_static_then_link(rng):
    dets = scene(rng)
    assert flag_static(dets, TC["static_match_radius_arcsec"], TC["static_min_frames"])
    assert sum(d.is_static for d in dets) == 40 * 8
    cands = [d for d in dets if not d.is_static]
    trs = link_tracklets(cands, TC)
    multi = [t for t in trs if t.kind == "MULTI_FRAME"]
    assert len(multi) == 2 and all(len(t.members) == 8 for t in multi)
    rates = sorted(t.rate_arcsec_s for t in multi)
    assert rates[0] == pytest.approx(0.5, rel=0.02) and rates[1] == pytest.approx(np.hypot(5, 2), rel=0.02)
    assert sum(d.status == "UNLINKED" for d in cands) == 8               # noise stays unlinked


def test_two_point_chains_rejected():
    trs = link_tracklets([det(0, 0, 0), det(1, 10, 0)], TC)
    assert trs == []


def test_streak_axis_must_agree_with_motion():
    a = [det(f, 60.0 * f, 0, morph="TRAILED", trail_pa_deg=0.0, trail_rate_arcsec_s=6.0) for f in range(4)]
    assert [t for t in link_tracklets(a, TC) if t.kind == "MULTI_FRAME"] == []   # axis N-S, motion E-W
    b = [det(f, 60.0 * f, 0, morph="TRAILED", trail_pa_deg=90.0, trail_rate_arcsec_s=6.0) for f in range(4)]
    assert len([t for t in link_tracklets(b, TC) if t.kind == "MULTI_FRAME"]) == 1
