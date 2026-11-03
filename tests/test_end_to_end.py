"""End-to-end run on the synthetic ground-truth sequence."""
import json
import os

import pytest
import yaml
from sst_pipeline.tools.refcat import import_csv
from sst_pipeline.tools.synthetic import generate

from sst_pipeline.astrometry.catalog_store import CatalogStore

from sst_pipeline.pipeline import run_pipeline
from sst_pipeline.utils.coordinate_utils import sep_arcsec


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    frames, out = str(d / "frames"), str(d / "out")
    truth = generate(frames)
    # operational path: the tiled local Gaia store, filled from the field's catalogue
    store = str(d / "refcat")
    import_csv(CatalogStore(store, mag_limit=16.0), os.path.join(frames, "reference_catalog.csv"))
    cfg = d / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"instrument": {"detector_ny": 1024},
                                   "astrometry": {"reference_catalog_dir": store}}))
    return run_pipeline(frames, out, str(cfg)), truth, out


def _truth_for(d, truth):
    objs = truth["frames"][d.frame_id]["objects"]
    name = min(objs, key=lambda k: sep_arcsec(objs[k]["ra"], objs[k]["dec"], d.ra_deg, d.dec_deg))
    return name, objs[name]


def test_frames_and_qc(run):
    res, truth, _ = run
    assert [s["frame_id"] for s in res.skipped_frames] == ["zz_bias_001"]
    q = res.qc_dataframe.set_index("frame_id")
    assert not q.loc["seq_011", "astrometry_pass"] and not q.loc["seq_011", "photometry_pass"]   # no TELESCOP
    assert q.loc["seq_010", "astrometry_pass"] and not q.loc["seq_010", "photometry_pass"]       # bleed/saturation
    assert q.loc["seq_008", "astrometry_pass"] and not q.loc["seq_008", "photometry_pass"]       # cloud
    assert q.loc["seq_003", "astrometry_pass"]                                                  # CRs cleaned
    good = q.drop(index=["seq_008", "seq_010", "seq_011"])
    assert good["photometry_pass"].all() and good["wcs_verified"].all()
    assert (good["astrometric_rms_arcsec"] < 0.1).all()
    assert (good["zeropoint_mag"] - truth["zp_true"]).abs().max() < 0.03


def test_tracklets_match_truth(run):
    res, truth, _ = run
    multi = [t for t in res.tracklets if t.kind == "MULTI_FRAME"]
    single = [t for t in res.tracklets if t.kind == "SINGLE_STREAK"]
    assert len(multi) == 2 and len(single) == 1
    for t in multi + single:
        names = {_truth_for(d, truth)[0] for d in t.members}
        assert len(names) == 1                       # one object per tracklet
        name = names.pop()
        o = _truth_for(t.members[0], truth)[1]
        assert t.rate_arcsec_s == pytest.approx(o["rate"], rel=0.01), name
        assert abs(((t.pa_deg - o["pa"]) + 90) % 180 - 90) < 0.5, name
        for d in t.members:
            _, o = _truth_for(d, truth)
            assert sep_arcsec(o["ra"], o["dec"], d.ra_deg, d.dec_deg) < 0.3, (name, d.frame_id)
            assert d.epoch_utc.startswith(truth["frames"][d.frame_id]["mid_exposure_utc"][:23])
            if d.photometry_ok:
                assert d.mag == pytest.approx(o["mag"], abs=0.08), (name, d.frame_id)


def test_no_star_reported_as_rso(run):
    res, truth, _ = run
    for t in res.tracklets:
        for d in t.members:
            assert _truth_for(d, truth)[0].startswith("RSO")


def test_outputs_written(run):
    res, _, out = run
    for f in ("qc_matrix.csv", "session_metadata.csv", "detections.csv", "tracklets.csv",
              "uncorrelated_tracks.csv", "observations_tdm.kvn", "observations_ades.psv",
              "qc_summary.json", "rejected_frames.log", "SST_QC_Report.xlsx"):
        assert os.path.getsize(os.path.join(out, f)) > 0, f
    tdm = open(os.path.join(out, "observations_tdm.kvn")).read()
    n_obs = sum(len(t.members) for t in res.tracklets if all(d.astrometry_verified for d in t.members))
    assert tdm.count("ANGLE_1 =") == n_obs == res.summary["observations_ades"]
    ades = [l for l in open(os.path.join(out, "observations_ades.psv")) if not l.startswith(("#", "!"))]
    assert ades[0].strip().split("|")[:6] == ["permID", "provID", "trkSub", "mode", "stn", "obsTime"]
    js = json.load(open(os.path.join(out, "qc_summary.json")))
    assert js["tracklets_multi_frame"] == 2
    assert js["reference_catalogue"] == ["GaiaDR3 (local store)"]
    assert ades[1].split("|")[10] == "Gaia3"          # astCat recognised from the store
