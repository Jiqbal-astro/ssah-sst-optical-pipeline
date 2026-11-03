"""Source extraction and trail geometry vs injected truth."""
import numpy as np
import pytest
from conftest import gauss_stamp, trail

from sst_pipeline.config import load_config
from sst_pipeline.detection.source_classifier import Morph, classify
from sst_pipeline.detection.source_extractor import extract_sources

CFG = load_config()
SE, SC = CFG["source_extraction"], CFG["source_classification"]
SIG = 1.5


def field(rng, n=300):
    img = rng.normal(500.0, 10.0, (n, n))
    for x, y, f in [(40, 40, 3e4), (250, 60, 2e4), (60, 250, 2.5e4), (230, 240, 1.5e4)]:
        gauss_stamp(img, x, y, f, SIG)
    return img


def run(img, raw=None):
    return extract_sources("t", img, img if raw is None else raw, SE, 1.5, 65535.0)


def test_one_streak_one_detection(rng):
    # a single trail must give a single detection (no edge/fragment duplicates)
    img = field(rng)
    trail(img, 80, 120, 220, 170, 8e4, SIG)
    tr = [s for s in run(img).sources if classify(s, SC) == Morph.TRAILED]
    assert len(tr) == 1
    s = tr[0]
    assert s.length == pytest.approx(np.hypot(140, 50), rel=0.02)
    assert np.hypot(s.x - 150, s.y - 145) < 0.3


def test_streak_crossing_star_is_merged_and_unbiased(rng):
    img = field(rng)
    trail(img, 40, 150, 260, 150, 9e4, SIG)
    gauss_stamp(img, 150, 151, 3e5, SIG)        # very bright star sitting on the trail
    gauss_stamp(img, 255, 151, 2e5, SIG)        # and one on its end
    tr = [s for s in run(img).sources if classify(s, SC) == Morph.TRAILED]
    assert len(tr) == 1
    s = tr[0]
    assert s.length == pytest.approx(220, abs=3)
    assert abs(s.x - 150) < 1.5 and abs(s.y - 150) < 0.3
    assert s.profile_flux == pytest.approx(9e4, rel=0.05)


def test_point_source_fwhm_and_centroid(rng):
    img = field(rng)
    gauss_stamp(img, 150.25, 140.75, 5e4, SIG)
    s = min(run(img).sources, key=lambda s: np.hypot(s.x - 150.25, s.y - 140.75))
    assert classify(s, SC) == Morph.POINT
    assert np.hypot(s.x - 150.25, s.y - 140.75) < 0.05
    assert s.fwhm == pytest.approx(SIG * 2.3548, rel=0.08)


def test_bleed_column_is_artifact(rng):
    img = field(rng)
    img[50:250, 149:152] = 65535.0
    bl = [s for s in run(img).sources if s.saturated and s.elongation > 5]
    assert bl and all(classify(s, SC) == Morph.ARTIFACT for s in bl)


def test_hot_pixel_is_artifact(rng):
    img = field(rng)
    img[150, 150] += 3000
    near = [s for s in run(img).sources if np.hypot(s.x - 150, s.y - 150) < 2]
    assert all(classify(s, SC) == Morph.ARTIFACT for s in near)   # rejected or not detected


def test_roles_depend_on_tracking_mode():
    from sst_pipeline.detection.source_classifier import reference_morph, target_morph
    assert reference_morph("sidereal") == Morph.POINT and target_morph("sidereal") == Morph.TRAILED
    assert reference_morph("rate") == Morph.TRAILED and target_morph("rate") == Morph.POINT
