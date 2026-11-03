import numpy as np
import pytest
from conftest import gauss_stamp

from sst_pipeline.calibration.calibration_manager import CalibrationFrames, calibrate_frame


def test_bias_dark_flat_arithmetic():
    raw = np.full((50, 50), 1000.0 + 2 * 10 + 300 * 1.25)
    flat = np.full((50, 50), 1.25)
    cal = CalibrationFrames(bias=np.full((50, 50), 1000.0), dark_1s=np.full((50, 50), 2.0), flat=flat)
    out, mask, steps = calibrate_frame(raw, 10.0, cal)
    assert np.allclose(out, 300.0) and steps == ["B", "D", "F"]


def test_shape_mismatch_not_applied():
    out, _, steps = calibrate_frame(np.ones((10, 10)), 1.0, CalibrationFrames(bias=np.ones((5, 5))))
    assert steps == [] and np.allclose(out, 1.0)


def test_cr_cleaning_preserves_stars(rng):
    pytest.importorskip("astroscrappy")
    img = rng.normal(500, 10, (200, 200))
    gauss_stamp(img, 100, 100, 5e4, 1.5)
    img[50, 60] += 4000                                   # cosmic ray
    out, mask, steps = calibrate_frame(img, 1.0, CalibrationFrames(), {"cosmic_ray_rejection": True},
                                       gain=1.5, read_noise_e=9.0)
    star_before = img[95:106, 95:106].sum() - 121 * 500
    star_after = out[95:106, 95:106].sum() - 121 * 500
    assert star_after == pytest.approx(star_before, rel=0.01)    # cleaning must not clip star cores
    assert mask[50, 60] and out[50, 60] < 600
