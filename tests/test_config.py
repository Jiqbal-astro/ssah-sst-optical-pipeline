import pytest
import yaml

from sst_pipeline.config import load_config


def test_every_section_the_code_reads_exists():
    cfg = load_config()
    for k in ("instrument", "site", "observation", "timing_calibration", "header_requirements", "calibration",
              "source_extraction", "source_classification", "qc", "astrometry", "photometry", "tracklet",
              "tle_tracking", "output"):
        assert k in cfg


def test_partial_override_merges(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"qc": {"snr": {"photometry_min": 20}}}))
    cfg = load_config(p)
    assert cfg["qc"]["snr"]["photometry_min"] == 20 and cfg["qc"]["snr"]["astrometry_min"] == 5.0


def test_invalid_mode_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"observation": {"tracking_mode": "stare"}}))
    with pytest.raises(ValueError):
        load_config(p)


def test_unknown_key_warns(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"timing_calibration": {"time_bais_ms": 12.0}}))
    with pytest.warns(UserWarning, match="time_bais_ms"):
        load_config(p)


def test_site_template_is_valid():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_config(os.path.join(root, "config", "site_template.yaml"))
