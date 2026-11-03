"""
config.py
---------
Loads the pipeline YAML. The packaged defaults.yaml is the single source of
defaults; a user config is deep-merged on top, so a partial file
(only the keys you want to change) is valid. Lists (e.g. header_requirements)
are replaced, not merged.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "defaults.yaml"

VALID_TRACKING_MODES = ("sidereal", "rate")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _unknown_keys(base: dict, override: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in (override or {}).items():
        if k not in base:
            out.append(prefix + str(k))
        elif isinstance(v, dict) and isinstance(base[k], dict):
            out += _unknown_keys(base[k], v, f"{prefix}{k}.")
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        unknown = _unknown_keys(cfg, user)
        if unknown:   # a misspelt key would otherwise be silently ignored
            warnings.warn(f"{path}: unknown config keys ignored: {', '.join(unknown)}", stacklevel=2)
        cfg = _deep_merge(cfg, user)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Fail fast on values that would silently produce wrong science."""
    mode = cfg["observation"]["tracking_mode"]
    if mode not in VALID_TRACKING_MODES:
        raise ValueError(f"observation.tracking_mode must be one of {VALID_TRACKING_MODES}, got {mode!r}")
    ref = cfg["timing_calibration"]["date_obs_reference"]
    if ref not in ("start", "mid", "end"):
        raise ValueError(f"timing_calibration.date_obs_reference must be start|mid|end, got {ref!r}")
    if cfg["timing_calibration"]["shutter_type"] not in ("global", "rolling"):
        raise ValueError("timing_calibration.shutter_type must be global|rolling")
    ins = cfg["instrument"]
    for k in ("gain_e_per_adu", "read_noise_e", "saturation_adu", "pixel_scale_arcsec"):
        if float(ins[k]) <= 0:
            raise ValueError(f"instrument.{k} must be > 0")
    if cfg["observation"]["ra_numeric_unit"] not in ("deg", "hour"):
        raise ValueError("observation.ra_numeric_unit must be deg|hour")
