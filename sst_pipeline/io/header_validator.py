"""
header_validator.py
-------------------
Validates FITS headers against config 'header_requirements'. Returns a
structured result (never raises) that feeds the QC matrix.

Types: float | string | datetime | angle_ra | angle_dec
  * datetime uses astropy.time (accepts ISO with any fractional precision)
    and, if DATE-OBS holds only a date, combines it with TIME-OBS / UT / UTC.
  * angle_ra / angle_dec accept decimal degrees or sexagesimal strings
    ('12 30 45.2', '+10:20:30') as written by ASCOM / MaxIm / N.I.N.A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from astropy.io import fits
from astropy.time import Time

from sst_pipeline.utils.coordinate_utils import parse_dec_deg, parse_ra_deg


@dataclass
class HeaderValidationResult:
    frame_id: str
    is_valid: bool
    missing_required: list[str] = field(default_factory=list)
    invalid_values: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved: dict[str, Any] = field(default_factory=dict)


def _get(header: fits.Header, keyword: str, aliases: list[str]):
    for kw in [keyword, *aliases]:
        if kw in header and header[kw] not in (None, ""):
            return header[kw], kw
    return None, None


def parse_obs_time(header: fits.Header) -> Optional[Time]:
    """DATE-OBS (ISO) -> astropy Time (UTC); falls back to DATE-OBS + TIME-OBS."""
    raw = header.get("DATE-OBS")
    if raw in (None, ""):
        return None
    s = str(raw).strip().replace(" ", "T").rstrip("Z")
    if "T" not in s:
        tod = header.get("TIME-OBS") or header.get("UT") or header.get("UTC")
        if tod in (None, ""):
            return None
        s = f"{s}T{str(tod).strip()}"
    try:
        return Time(s, format="isot", scale="utc")
    except ValueError:
        return None


def _check(value: Any, typ: str, header: fits.Header, ra_unit: str) -> tuple[bool, Any]:
    try:
        if typ == "float":
            return True, float(value)
        if typ == "string":
            s = str(value).strip()
            return bool(s), s
        if typ == "datetime":
            t = parse_obs_time(header)
            return t is not None, t
        if typ == "angle_ra":
            v = parse_ra_deg(value, ra_unit)
            return v is not None, v
        if typ == "angle_dec":
            v = parse_dec_deg(value)
            return v is not None, v
        return True, value
    except (TypeError, ValueError):
        return False, value


def validate_header(frame_id: str, header: fits.Header, rules: list[dict],
                    ra_numeric_unit: str = "deg") -> HeaderValidationResult:
    res = HeaderValidationResult(frame_id=frame_id, is_valid=True)
    for rule in rules:
        kw = rule["keyword"]
        required = bool(rule.get("required", False))
        typ = rule.get("type", "string")
        value, _ = _get(header, kw, rule.get("aliases", []) or [])
        if value is None:
            if required:
                res.missing_required.append(kw)
                res.is_valid = False
            else:
                res.warnings.append(f"{kw} not present (optional)")
            continue
        ok, coerced = _check(value, typ, header, ra_numeric_unit)
        if not ok:
            msg = f"{kw}={value!r} could not be parsed as {typ}"
            (res.invalid_values if required else res.warnings).append(msg)
            res.is_valid = res.is_valid and not required
            continue
        if typ == "float":
            lo, hi = rule.get("min"), rule.get("max")
            for bad, msg in ((lo is not None and coerced < lo, f"{kw}={coerced} < min {lo}"),
                             (hi is not None and coerced > hi, f"{kw}={coerced} > max {hi}")):
                if bad:
                    (res.invalid_values if required else res.warnings).append(msg)
                    res.is_valid = res.is_valid and not required
        res.resolved[kw] = coerced
    return res
