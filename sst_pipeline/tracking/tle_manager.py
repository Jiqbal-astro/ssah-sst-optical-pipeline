"""
tle_manager.py
--------------
Loads a local TLE catalogue (2-line or 3-line format, e.g. from
the sst-tle tool). Operational design: download on a
networked machine, run the pipeline offline against the file.

Line checksums are verified (corrupted lines are skipped, not silently
propagated), SGP4 initialisation errors are caught, and each object carries
its epoch so stale TLEs (errors grow ~1-3 km/day in LEO) are flagged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from astropy.time import Time
from sgp4.api import WGS72, Satrec

from sst_pipeline.utils.logging_config import get_logger

log = get_logger(__name__)


def tle_checksum_ok(line: str) -> bool:
    if len(line) < 69 or not line[68].isdigit():
        return False
    s = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return s % 10 == int(line[68])


@dataclass
class TLEObject:
    norad_id: int
    name: str
    line1: str
    line2: str
    satrec: Satrec = field(init=False, repr=False)
    epoch: Time = field(init=False, repr=False)

    def __post_init__(self):
        # WGS72 constants: the gravity model TLEs are generated with
        self.satrec = Satrec.twoline2rv(self.line1, self.line2, WGS72)
        self.epoch = Time(self.satrec.jdsatepoch, self.satrec.jdsatepochF, format="jd", scale="utc")

    def age_days(self, t: Time) -> float:
        return float((t - self.epoch).jd)


def parse_tle_file(path: str, verify_checksum: bool = True) -> list[TLEObject]:
    if not os.path.exists(path):
        log.error("TLE file not found: %s", path)
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\n\r") for l in f if l.strip() and not l.startswith("#")]
    objs, bad, i = [], 0, 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name, l1, l2, i = None, lines[i], lines[i + 1], i + 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, l1, l2, i = lines[i].strip().lstrip("0 ").strip() or lines[i].strip(), lines[i + 1], lines[i + 2], i + 3
        else:
            i += 1
            continue
        if verify_checksum and not (tle_checksum_ok(l1) and tle_checksum_ok(l2)):
            bad += 1
            continue
        try:
            nid = int(l1[2:7])
            objs.append(TLEObject(nid, name or f"NORAD-{nid}", l1, l2))
        except Exception:  # noqa: BLE001
            bad += 1
    log.info("TLE catalogue: %d objects from %s (%d rejected: checksum/format)", len(objs), path, bad)
    return objs


def load_catalog(cfg: dict) -> list[TLEObject]:
    path = cfg.get("tle_file_path", "")
    if not cfg.get("enabled", True) or not path:
        log.info("No TLE catalogue configured — known-object identification disabled")
        return []
    return parse_tle_file(path)
