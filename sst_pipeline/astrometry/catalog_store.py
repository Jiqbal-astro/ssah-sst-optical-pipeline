"""
catalog_store.py
----------------
Local, tiled Gaia DR3 reference catalogue with optional read-through
download.

Why: the ESA Gaia archive is often slow or busy, and SST operations must
not depend on it at run time. The sky is split into ~2 x 2 deg cells
(equal-dec bands, RA cells widened towards the poles). Each cell is one
compressed .npz file holding ra, dec (ICRS, epoch 2016.0), pmra, pmdec
(mas/yr) and G for stars brighter than the store's magnitude limit.

* the sst-refcat tool fills a region in advance
  (site-visible sky, GEO belt, a box or a single field), resumably.
* At run time the pipeline reads only the cells overlapping a frame.
* With --catalog, missing cells are downloaded once (VizieR mirrors
  first, then the ESA archive, each with a timeout) and saved, so every
  later run in that region is offline.

A cell that has been downloaded and holds no stars is stored as an empty
file, so it is not fetched again. A failed download stores nothing, so it
is retried next time.
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from sst_pipeline.utils.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_CELL_DEG = 2.0
GAIA_DR3_EPOCH = 2016.0
FIELDS = ("ra", "dec", "pmra", "pmdec", "mag")


# ---------------------------------------------------------------------------
# Cell geometry
# ---------------------------------------------------------------------------

def _n_dec(cell: float) -> int:
    return int(round(180.0 / cell))


def _n_ra(idec: int, cell: float) -> int:
    dh = 180.0 / _n_dec(cell)
    d0, d1 = -90.0 + idec * dh, -90.0 + (idec + 1) * dh
    cos_min_abs = math.cos(math.radians(min(abs(d0), abs(d1)))) if d0 * d1 > 0 else 1.0
    return max(1, int(round(360.0 * cos_min_abs / cell)))


def cell_of(ra: float, dec: float, cell: float = DEFAULT_CELL_DEG) -> str:
    nd = _n_dec(cell)
    idec = min(int((dec + 90.0) / (180.0 / nd)), nd - 1)
    nra = _n_ra(idec, cell)
    ira = min(int((ra % 360.0) / (360.0 / nra)), nra - 1)
    return f"d{idec:03d}_r{ira:03d}"


def cell_bounds(key: str, cell: float = DEFAULT_CELL_DEG) -> tuple[float, float, float, float]:
    idec, ira = int(key[1:4]), int(key[6:9])
    dh = 180.0 / _n_dec(cell)
    rw = 360.0 / _n_ra(idec, cell)
    return ira * rw, (ira + 1) * rw, -90.0 + idec * dh, -90.0 + (idec + 1) * dh


def cells_for_cone(ra: float, dec: float, radius_deg: float, cell: float = DEFAULT_CELL_DEG) -> list[str]:
    """Every cell that can contain a point within radius_deg of (ra, dec)."""
    nd = _n_dec(cell)
    dh = 180.0 / nd
    d_lo, d_hi = max(dec - radius_deg, -90.0), min(dec + radius_deg, 90.0)
    keys = []
    for idec in range(int((d_lo + 90) / dh), min(int((d_hi + 90) / dh), nd - 1) + 1):
        nra = _n_ra(idec, cell)
        rw = 360.0 / nra
        band_lo, band_hi = -90 + idec * dh, -90 + (idec + 1) * dh
        max_abs = max(abs(max(band_lo, d_lo)), abs(min(band_hi, d_hi)))
        if max_abs >= 89.9 or d_hi >= 90.0 or d_lo <= -90.0:
            ras = range(nra)
        else:
            sin_r = math.sin(math.radians(radius_deg))
            # max RA extent of the cone is asin(sin r / cos dec0); use the larger |dec| to stay conservative
            cos_d = math.cos(math.radians(max(max_abs, abs(dec))))
            if sin_r >= cos_d:
                ras = range(nra)
            else:
                hw = math.degrees(math.asin(sin_r / cos_d))
                i0 = math.floor(((ra - hw) % 360.0) / rw)
                span = int(math.ceil((2 * hw) / rw)) + 1
                ras = sorted({(i0 + k) % nra for k in range(span + 1)})
                if len(ras) >= nra:
                    ras = range(nra)
        keys += [f"d{idec:03d}_r{ira:03d}" for ira in ras]
    return keys


def cells_for_box(ra_min, ra_max, dec_min, dec_max, cell: float = DEFAULT_CELL_DEG) -> list[str]:
    """Cells overlapping an RA/Dec box (ra_min > ra_max wraps through 0)."""
    nd = _n_dec(cell)
    dh = 180.0 / nd
    keys = []
    for idec in range(int((max(dec_min, -90) + 90) / dh), min(int((min(dec_max, 90) + 90) / dh), nd - 1) + 1):
        nra = _n_ra(idec, cell)
        rw = 360.0 / nra
        for ira in range(nra):
            a0, a1 = ira * rw, (ira + 1) * rw
            if ra_min <= ra_max:
                hit = a1 > ra_min and a0 < ra_max
            else:
                hit = a1 > ra_min or a0 < ra_max
            if hit or (ra_min == 0 and ra_max >= 360):
                keys.append(f"d{idec:03d}_r{ira:03d}")
    return keys


# ---------------------------------------------------------------------------
# Online fetchers (each returns a dict of arrays for exactly one cell)
# ---------------------------------------------------------------------------

def _with_timeout(fn: Callable, timeout_s: float):
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout_s)
    except FutTimeout:
        raise TimeoutError(f"no answer within {timeout_s:.0f} s")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def fetch_vizier(key: str, mag_limit: float, cell: float, timeout_s: float,
                 server: str = "vizier.cds.unistra.fr") -> dict:
    """Gaia DR3 (VizieR I/355/gaiadr3) for one cell. row_limit=-1 is essential:
    astroquery's default (50 rows) silently truncates."""
    from astroquery.vizier import Vizier
    a0, a1, d0, d1 = cell_bounds(key, cell)
    v = Vizier(columns=["RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Gmag"],
               column_filters={"RA_ICRS": f"{a0:.6f}..{a1:.6f}", "DE_ICRS": f"{d0:.6f}..{d1:.6f}",
                               "Gmag": f"<{mag_limit:.2f}"},
               row_limit=-1, timeout=timeout_s, vizier_server=server)
    res = _with_timeout(lambda: v.query_constraints(catalog="I/355/gaiadr3"), timeout_s + 5)
    if res is None or len(res) == 0:
        return {f: np.array([]) for f in FIELDS}
    t = res[0]
    missing = {"RA_ICRS", "DE_ICRS", "Gmag"} - set(t.colnames)
    if missing:
        raise RuntimeError(f"VizieR answer lacks columns {missing}; got {t.colnames}")

    def col(name):
        c = t[name]
        return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, float)
    return _trim({"ra": col("RA_ICRS"), "dec": col("DE_ICRS"),
                  "pmra": col("pmRA") if "pmRA" in t.colnames else np.zeros(len(t)),
                  "pmdec": col("pmDE") if "pmDE" in t.colnames else np.zeros(len(t)),
                  "mag": col("Gmag")}, key, cell, mag_limit)


def fetch_esa(key: str, mag_limit: float, cell: float, timeout_s: float) -> dict:
    from astroquery.gaia import Gaia
    a0, a1, d0, d1 = cell_bounds(key, cell)
    adql = (f"SELECT ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source "
            f"WHERE ra >= {a0:.6f} AND ra < {a1:.6f} AND dec >= {d0:.6f} AND dec < {d1:.6f} "
            f"AND phot_g_mean_mag < {mag_limit:.2f}")
    t = _with_timeout(lambda: Gaia.launch_job_async(adql).get_results(), timeout_s)
    f = lambda c: np.asarray(t[c].filled(np.nan) if hasattr(t[c], "filled") else t[c], float)  # noqa: E731
    return _trim({"ra": f("ra"), "dec": f("dec"), "pmra": f("pmra"), "pmdec": f("pmdec"),
                  "mag": f("phot_g_mean_mag")}, key, cell, mag_limit)


def _trim(arr: dict, key: str, cell: float, mag_limit: float) -> dict:
    a0, a1, d0, d1 = cell_bounds(key, cell)
    ok = ((arr["ra"] >= a0) & (arr["ra"] < a1) & (arr["dec"] >= d0) & (arr["dec"] < d1)
          & np.isfinite(arr["mag"]) & (arr["mag"] < mag_limit))
    return {k: np.asarray(v)[ok] for k, v in arr.items()}


def make_fetcher(sources: list[str], vizier_servers: list[str], timeout_s: float, cell: float):
    """Returns fetch(key, mag_limit) -> (arrays, source_name); tries sources in order."""
    def fetch(key: str, mag_limit: float):
        errors = []
        for src in sources:
            targets = vizier_servers if src == "vizier" else [None]
            for srv in targets:
                name = f"vizier:{srv}" if srv else src
                try:
                    if src == "vizier":
                        return fetch_vizier(key, mag_limit, cell, timeout_s, srv), name
                    if src == "esa":
                        return fetch_esa(key, mag_limit, cell, timeout_s), name
                    raise ValueError(f"unknown catalogue source '{src}'")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("all catalogue sources failed — " + " | ".join(errors))
    return fetch


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class CatalogStore:
    def __init__(self, root: str, mag_limit: float = 16.0, cell: float = DEFAULT_CELL_DEG):
        self.root = root
        self.cells_dir = os.path.join(root, "cells")
        self._lock = threading.Lock()
        man = os.path.join(root, "manifest.json")
        if os.path.exists(man):
            with open(man) as f:
                self.manifest = json.load(f)
            self.cell = float(self.manifest["cell_deg"])
            self.mag_limit = float(self.manifest["mag_limit"])
        else:
            self.cell, self.mag_limit = float(cell), float(mag_limit)
            self.manifest = {"catalog": "Gaia DR3", "epoch": GAIA_DR3_EPOCH, "cell_deg": self.cell,
                             "mag_limit": self.mag_limit, "created": _now(), "cells": {}}

    # --- persistence -----------------------------------------------------
    def _path(self, key):
        return os.path.join(self.cells_dir, f"{key}.npz")

    def has(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def save(self, key: str, arr: dict, source: str) -> None:
        os.makedirs(self.cells_dir, exist_ok=True)
        tmp = self._path(key) + ".tmp.npz"
        np.savez_compressed(tmp, ra=np.asarray(arr["ra"], np.float64), dec=np.asarray(arr["dec"], np.float64),
                            pmra=np.nan_to_num(np.asarray(arr["pmra"], np.float32)),
                            pmdec=np.nan_to_num(np.asarray(arr["pmdec"], np.float32)),
                            mag=np.asarray(arr["mag"], np.float32))
        os.replace(tmp, self._path(key))          # atomic: no half-written cells
        with self._lock:
            self.manifest["cells"][key] = {"n": int(len(arr["ra"])), "source": source, "date": _now()}
            self._write_manifest()

    def _write_manifest(self):
        os.makedirs(self.root, exist_ok=True)
        tmp = os.path.join(self.root, "manifest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=0)
        os.replace(tmp, os.path.join(self.root, "manifest.json"))

    def load(self, key: str) -> dict:
        with np.load(self._path(key)) as z:
            return {k: z[k] for k in FIELDS}

    # --- query -----------------------------------------------------------
    def query_cone(self, ra: float, dec: float, radius_deg: float,
                   fetch: Optional[Callable] = None) -> tuple[dict, float]:
        """Stars within the cells covering the cone; returns (arrays, coverage 0-1)."""
        keys = cells_for_cone(ra, dec, radius_deg, self.cell)
        parts, have = [], 0
        for k in keys:
            if not self.has(k) and fetch is not None:
                try:
                    arr, src = fetch(k, self.mag_limit)
                    self.save(k, arr, src)
                    log.info("Reference cell %s: %d stars from %s (saved)", k, len(arr["ra"]), src)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Reference cell %s unavailable: %s", k, exc)
            if self.has(k):
                parts.append(self.load(k))
                have += 1
        if not parts:
            return {f: np.array([]) for f in FIELDS}, 0.0
        return {f: np.concatenate([p[f] for p in parts]) for f in FIELDS}, have / len(keys)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
