"""
reference_catalog.py
--------------------
Reference stars for WCS verification/refinement and photometric zero-point.

Sources (in order): a single local CSV, else the tiled local Gaia DR3 store
(astrometry/catalog_store.py), which can fill missing cells online
(VizieR mirrors, then ESA) when the pipeline runs with --catalog.

Notes:
  * Downloads remove the client row limit (astroquery defaults silently
    truncate to 50 rows) and apply the magnitude cut server-side.
  * Gaia DR3 positions are at epoch J2016.0; proper motions are applied to
    the observation epoch (high-PM stars move >1"/decade).
  * Downloaded once per sky cell and kept; runs are offline afterwards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sst_pipeline.utils.coordinate_utils import sep_arcsec
from sst_pipeline.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class RefCatalog:
    ra: np.ndarray
    dec: np.ndarray
    mag: np.ndarray
    pmra: np.ndarray          # mas/yr, includes cos(dec)
    pmdec: np.ndarray         # mas/yr
    ref_epoch: np.ndarray     # Julian year
    band: str = "G"
    source: str = ""

    def __len__(self):
        return len(self.ra)

    def at_epoch(self, jyear: float) -> tuple[np.ndarray, np.ndarray]:
        dt = jyear - self.ref_epoch
        dec = self.dec + np.nan_to_num(self.pmdec) * dt / 3.6e6
        ra = self.ra + np.nan_to_num(self.pmra) * dt / 3.6e6 / np.cos(np.radians(self.dec))
        return ra % 360.0, dec


def _from_df(df: pd.DataFrame, band: str, source: str) -> RefCatalog:
    cols = {c.lower(): c for c in df.columns}

    def col(name, default):
        return df[cols[name]].to_numpy(float) if name in cols else np.full(len(df), default)
    mag_col = next((c for c in ("mag", "phot_g_mean_mag", "g") if c in cols), None)
    if mag_col is None or "ra" not in cols or "dec" not in cols:
        raise ValueError("reference catalogue needs ra, dec and mag (or phot_g_mean_mag) columns")
    return RefCatalog(ra=col("ra", np.nan), dec=col("dec", np.nan),
                      mag=df[cols[mag_col]].to_numpy(float),
                      pmra=col("pmra", 0.0), pmdec=col("pmdec", 0.0),
                      ref_epoch=col("ref_epoch", 2016.0), band=band, source=source)


def load_local_catalog(path: str, band: str = "G") -> RefCatalog:
    return _from_df(pd.read_csv(path), band, f"local:{os.path.basename(path)}")


def get_reference_catalog(center_ra: float, center_dec: float, radius_deg: float,
                          cfg: dict, allow_online: bool, _memo: dict = {}) -> Optional[RefCatalog]:  # noqa: B006
    """Reference stars around a field.

    Priority: (1) reference_catalog_path — one small CSV (tests, special
    fields); (2) reference_catalog_dir — the tiled local Gaia DR3 store; with
    allow_online (--catalog) missing cells are downloaded once and saved.
    """
    path = cfg.get("reference_catalog_path", "")
    if path:
        if path not in _memo:
            _memo[path] = load_local_catalog(path) if os.path.exists(path) else None
            if _memo[path] is None:
                log.error("reference_catalog_path '%s' not found", path)
            else:
                log.info("Reference catalogue: %d stars from %s", len(_memo[path]), path)
        cat = _memo[path]
        if cat is None:
            return None
        keep = sep_arcsec(cat.ra, cat.dec, center_ra, center_dec) <= radius_deg * 3600
        return RefCatalog(cat.ra[keep], cat.dec[keep], cat.mag[keep], cat.pmra[keep],
                          cat.pmdec[keep], cat.ref_epoch[keep], cat.band, cat.source)

    root = cfg.get("reference_catalog_dir", "")
    if not root:
        return None
    from sst_pipeline.astrometry.catalog_store import CatalogStore, make_fetcher
    key = ("store", os.path.abspath(root))
    if key not in _memo:
        _memo[key] = CatalogStore(root, mag_limit=cfg.get("catalog_mag_limit", 16.0))
    store = _memo[key]
    fetch = None
    if allow_online:
        fetch = make_fetcher(cfg.get("online_sources", ["vizier", "esa"]),
                             cfg.get("vizier_servers", ["vizier.cds.unistra.fr"]),
                             float(cfg.get("online_timeout_s", 60)), store.cell)
    arr, coverage = store.query_cone(center_ra, center_dec, radius_deg, fetch)
    if coverage < 1.0:
        log.warning("Reference store covers %.0f%% of this field (%s)%s", 100 * coverage, root,
                    "" if allow_online else " — fill it with sst-refcat "
                                             "or run with --catalog")
    if len(arr["ra"]) == 0:
        return None
    keep = sep_arcsec(arr["ra"], arr["dec"], center_ra, center_dec) <= radius_deg * 3600
    n = int(keep.sum())
    return RefCatalog(arr["ra"][keep], arr["dec"][keep], arr["mag"][keep].astype(float),
                      arr["pmra"][keep].astype(float), arr["pmdec"][keep].astype(float),
                      np.full(n, store.manifest.get("epoch", 2016.0)), "G", "GaiaDR3 (local store)")
