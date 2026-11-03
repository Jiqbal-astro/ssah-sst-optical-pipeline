"""Local tiled reference store: geometry, read-through fill, fallbacks, timeouts (offline, simulated servers)."""
import time

import numpy as np
import pytest
from sst_pipeline.tools.refcat import geo_belt_dec_range, import_csv, site_visible_dec_range

from sst_pipeline.astrometry import catalog_store as cs
from sst_pipeline.astrometry.catalog_store import (CatalogStore, cell_bounds, cell_of, cells_for_box,
                                                   cells_for_cone, make_fetcher)
from sst_pipeline.astrometry.reference_catalog import get_reference_catalog
from sst_pipeline.utils.coordinate_utils import from_tangent_plane


def fake_cell(key, mag_limit, cell=cs.DEFAULT_CELL_DEG, n=50, seed=None):
    a0, a1, d0, d1 = cell_bounds(key, cell)
    r = np.random.default_rng(seed if seed is not None else abs(hash(key)) % 2**32)
    return {"ra": r.uniform(a0, a1, n), "dec": r.uniform(d0, d1, n), "pmra": np.zeros(n),
            "pmdec": np.zeros(n), "mag": r.uniform(9, mag_limit - 0.01, n)}


@pytest.mark.parametrize("ra,dec,rad", [(150, 20, 0.6), (359.8, 5, 0.8), (0.1, -30, 1.2),
                                         (45, 88.7, 1.0), (200, -89.5, 0.7), (10, 0, 3.0)])
def test_cone_cells_cover_every_point(ra, dec, rad):
    keys = set(cells_for_cone(ra, dec, rad))
    r = np.random.default_rng(1)
    for _ in range(3000):
        ang, d = r.uniform(0, 2 * np.pi), rad * np.sqrt(r.uniform()) * 3600
        pra, pdec = from_tangent_plane(d * np.sin(ang), d * np.cos(ang), ra, dec)
        assert cell_of(float(pra), float(pdec)) in keys


def test_cells_tile_the_sphere():
    keys = cells_for_box(0, 360, -90, 90)
    area = sum((np.radians(b[1] - b[0])) * (np.sin(np.radians(b[3])) - np.sin(np.radians(b[2])))
               for b in map(cell_bounds, keys))
    assert area == pytest.approx(4 * np.pi, rel=1e-9) and len(keys) == len(set(keys))


def test_read_through_fill_then_offline(tmp_path):
    calls = []

    def fetch(k, m):
        calls.append(k)
        return fake_cell(k, m), "fake"
    st = CatalogStore(str(tmp_path), mag_limit=16)
    arr, cov = st.query_cone(150, 20, 0.6, fetch)
    assert cov == 1.0 and len(arr["ra"]) > 0 and len(calls) == len(cells_for_cone(150, 20, 0.6))
    n_calls = len(calls)
    st2 = CatalogStore(str(tmp_path))                    # new session, reads manifest
    arr2, cov2 = st2.query_cone(150, 20, 0.6, fetch)
    assert len(calls) == n_calls and cov2 == 1.0       # nothing re-downloaded
    assert np.array_equal(np.sort(arr["ra"]), np.sort(arr2["ra"]))
    _, cov3 = st2.query_cone(10, -40, 0.6, None)       # offline, never fetched
    assert cov3 == 0.0


def test_failed_cell_not_saved_and_retried(tmp_path):
    st = CatalogStore(str(tmp_path))
    def bad(k, m):
        raise RuntimeError("server busy")
    _, cov = st.query_cone(150, 20, 0.5, bad)
    assert cov == 0.0 and not any(st.has(k) for k in cells_for_cone(150, 20, 0.5))


def test_empty_cell_is_remembered(tmp_path):
    st = CatalogStore(str(tmp_path))
    n = []
    def empty(k, m):
        n.append(k)
        return {f: np.array([]) for f in cs.FIELDS}, "fake"
    st.query_cone(150, 20, 0.5, empty)
    first = len(n)
    st.query_cone(150, 20, 0.5, empty)
    assert len(n) == first


def test_fallback_order_and_timeout(monkeypatch):
    seen = []
    def slow_vizier(key, m, cell, timeout_s, server):
        seen.append(server)
        if server == "vizier.cds.unistra.fr":
            return cs._with_timeout(lambda: time.sleep(5), 0.2)     # busy server -> TimeoutError
        raise ConnectionError("mirror down")
    def esa(key, m, cell, timeout_s):
        seen.append("esa")
        return fake_cell(key, m)
    monkeypatch.setattr(cs, "fetch_vizier", slow_vizier)
    monkeypatch.setattr(cs, "fetch_esa", esa)
    f = make_fetcher(["vizier", "esa"], ["vizier.cds.unistra.fr", "vizier.cfa.harvard.edu"], 0.2, 2.0)
    t0 = time.time()
    arr, src = f("d055_r075", 16)
    assert src == "esa" and seen == ["vizier.cds.unistra.fr", "vizier.cfa.harvard.edu", "esa"]
    assert time.time() - t0 < 3


def test_vizier_request_removes_row_limit(monkeypatch):
    captured = {}

    class FakeVizier:
        def __init__(self, **kw):
            captured.update(kw)
        def query_constraints(self, catalog):
            from astropy.table import Table
            captured["catalog"] = catalog
            a0, a1, d0, d1 = cell_bounds("d055_r075")
            return [Table({"RA_ICRS": [a0 + 0.1, a0 - 1], "DE_ICRS": [d0 + 0.1, d0 + 0.1],
                           "pmRA": [1.0, 0], "pmDE": [2.0, 0], "Gmag": [12.0, 12.0]})]
    import astroquery.vizier
    monkeypatch.setattr(astroquery.vizier, "Vizier", FakeVizier)
    arr = cs.fetch_vizier("d055_r075", 16.0, 2.0, 30)
    assert captured["row_limit"] == -1 and captured["catalog"] == "I/355/gaiadr3"
    assert captured["column_filters"]["Gmag"] == "<16.00"
    assert len(arr["ra"]) == 1          # the out-of-cell row is trimmed


def test_geo_belt_and_visible_ranges():
    d0, d1 = geo_belt_dec_range(25.3573, 50.0, 20.0)
    assert -6.5 < d0 < d1 < -3.5        # GEO ring seen ~4 deg south of the equator from 25 N
    assert site_visible_dec_range(25.3573, 20.0) == pytest.approx((-44.6427, 90.0))


def test_import_csv_and_pipeline_lookup(tmp_path):
    import pandas as pd
    r = np.random.default_rng(3)
    df = pd.DataFrame({"ra": r.uniform(149, 151, 500), "dec": r.uniform(19, 21, 500),
                       "pmra": 0.0, "pmdec": 0.0, "phot_g_mean_mag": r.uniform(8, 18, 500)})
    df.to_csv(tmp_path / "g.csv", index=False)
    st = CatalogStore(str(tmp_path / "store"), mag_limit=16)
    n = import_csv(st, str(tmp_path / "g.csv"))
    assert n == int((df.phot_g_mean_mag < 16).sum())
    cat = get_reference_catalog(150, 20, 0.5, {"reference_catalog_dir": str(tmp_path / "store")}, False)
    assert cat is not None and cat.source.startswith("Gaia") and (cat.mag < 16).all()
    from sst_pipeline.utils.coordinate_utils import sep_arcsec
    inside = (df.phot_g_mean_mag < 16) & (sep_arcsec(df.ra, df.dec, 150, 20) <= 0.5 * 3600)
    assert len(cat) == int(inside.sum())


def test_single_csv_path_still_supported(tmp_path):
    import pandas as pd
    pd.DataFrame({"ra": [150.0, 150.1, 170.0], "dec": [20.0, 20.1, 20.0], "mag": [10.0, 11.0, 12.0]}).to_csv(
        tmp_path / "c.csv", index=False)
    cat = get_reference_catalog(150, 20, 0.5, {"reference_catalog_path": str(tmp_path / "c.csv")}, False)
    assert len(cat) == 2
