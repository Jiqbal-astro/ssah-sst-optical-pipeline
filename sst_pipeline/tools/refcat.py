"""
sst-refcat — build the local reference catalogue
------------------------------------------------
Build (or extend) the local tiled Gaia DR3 reference catalogue used for
plate verification and photometric zero-points, so pipeline runs never
wait on a catalogue server.

Run it on a networked machine, ideally off-peak; it is resumable (cells
already present are skipped) — just run the same command again after an
interruption. Copy the folder to the telescope computer afterwards.

Regions (pick one, or several in separate runs):
  --geo-belt             declinations where GEO objects appear from the site
  --site-visible         every declination that rises above --min-alt
  --box RA0 RA1 DEC0 DEC1   (RA0 > RA1 wraps through 0)
  --around RA DEC RADIUS    one field (degrees)
  --from-csv FILE        import a CSV you already have (e.g. a Gaia archive
                         export: ra, dec, pmra, pmdec, phot_g_mean_mag) — no download

Examples
  sst-refcat --geo-belt --dry-run
  sst-refcat --geo-belt --mag-limit 16
  sst-refcat --around 150 20 1.5
  sst-refcat --from-csv gaia_export.csv

Sources are tried per cell in order: VizieR (CDS, then the CfA mirror),
then the ESA Gaia archive. Use --source to force one.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from sst_pipeline.astrometry.catalog_store import (DEFAULT_CELL_DEG, CatalogStore, cell_of,
                                                   cells_for_box, cells_for_cone, make_fetcher)
from sst_pipeline.config import load_config

R_EARTH, R_GEO = 6378.137, 42164.0


def geo_belt_dec_range(lat_deg: float, height_m: float, min_alt_deg: float) -> tuple[float, float]:
    """Topocentric declination span of the equatorial GEO ring from a site
    (parallax shifts it ~ -4 deg at latitude 25 N), over all ring points
    above min_alt. Spherical Earth: adequate for choosing catalogue cells."""
    lat = math.radians(lat_deg)
    r_site = R_EARTH + height_m / 1000.0
    site = np.array([r_site * math.cos(lat), 0.0, r_site * math.sin(lat)])
    up = site / np.linalg.norm(site)
    decs = []
    for lon in np.radians(np.arange(-90, 90.01, 0.25)):     # ring longitude relative to the site
        v = np.array([R_GEO * math.cos(lon), R_GEO * math.sin(lon), 0.0]) - site
        rng = np.linalg.norm(v)
        if math.degrees(math.asin(np.dot(v, up) / rng)) >= min_alt_deg:
            decs.append(math.degrees(math.asin(v[2] / rng)))
    return min(decs), max(decs)


def site_visible_dec_range(lat_deg: float, min_alt_deg: float) -> tuple[float, float]:
    reach = 90.0 - min_alt_deg
    return max(lat_deg - reach, -90.0), min(lat_deg + reach, 90.0)


def import_csv(store: CatalogStore, path: str) -> int:
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    magc = next((cols[c] for c in ("mag", "phot_g_mean_mag", "gmag") if c in cols), None)
    if not ({"ra", "dec"} <= set(cols)) or magc is None:
        sys.exit("CSV needs ra, dec and mag / phot_g_mean_mag columns")
    get = lambda n: df[cols[n]].to_numpy(float) if n in cols else np.zeros(len(df))  # noqa: E731
    ra, dec, mag = get("ra"), get("dec"), df[magc].to_numpy(float)
    pmra, pmdec = get("pmra"), get("pmdec")
    keep = np.isfinite(mag) & (mag < store.mag_limit)
    keys = np.array([cell_of(a, d, store.cell) for a, d in zip(ra, dec)])
    n = 0
    for k in np.unique(keys[keep]):
        m = keep & (keys == k)
        store.save(k, {"ra": ra[m], "dec": dec[m], "pmra": pmra[m], "pmdec": pmdec[m], "mag": mag[m]},
                   f"import:{os.path.basename(path)}")
        n += int(m.sum())
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--geo-belt", action="store_true")
    g.add_argument("--site-visible", action="store_true")
    g.add_argument("--box", nargs=4, type=float, metavar=("RA0", "RA1", "DEC0", "DEC1"))
    g.add_argument("--around", nargs=3, type=float, metavar=("RA", "DEC", "RADIUS"))
    g.add_argument("--from-csv")
    p.add_argument("--out", default=None, help="store folder (default: astrometry.reference_catalog_dir)")
    p.add_argument("--config", default=None)
    p.add_argument("--mag-limit", type=float, default=None, help="G limit for a NEW store (default from config)")
    p.add_argument("--min-alt", type=float, default=20.0, help="minimum altitude for --geo-belt/--site-visible")
    p.add_argument("--margin", type=float, default=2.0, help="extra declination margin for --geo-belt (deg); "
                                                              "use ~15 to include drifting/inclined GEO objects")
    p.add_argument("--source", choices=["auto", "vizier", "esa"], default="auto")
    p.add_argument("--workers", type=int, default=3, help="parallel downloads (be polite: <= 4)")
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    acfg, site = cfg["astrometry"], cfg["site"]
    out = a.out or acfg.get("reference_catalog_dir") or "data/refcat"
    store = CatalogStore(out, mag_limit=a.mag_limit or acfg.get("catalog_mag_limit", 16.0),
                         cell=DEFAULT_CELL_DEG)
    if a.mag_limit and abs(a.mag_limit - store.mag_limit) > 1e-6:
        print(f"Note: existing store at {out} has G < {store.mag_limit}; keeping that limit.")

    if a.from_csv:
        n = import_csv(store, a.from_csv)
        print(f"Imported {n} stars (G < {store.mag_limit}) into {out}")
        return 0

    if a.geo_belt:
        d0, d1 = geo_belt_dec_range(site["latitude_deg"], site["height_m"], a.min_alt)
        d0, d1 = d0 - a.margin, d1 + a.margin
        keys, what = cells_for_box(0, 360, d0, d1, store.cell), f"GEO belt, Dec {d0:+.1f}..{d1:+.1f}"
    elif a.site_visible:
        d0, d1 = site_visible_dec_range(site["latitude_deg"], a.min_alt)
        keys, what = cells_for_box(0, 360, d0, d1, store.cell), f"site-visible sky, Dec {d0:+.1f}..{d1:+.1f}"
    elif a.box:
        keys, what = cells_for_box(*a.box, cell=store.cell), "box"
    else:
        keys, what = cells_for_cone(*a.around, cell=store.cell), "field"

    todo = [k for k in keys if not store.has(k)]
    print(f"{what}: {len(keys)} cells of ~{store.cell:.0f} deg, {len(keys) - len(todo)} already stored, "
          f"{len(todo)} to download (G < {store.mag_limit}) -> {out}")
    if a.dry_run or not todo:
        return 0

    sources = ["vizier", "esa"] if a.source == "auto" else [a.source]
    fetch = make_fetcher(sources, acfg.get("vizier_servers", ["vizier.cds.unistra.fr"]),
                         a.timeout or float(acfg.get("online_timeout_s", 60)), store.cell)

    def job(k):
        last = None
        for attempt in range(a.retries + 1):
            try:
                arr, src = fetch(k, store.mag_limit)
                store.save(k, arr, src)
                return k, len(arr["ra"]), src, None
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(min(30, 5 * (attempt + 1)))
        return k, 0, "", last

    t0, done, stars, failed = time.time(), 0, 0, []
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for fut in as_completed([ex.submit(job, k) for k in todo]):
            k, n, src, err = fut.result()
            done += 1
            if err:
                failed.append(k)
                print(f"  [{done}/{len(todo)}] {k} FAILED: {err}")
            else:
                stars += n
                rate = (time.time() - t0) / done
                print(f"  [{done}/{len(todo)}] {k}: {n} stars ({src})  ETA {rate * (len(todo) - done) / 60:.0f} min")
    print(f"Done: {stars} stars added; {len(failed)} cells failed"
          + (" — re-run the same command to retry them." if failed else "."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
