"""
sst-tle — download a TLE catalogue
----------------------------------
Download a TLE catalogue from CelesTrak's GP API and save it for offline use.
Run on a networked machine before each session (LEO TLE errors grow by
~1-3 km/day; refresh at least daily for LEO work).

    sst-tle --group active --out data/tle_catalog.txt
    sst-tle --catnr 25544 --out data/iss.txt

Groups (CelesTrak): active, stations, visual, gps-ops, galileo, geo,
starlink, cosmos-1408-debris, fengyun-1c-debris, iridium-33-debris, ...
URL format: https://celestrak.org/NORAD/elements/gp.php?GROUP=<g>&FORMAT=tle
"""

import argparse
import os
import sys
import urllib.request
from datetime import datetime, timezone

from sst_pipeline.tracking.tle_manager import tle_checksum_ok

BASE = "https://celestrak.org/NORAD/elements/gp.php"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--group", default="active")
    g.add_argument("--catnr", type=int, help="single NORAD catalogue number")
    p.add_argument("--out", default="data/tle_catalog.txt")
    a = p.parse_args()
    url = f"{BASE}?CATNR={a.catnr}&FORMAT=tle" if a.catnr else f"{BASE}?GROUP={a.group}&FORMAT=tle"
    print(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "SSAH-SST-Pipeline/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Download failed: {exc}")
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    l1 = [l for l in lines if l.startswith("1 ")]
    bad = sum(not tle_checksum_ok(l) for l in lines if l.startswith(("1 ", "2 ")))
    if not l1:
        sys.exit(f"No TLEs returned (server said: {text[:200]!r})")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"# CelesTrak GP {('CATNR=' + str(a.catnr)) if a.catnr else ('GROUP=' + a.group)}\n")
        f.write(f"# Downloaded {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} from {url}\n")
        f.write("\n".join(lines) + "\n")
    print(f"Saved {len(l1)} objects to {a.out} ({bad} lines with bad checksum)")
    print(f'Set tle_tracking.tle_file_path: "{os.path.abspath(a.out)}" in your config.')


if __name__ == "__main__":
    main()
