"""
sst-demo — end-to-end demonstration on synthetic data
-----------------------------------------------------
Generates the ground-truth sequence, loads its reference stars into a
local tiled store (as on the real system), runs the pipeline and compares
the recovered objects with the truth.

    sst-demo                 # writes into ./sst_demo/
    sst-demo --dir /tmp/demo
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import yaml

from sst_pipeline.astrometry.catalog_store import CatalogStore
from sst_pipeline.pipeline import run_pipeline
from sst_pipeline.tools.refcat import import_csv
from sst_pipeline.tools.synthetic import generate
from sst_pipeline.utils.coordinate_utils import sep_arcsec


def main(argv=None):
    p = argparse.ArgumentParser(prog="sst-demo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="sst_demo", help="working folder (recreated)")
    a = p.parse_args(argv)
    root = os.path.abspath(a.dir)
    shutil.rmtree(root, ignore_errors=True)
    frames, out, store = (os.path.join(root, d) for d in ("frames", "output", "refcat"))
    truth = generate(frames)
    import_csv(CatalogStore(store, mag_limit=16.0), os.path.join(frames, "reference_catalog.csv"))
    cfg = os.path.join(root, "demo_config.yaml")
    with open(cfg, "w") as f:
        yaml.safe_dump({"instrument": {"detector_nx": 1024, "detector_ny": 1024},
                        "astrometry": {"reference_catalog_dir": store}}, f)
    res = run_pipeline(frames, out, config_path=cfg, session_label="synthetic demo")

    print("\n--- Summary ---")
    for k, v in res.summary.items():
        print(f"  {k:30s}: {v}")
    print("\n--- Recovered objects vs truth (mid-exposure positions) ---")
    for tr in res.tracklets:
        errs = []
        for d in tr.members:
            objs = truth["frames"][d.frame_id]["objects"]
            best = min(objs.values(), key=lambda o: sep_arcsec(o["ra"], o["dec"], d.ra_deg, d.dec_deg))
            errs.append(sep_arcsec(best["ra"], best["dec"], d.ra_deg, d.dec_deg))
        print(f"  {tr.tracklet_id} {tr.kind:13s} n={len(tr.members):2d} rate={tr.rate_arcsec_s:8.3f}\"/s "
              f"PA={tr.pa_deg:6.2f}  error vs truth: median {np.median(errs):.3f}\" max {np.max(errs):.3f}\"")
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
