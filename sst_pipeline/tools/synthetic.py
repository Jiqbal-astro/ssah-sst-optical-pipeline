"""
sst-synthetic — synthetic sequence with ground truth
----------------------------------------------------
Physically consistent synthetic sequence WITH GROUND TRUTH, so the pipeline
can be validated quantitatively.

Scene (sidereal tracking, 1024x1024, 0.82"/px, 2 s exposures every 10 s):
  * fixed star field: catalogued stars (G=9.5-16, written to
    reference_catalog.csv) + faint uncatalogued stars (G=16.3-17.5) that
    must be rejected by the static-sky test
  * RSO-A  : trailed, 6"/s  (multi-frame tracklet)
  * RSO-B  : point-like, 0.6"/s (only separable from stars by motion)
  * RSO-C  : one-frame fast streak, 150"/s (single-streak tracklet)
  * QC cases: cosmic-ray frame, cloud frame, saturated/bleeding star frame,
    missing-TELESCOP frame, plus a BIAS frame that must be skipped
  * header WCS deliberately offset ~29" and rotated 0.05 deg from truth
Noise: Poisson (in electrons) + read noise, bias pedestal 1000 ADU.

Writes FITS + reference_catalog.csv + truth.json into the output folder.
"""

from __future__ import annotations

import json
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from sst_pipeline.utils.coordinate_utils import from_tangent_plane

N, SCALE, GAIN, RN_E, BIAS = 1024, 0.82, 1.47, 9.3, 1000.0
RA0, DEC0 = 150.0, 20.0
ZP_TRUE = 23.5                # mag giving 1 ADU/s
SKY_ADU_S = 300.0
FWHM_PX = 3.4
EXPTIME, CADENCE, NFRAMES = 2.0, 10.0, 12
T0 = Time("2026-03-15T20:00:00", scale="utc")


def make_wcs(ra, dec, rot_deg):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [N / 2 + 0.5, N / 2 + 0.5]
    c, s, sc = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg)), SCALE / 3600
    w.wcs.cd = [[-sc * c, sc * s], [sc * s, sc * c]]
    w.pixel_shape = (N, N)
    return w


def add_gauss(img, x, y, flux, sigma=FWHM_PX / 2.3548):
    r = int(np.ceil(5 * sigma)) + 1
    x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, N)
    y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, N)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    img[y0:y1, x0:x1] += flux / (2 * np.pi * sigma ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def flux_of(mag, t=EXPTIME):
    return 10 ** (-0.4 * (mag - ZP_TRUE)) * t


class Mover:
    def __init__(self, name, xi0, eta0, rate, pa, mag, frames=None):
        self.name, self.xi0, self.eta0, self.mag, self.frames = name, xi0, eta0, mag, frames
        self.ve, self.vn = rate * np.sin(np.radians(pa)), rate * np.cos(np.radians(pa))
        self.rate, self.pa = rate, pa

    def radec(self, dt):  # dt seconds from reference epoch
        return from_tangent_plane(self.xi0 + self.ve * dt, self.eta0 + self.vn * dt, RA0, DEC0)


def render(true_wcs, stars, movers, t_start, t_ref, frame_k, rng, transparency=1.0,
           sky_factor=1.0, cosmic=0, bleed=False):
    sig = np.zeros((N, N))
    xs, ys = true_wcs.world_to_pixel(SkyCoord(stars["ra"] * u.deg, stars["dec"] * u.deg))
    for x, y, m in zip(xs, ys, stars["mag"]):
        if -10 < x < N + 10 and -10 < y < N + 10:
            add_gauss(sig, x, y, flux_of(m) * transparency)
    truth = {}
    for mv in movers:
        if mv.frames is not None and frame_k not in mv.frames:
            continue
        dt0 = (t_start - t_ref).sec
        ra_mid, dec_mid = mv.radec(dt0 + EXPTIME / 2)
        xm, ym = true_wcs.world_to_pixel(SkyCoord(ra_mid * u.deg, dec_mid * u.deg))
        if not (15 < xm < N - 15 and 15 < ym < N - 15):
            continue
        nstep = max(int(mv.rate * EXPTIME / SCALE * 4), 1)
        ts = dt0 + (np.arange(nstep) + 0.5) / nstep * EXPTIME
        ra, dec = mv.radec(ts)
        px, py = true_wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        for x, y in zip(px, py):
            add_gauss(sig, x, y, flux_of(mv.mag) * transparency / nstep)
        truth[mv.name] = {"ra": float(ra_mid), "dec": float(dec_mid), "x": float(xm), "y": float(ym),
                          "mag": mv.mag, "rate": mv.rate, "pa": mv.pa}
    if bleed:
        x, y = N * 0.3, N * 0.6
        add_gauss(sig, x, y, flux_of(5.0))
        sig[int(y) - 300:int(y) + 300, int(x) - 1:int(x) + 2] = 1e7
    elec = rng.poisson(np.clip((sig + SKY_ADU_S * EXPTIME * sky_factor) * GAIN, 0, None)).astype(float)
    adu = elec / GAIN + rng.normal(0, RN_E / GAIN, (N, N)) + BIAS
    if cosmic:
        cx, cy = rng.integers(5, N - 5, cosmic), rng.integers(5, N - 5, cosmic)
        adu[cy, cx] += rng.uniform(800, 5000, cosmic)
    return np.clip(adu, 0, 65535), truth


def header(t_start, hdr_wcs, obj="FIELD-A", telescop=True, imagetyp="Light"):
    h = hdr_wcs.to_header()
    h["DATE-OBS"] = t_start.isot
    h["EXPTIME"] = EXPTIME
    h["OBJECT"] = obj
    if telescop:
        h["TELESCOP"] = "SSAH-T1"
    h["INSTRUME"] = "SSAH-CCD-01"
    h["FILTER"] = "CLEAR"
    h["IMAGETYP"] = imagetyp
    c = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    h["OBJCTRA"] = c.ra.to_string(unit=u.hour, sep=" ", precision=2, pad=True)
    h["OBJCTDEC"] = c.dec.to_string(sep=" ", precision=1, alwayssign=True, pad=True)
    h["CCD-TEMP"] = -15.0
    return h


def generate(out_dir: str, seed: int = 7) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    true_wcs = make_wcs(RA0, DEC0, 0.30)
    hdr_ra, hdr_dec = from_tangent_plane(25.0, -15.0, RA0, DEC0)   # ~29" pointing error
    hdr_wcs = make_wcs(float(hdr_ra), float(hdr_dec), 0.35)

    half = N * SCALE / 2 + 30
    def field(n, lo, hi):
        xi, eta = rng.uniform(-half, half, n), rng.uniform(-half, half, n)
        ra, dec = from_tangent_plane(xi, eta, RA0, DEC0)
        return {"ra": ra, "dec": dec, "mag": rng.uniform(lo, hi, n)}
    cat, faint = field(220, 9.5, 16.0), field(60, 16.3, 17.5)
    stars = {k: np.concatenate([cat[k], faint[k]]) for k in cat}
    import pandas as pd
    pd.DataFrame({"ra": cat["ra"], "dec": cat["dec"], "mag": cat["mag"],
                  "pmra": 0.0, "pmdec": 0.0, "ref_epoch": 2016.0}).to_csv(
        os.path.join(out_dir, "reference_catalog.csv"), index=False)

    t_ref = T0 + TimeDelta((NFRAMES // 2 * CADENCE + EXPTIME / 2) * u.s)
    movers = [Mover("RSO-A", -40.0, 30.0, 6.0, 60.0, 12.5),
              Mover("RSO-B", 60.0, -80.0, 0.6, 200.0, 13.0),
              # RSO-C placed so that its frame-4 midpoint (dt = -20 s) is near the centre
              Mover("RSO-C", 150 * np.sin(np.radians(115)) * 20 - 50,
                    150 * np.cos(np.radians(115)) * 20 + 40, 150.0, 115.0, 10.5, frames={4})]
    special = {3: dict(cosmic=400), 8: dict(transparency=0.25, sky_factor=1.6),
               10: dict(bleed=True)}
    truth = {"zp_true": ZP_TRUE, "exptime": EXPTIME, "frames": {}, "true_wcs_center": [RA0, DEC0]}
    for k in range(NFRAMES):
        t_start = T0 + TimeDelta(k * CADENCE * u.s)
        img, tr = render(true_wcs, stars, movers, t_start, t_ref, k, rng, **special.get(k, {}))
        fid = f"seq_{k:03d}"
        h = header(t_start, hdr_wcs, telescop=(k != 11))
        fits.PrimaryHDU(img.astype(np.float32), header=h).writeto(os.path.join(out_dir, fid + ".fits"),
                                                                 overwrite=True)
        truth["frames"][fid] = {"mid_exposure_utc": (t_start + TimeDelta(EXPTIME / 2 * u.s)).isot,
                                "objects": tr}
    bias = rng.normal(BIAS, RN_E / GAIN, (N, N))
    fits.PrimaryHDU(bias.astype(np.float32),
                    header=header(T0, hdr_wcs, obj="BIAS", imagetyp="Bias Frame")).writeto(
        os.path.join(out_dir, "zz_bias_001.fits"), overwrite=True)
    truth["special_frames"] = {"cosmic": "seq_003", "cloud": "seq_008", "bleed": "seq_010",
                               "bad_header": "seq_011", "bias": "zz_bias_001", "single_streak": "seq_004"}
    with open(os.path.join(out_dir, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    return truth


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="sst-synthetic", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", nargs="?", default="sst_demo/frames")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    generate(a.out_dir, a.seed)
    print(f"Synthetic sequence, reference_catalog.csv and truth.json written to {a.out_dir}/")


if __name__ == "__main__":
    main()
