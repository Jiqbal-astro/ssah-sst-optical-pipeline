# SSAH SST Optical Data Processing Pipeline

[![tests](https://github.com/YOUR-GITHUB-USER/ssah-sst-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR-GITHUB-USER/ssah-sst-pipeline/actions/workflows/tests.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A Python pipeline for optical **Space Surveillance and Tracking (SST)** at the
Sharjah Astronomical Observatory (SSAH). It turns raw FITS frames into
quality-controlled, star-referenced positions and magnitudes of resident space
objects (RSOs), links them into tracklets, correlates them with a TLE catalogue,
and exports CCSDS TDM and ADES files.

**Documentation:** [Technical Reference Manual (PDF)](docs/manual/SSAH_SST_Pipeline_Manual.pdf) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

## Features

- **Dual-track quality control.** Every frame gets independent photometry and
  astrometry decisions, so trailed or marginal frames still contribute positions.
- **Verified astrometry.** The header WCS is checked and refitted (TAN + SIP)
  against Gaia DR3 stars. Positions are reported only from verified plates.
- **Correct epochs.** Every measurement is stamped at mid-exposure, with
  calibrated time bias and rolling-shutter correction, at microsecond precision.
- **Robust trail measurement.** Length, midpoint and flux come from the
  along-track profile, so stars crossing a trail do not bias them.
- **Sidereal or target-rate tracking.** Both modes are supported.
- **Offline by design.** Reference stars come from a local tiled Gaia DR3 store
  that is built once. TLEs come from a local file.
- **Known objects and UCTs.** SGP4 correlation uses separate along-track and
  cross-track tolerances. It reports a time offset that can be used to calibrate
  the camera clock.
- **Validated.** The test suite includes an end-to-end run on synthetic data with
  ground truth and a cross-check of SGP4 predictions against skyfield.

## Installation

Python 3.10 or later:

```bash
git clone https://github.com/YOUR-GITHUB-USER/ssah-sst-pipeline
cd ssah-sst-pipeline
pip install -e ".[all]"        # pipeline + online catalogue tools + tests
```

Optional: install astrometry.net `solve-field` and its index files to plate-solve
frames that have no WCS in the header.

## Quick start

```bash
sst-demo
```

This generates a synthetic 12-frame sequence with known objects, runs the
pipeline, and prints the recovered objects against the truth. You should see
2 multi-frame tracklets and 1 single-frame streak, with position errors of about
0.2″ or better.

## Running on real data

1. **Configure the site.** Copy `config/site_template.yaml`, for example to
   `config/ssah_site.yaml`, and replace every `[REQUIRED]` value: camera gain,
   read noise and non-linearity limit, surveyed site coordinates, what DATE-OBS
   marks, and the measured time bias.
2. **Build the reference catalogue once,** on a networked machine:
   ```bash
   sst-refcat --geo-belt --dry-run      # preview
   sst-refcat --geo-belt                # GEO band seen from the site
   sst-refcat --site-visible            # whole sky above 20° altitude
   ```
   The build is resumable. Copy `data/refcat/` to the telescope computer.
3. **Download TLEs** (refresh daily for LEO work):
   ```bash
   sst-tle --group active --out data/tle_catalog.txt
   ```
4. **Run:**
   ```bash
   sst-pipeline /path/to/fits /path/to/output --config config/ssah_site.yaml
   ```

| Option | Meaning |
|---|---|
| `--config FILE` | Site YAML, merged over the packaged defaults. Unknown keys trigger a warning. |
| `--catalog` | Download any reference-store cells missing for the observed fields, and keep them. |
| `--solve-wcs` | Run local `solve-field` for frames without a header WCS. |
| `--pattern GLOB` | FITS file pattern (default `*.fit*`). |
| `--label TEXT` | Free-text session label written to the summary. |

## Outputs

| File | Content |
|---|---|
| `qc_matrix.csv`, `rejected_frames.log` | Per-frame QC metrics and pass/fail decisions for each track, with reasons |
| `session_metadata.csv` | Parsed header values for every frame |
| `detections.csv` | Every non-star detection: position, epoch, uncertainties, magnitude, status, TLE O−C |
| `tracklets.csv`, `uncorrelated_tracks.csv` | Linked tracklets; those not matched to the TLE catalogue (UCTs) |
| `observations_tdm.kvn` | CCSDS Tracking Data Message (RADEC, ICRF, UTC) |
| `observations_ades.psv` | ADES PSV, for archiving (the MPC does not accept artificial satellites) |
| `qc_summary.json`, `SST_QC_Report.xlsx`, `plot_*.png`, `pipeline_*.log` | Summary, workbook, diagnostic plots, session log |

## Repository layout

```
sst_pipeline/            Python package
  pipeline.py            orchestrator and `sst-pipeline` command
  defaults.yaml          every threshold and parameter, with comments
  io/ calibration/ detection/ qc/ astrometry/ photometry/
  tracking/ output/ reporting/ utils/
  tools/                 sst-refcat, sst-tle, sst-synthetic, sst-demo
config/site_template.yaml  the file you copy and fill in
docs/manual/             technical reference manual (LaTeX source + PDF)
tests/                   78 tests (pytest)
```

## Testing

```bash
pytest          # about 40 s, including an end-to-end run
```

## Status and disclaimer

The pipeline has been validated on synthetic data with ground truth (see the
manual, *Validation*). Its thresholds, time bias and photometric performance
still have to be established on real observations for each site. The software
is provided "as is", without warranty. It is **not** a certified system for
collision avoidance or other safety-critical decisions.

## Citing

If you use this software in published work, please cite it. GitHub's
**Cite this repository** button provides the reference generated from
[`CITATION.cff`](CITATION.cff).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately, as
described in [SECURITY.md](SECURITY.md).

## License

Released under the [BSD 3-Clause License](LICENSE).

Third-party packages are used under their own licenses and are not bundled.
Most are BSD- or MIT-licensed (Astropy, Photutils, Astroscrappy, sgp4,
Astroquery, Skyfield). SEP is LGPL-3.0-or-later.

## Acknowledgements

Developed by Javed Iqbal during an internship at the Sharjah Astronomical
Observatory (SSAH).

The pipeline relies on data and services from:
- **Gaia DR3** (ESA, processed by the Gaia DPAC);
- the **VizieR** catalogue service (CDS, Strasbourg);
- **CelesTrak** (TLE distribution);
- **astrometry.net**, optionally.

Publications that use results based on Gaia data should include the
acknowledgement text that ESA requests; see the
[Gaia credit guidelines](https://www.cosmos.esa.int/web/gaia-users/credits).
