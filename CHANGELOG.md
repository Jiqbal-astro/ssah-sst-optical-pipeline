# Changelog

## 2.0.0 — November 2026

First public release, under the BSD 3-Clause License. This is a major revision
following a full scientific and logical review of 1.0.0.

### Fixed

- **TLE prediction.** The topocentric vector was transformed to ICRS together
  with its distance, which moved the origin to the barycentre. Predicted
  positions were off by more than 100°, and altitudes had the wrong sign.
  Predictions now agree with skyfield to about 1″.
- **Epochs.** Point sources were stamped at exposure start while trails used
  mid-exposure. All measurements now use mid-exposure, with a configurable
  DATE-OBS convention, time bias and rolling shutter, at microsecond precision.
- **Trail detection.** Canny/Hough detected both edges of a trail and split it
  into fragments, producing one false streak per edge and fragment. It is
  replaced by segmentation moments, along-track half-maximum refinement and
  fragment merging.
- **Trail endpoints.** Endpoints were reported as extra observations sharing the
  midpoint epoch. Each object now gives one observation per frame.
- **Tracklets.** Tracklets were split only by time gaps, so different objects and
  stars were merged. They are now built by kinematic linking (constant-rate
  model, at least 3 positions, RMS cut, consistency with the trail axis).
- **False unknown objects.** The static sky was built without the stars and
  linking was by nearest neighbour, so stars came back as "unknown objects".
  Static-sky rejection now uses all detections.
- **Trail photometry.** Pixels were sampled about 4 times, a −1.5 mag bias.
  Exact apertures, aperture correction and the CCD-equation uncertainty are now
  used.
- **Cosmic-ray fallback.** The fallback clipped about 20 % of star-core flux. It
  is removed, and astroscrappy is now a required dependency.
- **QC metrics.**
  - Photometric RMS was never computed; it now is.
  - Extinction was double-counted on frame zero-points; it no longer is.
  - Saturation and hot pixels were measured after calibration; they are now
    measured on raw data.
  - The cosmic-ray fraction was measured after cleaning; it is now measured
    before.
- **Header WCS.** The mount's pointing-model WCS was accepted without checks. It
  is now verified and refitted against Gaia DR3.
- **Reference catalogue.** Gaia queries were silently truncated to 50 rows and
  ignored proper motion. Both are fixed.
- **Header parsing.** Sexagesimal RA/Dec and DATE-OBS + TIME-OBS headers now
  parse. The pixel scale is taken from the CD-matrix determinant.
- **Calibration frames** (bias, dark, flat) found in the science folder are
  skipped.
- **Command line and configuration.** The `--solve-wcs` flag was ignored.
  Config sections the code read were missing. The demo script crashed. The ADES
  file was invalid. TLE URLs and sample-TLE checksums were wrong.

### Added

- Local tiled Gaia DR3 reference store with resumable builder (`sst-refcat`).
  It is filled from VizieR mirrors or the ESA archive, with timeouts.
- CCSDS TDM output.
- QC dimensions for seeing (FWHM), tracking ellipticity and transparency.
- Support for target-rate tracking.
- Along-track and cross-track O−C and the time-offset estimate.
- Light-time and aberration convention for star-referenced predictions.
- Packaging (`pyproject.toml`) with console commands.
- Site template, with warnings for unknown config keys.
- Synthetic data generator with ground truth.
- 78 tests, with continuous integration.
- Open-source project files: LICENSE, CITATION.cff, CONTRIBUTING, Code of
  Conduct, security policy, and issue and pull-request templates.

### Removed

- Online plate solving via nova.astrometry.net (operation is offline by design).
- Live TLE client, which stored credentials in configuration.
- Separate moving-object detector, now part of the tracklet linker.
- Hough streak detector.

## 1.0.0 — June 2026

Initial internal release.
