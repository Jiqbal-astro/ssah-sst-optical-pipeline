# Contributing

Thank you for your interest in improving the SSAH SST pipeline. Bug reports,
test data, documentation fixes and code contributions are all welcome.

## Reporting a problem

Open an issue using the **Bug report** template. Include:

- the pipeline version (`python -c "import sst_pipeline; print(sst_pipeline.__version__)"`);
- the command you ran and your site configuration file (remove any private paths);
- the relevant part of `pipeline_<UTC>.log` and, if possible, `qc_matrix.csv`;
- a FITS header (`fitsheader frame.fits`) if the problem is frame-specific.

Please do not attach large FITS files to issues; link to them instead.

## Development setup

```bash
git clone https://github.com/YOUR-GITHUB-USER/ssah-sst-pipeline
cd ssah-sst-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
pytest
```

All tests must pass before a pull request is merged. Continuous integration
runs them on Python 3.10, 3.11 and 3.12.

## Guidelines for changes

- **Science first.** A change that affects positions, epochs, magnitudes or QC
  decisions needs a test that checks the *value* against a known answer.
  Checking only that the output has a plausible range is not enough. The
  synthetic generator (`sst_pipeline/tools/synthetic.py`) provides ground truth
  for end-to-end checks.
- **No hard-coded thresholds.** New parameters go in
  `sst_pipeline/defaults.yaml`, with a comment giving the unit and the reasoning.
- **Cite methods.** If you implement a published method, name the paper in the
  module docstring and add it to the manual's references.
- **Keep it offline.** Runtime stages must not require network access.
  Downloads belong in the `tools/` commands.
- **Style.** Follow PEP 8, add type hints to public functions, and give every
  module a docstring that explains its method. Run `pyflakes sst_pipeline tests`
  before committing.
- **Documentation.** Update `CHANGELOG.md`, and update the manual
  (`docs/manual/`) when behaviour or configuration changes. Rebuild the manual
  with `latexmk -pdf SSAH_SST_Pipeline_Manual.tex`.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make focused commits with clear messages.
3. Make sure `pytest` passes and add tests for new behaviour.
4. Open a pull request using the template, and describe the scientific effect
   of the change.

By contributing, you agree that your contributions are licensed under the
project's [BSD 3-Clause License](LICENSE).
