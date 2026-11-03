"""
fits_loader.py
---------------
Thin, defensive wrapper around astropy.io.fits for batch ingestion of
SST FITS frames. Keeps file handling in one place so the rest of the
pipeline never touches raw I/O directly.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.io import fits


@dataclass
class FitsFrame:
    """Container for one loaded FITS frame and its essential metadata."""

    path: str
    data: np.ndarray
    header: fits.Header
    frame_id: str = field(init=False)
    load_error: Optional[str] = None

    def __post_init__(self):
        name = os.path.basename(self.path)
        for ext in (".fits.gz", ".fit.gz", ".fits.fz", ".fits", ".fit", ".fts"):
            if name.lower().endswith(ext):
                name = name[: -len(ext)]
                break
        self.frame_id = name


def discover_fits_files(directory: str, pattern: str = "*.fit*") -> list[str]:
    """
    Find FITS files in a directory (non-recursive by default).
    Accepts .fit, .fits, .fts and gzip/bzip2-compressed variants.
    """
    files = set(glob.glob(os.path.join(directory, pattern)))
    for ext in ("*.fts", "*.fits.gz", "*.fit.gz", "*.fits.fz"):
        files.update(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def load_frame(path: str, extension: int = 0) -> FitsFrame:
    """
    Load a single FITS frame. Never raises on a malformed file —
    instead returns a FitsFrame with `load_error` set, so a single
    corrupt frame never stops a batch run.
    """
    try:
        with fits.open(path, memmap=False, ignore_missing_simple=True) as hdul:
            hdu = hdul[extension]
            data = hdu.data
            header = hdu.header
            if data is None:
                # some frames store the image in extension 1 (compressed HDU)
                for h in hdul:
                    if h.data is not None:
                        data = h.data
                        header = h.header
                        break
            if data is None:
                return FitsFrame(path=path, data=np.array([]), header=header,
                                  load_error="No image data found in any HDU")
            return FitsFrame(path=path, data=np.asarray(data, dtype=np.float64),
                              header=header)
    except Exception as exc:  # noqa: BLE001 - intentionally broad for batch safety
        empty_header = fits.Header()
        return FitsFrame(path=path, data=np.array([]), header=empty_header,
                          load_error=f"{type(exc).__name__}: {exc}")


def load_batch(directory: str, pattern: str = "*.fit*") -> list[FitsFrame]:
    """Load every FITS file matching `pattern` in `directory`."""
    paths = discover_fits_files(directory, pattern)
    return [load_frame(p) for p in paths]
