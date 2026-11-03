"""
source_classifier.py
--------------------
Morphology (POINT / TRAILED / ARTIFACT / EXTENDED) plus the ROLE a source
plays, which depends on how the telescope was tracking:

  tracking_mode   stars look    targets look   reference stars for WCS/ZP
  -------------   ----------    ------------   --------------------------
  sidereal        POINT         TRAILED        POINT sources
  rate (target)   TRAILED       POINT          TRAILED sources (midpoints)

Morphology alone cannot separate a slow point-like RSO from a star in
sidereal mode; that is done later by reference-catalogue matching and the
multi-frame static-sky test (tracking/linker.py).
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from sst_pipeline.detection.source_extractor import Source


class Morph(str, Enum):
    POINT = "POINT"
    TRAILED = "TRAILED"
    ARTIFACT = "ARTIFACT"
    EXTENDED = "EXTENDED"


def classify(src: Source, cfg: dict) -> Morph:
    # Saturated, column/row-aligned "trail" = charge bleeding from a bright star
    if src.saturated and src.elongation >= cfg.get("elongation_threshold", 2.5):
        ang = np.degrees(src.theta) % 90.0
        if min(ang, 90.0 - ang) <= cfg.get("bleed_axis_tolerance_deg", 1.0):
            return Morph.ARTIFACT
    if src.elongation >= cfg.get("elongation_threshold", 2.5) and \
            src.length >= cfg.get("min_streak_length_px", 8.0):
        return Morph.TRAILED
    if src.fwhm < cfg.get("min_fwhm_px", 1.2) or src.npix < 3:
        return Morph.ARTIFACT       # hot pixel / cosmic-ray hit: sharper than the PSF
    if src.fwhm > cfg.get("max_point_fwhm_px", 15.0):
        return Morph.EXTENDED
    return Morph.POINT


def reference_morph(tracking_mode: str) -> Morph:
    """Morphology of field stars for this tracking mode."""
    return Morph.POINT if tracking_mode == "sidereal" else Morph.TRAILED


def target_morph(tracking_mode: str) -> Morph:
    """Morphology an RSO is expected to have for this tracking mode."""
    return Morph.TRAILED if tracking_mode == "sidereal" else Morph.POINT
