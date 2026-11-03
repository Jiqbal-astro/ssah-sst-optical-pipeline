"""
metadata_extractor.py
---------------------
One provenance row per frame (pass or fail). Values come from the header
validator's parsed results where available, so RA/Dec in sexagesimal form
and DATE-OBS + TIME-OBS headers are handled consistently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from astropy.io import fits

from sst_pipeline.io.header_validator import HeaderValidationResult, parse_obs_time
from sst_pipeline.utils.coordinate_utils import format_hmsdms, parse_dec_deg, parse_ra_deg


@dataclass
class FrameMetadata:
    frame_id: str
    file_path: str
    date_obs: str = ""
    mjd_obs: float = float("nan")
    exptime_s: float = float("nan")
    object_name: str = ""
    image_type: str = ""
    telescope: str = ""
    instrument: str = ""
    filter_name: str = ""
    ccd_temp_c: float = float("nan")
    header_gain: float = float("nan")
    ra_deg: float = float("nan")
    dec_deg: float = float("nan")
    ra_hms: str = ""
    dec_dms: str = ""
    airmass_header: float = float("nan")
    wcs_in_header: bool = False
    wcs_ctype: str = ""
    header_pixel_scale_arcsec: float = float("nan")
    observer: str = ""
    naxis1: int = 0
    naxis2: int = 0
    header_valid: bool = True
    header_missing: str = ""
    header_invalid: str = ""
    header_warnings: str = ""


def _f(header, *keys) -> float:
    for k in keys:
        try:
            v = header.get(k)
            if v not in (None, ""):
                return float(v)
        except (TypeError, ValueError):
            continue
    return float("nan")


def _s(header, *keys, default="") -> str:
    for k in keys:
        v = header.get(k)
        if v not in (None, ""):
            return str(v).strip()
    return default


def _pixel_scale(header: fits.Header) -> float:
    """sqrt(|det CD|) handles rotation; CD1_1 alone does not."""
    try:
        if "CD1_1" in header:
            cd = np.array([[header.get("CD1_1", 0.0), header.get("CD1_2", 0.0)],
                           [header.get("CD2_1", 0.0), header.get("CD2_2", 0.0)]], float)
            return float(np.sqrt(abs(np.linalg.det(cd))) * 3600)
        if "CDELT1" in header and "CDELT2" in header:
            return float(np.sqrt(abs(header["CDELT1"] * header["CDELT2"])) * 3600)
    except (TypeError, ValueError):
        pass
    return float("nan")


def extract_metadata(frame_id: str, file_path: str, header: fits.Header,
                     hv: HeaderValidationResult, ra_numeric_unit: str = "deg") -> FrameMetadata:
    t = parse_obs_time(header)
    ra = hv.resolved.get("RA")
    dec = hv.resolved.get("DEC")
    if ra is None:
        ra = parse_ra_deg(header.get("RA") or header.get("OBJCTRA"), ra_numeric_unit)
    if dec is None:
        dec = parse_dec_deg(header.get("DEC") or header.get("OBJCTDEC"))
    ra_hms, dec_dms = format_hmsdms(ra, dec) if (ra is not None and dec is not None) else ("", "")
    wcs = all(k in header for k in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"))
    return FrameMetadata(
        frame_id=frame_id, file_path=file_path,
        date_obs=t.utc.isot if t is not None else "",
        mjd_obs=float(t.mjd) if t is not None else float("nan"),
        exptime_s=_f(header, "EXPTIME", "EXPOSURE"),
        object_name=_s(header, "OBJECT", "TARGET"),
        image_type=_s(header, "IMAGETYP", "IMGTYPE", "FRAMETYP"),
        telescope=_s(header, "TELESCOP", "TELESCOPE"),
        instrument=_s(header, "INSTRUME", "INSTRUMENT", "CAMERA"),
        filter_name=_s(header, "FILTER", "FLTNAME", default="CLEAR"),
        ccd_temp_c=_f(header, "CCD-TEMP", "CCDTEMP", "TEMPERAT"),
        header_gain=_f(header, "GAIN", "EGAIN"),
        ra_deg=float(ra) if ra is not None else float("nan"),
        dec_deg=float(dec) if dec is not None else float("nan"),
        ra_hms=ra_hms, dec_dms=dec_dms,
        airmass_header=_f(header, "AIRMASS"),
        wcs_in_header=wcs,
        wcs_ctype=f"{header.get('CTYPE1','')}/{header.get('CTYPE2','')}" if wcs else "",
        header_pixel_scale_arcsec=_pixel_scale(header),
        observer=_s(header, "OBSERVER"),
        naxis1=int(header.get("NAXIS1", 0) or 0), naxis2=int(header.get("NAXIS2", 0) or 0),
        header_valid=hv.is_valid,
        header_missing="; ".join(hv.missing_required),
        header_invalid="; ".join(hv.invalid_values),
        header_warnings="; ".join(hv.warnings),
    )


def build_metadata_table(rows: list[FrameMetadata]) -> pd.DataFrame:
    return pd.DataFrame([asdict(m) for m in rows])
