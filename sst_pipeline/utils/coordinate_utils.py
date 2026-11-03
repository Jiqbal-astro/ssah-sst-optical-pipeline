"""
coordinate_utils.py
-------------------
Coordinate helpers. All sky positions are ICRS RA/Dec in degrees.
Separations use the Vincenty formula (exact at all scales, no RA-wrap or
pole problems — the small-angle "dRA*cos(dec)" shortcut used previously
breaks across RA=0/360).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

_SEXA = re.compile(r"^\s*([+-]?)\s*(\d+(?:\.\d*)?)[\s:hd°]+(\d+(?:\.\d*)?)[\s:m']+(\d+(?:\.\d*)?)\s*[s\"]?\s*$")


def _parse_sexagesimal(text: str) -> Optional[float]:
    m = _SEXA.match(str(text))
    if not m:
        return None
    sign = -1.0 if m.group(1) == "-" else 1.0
    a, b, c = (float(m.group(i)) for i in (2, 3, 4))
    return sign * (a + b / 60.0 + c / 3600.0)


def parse_ra_deg(value, numeric_unit: str = "deg") -> Optional[float]:
    """RA keyword -> degrees. Sexagesimal strings ('12 30 45.2', '12:30:45')
    are hours; bare numbers use `numeric_unit` (telescope software differs)."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        v = float(value)
        return (v * 15.0 if numeric_unit == "hour" else v) % 360.0
    s = str(value).strip()
    sexa = _parse_sexagesimal(s)
    if sexa is not None:
        return (sexa * 15.0) % 360.0
    try:
        v = float(s)
    except ValueError:
        return None
    return (v * 15.0 if numeric_unit == "hour" else v) % 360.0


def parse_dec_deg(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        v = float(value)
    else:
        s = str(value).strip()
        sexa = _parse_sexagesimal(s)
        if sexa is not None:
            v = sexa
        else:
            try:
                v = float(s)
            except ValueError:
                return None
    return v if -90.0 <= v <= 90.0 else None


def sep_arcsec(ra1, dec1, ra2, dec2):
    """Vincenty great-circle separation in arcsec (vectorised)."""
    ra1, dec1, ra2, dec2 = (np.radians(np.asarray(a, dtype=float)) for a in (ra1, dec1, ra2, dec2))
    dra = ra2 - ra1
    num = np.hypot(np.cos(dec2) * np.sin(dra),
                   np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra))
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    out = np.degrees(np.arctan2(num, den)) * 3600.0
    return float(out) if np.ndim(out) == 0 else out


def position_angle_deg(ra1, dec1, ra2, dec2):
    """Position angle of point 2 as seen from point 1, east of north, degrees."""
    ra1, dec1, ra2, dec2 = (np.radians(np.asarray(a, dtype=float)) for a in (ra1, dec1, ra2, dec2))
    dra = ra2 - ra1
    pa = np.arctan2(np.sin(dra), np.cos(dec1) * np.tan(dec2) - np.sin(dec1) * np.cos(dra))
    out = np.degrees(pa) % 360.0
    return float(out) if np.ndim(out) == 0 else out


def to_tangent_plane(ra, dec, ra0, dec0):
    """Gnomonic projection about (ra0, dec0); returns (xi, eta) in arcsec.
    xi points East, eta North. Avoids RA wrap-around in linear fits."""
    ra, dec = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    a0, d0 = np.radians(ra0), np.radians(dec0)
    cosc = np.sin(d0) * np.sin(dec) + np.cos(d0) * np.cos(dec) * np.cos(ra - a0)
    xi = np.cos(dec) * np.sin(ra - a0) / cosc
    eta = (np.cos(d0) * np.sin(dec) - np.sin(d0) * np.cos(dec) * np.cos(ra - a0)) / cosc
    k = np.degrees(1.0) * 3600.0
    return xi * k, eta * k


def from_tangent_plane(xi_arcsec, eta_arcsec, ra0, dec0):
    k = np.degrees(1.0) * 3600.0
    xi, eta = np.asarray(xi_arcsec, float) / k, np.asarray(eta_arcsec, float) / k
    a0, d0 = np.radians(ra0), np.radians(dec0)
    denom = np.cos(d0) - eta * np.sin(d0)
    ra = a0 + np.arctan2(xi, denom)
    dec = np.arctan2(np.sin(d0) + eta * np.cos(d0), np.hypot(xi, denom))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def compute_airmass(ra_deg: float, dec_deg: float, obs_time: Time,
                    lat_deg: float, lon_deg: float, height_m: float) -> float:
    """Airmass from site/time/pointing (Rozenberg 1966, valid to the horizon).
    Returns NaN below the horizon."""
    site = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=height_m * u.m)
    alt = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").transform_to(
        AltAz(obstime=obs_time, location=site)).alt.deg
    if alt <= 0:
        return float("nan")
    cz = np.cos(np.radians(90.0 - alt))
    return float(1.0 / (cz + 0.025 * np.exp(-11.0 * cz)))


def format_hmsdms(ra_deg: float, dec_deg: float) -> tuple[str, str]:
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return (c.ra.to_string(unit=u.hour, sep=":", precision=3, pad=True),
            c.dec.to_string(sep=":", precision=2, alwayssign=True, pad=True))


def pixel_errors_to_sky(wcs, sx_px: float, sy_px: float) -> tuple[float, float]:
    """Propagate (x,y) centroid sigmas to (RA*cos(dec), Dec) sigmas in arcsec
    through the local linear WCS (accounts for rotation and flips)."""
    cd = wcs.pixel_scale_matrix * 3600.0          # arcsec / px, rows = (RA, Dec)
    cov_px = np.diag([sx_px ** 2, sy_px ** 2])
    cov_sky = cd @ cov_px @ cd.T
    return float(np.sqrt(cov_sky[0, 0])), float(np.sqrt(cov_sky[1, 1]))
