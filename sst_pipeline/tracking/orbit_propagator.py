"""
orbit_propagator.py
-------------------
SGP4 (Vallado et al. 2006) prediction of where catalogued objects appear
on a star-referenced image.

Chain:
  1. SGP4 -> TEME position/velocity (whole catalogue at once, SatrecArray)
  2. light-time: r(t - rho/c) ~= r - v*rho/c        (LEO: ~25 m shift)
  3. TEME -> GCRS (astropy, IAU 2006/2000A), minus site GCRS position
     -> topocentric line of sight in GCRS axes
  4. The plate solution maps the APPARENT star field onto catalogue (ICRS,
     aberration-free) positions, i.e. it removes the annual aberration that
     stars show (~20"). An Earth-orbiting object shows no annual aberration,
     so the measured satellite position is shifted by the same -annual
     aberration. The prediction must therefore be the line-of-sight unit
     vector transformed GCRS -> ICRS as a DIRECTION (no distance): astropy
     then removes annual aberration exactly as the plate solution does.
     (Diurnal aberration affects stars and satellite alike and cancels.)

Transforming the topocentric vector WITH its distance would make astropy
shift the origin to the solar-system barycentre (positions wrong by tens of
degrees); only the direction is transformed. A simple GAST rotation of TEME
gives an Earth-fixed frame, not GCRS, and is not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.constants import c as C_LIGHT
from astropy.coordinates import (GCRS, ITRS, TEME, CartesianRepresentation, EarthLocation,
                                 SkyCoord, UnitSphericalRepresentation)
from astropy.time import Time, TimeDelta
from sgp4.api import SatrecArray

from sst_pipeline.tracking.tle_manager import TLEObject
from sst_pipeline.utils.coordinate_utils import position_angle_deg, sep_arcsec

C_KM_S = C_LIGHT.to(u.km / u.s).value


@dataclass
class Predictions:
    norad_id: np.ndarray
    name: np.ndarray
    ra: np.ndarray            # star-referenced (astrometric-equivalent) ICRS, deg
    dec: np.ndarray
    alt: np.ndarray           # deg
    az: np.ndarray            # deg, N=0 E=90
    range_km: np.ndarray
    rate_arcsec_s: np.ndarray
    pa_deg: np.ndarray        # direction of motion, E of N
    tle_age_days: np.ndarray
    valid: np.ndarray


class Propagator:
    def __init__(self, objects: list[TLEObject], lat_deg: float, lon_deg: float, height_m: float):
        self.objects = objects
        self.arr = SatrecArray([o.satrec for o in objects]) if objects else None
        self.site = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=height_m * u.m)
        self.lat, self.lon = np.radians(lat_deg), np.radians(lon_deg)
        self.ids = np.array([o.norad_id for o in objects])
        self.names = np.array([o.name for o in objects], dtype=object)
        self.epoch_jd = np.array([o.satrec.jdsatepoch + o.satrec.jdsatepochF for o in objects])

    def _sgp4(self, t: Time):
        e, r, v = self.arr.sgp4(np.array([t.jd1]), np.array([t.jd2]))
        return e[:, 0], r[:, 0, :], v[:, 0, :]

    def _los_gcrs(self, t: Time, r_teme: np.ndarray) -> np.ndarray:
        """Topocentric line-of-sight vectors (km) in GCRS axes, shape (N,3)."""
        sat = TEME(CartesianRepresentation(r_teme.T * u.km), obstime=t).transform_to(GCRS(obstime=t))
        s = sat.cartesian.xyz.to_value(u.km).T
        site = self.site.get_gcrs(t).cartesian.xyz.to_value(u.km)
        return s - site

    def predict(self, t: Time, light_time: bool = True, star_referenced: bool = True) -> Predictions:
        """light_time / star_referenced can be switched off only to compare with
        geometric ephemerides (tests); operational matching uses both."""
        n = len(self.objects)
        if n == 0:
            z = np.array([])
            return Predictions(z, z, z, z, z, z, z, z, z, z, z.astype(bool))
        err, r, v = self._sgp4(t)
        los = self._los_gcrs(t, r)
        rho = np.linalg.norm(los, axis=1)
        r_lt = r - v * (rho / C_KM_S)[:, None] if light_time else r   # light-time correction
        los = self._los_gcrs(t, r_lt)
        rho = np.linalg.norm(los, axis=1)

        if star_referenced:
            uvec = CartesianRepresentation((los / rho[:, None]).T * u.one)
            icrs = SkyCoord(uvec.represent_as(UnitSphericalRepresentation), frame=GCRS(obstime=t)).icrs
            ra, dec = icrs.ra.deg, icrs.dec.deg
        else:   # geometric direction in GCRS (= ICRS-aligned) axes
            ra = np.degrees(np.arctan2(los[:, 1], los[:, 0])) % 360
            dec = np.degrees(np.arcsin(los[:, 2] / rho))

        # Alt/Az from Earth-fixed topocentric vector (ENU)
        itrs = TEME(CartesianRepresentation(r_lt.T * u.km), obstime=t).transform_to(ITRS(obstime=t))
        d = itrs.cartesian.xyz.to_value(u.km).T - self.site.get_itrs(t).cartesian.xyz.to_value(u.km)
        sl, cl, so, co = np.sin(self.lat), np.cos(self.lat), np.sin(self.lon), np.cos(self.lon)
        e_ = -so * d[:, 0] + co * d[:, 1]
        n_ = -sl * co * d[:, 0] - sl * so * d[:, 1] + cl * d[:, 2]
        up = cl * co * d[:, 0] + cl * so * d[:, 1] + sl * d[:, 2]
        alt = np.degrees(np.arctan2(up, np.hypot(e_, n_)))
        az = np.degrees(np.arctan2(e_, n_)) % 360.0

        # apparent angular rate and direction by central difference (+-0.5 s)
        dirs = []
        for dt in (-0.5, 0.5):
            tt = t + TimeDelta(dt * u.s)
            _, r2, _ = self._sgp4(tt)
            l2 = self._los_gcrs(tt, r2)
            dirs.append((np.degrees(np.arctan2(l2[:, 1], l2[:, 0])) % 360,
                         np.degrees(np.arcsin(l2[:, 2] / np.linalg.norm(l2, axis=1)))))
        rate = sep_arcsec(dirs[0][0], dirs[0][1], dirs[1][0], dirs[1][1])
        pa = position_angle_deg(dirs[0][0], dirs[0][1], dirs[1][0], dirs[1][1])

        return Predictions(self.ids, self.names, ra, dec, alt, az, rho, np.atleast_1d(rate),
                           np.atleast_1d(pa), (t.jd - self.epoch_jd), err == 0)
