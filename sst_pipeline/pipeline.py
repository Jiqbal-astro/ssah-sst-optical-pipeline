"""
pipeline.py — SSAH SST Optical Pipeline v2.0.0
=============================================

Per frame
  01 load FITS, validate header, extract metadata, skip non-science frames
  02 calibrate (bias, dark, flat, L.A.Cosmic)
  03 SEP extraction (+ trail geometry) and morphology / role classification
  04 QC metrics + dual-track QC (pre-astrometry)
  05 initial WCS (header, or solve-field with --solve-wcs)
  06 reference stars -> WCS verification / refit -> astrometric QC
  07 photometric zero-point from the same stars -> photometric QC
  08 detections: sky position at mid-exposure epoch, photometry
Per session
  09 static-sky rejection and RSO candidate selection
  10 kinematic tracklet linking
  11 TLE / SGP4 correlation (known objects vs UCTs)
  12 outputs: QC matrix, metadata, detections, tracklets, UCTs, CCSDS TDM,
     ADES, Excel workbook, plots, log

CLI:
  sst-pipeline FITS_DIR OUT_DIR [--config site.yaml] [--solve-wcs] [--catalog]
               [--pattern "*.fits"] [--label TEXT]
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sst_pipeline import __version__
from sst_pipeline.config import load_config


@dataclass
class PipelineResults:
    summary: dict = field(default_factory=dict)
    qc_dataframe: Any = None
    metadata: list = field(default_factory=list)
    detections: list = field(default_factory=list)
    tracklets: list = field(default_factory=list)
    artefacts: dict = field(default_factory=dict)
    skipped_frames: list = field(default_factory=list)


def run_pipeline(fits_dir: str, output_dir: str, config_path: str | None = None,
                 solve_wcs: bool = False, use_online_catalog: bool = False,
                 file_pattern: str = "*.fit*", session_label: str = "") -> PipelineResults:
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)

    from sst_pipeline.utils.logging_config import get_logger, setup_logging
    setup_logging(output_dir)
    log = get_logger("sst_pipeline")
    log.info("SSAH SST Optical Pipeline v%s | %s -> %s | mode=%s solve_wcs=%s online_catalog=%s",
             __version__, fits_dir, output_dir, cfg["observation"]["tracking_mode"],
             solve_wcs, use_online_catalog)

    from sst_pipeline.astrometry.position_extractor import make_detection
    from sst_pipeline.astrometry.reference_catalog import get_reference_catalog
    from sst_pipeline.astrometry.wcs_solver import WCSSolution, header_wcs, refine_wcs, run_solve_field
    from sst_pipeline.calibration.calibration_manager import (calibrate_frame, load_calibration_frames,
                                                              save_calibrated_frame)
    from sst_pipeline.detection.source_classifier import Morph, classify, reference_morph, target_morph
    from sst_pipeline.detection.source_extractor import extract_sources
    from sst_pipeline.io.fits_loader import load_batch
    from sst_pipeline.io.header_validator import parse_obs_time, validate_header
    from sst_pipeline.io.metadata_extractor import build_metadata_table, extract_metadata
    from sst_pipeline.photometry.aperture_photometry import (point_photometry, solve_zeropoint,
                                                             trail_photometry)
    from sst_pipeline.qc.metrics import FrameMetrics, compute_frame_metrics
    from sst_pipeline.qc.qc_matrix import (apply_astrometric_qc, apply_photometric_qc, build_qc_matrix,
                                           evaluate_frame, summarize)
    from sst_pipeline.utils.coordinate_utils import compute_airmass
    from sst_pipeline.utils.timing_calibration import build_timing_model

    ins, site, qc = cfg["instrument"], cfg["site"], cfg["qc"]
    acfg, pcfg, ccfg = cfg["astrometry"], cfg["photometry"], cfg["source_classification"]
    gain, sat, scale0 = float(ins["gain_e_per_adu"]), float(ins["saturation_adu"]), float(ins["pixel_scale_arcsec"])
    mode = cfg["observation"]["tracking_mode"]
    ref_m, tgt_m = reference_morph(mode), target_morph(mode)
    sci_types = {s.upper() for s in cfg["observation"]["science_image_types"]}
    ra_unit = cfg["observation"]["ra_numeric_unit"]
    timing = build_timing_model(cfg["timing_calibration"], ins)
    cal = load_calibration_frames(cfg["calibration"])
    allow_unverified = not qc.get("require_verified_wcs", True)

    frames = load_batch(fits_dir, pattern=file_pattern)
    if not frames:
        raise FileNotFoundError(f"No FITS files in '{fits_dir}' matching '{file_pattern}'")
    log.info("%d FITS files found", len(frames))

    metadata, qc_rows, history, dets, skipped = [], [], [], [], []
    ref_sources: set[str] = set()
    frame_index = -1

    for frame in frames:
        fid = frame.frame_id
        hv = validate_header(fid, frame.header, cfg["header_requirements"], ra_unit)
        meta = extract_metadata(fid, frame.path, frame.header, hv, ra_unit)
        if meta.image_type.strip().upper() not in sci_types:
            log.info("[%s] skipped: IMAGETYP=%s is not a science frame", fid, meta.image_type)
            skipped.append({"frame_id": fid, "reason": f"IMAGETYP={meta.image_type}"})
            continue
        metadata.append(meta)
        frame_index += 1
        if frame.load_error or frame.data.ndim != 2:
            m = FrameMetrics(frame_id=fid, error=frame.load_error or f"data ndim={frame.data.ndim}")
            row = evaluate_frame(m, hv, history, qc)
            qc_rows.append(row)
            log.warning("[%s] load error: %s", fid, m.error)
            continue
        try:
            raw = frame.data
            exptime = meta.exptime_s if np.isfinite(meta.exptime_s) else 0.0
            data, cr_mask, steps = calibrate_frame(raw, exptime, cal, cfg["calibration"], gain,
                                                   float(ins["read_noise_e"]), sat)
            if cfg["output"].get("save_calibrated_frames"):
                save_calibrated_frame(data, frame.header, os.path.join(output_dir, f"cal_{fid}.fits"), steps)
            ext = extract_sources(fid, data, raw, cfg["source_extraction"], gain, sat,
                                  ccfg.get("elongation_threshold", 2.5))
            morphs = [classify(s, ccfg) for s in ext.sources]
            metrics = compute_frame_metrics(fid, raw, data, cr_mask, ext, morphs, ref_m, sat, scale0,
                                            qc["artifacts"])
            if ext.error:
                metrics.error = ext.error
            row = evaluate_frame(metrics, hv, history, qc)
            history.append(metrics)
            qc_rows.append(row)
            log.info("[%s] %d src (%d ref, %d trailed) SNRmax=%.0f FWHM=%.2f\" | QC phot=%s astro=%s",
                     fid, metrics.n_sources, metrics.n_reference, sum(m == Morph.TRAILED for m in morphs),
                     metrics.max_snr, metrics.fwhm_arcsec, row.photometry_pass, row.astrometry_pass)

            date_obs = parse_obs_time(frame.header)
            if not row.astrometry_pass or date_obs is None or exptime <= 0:
                continue

            # ---- WCS ---------------------------------------------------------
            w0, wsrc = (header_wcs(frame.header, data.shape), "header") if acfg.get("use_header_wcs", True) else (None, "")
            if w0 is None and solve_wcs:
                w0, wsrc = run_solve_field(fid, data, frame.header, os.path.join(output_dir, "solve_field"),
                                           scale0, meta.ra_deg if np.isfinite(meta.ra_deg) else None,
                                           meta.dec_deg if np.isfinite(meta.dec_deg) else None, acfg), "solve-field"
            if w0 is None:
                apply_astrometric_qc(row, "none", False, 0, float("nan"), qc)
                row.fail("astrometry", "no WCS (no header WCS; run with --solve-wcs)")
                continue
            ny, nx = data.shape
            ctr = w0.pixel_to_world(nx / 2, ny / 2)
            radius_deg = 0.5 * np.hypot(nx, ny) * scale0 / 3600 + acfg["initial_offset_search_arcsec"] / 3600
            refcat = get_reference_catalog(ctr.ra.deg, ctr.dec.deg, radius_deg, acfg, use_online_catalog)
            ref_idx = np.array([i for i, (s, mo) in enumerate(zip(ext.sources, morphs))
                                if mo == ref_m and not s.saturated], int)
            if refcat is not None and len(refcat) and len(ref_idx):
                ref_sources.add(refcat.source)
                cra, cdec = refcat.at_epoch(date_obs.jyear)
                S = ext.sources
                sol = refine_wcs(w0, wsrc, np.array([S[i].x for i in ref_idx]),
                                 np.array([S[i].y for i in ref_idx]),
                                 np.array([S[i].flux for i in ref_idx]), cra, cdec, refcat.mag,
                                 data.shape, acfg)
            else:
                sol = WCSSolution(True, w0, wsrc, message="no reference catalogue available")
            apply_astrometric_qc(row, sol.source, sol.verified, sol.n_matched, sol.rms_arcsec, qc)
            log.info("[%s] WCS %s verified=%s: %s", fid, sol.source, sol.verified, sol.message)
            if not row.astrometry_pass:
                continue
            wcs = sol.wcs
            star_global = set(ref_idx[sol.matched_det_idx].tolist()) if len(sol.matched_det_idx) else set()

            # ---- photometric zero-point -------------------------------------
            fwhm = metrics.fwhm_px if np.isfinite(metrics.fwhm_px) else \
                float(np.median([s.fwhm for s in ext.sources])) if ext.sources else 3.0
            airmass = meta.airmass_header if np.isfinite(meta.airmass_header) else \
                compute_airmass(ctr.ra.deg, ctr.dec.deg, date_obs, site["latitude_deg"],
                                site["longitude_deg"], site["height_m"])
            zp, zp_err, band, zp_src = float("nan"), float("nan"), meta.filter_name, ""
            if sol.verified and len(sol.matched_det_idx):
                gi = ref_idx[sol.matched_det_idx]
                mags = refcat.mag[sol.matched_cat_idx]
                sel = [k for k, i in enumerate(gi) if ext.sources[i].snr >= pcfg["zp_star_min_snr"]]
                if ref_m == Morph.POINT:
                    ph = point_photometry(data, [ext.sources[gi[k]].x for k in sel],
                                          [ext.sources[gi[k]].y for k in sel], fwhm, pcfg, gain, exptime)
                else:
                    ph = [trail_photometry(data, ext.sources[gi[k]].x, ext.sources[gi[k]].y,
                                           ext.sources[gi[k]].length, ext.sources[gi[k]].theta, fwhm,
                                           pcfg, gain, exptime) for k in sel]
                zp, zp_err, zp_rms, nzp = solve_zeropoint(mags[sel], np.array([p.m_inst for p in ph]),
                                                          np.array([p.m_inst_err for p in ph]),
                                                          pcfg["min_zp_stars"])
                apply_photometric_qc(row, zp, zp_rms, nzp, qc, pcfg["min_zp_stars"])
                band, zp_src = refcat.band, f"frame ({nzp} {refcat.band} stars)"
            else:
                apply_photometric_qc(row, float("nan"), float("nan"), 0, qc, pcfg["min_zp_stars"])
            if not np.isfinite(zp) and np.isfinite(airmass):
                zp = pcfg["fallback_zeropoint_mag"] - pcfg["extinction_coeff"] * airmass
                zp_src = "config fallback (uncalibrated)"

            # ---- detections ---------------------------------------------------
            pt_idx = []
            for i, (s, mo) in enumerate(zip(ext.sources, morphs)):
                if mo in (Morph.ARTIFACT, Morph.EXTENDED):
                    continue
                d = make_detection(fid, frame_index, s, mo, wcs, date_obs, exptime, timing, ny,
                                   sol.rms_ra_arcsec, sol.rms_dec_arcsec, airmass, sol.verified)
                d.is_catalog_star = i in star_global
                dets.append(d)
                if d.is_catalog_star:
                    continue
                if mo == Morph.TRAILED:
                    p = trail_photometry(data, s.x, s.y, s.length, s.theta, fwhm, pcfg, gain, exptime)
                    note = ""
                    if np.isfinite(s.profile_flux) and s.profile_flux > 0 and \
                            p.flux - s.profile_flux > max(3 * p.flux_err, 0.1 * s.profile_flux):
                        # stars inside the rectangular aperture: use the robust plateau flux
                        p.flux, p.m_inst = s.profile_flux, -2.5 * np.log10(s.profile_flux / exptime)
                        note = "; trail profile flux (aperture star-contaminated)"
                    _set_mag(d, p, zp, zp_err, band, zp_src + note, row.photometry_pass)
                else:
                    pt_idx.append((d, s))
            if pt_idx:
                ph = point_photometry(data, [s.x for _, s in pt_idx], [s.y for _, s in pt_idx],
                                      fwhm, pcfg, gain, exptime)
                for (d, _), p in zip(pt_idx, ph):
                    _set_mag(d, p, zp, zp_err, band, zp_src, row.photometry_pass)
        except Exception as exc:  # noqa: BLE001 — one bad frame never stops a batch
            log.exception("[%s] processing error", fid)
            if qc_rows and qc_rows[-1].frame_id == fid:
                qc_rows[-1].fail("astrometry", f"processing error: {exc}")
                qc_rows[-1].fail("photometry", f"processing error: {exc}")

    # ---- session stages --------------------------------------------------------
    from sst_pipeline.tracking.known_object_matcher import identify
    from sst_pipeline.tracking.linker import flag_static, link_tracklets
    from sst_pipeline.tracking.orbit_propagator import Propagator
    from sst_pipeline.tracking.tle_manager import load_catalog

    tcfg = cfg["tracklet"]
    static_ok = flag_static([d for d in dets if not d.is_catalog_star],
                            tcfg["static_match_radius_arcsec"], tcfg["static_min_frames"])
    if not static_ok:
        log.warning("Fewer than %d astrometric frames: static-sky test unavailable; "
                    "only %s-morphology sources are RSO candidates", tcfg["static_min_frames"], tgt_m.value)
    cands = []
    for d in dets:
        if d.is_catalog_star:
            d.status = "STAR"
        elif d.is_static:
            d.status = "STATIC"
        elif d.morphology == tgt_m.value or (d.morphology == ref_m.value and static_ok):
            d.is_candidate = True
            cands.append(d)
    tracklets = link_tracklets(cands, tcfg)

    tle_cfg = cfg["tle_tracking"]
    prop = Propagator(load_catalog(tle_cfg), site["latitude_deg"], site["longitude_deg"], site["height_m"])
    reported = [d for t in tracklets for d in t.members]
    identify(reported, tracklets, prop, tle_cfg)

    # ---- outputs -------------------------------------------------------------
    from sst_pipeline.output.writers import (write_ades_psv, write_detections_csv, write_tdm,
                                             write_tracklets_csv)
    from sst_pipeline.reporting.report_generator import (make_plots, summary_frame, write_excel,
                                                         write_json, write_rejected_log)
    import pandas as pd

    qc_df = build_qc_matrix(qc_rows)
    meta_df = build_metadata_table(metadata)
    cand_rows = [d for d in dets if not d.is_catalog_star]
    out = cfg["output"]
    P = lambda name: os.path.join(output_dir, name)  # noqa: E731
    arts = {
        "metadata_csv": P("session_metadata.csv"), "qc_matrix_csv": P("qc_matrix.csv"),
        "rejected_frames_log": write_rejected_log(qc_df, P("rejected_frames.log")) if len(qc_df) else None,
        "detections_csv": write_detections_csv(cand_rows, P("detections.csv")),
        "tracklets_csv": write_tracklets_csv(tracklets, P("tracklets.csv")),
        "uct_csv": write_tracklets_csv(tracklets, P("uncorrelated_tracks.csv"), uct_only=True),
    }
    meta_df.to_csv(arts["metadata_csv"], index=False)
    qc_df.to_csv(arts["qc_matrix_csv"], index=False)
    ast_cat = "Gaia3" if any(s.startswith("Gaia") for s in ref_sources) else "UNK"
    n_tdm = write_tdm(tracklets, P("observations_tdm.kvn"), site, allow_unverified)
    n_ades = write_ades_psv(tracklets, P("observations_ades.psv"), site, out, ins, ast_cat, allow_unverified)
    arts["tdm"], arts["ades_psv"] = P("observations_tdm.kvn"), P("observations_ades.psv")
    if skipped:
        pd.DataFrame(skipped).to_csv(P("skipped_frames.csv"), index=False)
        arts["skipped_csv"] = P("skipped_frames.csv")

    summary = summarize(qc_df)
    summary.update({
        "pipeline_version": __version__, "session_label": session_label,
        "tracking_mode": mode, "frames_found": len(frames), "frames_skipped_non_science": len(skipped),
        "reference_catalogue": sorted(ref_sources),
        "rso_candidates": len(cands),
        "tracklets_multi_frame": sum(t.kind == "MULTI_FRAME" for t in tracklets),
        "tracklets_single_streak": sum(t.kind == "SINGLE_STREAK" for t in tracklets),
        "tracklets_identified": sum(t.identified for t in tracklets),
        "uncorrelated_tracks": sum(not t.identified for t in tracklets),
        "unlinked_candidates": sum(d.status == "UNLINKED" for d in cands),
        "tle_objects_loaded": len(prop.objects),
        "observations_tdm": n_tdm, "observations_ades": n_ades,
        "elapsed_s": round(time.time() - t_start, 1),
    })
    arts["qc_summary_json"] = write_json(summary, P("qc_summary.json"))
    if out.get("generate_plots", True) and len(qc_df):
        arts["plots"] = make_plots(qc_df, qc, output_dir)
    try:
        arts["excel"] = write_excel({
            "Summary": summary_frame(summary), "QC Matrix": qc_df, "Metadata": meta_df,
            "Tracklets": pd.DataFrame([t.to_dict() for t in tracklets]),
            "Detections": pd.DataFrame([d.to_dict() for d in cand_rows]),
        }, P("SST_QC_Report.xlsx"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Excel report failed: %s", exc)

    log.info("Done in %.1fs: %s", summary["elapsed_s"],
             {k: summary[k] for k in ("photometry_pass", "astrometry_pass", "tracklets_multi_frame",
                                      "tracklets_single_streak", "uncorrelated_tracks")})
    return PipelineResults(summary=summary, qc_dataframe=qc_df, metadata=metadata, detections=dets,
                           tracklets=tracklets, artefacts=arts, skipped_frames=skipped)


def _set_mag(d, p, zp, zp_err, band, zp_src, phot_pass):
    d.flux_adu = p.flux
    if np.isfinite(p.m_inst) and np.isfinite(zp) and not d.saturated:
        d.mag = float(p.m_inst + zp)
        d.mag_err = float(np.hypot(p.m_inst_err, zp_err if np.isfinite(zp_err) else 0.0))
        d.mag_band, d.zp_source = band, zp_src
    d.photometry_ok = bool(phot_pass and not d.saturated and d.mag is not None
                           and not zp_src.startswith("config"))


def main():
    import argparse
    p = argparse.ArgumentParser(prog="sst-pipeline",
                                description=f"SSAH SST Optical Pipeline v{__version__}")
    p.add_argument("fits_dir")
    p.add_argument("output_dir")
    p.add_argument("--config", default=None, help="YAML merged over the packaged defaults (see config/site_template.yaml)")
    p.add_argument("--solve-wcs", action="store_true", help="run local astrometry.net when no header WCS")
    p.add_argument("--catalog", action="store_true",
                   help="download reference-store cells missing for these fields (VizieR, then ESA); saved for reuse")
    p.add_argument("--pattern", default="*.fit*")
    p.add_argument("--label", default="", help="free-text session label for the summary")
    a = p.parse_args()
    r = run_pipeline(a.fits_dir, a.output_dir, a.config, a.solve_wcs, a.catalog, a.pattern, a.label)
    print("\n".join(f"  {k:32s}: {v}" for k, v in r.summary.items()))


if __name__ == "__main__":
    main()
