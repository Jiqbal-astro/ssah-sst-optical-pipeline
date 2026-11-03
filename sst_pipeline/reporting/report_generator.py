"""
report_generator.py
-------------------
QC artefacts: qc_matrix.csv, qc_summary.json, rejected_frames.log,
diagnostic plots (thresholds drawn from the config) and a multi-sheet
Excel workbook.
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

from sst_pipeline.utils.logging_config import get_logger  # noqa: E402

log = get_logger(__name__)
NAVY, TEAL, RED, GOLD, GREEN = "#1B3A6B", "#1E6B6B", "#A53B3B", "#B8860B", "#3BA56E"


def write_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
    return path


def write_rejected_log(df: pd.DataFrame, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'frame_id':<32} | photometry | astrometry\n" + "-" * 120 + "\n")
        for _, r in df[~(df["photometry_pass"] & df["astrometry_pass"])].iterrows():
            f.write(f"{r['frame_id']:<32} | {r['reject_reasons_photometry'] or 'PASS'} | "
                    f"{r['reject_reasons_astrometry'] or 'PASS'}\n")
    return path


def _save(fig, out_dir, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def make_plots(df: pd.DataFrame, qc: dict, out_dir: str) -> list[str]:
    paths = []
    x = np.arange(len(df))
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        r = [100 * df["photometry_pass"].mean(), 100 * df["astrometry_pass"].mean()]
        ax.bar(["Photometry", "Astrometry"], r, color=[NAVY, TEAL])
        for i, v in enumerate(r):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center")
        ax.set_ylim(0, 110); ax.set_ylabel("% frames passing"); ax.set_title("QC pass rate by track")
        paths.append(_save(fig, out_dir, "plot_01_pass_rates.png"))

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(x, df["snr"], "o-", color=TEAL, ms=3)
        ax.axhline(qc["snr"]["photometry_min"], color=RED, ls="--", label="photometry min")
        ax.axhline(qc["snr"]["astrometry_min"], color=GOLD, ls=":", label="astrometry min")
        ax.set_yscale("log"); ax.set_xlabel("frame"); ax.set_ylabel("max SNR"); ax.legend()
        paths.append(_save(fig, out_dir, "plot_02_snr.png"))

        fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        axs[0].plot(x, df["bkg_level"], "o-", color=NAVY, ms=3); axs[0].set_ylabel("sky level (ADU)")
        axs[1].bar(x, df["bkg_rms"], color=[GREEN if s else RED for s in df["bkg_stable"]])
        axs[1].set_ylabel("sky RMS (ADU)"); axs[1].set_xlabel("frame")
        paths.append(_save(fig, out_dir, "plot_03_background.png"))

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(x, df["fwhm_arcsec"], "o-", color=TEAL, ms=3, label="FWHM")
        ax.axhline(qc["seeing"]["photometry_max_fwhm_arcsec"], color=RED, ls="--", label="photometry max")
        ax.plot(x, df["astrometric_rms_arcsec"], "s", color=NAVY, ms=4, label="plate RMS")
        ax.axhline(qc["rms_thresholds"]["astrometric_rms_max_arcsec"], color=GOLD, ls=":", label="RMS max")
        ax.set_ylabel("arcsec"); ax.set_xlabel("frame"); ax.legend(fontsize=8)
        paths.append(_save(fig, out_dir, "plot_04_seeing_astrometry.png"))

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(x, df["n_reference_stars"], "o-", color=TEAL, ms=3)
        ax.set_ylabel("reference stars"); ax.set_xlabel("frame"); ax.set_title("Transparency proxy")
        paths.append(_save(fig, out_dir, "plot_05_star_counts.png"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Plotting failed: %s", exc)
    return paths


def write_excel(sheets: dict[str, pd.DataFrame], path: str) -> str:
    wb = Workbook()
    wb.remove(wb.active)
    hdr_fill = PatternFill("solid", fgColor="1B3A6B")
    ok_fill, bad_fill = PatternFill("solid", fgColor="D9F2E3"), PatternFill("solid", fgColor="F8D7D7")
    for name, df in sheets.items():
        ws = wb.create_sheet(name[:31])
        if df is None or df.empty:
            ws["A1"] = "No rows."
            continue
        ws.append(list(df.columns))
        for c in ws[1]:
            c.font, c.fill = Font(bold=True, color="FFFFFF"), hdr_fill
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for rec in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and not np.isfinite(v)) else
                       (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for v in rec])
        for j, col in enumerate(df.columns, 1):
            ws.column_dimensions[get_column_letter(j)].width = min(max(len(str(col)) + 2, 12), 45)
            if col.endswith("_pass") or col in ("identified", "astrometry_verified", "header_valid"):
                for i in range(2, ws.max_row + 1):
                    v = ws.cell(i, j).value
                    ws.cell(i, j).fill = ok_fill if v is True else bad_fill if v is False else PatternFill()
        ws.freeze_panes = "B2"
    wb.save(path)
    return path


def summary_frame(summary: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(summary), "value": [str(v) for v in summary.values()]})
