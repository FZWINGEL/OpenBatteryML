from __future__ import annotations

"""
Parser for the CALCE battery dataset that writes the unified BDUS Parquet tables
(cells.parquet, cycles.parquet, eis.parquet).

The raw CALCE dump used in the BatteryLife repo ships each cell as a .zip that
contains one or more .txt / .xls / .xlsx files. These files hold time-series of
voltage, current, and time (no temperature) for charge + discharge segments and
an integer cycle index. This parser:

1. Inflates each .zip (if needed) or directly reads loose files.
2. Concatenates all channels/sheets per cell into a single DataFrame.
3. Sorts and re-numbers cycles to ensure strictly increasing logical numbers.
4. Computes discharge capacity per cycle via current integration to derive SOH.
5. Cleans outlier cycles using a median filter on capacity (same idea as the
   original BatteryLife implementation).
6. Resamples the CHARGE voltage curve to a fixed length (1024) for
   `voltage_resampled`.
7. Engineers CC charge time and a (missing) temperature feature (filled with 0).
8. Calculates RUL after all cycles for the cell.
9. Validates and writes the BDUS tables with Pandera schemas.

Assumptions / CALCE specifics:
- Rated capacity: 1.10 Ah for "CS" cells, 1.35 Ah for "CX" cells (mirrors the
  BatteryLife code). Override by editing `_infer_rated_capacity`.
- Chemistry: LCO/Graphite (as used in prior works). Adjust if needed.
- End-of-life SOH threshold: 0.7 (common choice). Change `_EOL_SOH` if desired.
- No EIS data in CALCE -> `has_eis_data` is always False and eis.parquet is
  produced empty but valid.

Usage (from project root):

    python -m src.parsers.calce_parser --raw data/raw/calce --out data/processed

or via `src/main.py` with the "calce" option.
"""

import logging
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import medfilt

from utils.processing_helpers import resample_voltage_current
from utils.schema import cells_schema, cycles_schema, eis_schema

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


_EOL_SOH = 0.7  # configurable end-of-life threshold for SOH
_NUM_RESAMPLE_POINTS = 1024
_MEDFILT_KERNEL = 21  # must be odd


class CalceParser:
    def __init__(self, raw_data_dir: str | os.PathLike[str], processed_data_dir: str | os.PathLike[str]):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)

        self.cells_df = pd.DataFrame(columns=cells_schema.columns.keys())
        self.cycles_df = pd.DataFrame(columns=cycles_schema.columns.keys())
        self.eis_df = pd.DataFrame(columns=eis_schema.columns.keys())

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def execute(self) -> None:
        self._process_all_cells()
        self._finalise_and_save()

    # ---------------------------------------------------------------------
    # Core processing
    # ---------------------------------------------------------------------
    def _process_all_cells(self) -> None:
        # Look for .zip archives *or* already extracted folders/files
        zips = sorted(self.raw_data_dir.glob("*.zip"))
        folders = [p for p in self.raw_data_dir.iterdir() if p.is_dir()]
        loose_files = [p for p in self.raw_data_dir.glob("*.xlsx")] + [p for p in self.raw_data_dir.glob("*.xls")] + [p for p in self.raw_data_dir.glob("*.txt")]

        if not zips and not folders and not loose_files:
            LOGGER.warning("No CALCE raw files found in %s", self.raw_data_dir)
            return

        # Process zips individually
        for z in zips:
            cell_name = z.stem
            with TemporaryExtraction(z) as extract_dir:
                self._process_cell_dir(extract_dir, cell_name)

        # Process standalone folders (assumed one cell per folder name)
        for f in folders:
            self._process_cell_dir(f, f.name)

        # Process loose files as one pseudo-cell
        if loose_files:
            tmp_dir = self.raw_data_dir / "__loose_files__"
            tmp_dir.mkdir(exist_ok=True)
            for lf in loose_files:
                shutil.copy2(lf, tmp_dir / lf.name)
            self._process_cell_dir(tmp_dir, "calce_loose")
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _process_cell_dir(self, directory: Path, raw_cell_id: str) -> None:
        try:
            df = self._load_all_files_in_dir(directory)
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Failed to load data for %s – %s", raw_cell_id, exc)
            return

        if df.empty:
            LOGGER.warning("No usable data found for %s", raw_cell_id)
            return

        # Ensure cycles are monotonically increasing integers
        df["Cycle_Index"] = _organize_cycle_index(df["Cycle_Index"].to_numpy())

        # Group by (date, Cycle_Index) to form a logical cycle
        cycles_raw: List[Tuple[pd.DataFrame, int]] = []
        for (date, idx), grp in df.groupby(["date", "Cycle_Index"], sort=True):
            cycles_raw.append((grp.reset_index(drop=True), idx))

        # Build cleaned list & compute capacities
        capacities = []
        cycle_objs: List[dict] = []
        rated_capacity = self._infer_rated_capacity(raw_cell_id)
        full_cell_id = f"calce_{raw_cell_id}"
        self._add_cell_metadata(full_cell_id, rated_capacity)

        logical_num = 0
        for grp, _ in cycles_raw:
            I = grp["Current(A)"].to_numpy(dtype=np.float32)
            t = grp["Test_Time(s)"].to_numpy(dtype=np.float32)
            V = grp["Voltage(V)"].to_numpy(dtype=np.float32)

            qd = _integrate_capacity(I, t, is_charge=False)
            discharge_capacity = float(qd.max()) if qd.size else np.nan
            capacities.append(discharge_capacity)

        # Median filter outliers (same idea as original CALCE script)
        if len(capacities) >= _MEDFILT_KERNEL:
            med = medfilt(capacities, kernel_size=_MEDFILT_KERNEL)
        else:  # fallback: no filtering
            med = np.array(capacities)
        ths = float(np.median(np.abs(np.array(capacities) - med))) if capacities else 0.0
        keep_mask = np.abs(np.array(capacities) - med) < 3 * ths if ths > 0 else np.ones(len(capacities), dtype=bool)

        # Re-loop, create rows
        for i, ((grp, _), cap, keep) in enumerate(zip(cycles_raw, capacities, keep_mask)):
            if not keep or (np.isnan(cap)) or cap <= 0.1:
                continue
            logical_num += 1

            # Extract charge segment for resampling
            charge_mask = grp["Current(A)"].to_numpy(dtype=np.float32) > 0
            charge_df = grp[charge_mask].copy()
            if charge_df.empty:
                charge_df = grp.copy()
            # Construct simple DF for resampling helper
            rc_df = pd.DataFrame({
                "voltage": charge_df["Voltage(V)"].to_numpy(dtype=np.float32),
                "current": charge_df["Current(A)"].to_numpy(dtype=np.float32),
            })
            rc_df.index = np.arange(len(rc_df))  # ensure numeric index
            resampled = resample_voltage_current(rc_df, num_points=_NUM_RESAMPLE_POINTS)

            temp_avg_c = 0.0  # CALCE files usually lack temperature
            cc_time = _calculate_cc_time(charge_df)
            soh = cap / rated_capacity if rated_capacity > 0 else np.nan

            cycle_objs.append({
                "cell_id": full_cell_id,
                "cycle_number": logical_num,
                "soh": float(soh),
                "rul": np.nan,  # filled later
                "voltage_resampled": resampled["voltage"].to_numpy(dtype=np.float32),
                "temperature_avg_c": float(temp_avg_c),
                "cc_charge_time_s": int(cc_time),
                "has_eis_data": False,
            })

        if not cycle_objs:
            LOGGER.warning("All cycles for %s were filtered out or invalid", full_cell_id)
            return

        df_new = pd.DataFrame(cycle_objs)
        df_new = df_new.reindex(columns=self.cycles_df.columns, fill_value=np.nan)
        self.cycles_df = pd.concat([self.cycles_df, df_new], ignore_index=True)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _load_all_files_in_dir(self, d: Path) -> pd.DataFrame:
        files = [p for ext in ("*.txt", "*.xlsx", "*.xls") for p in d.glob(ext)]
        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            if f.suffix.lower() == ".txt":
                dfs.append(_load_txt(f))
            else:
                dfs.append(_load_excel(f))
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values(["date", "Test_Time(s)"]).reset_index(drop=True)
        return df

    def _infer_rated_capacity(self, raw_cell_id: str) -> float:
        up = raw_cell_id.upper()
        if "CS" in up:
            return 1.10
        return 1.35

    def _add_cell_metadata(self, cell_id: str, rated_capacity: float) -> None:
        if cell_id in self.cells_df["cell_id"].values:
            return
        info = {
            "cell_id": cell_id,
            "dataset_source": "calce",
            "chemistry": "LCO",
            "rated_capacity_ah": float(rated_capacity),
            "end_of_life_soh": float(_EOL_SOH),
        }
        self.cells_df = pd.concat([self.cells_df, pd.DataFrame([info])], ignore_index=True)

    def _finalise_and_save(self) -> None:
        if self.cycles_df.empty:
            LOGGER.warning("No CALCE cycles parsed – nothing to save.")
            return

        # Compute RUL per cell
        merged = self.cycles_df.merge(self.cells_df[["cell_id", "end_of_life_soh"]], on="cell_id", how="left")

        def _calc_rul(grp: pd.DataFrame) -> pd.Series:
            eol = grp["end_of_life_soh"].iloc[0]
            eol_cycle = grp.loc[grp["soh"] <= eol, "cycle_number"].min()
            if pd.isna(eol_cycle):
                rul_vals = grp["cycle_number"].max() - grp["cycle_number"]
            else:
                rul_vals = eol_cycle - grp["cycle_number"]
                rul_vals.loc[grp["cycle_number"] >= eol_cycle] = 0
            return rul_vals.astype(int)

        rul_all = (
            merged.groupby("cell_id", sort=False, group_keys=False).apply(_calc_rul)
        )
        merged.loc[rul_all.index, "rul"] = rul_all
        self.cycles_df = merged[self.cycles_df.columns]
        self.cycles_df.loc[:, "has_eis_data"] = self.cycles_df["has_eis_data"].astype(bool)

        # Ensure output dir
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Validating and writing cells.parquet …")
        cells_schema.validate(self.cells_df)
        self.cells_df.to_parquet(self.processed_data_dir / "cells.parquet", index=False)

        LOGGER.info("Validating and writing cycles.parquet …")
        _tmp = self.cycles_df.copy()
        _tmp.loc[:, "has_eis_data"] = _tmp["has_eis_data"].astype(bool)
        cycles_schema.validate(_tmp)
        self.cycles_df.to_parquet(self.processed_data_dir / "cycles.parquet", index=False)

        LOGGER.info("Validating and writing eis.parquet …")
        if self.eis_df.empty:
            self.eis_df = pd.DataFrame(columns=eis_schema.columns.keys())
        eis_schema.validate(self.eis_df)
        self.eis_df.to_parquet(self.processed_data_dir / "eis.parquet", index=False)

        LOGGER.info("CALCE processed data saved to %s", self.processed_data_dir)


# -------------------------------------------------------------------------
# Standalone helpers (module-level)
# -------------------------------------------------------------------------

def _integrate_capacity(I: np.ndarray, t: np.ndarray, is_charge: bool) -> np.ndarray:
    """Integrate current over time to get capacity in Ah. Follows sign rules.
    Vectorised equivalent of the numba version in BatteryLife."""
    Q = np.zeros_like(I, dtype=np.float32)
    dt = np.diff(t, prepend=t[0]) / 3600.0  # hours
    if is_charge:
        mask = I > 0
        Q[mask] = np.cumsum(I[mask] * dt[mask])
        Q[~mask] = np.maximum.accumulate(Q[mask])[-1] if mask.any() else 0
    else:
        mask = I < 0
        Q[mask] = np.cumsum(-I[mask] * dt[mask])
        Q[~mask] = np.maximum.accumulate(Q[mask])[-1] if mask.any() else 0
    return Q


def _organize_cycle_index(cycle_index: np.ndarray) -> np.ndarray:
    current_cycle = cycle_index[0]
    prev_value = cycle_index[0]
    for i in range(1, len(cycle_index)):
        if cycle_index[i] != prev_value:
            current_cycle += 1
            prev_value = cycle_index[i]
        cycle_index[i] = current_cycle
    return cycle_index


def _calculate_cc_time(charge_df: pd.DataFrame) -> int:
    if charge_df.empty:
        return 0
    I = charge_df["Current(A)"].to_numpy(dtype=np.float32)
    t = charge_df["Test_Time(s)"].to_numpy(dtype=np.float32)
    if I.size == 0:
        return int(t.max()) if t.size else 0
    max_I = I.max()
    if max_I == 0:
        return int(t.max())
    threshold = max_I * 0.99
    below = np.where(I < threshold)[0]
    if below.size:
        return int(t[below[0]])
    return int(t.max())


_DATE_PAT_1 = re.compile(r"C[XS]2?_\d+_(\d+)_(\d+)B?_(\d+)", re.IGNORECASE)
_DATE_PAT_2 = re.compile(r"(\d+)_(\d+)_(\d+)_CX2_32", re.IGNORECASE)


def _extract_date_from_filename(name: str) -> str:
    up = name.upper()
    m = _DATE_PAT_1.findall(up)
    if not m:
        m = _DATE_PAT_2.findall(up)
    if not m:
        return "1970-01-01"
    month, day, year = map(int, m[0])
    return f"{year:04d}-{month:02d}-{day:02d}"


def _load_excel(path: Path) -> pd.DataFrame:
    cache = path.with_name(path.stem + "_cache").with_suffix(".csv")
    if cache.exists():
        return pd.read_csv(cache)

    xls = pd.ExcelFile(path)
    frames = []
    for sheet in xls.sheet_names:
        if sheet.lower().startswith("channel") or sheet.lower().startswith("sheet"):
            frames.append(xls.parse(sheet))
        else:
            # Some files use odd sheet names; just parse all
            frames.append(xls.parse(sheet))
    df = pd.concat(frames, ignore_index=True)
    date = _extract_date_from_filename(path.stem)
    df["date"] = date
    cols = ["date", "Cycle_Index", "Test_Time(s)", "Current(A)", "Voltage(V)"]
    df = df[cols]
    df.to_csv(cache, index=False)
    return df


def _load_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    date = _extract_date_from_filename(path.stem)
    out = pd.DataFrame({
        "date": date,
        "Cycle_Index": df["Charge count"] // 2 + 1,
        "Test_Time(s)": df["Time"],
        "Current(A)": df["mA"] / 1000.0,
        "Voltage(V)": df["mV"] / 1000.0,
    })
    return out


class TemporaryExtraction:
    """Context manager to extract a zip file into a temp folder and cleanup."""

    def __init__(self, zip_path: Path):
        self.zip_path = zip_path
        self.tmp_dir = zip_path.parent / f"__tmp_extract_{zip_path.stem}__"

    def __enter__(self) -> Path:
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.tmp_dir)
        # Special case from original CALCE code: cx2_8 folder lowercase
        special = self.tmp_dir / "cx2_8"
        if special.exists():
            special.rename(self.tmp_dir / "CX2_8")
        return self.tmp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Parse CALCE dataset into BDUS Parquet tables.")
    p.add_argument("--raw", required=True, help="Directory containing CALCE raw .zip/.txt/.xls(x) files")
    p.add_argument("--out", required=True, help="Directory to write processed Parquet files")
    args = p.parse_args()

    CalceParser(args.raw, args.out).execute()
