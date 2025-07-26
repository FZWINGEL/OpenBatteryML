# src/parsers/nasa_pcoe_parser.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Any, List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat

from utils.processing_helpers import resample_voltage_current
from utils.schema import cells_schema, cycles_schema, eis_schema

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


class NasaPcoeParser:
    """Parser that converts NASA PCoE .mat files into BDUS-compliant Parquet tables."""

    # Fallback candidates found in different MAT versions
    _MAT_TEMPERATURE_KEYS: List[str] = ["Temperature_measured", "Temperature_measured_C"]
    _MAT_TIME_KEYS: List[str] = ["Relative_Time", "Time"]

    def __init__(self, raw_data_dir: str | os.PathLike[str], processed_data_dir: str | os.PathLike[str]):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)

        # Working dataframes (column order must match schemas)
        self.cells_df = pd.DataFrame(columns=cells_schema.columns.keys())
        self.cycles_df = pd.DataFrame(columns=cycles_schema.columns.keys())
        self.eis_df = pd.DataFrame(columns=eis_schema.columns.keys())

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def execute(self) -> None:
        """Run the complete parse → transform → save pipeline."""
        self._process_all_mat_files()
        self._finalise_and_save()

    # --------------------------------------------------------------------- #
    # Internal main steps
    # --------------------------------------------------------------------- #
    def _process_all_mat_files(self) -> None:
        for file_path in sorted(self.raw_data_dir.glob("*.mat")):
            LOGGER.info("Processing file %s", file_path.name)
            self._process_single_file(file_path)

    def _process_single_file(self, file_path: Path) -> None:
        mat_data = self._safe_load_mat(file_path)
        if mat_data is None:
            return

        cell_id_raw = file_path.stem
        mat_key = cell_id_raw if cell_id_raw in mat_data else next(iter(mat_data), None)
        if mat_key is None:
            LOGGER.error("MAT file %s empty – skipping", file_path.name)
            return

        cycles_raw = self._extract_cycles(mat_data, mat_key, file_path.name)
        if cycles_raw is None:
            return

        full_cell_id = f"nasa_pcoe_{cell_id_raw}"
        self._ensure_cell_metadata(full_cell_id)

        rated_capacity = self._get_rated_capacity(full_cell_id)
        if rated_capacity is None:
            # Should not happen, but keeps us defensive
            LOGGER.error("No rated_capacity found for %s – metadata insertion failed", full_cell_id)
            return

        cycles_rows: List[Dict[str, Any]] = []
        eis_rows: List[Dict[str, Any]] = []

        charge_cycle_cache: Optional[Dict[str, Any]] = None
        logical_cycle_number = 0

        for idx, cycle_struct in enumerate(cycles_raw):
            cycle_type = self._get_cycle_type(cycle_struct)
            if cycle_type is None:
                LOGGER.warning("Unexpected cycle struct type %s in %s – skipping", type(cycle_struct), full_cell_id)
                continue

            # Cache charge cycles, discharge follows
            if cycle_type == "charge":
                charge_cycle_cache = cycle_struct
                continue
            if cycle_type != "discharge":
                # ignore impedance here, handled when following discharge
                continue

            if charge_cycle_cache is None:
                LOGGER.warning("Discharge cycle %d in %s has no preceding charge cycle – skipping", idx, full_cell_id)
                continue

            logical_cycle_number += 1

            # ------------------ Charge part ------------------ #
            charge_df = self._mat_to_dataframe(charge_cycle_cache.get("data", {}))
            resampled_df = resample_voltage_current(charge_df)
            temp_avg_c = float(charge_df["temperature"].mean()) if not charge_df.empty else 0.0
            cc_charge_time_s = self._calculate_cc_charge_time(charge_df)

            # ----------------- Discharge part ---------------- #
            discharge_data = cycle_struct.get("data", {})
            capacity_value = self._extract_capacity(discharge_data)
            soh = capacity_value / rated_capacity if not np.isnan(capacity_value) else np.nan

            # ------------------ EIS part --------------------- #
            has_eis, eis_row = self._maybe_extract_eis(
                next_cycle=cycles_raw[idx + 1] if (idx + 1) < len(cycles_raw) else None,
                cell_id=full_cell_id,
                logical_cycle_number=logical_cycle_number,
            )
            if eis_row is not None:
                eis_rows.append(eis_row)

            # ----------------- Collect row ------------------- #
            cycles_rows.append(
                {
                    "cell_id": full_cell_id,
                    "cycle_number": logical_cycle_number,
                    "soh": soh,
                    "rul": np.nan,  # filled later
                    "voltage_resampled": resampled_df["voltage"].to_numpy(dtype=np.float32),
                    "temperature_avg_c": temp_avg_c,
                    "cc_charge_time_s": cc_charge_time_s,
                    "has_eis_data": bool(has_eis),
                }
            )
            charge_cycle_cache = None  # reset for next pair

        # Append to master frames (preserving column order)
        if cycles_rows:
            df_new_cycles = pd.DataFrame(cycles_rows).reindex(columns=self.cycles_df.columns, fill_value=np.nan)
            self.cycles_df = pd.concat([self.cycles_df, df_new_cycles], ignore_index=True)

        if eis_rows:
            df_new_eis = pd.DataFrame(eis_rows).reindex(columns=self.eis_df.columns, fill_value=np.nan)
            self.eis_df = pd.concat([self.eis_df, df_new_eis], ignore_index=True)

    def _finalise_and_save(self) -> None:
        if self.cycles_df.empty:
            LOGGER.warning("No cycle data parsed – nothing to save.")
            return

        # ----------------- Compute RUL ----------------- #
        merged = self.cycles_df.merge(
            self.cells_df[["cell_id", "end_of_life_soh"]], on="cell_id", how="left"
        )

        def _calc_rul(group: pd.DataFrame) -> pd.Series:
            eol_soh = group["end_of_life_soh"].iloc[0]
            eol_cycle = group.loc[group["soh"] <= eol_soh, "cycle_number"].min()
            if pd.isna(eol_cycle):
                # Never hit EOL -> RUL = max_cycle - current
                return (group["cycle_number"].max() - group["cycle_number"]).astype(int)
            rul_vals = (eol_cycle - group["cycle_number"]).astype(int)
            rul_vals.loc[group["cycle_number"] >= eol_cycle] = 0
            return rul_vals

        rul_series_parts = [ _calc_rul(g) for _, g in merged.groupby("cell_id", sort=False) ]
        rul_all = pd.concat(rul_series_parts).sort_index()
        merged.loc[rul_all.index, "rul"] = rul_all

        # Restore original column order
        self.cycles_df = merged[self.cycles_df.columns]

        # Ensure boolean dtype is correct (Pandera needs strict bool for validation step)
        self.cycles_df.loc[:, "has_eis_data"] = self.cycles_df["has_eis_data"].map(bool).astype("boolean")

        # ----------------- Save ----------------- #
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Validating and writing cells.parquet …")
        cells_schema.validate(self.cells_df)
        self.cells_df.to_parquet(self.processed_data_dir / "cells.parquet", index=False)

        LOGGER.info("Validating and writing cycles.parquet …")
        _cycles_for_validation = self.cycles_df.copy()
        _cycles_for_validation.loc[:, "has_eis_data"] = _cycles_for_validation["has_eis_data"].astype(bool)
        cycles_schema.validate(_cycles_for_validation)
        self.cycles_df.to_parquet(self.processed_data_dir / "cycles.parquet", index=False)

        LOGGER.info("Validating and writing eis.parquet …")
        if self.eis_df.empty:
            self.eis_df = pd.DataFrame(columns=eis_schema.columns.keys())
        eis_schema.validate(self.eis_df)
        self.eis_df.to_parquet(self.processed_data_dir / "eis.parquet", index=False)

        LOGGER.info("All processed data saved to %s", self.processed_data_dir)

    # --------------------------------------------------------------------- #
    # Helpers – MAT handling
    # --------------------------------------------------------------------- #
    @staticmethod
    def _safe_load_mat(path: Path) -> Optional[Dict[str, Any]]:
        try:
            return loadmat(path, simplify_cells=True)
        except Exception as exc:
            LOGGER.error("Could not load %s – %s", path.name, exc)
            return None

    @staticmethod
    def _extract_cycles(mat_data: Dict[str, Any], key: str, fname: str) -> Optional[List[Any]]:
        try:
            raw = mat_data[key]["cycle"]
        except (KeyError, TypeError):
            LOGGER.error("File %s does not contain expected 'cycle' structure – skipping", fname)
            return None

        if not isinstance(raw, (list, np.ndarray)):
            raw = [raw]

        return raw.tolist() if isinstance(raw, np.ndarray) else list(raw)

    @staticmethod
    def _get_cycle_type(cycle_struct: Any) -> Optional[str]:
        """Return lower-case cycle type or None if not identifiable as dict/str."""
        if isinstance(cycle_struct, dict):
            return str(cycle_struct.get("type", "")).lower()
        if isinstance(cycle_struct, (str, bytes)):
            # Handles edge-case in original code
            return str(cycle_struct).lower()
        return None

    # --------------------------------------------------------------------- #
    # Helpers – Data extraction / transformation
    # --------------------------------------------------------------------- #
    def _mat_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert a MAT 'data' dict to a normalized DataFrame with columns: voltage, current, temperature, time."""
        voltage = np.asarray(data.get("Voltage_measured", []), dtype=np.float32).flatten()
        current = np.asarray(data.get("Current_measured", []), dtype=np.float32).flatten()

        temperature = self._first_present(data, self._MAT_TEMPERATURE_KEYS)
        time_vec = self._first_present(data, self._MAT_TIME_KEYS)

        max_len = max(len(voltage), len(current), len(temperature), len(time_vec), 1)

        def pad(arr: np.ndarray) -> np.ndarray:
            if len(arr) == max_len:
                return arr
            if len(arr) == 0:
                return np.zeros(max_len, dtype=np.float32)
            return np.pad(arr, (0, max_len - len(arr)), mode="edge")

        return pd.DataFrame(
            {
                "voltage": pad(voltage),
                "current": pad(current),
                "temperature": pad(temperature),
                "time": pad(time_vec),
            }
        )

    @staticmethod
    def _first_present(data: Dict[str, Any], candidates: List[str]) -> np.ndarray:
        """Return flattened float32 array for the first present key. Fallback to zeros matching another array length."""
        for key in candidates:
            if key in data:
                return np.asarray(data[key], dtype=np.float32).flatten()

        # Fallback: match length of any numeric array we can find, else empty
        for v in data.values():
            try:
                arr = np.asarray(v, dtype=np.float32).flatten()
                return np.zeros_like(arr, dtype=np.float32)
            except Exception:
                continue
        return np.array([], dtype=np.float32)

    @staticmethod
    def _extract_capacity(discharge_data: Dict[str, Any]) -> float:
        capacity_raw = discharge_data.get("Capacity")
        if capacity_raw is None:
            return np.nan
        arr = np.asarray(capacity_raw, dtype=np.float32).flatten()
        return float(arr[-1]) if arr.size else np.nan

    def _maybe_extract_eis(
        self,
        next_cycle: Any,
        cell_id: str,
        logical_cycle_number: int,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Return (has_eis, eis_row)."""
        if not isinstance(next_cycle, dict) or str(next_cycle.get("type", "")).lower() != "impedance":
            return False, None

        eis_data_struct = next_cycle.get("data", {})
        impedance_vec = eis_data_struct.get("Rectified_impedance") or eis_data_struct.get("Battery_impedance")
        if impedance_vec is None:
            return True, None

        impedance_arr = np.asarray(impedance_vec, dtype=np.complex64).flatten()
        return True, {
            "cell_id": cell_id,
            "cycle_number": logical_cycle_number,
            "frequency_hz": np.array([], dtype=np.float32),  # NASA file often lacks freq
            "impedance_real_ohm": np.real(impedance_arr).astype(np.float32),
            "impedance_imag_ohm": np.imag(impedance_arr).astype(np.float32),
        }

    @staticmethod
    def _calculate_cc_charge_time(df: pd.DataFrame) -> int:
        """Approximate duration of constant-current phase (first 99% of max current)."""
        try:
            if df.empty:
                return 0
            max_current = df["current"].max()
            if max_current == 0:
                return int(df["time"].max())
            threshold = max_current * 0.99
            cc_end_time = df.loc[df["current"] < threshold, "time"].min()
            return int(cc_end_time) if pd.notna(cc_end_time) else int(df["time"].max())
        except Exception:
            return int(df.get("time", pd.Series([0])).max())

    # --------------------------------------------------------------------- #
    # Helpers – Metadata / config
    # --------------------------------------------------------------------- #
    def _ensure_cell_metadata(self, full_cell_id: str) -> None:
        """Insert static cell metadata once per cell."""
        if full_cell_id in self.cells_df["cell_id"].values:
            return
        cell_info = {
            "cell_id": full_cell_id,
            "dataset_source": "nasa_pcoe",
            "chemistry": "LCO",
            "rated_capacity_ah": 2.0,
            "end_of_life_soh": 0.8,
        }
        self.cells_df = pd.concat([self.cells_df, pd.DataFrame([cell_info])], ignore_index=True)

    def _get_rated_capacity(self, full_cell_id: str) -> Optional[float]:
        try:
            return float(
                self.cells_df.loc[self.cells_df["cell_id"] == full_cell_id, "rated_capacity_ah"].iloc[0]
            )
        except IndexError:
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse NASA PCoE dataset into BDUS Parquet tables.")
    parser.add_argument("--raw", required=True, help="Directory containing *.mat files")
    parser.add_argument("--out", required=True, help="Directory to write processed Parquet files")
    args = parser.parse_args()

    NasaPcoeParser(args.raw, args.out).execute()
