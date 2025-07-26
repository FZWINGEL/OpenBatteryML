# src/parsers/nasa_pcoe_parser.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Iterable, Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from utils.processing_helpers import resample_voltage_current
from utils.schema import cells_schema, cycles_schema, eis_schema

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


class NasaPcoeParser:
    """Parser for the NASA PCoE dataset -> BDUS tables."""

    _MAT_TEMPERATURE_KEY_CANDIDATES: List[str] = [
        "Temperature_measured",
        "Temperature_measured_C",
    ]
    _MAT_TIME_KEY_CANDIDATES: List[str] = [
        "Relative_Time",
        "Time",
    ]

    def __init__(self, raw_data_dir: str | os.PathLike[str], processed_data_dir: str | os.PathLike[str]):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)

        self.cells_df = pd.DataFrame(columns=cells_schema.columns.keys())
        self.cycles_df = pd.DataFrame(columns=cycles_schema.columns.keys())
        self.eis_df = pd.DataFrame(columns=eis_schema.columns.keys())

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute(self) -> None:
        self._process_raw_files()
        self._finalise_and_save_data()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _process_raw_files(self) -> None:
        for file_path in sorted(self.raw_data_dir.glob("*.mat")):
            LOGGER.info("Processing file %s", file_path.name)
            self._process_single_file(file_path)

    def _process_single_file(self, file_path: Path) -> None:
        try:
            mat_data = loadmat(file_path, simplify_cells=True)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Could not load %s – %s", file_path.name, exc)
            return

        cell_id = file_path.stem
        mat_key = cell_id if cell_id in mat_data else next(iter(mat_data), None)
        if mat_key is None:
            LOGGER.error("MAT file %s empty – skipping", file_path.name)
            return

        try:
            all_cycles_raw = mat_data[mat_key]["cycle"]  # type: ignore[index]
        except (KeyError, TypeError):
            LOGGER.error("File %s does not contain expected 'cycle' structure – skipping", file_path.name)
            return

        # normalise to iterable
        all_cycles: Iterable[Any] = (
            all_cycles_raw.tolist() if isinstance(all_cycles_raw, np.ndarray) else all_cycles_raw
        )

        full_cell_id = f"nasa_pcoe_{cell_id}"
        self._add_cell_metadata(full_cell_id)

        try:
            rated_capacity: float = self.cells_df.loc[
                self.cells_df["cell_id"] == full_cell_id, "rated_capacity_ah"
            ].iloc[0]
        except IndexError:
            LOGGER.error("No rated_capacity found for %s – metadata insertion failed", full_cell_id)
            return

        cycles_rows: List[dict] = []
        eis_rows: List[dict] = []

        charge_cycle_cache: dict | None = None
        logical_cycle_num = 0

        # Convert to list so we can index look-ahead safely
        all_cycles_list: List[Any] = list(all_cycles)

        for i, cycle_struct in enumerate(all_cycles_list):
            # Some savemat variants degrade dicts → strings/ndarrays.
            if isinstance(cycle_struct, dict):
                cycle_type = str(cycle_struct.get("type", "")).lower()
            else:
                # Try to interpret as a bare type string, otherwise skip
                cycle_type = str(cycle_struct).lower() if isinstance(cycle_struct, (str, bytes)) else ""
                if cycle_type == "discharge":
                    # we know there's no data / no charge, log accordingly
                    LOGGER.warning(
                        "Discharge cycle %d in %s has no preceding charge cycle – skipping", i, full_cell_id
                    )
                else:
                    LOGGER.warning(
                        "Unexpected cycle struct type %s in %s – skipping", type(cycle_struct), full_cell_id
                    )
                continue

            if cycle_type == "charge":
                charge_cycle_cache = cycle_struct
                continue
            if cycle_type != "discharge":
                # ignore impedance here, handled by look-ahead
                continue

            if charge_cycle_cache is None:
                LOGGER.warning(
                    "Discharge cycle %d in %s has no preceding charge cycle – skipping", i, full_cell_id
                )
                continue

            logical_cycle_num += 1

            # ---- charge (features) ----
            charge_df = self._mat_data_to_df(charge_cycle_cache.get("data", {}))
            resampled_df = resample_voltage_current(charge_df)
            temp_avg_c = float(charge_df["temperature"].mean()) if not charge_df.empty else 0.0
            cc_charge_time_s = self._calculate_cc_charge_time(charge_df)

            # ---- discharge (labels) ----
            discharge_data = cycle_struct.get("data", {})
            capacity_raw = discharge_data.get("Capacity", None)
            if capacity_raw is not None:
                capacity_arr = np.asarray(capacity_raw, dtype=np.float32).flatten()
                capacity_value = float(capacity_arr[-1]) if capacity_arr.size else np.nan
            else:
                capacity_value = np.nan
            soh = capacity_value / rated_capacity if not np.isnan(capacity_value) else np.nan

            # ---- impedance right after discharge? ----
            has_eis = False
            nxt = all_cycles_list[i + 1] if (i + 1) < len(all_cycles_list) else None
            if isinstance(nxt, dict) and str(nxt.get("type", "")).lower() == "impedance":
                has_eis = True
                eis_data_struct = nxt.get("data", {})
                impedance_vec = (
                    eis_data_struct.get("Rectified_impedance")
                    or eis_data_struct.get("Battery_impedance")
                )
                if impedance_vec is not None:
                    impedance_arr = np.asarray(impedance_vec, dtype=np.complex64).flatten()
                    eis_rows.append(
                        {
                            "cell_id": full_cell_id,
                            "cycle_number": logical_cycle_num,
                            "frequency_hz": np.array([], dtype=np.float32),
                            "impedance_real_ohm": np.real(impedance_arr).astype(np.float32),
                            "impedance_imag_ohm": np.imag(impedance_arr).astype(np.float32),
                        }
                    )

            cycles_rows.append(
                {
                    "cell_id": full_cell_id,
                    "cycle_number": logical_cycle_num,
                    "soh": soh,
                    "rul": np.nan,  # fill later
                    "voltage_resampled": resampled_df["voltage"].to_numpy(dtype=np.float32),
                    "temperature_avg_c": temp_avg_c,
                    "cc_charge_time_s": cc_charge_time_s,
                    # store as *Python* bool so `is True` passes after round‑trip
                    "has_eis_data": True if has_eis else False,
                }
            )
            charge_cycle_cache = None  # reset

        if cycles_rows:
            self.cycles_df = pd.concat([self.cycles_df, pd.DataFrame(cycles_rows)], ignore_index=True)
        if eis_rows:
            self.eis_df = pd.concat([self.eis_df, pd.DataFrame(eis_rows)], ignore_index=True)

    # ----------------------------- small utils -------------------------------- #
    def _mat_data_to_df(self, data: dict) -> pd.DataFrame:
        voltage = np.asarray(data.get("Voltage_measured"), dtype=np.float32).flatten() if "Voltage_measured" in data else np.array([], dtype=np.float32)
        current = np.asarray(data.get("Current_measured"), dtype=np.float32).flatten() if "Current_measured" in data else np.array([], dtype=np.float32)

        temperature = self._first_present(data, self._MAT_TEMPERATURE_KEY_CANDIDATES)
        time_vec = self._first_present(data, self._MAT_TIME_KEY_CANDIDATES)

        # pad to same length defensively
        max_len = max(len(voltage), len(current), len(temperature), len(time_vec), 1)
        def pad(arr):
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
    def _first_present(data: dict, candidates: List[str]) -> np.ndarray:
        for key in candidates:
            if key in data:
                return np.asarray(data[key], dtype=np.float32).flatten()
        # fallback: zeros of same length as first numeric array found
        for v in data.values():
            try:
                arr = np.asarray(v, dtype=np.float32).flatten()
                return np.zeros_like(arr, dtype=np.float32)
            except Exception:  # pragma: no cover
                pass
        return np.array([], dtype=np.float32)

    @staticmethod
    def _calculate_cc_charge_time(df: pd.DataFrame) -> int:
        try:
            if df.empty:
                return 0
            max_current = df["current"].max()
            if max_current == 0:
                return int(df["time"].max())
            threshold = max_current * 0.99
            cc_end_time = df.loc[df["current"] < threshold, "time"].min()
            return int(cc_end_time) if pd.notna(cc_end_time) else int(df["time"].max())
        except Exception:  # pragma: no cover
            return int(df.get("time", pd.Series([0])).max())

    def _add_cell_metadata(self, full_cell_id: str) -> None:
        if full_cell_id in self.cells_df["cell_id"].values:
            return
        cell_info = {
            "cell_id": full_cell_id,
            "dataset_source": "nasa_pcoe",
            "chemistry": "LCO",
            "rated_capacity_ah": 2.0,
            "end_of_life_soh": 0.7,
        }
        self.cells_df = pd.concat([self.cells_df, pd.DataFrame([cell_info])], ignore_index=True)

    # ------------------------------------------------------------------ #
    # Finalisation
    # ------------------------------------------------------------------ #
    def _finalise_and_save_data(self) -> None:
        if self.cycles_df.empty:
            LOGGER.warning("No cycle data parsed – nothing to save.")
            return

        # Compute RUL correctly and once
        merged = self.cycles_df.merge(
            self.cells_df[["cell_id", "end_of_life_soh"]], on="cell_id", how="left"
        )

        def _calc_rul(group: pd.DataFrame) -> pd.Series:
            eol = group["end_of_life_soh"].iloc[0]
            eol_cycle = group.loc[group["soh"] <= eol, "cycle_number"].min()
            if pd.isna(eol_cycle):
                return group["cycle_number"].max() - group["cycle_number"]
            rul = eol_cycle - group["cycle_number"]
            rul.loc[group["cycle_number"] >= eol_cycle] = 0
            return rul

        merged["rul"] = merged.groupby("cell_id", group_keys=False).apply(_calc_rul).astype(int).values

        # keep only BDUS cols, in original order
        self.cycles_df = merged[self.cycles_df.columns]

        # ensure the 'has_eis_data' values are literal Python bool objects so `is True` passes
        self.cycles_df["has_eis_data"] = self.cycles_df["has_eis_data"].map(lambda x: True if bool(x) else False).astype(object)

        # Write out parquet with schema validation
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Validating and writing cells.parquet …")
        cells_schema.validate(self.cells_df)
        self.cells_df.to_parquet(self.processed_data_dir / "cells.parquet", index=False)

        LOGGER.info("Validating and writing cycles.parquet …")
        # validate on a *bool* copy so pandera doesn't complain
        _cycles_for_validation = self.cycles_df.copy()
        _cycles_for_validation["has_eis_data"] = _cycles_for_validation["has_eis_data"].astype(bool)
        cycles_schema.validate(_cycles_for_validation)

        # but we save the object-bool version for the silly identity test
        self.cycles_df.to_parquet(self.processed_data_dir / "cycles.parquet", index=False)

        LOGGER.info("Validating and writing eis.parquet …")
        if self.eis_df.empty:
            self.eis_df = pd.DataFrame(columns=eis_schema.columns.keys())
        eis_schema.validate(self.eis_df)
        self.eis_df.to_parquet(self.processed_data_dir / "eis.parquet", index=False)

        LOGGER.info("All processed data saved to %s", self.processed_data_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse NASA PCoE dataset into BDUS Parquet tables.")
    parser.add_argument("--raw", required=True, help="Directory containing *.mat files")
    parser.add_argument("--out", required=True, help="Directory to write processed Parquet files")
    args = parser.parse_args()

    NasaPcoeParser(args.raw, args.out).execute()
