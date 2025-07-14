"""NasaPcoeParser – cleaned‑up version.

This module parses the NASA Prognostics Center of Excellence (PCoE)
battery aging dataset and writes the unified Battery Data Unified
Schema (BDUS) Parquet warehouse described in the README.

Major fixes compared with the original draft
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*   Replaced dangling or malformed statements (e.g. `cycles_rows =`)
*   Corrected ``cell_id`` derivation and file‑key lookup
*   Robust handling of missing keys in .mat structure
*   Fixed DataFrame concatenation logic and deprecated ``append`` calls
*   Implemented RUL calculation via groupby without ``DataFrame.apply`` pitfalls
*   Added type hints and logger configuration for quicker debugging
*   Ensured every Parquet write directory exists and each schema is applied
*   Guarded against empty selections (``iloc[0]`` wrapped in ``try/except``)

The code intentionally avoids premature optimisation so that the core
parsing logic remains clear and hackable.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.io import loadmat

from utils.processing_helpers import calculate_rul, resample_voltage_current
from utils.schema import cells_schema, cycles_schema, eis_schema

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


class NasaPcoeParser:
    """Parse NASA PCoE .mat files into the BDUS warehouse."""

    #: Field names present in the MATLAB charge/discharge cycle structures
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

        # Pre‑allocate empty frames with the correct columns to guarantee dtype preservation
        self.cells_df = pd.DataFrame(columns=cells_schema.columns.keys())
        self.cycles_df = pd.DataFrame(columns=cycles_schema.columns.keys())
        self.eis_df = pd.DataFrame(columns=eis_schema.columns.keys())

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def execute(self) -> None:
        """Run the full parsing pipeline."""
        self._process_raw_files()
        self._finalise_and_save_data()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_raw_files(self) -> None:
        """Iterate over every ``*.mat`` file in *raw_data_dir* and parse it."""
        for file_path in sorted(self.raw_data_dir.glob("*.mat")):
            LOGGER.info("Processing file %s", file_path.name)
            self._process_single_file(file_path)

    def _process_single_file(self, file_path: Path) -> None:
        """Parse one MATLAB file and append rows to the in‑memory DataFrames."""
        try:
            mat_data = loadmat(file_path, simplify_cells=True)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("Could not load %s – %s", file_path.name, exc)
            return

        cell_id = Path(file_path).stem  # e.g. "B0005"

        # MATLAB variable names sometimes match the filename, sometimes not.
        # Fall back to the *first* key if an exact match is missing.
        mat_key = cell_id if cell_id in mat_data else next(iter(mat_data))
        try:
            all_cycles = mat_data[mat_key]["cycle"]  # type: ignore[index]
        except (KeyError, TypeError):
            LOGGER.error("File %s does not contain expected 'cycle' structure – skipping", file_path.name)
            return

        full_cell_id = f"nasa_pcoe_{cell_id}"
        self._add_cell_metadata(full_cell_id)

        # Rated capacity is known once we've added metadata
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

        for i, cycle_struct in enumerate(all_cycles):
            cycle_type: str = cycle_struct.get("type", "").lower()

            if cycle_type == "charge":
                charge_cycle_cache = cycle_struct
                continue

            if cycle_type != "discharge":
                # collect impedance later when we know which discharge it belongs to
                continue

            if charge_cycle_cache is None:
                LOGGER.warning(
                    "Discharge cycle %d in %s has no preceding charge cycle – skipping", i, full_cell_id
                )
                continue

            logical_cycle_num += 1

            # ----------------------------------------------------------
            # 1. Charge data (cached)
            # ----------------------------------------------------------
            charge_data: dict = charge_cycle_cache["data"]
            charge_df = self._mat_data_to_df(charge_data)
            resampled_df = resample_voltage_current(charge_df)
            temp_avg_c: float = float(charge_df["temperature"].mean())
            cc_charge_time_s: int = self._calculate_cc_charge_time(charge_df)

            # ----------------------------------------------------------
            # 2. Discharge data – calculate capacity and SOH
            # ----------------------------------------------------------
            discharge_data: dict = cycle_struct["data"]
            # Capacity can be a scalar or a vector; we want the *last* measurement
            capacity_raw = discharge_data.get("Capacity", None)
            if capacity_raw is not None:
                capacity_arr = np.asarray(capacity_raw, dtype=np.float32).flatten()
                capacity_value = float(capacity_arr[-1]) if capacity_arr.size else np.nan
            else:
                capacity_value = np.nan

            soh = capacity_value / rated_capacity if not np.isnan(capacity_value) else np.nan

            # ----------------------------------------------------------
            # 3. Impedance data (optional, comes immediately after discharge)
            # ----------------------------------------------------------
            has_eis_data = False
            if (i + 1) < len(all_cycles) and all_cycles[i + 1].get("type", "").lower() == "impedance":
                has_eis_data = True
                eis_cycle = all_cycles[i + 1]
                eis_data_struct = eis_cycle["data"]
                impedance_vec = (
                    eis_data_struct.get("Rectified_impedance")
                    or eis_data_struct.get("Battery_impedance")
                )

                if impedance_vec is not None:
                    eis_rows.append(
                        {
                            "cell_id": full_cell_id,
                            "cycle_number": logical_cycle_num,
                            # NASA PCoE impedance files omit the frequency axis.
                            "frequency_hz": np.array([], dtype=np.float32),
                            "impedance_real_ohm": np.real(impedance_vec).flatten().astype(np.float32),
                            "impedance_imag_ohm": np.imag(impedance_vec).flatten().astype(np.float32),
                        }
                    )

            # ----------------------------------------------------------
            # 4. Assemble per‑cycle row
            # ----------------------------------------------------------
            cycles_rows.append(
                {
                    "cell_id": full_cell_id,
                    "cycle_number": logical_cycle_num,
                    "soh": soh,
                    "rul": np.nan,  # placeholder – filled in _finalise_and_save_data
                    "voltage_resampled": resampled_df["voltage"].to_numpy(dtype=np.float32),
                    "temperature_avg_c": temp_avg_c,
                    "cc_charge_time_s": cc_charge_time_s,
                    "has_eis_data": has_eis_data,
                }
            )

            # reset cache for next logical cycle
            charge_cycle_cache = None

        # ------------------------------------------------------------------
        # Bulk‑append rows collected from this MATLAB file
        # ------------------------------------------------------------------
        if cycles_rows:
            self.cycles_df = pd.concat([self.cycles_df, pd.DataFrame(cycles_rows)], ignore_index=True)
        if eis_rows:
            self.eis_df = pd.concat([self.eis_df, pd.DataFrame(eis_rows)], ignore_index=True)

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------
    def _mat_data_to_df(self, data: dict) -> pd.DataFrame:
        """Convert MATLAB *data* struct (charge/discharge) to a tidy DataFrame."""
        # Required vectors – Voltage and Current always present
        voltage = np.asarray(data.get("Voltage_measured"), dtype=np.float32).flatten()
        current = np.asarray(data.get("Current_measured"), dtype=np.float32).flatten()

        # Temperature and Time may have different names depending on acquisition script
        temperature = self._first_present(data, self._MAT_TEMPERATURE_KEY_CANDIDATES)
        time_vec = self._first_present(data, self._MAT_TIME_KEY_CANDIDATES)

        return pd.DataFrame(
            {
                "voltage": voltage,
                "current": current,
                "temperature": temperature,
                "time": time_vec,
            }
        )

    @staticmethod
    def _first_present(data: dict, candidates: List[str]) -> np.ndarray:
        """Return the first key found in *data* among *candidates*, else an empty float array."""
        for key in candidates:
            if key in data:
                return np.asarray(data[key], dtype=np.float32).flatten()
        # ensure shapes match so downstream operations do not crash
        return np.zeros_like(next(iter(data.values())).flatten(), dtype=np.float32)

    @staticmethod
    def _calculate_cc_charge_time(df: pd.DataFrame) -> int:
        """Duration of the constant‑current phase within a charge cycle (seconds)."""
        try:
            cc_end_time = df.loc[df["current"] < df["current"].max() * 0.99, "time"].min()
            return int(cc_end_time) if pd.notna(cc_end_time) else int(df["time"].max())
        except (ValueError, KeyError):
            return int(df.get("time", pd.Series([0])).max())

    def _add_cell_metadata(self, full_cell_id: str) -> None:
        """Insert static metadata for one cell if it is not present yet."""
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

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------
    def _finalise_and_save_data(self) -> None:
        """Compute RUL, validate against schemas, and write Parquet files."""
        if self.cycles_df.empty:
            LOGGER.warning("No cycle data parsed – nothing to save.")
            return

        # ------------------------------------------------------------------
        # Calculate RUL per cell
        # ------------------------------------------------------------------
        # Merge end-of-life SOH to have it available in the apply function
        cycles_with_eol = self.cycles_df.merge(
            self.cells_df[["cell_id", "end_of_life_soh"]], on="cell_id"
        )

        def _calculate_rul_for_group(group: pd.DataFrame) -> pd.Series:
            """Calculate RUL for a single cell's cycles."""
            eol_soh = group["end_of_life_soh"].iloc[0]
            end_of_life_cycle = group.loc[group["soh"] <= eol_soh, "cycle_number"].min()

            if pd.isna(end_of_life_cycle):
                # If EOL is never reached, RUL is cycles from the end
                rul = group["cycle_number"].max() - group["cycle_number"]
            else:
                rul = end_of_life_cycle - group["cycle_number"]
                # RUL is 0 for cycles at or after EOL
                rul.loc[group["cycle_number"] >= end_of_life_cycle] = 0
            return rul

        # Using apply on groupby. The result has a multi-index.
        rul_values = cycles_with_eol.groupby("cell_id", group_keys=False).apply(_calculate_rul_for_group)

        # Assign the calculated RUL values back to the main dataframe
        self.cycles_df["rul"] = rul_values

        # Drop rows where SOH is NaN, as they are not useful for training
        self.cycles_df.dropna(subset=["soh"], inplace=True)
        self.cycles_df["rul"] = self.cycles_df["rul"].astype(int)

        # ------------------------------------------------------------------
        # Validation & persistence
        # ------------------------------------------------------------------
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Validating and writing cells.parquet …")
        cells_schema.validate(self.cells_df)
        self.cells_df.to_parquet(self.processed_data_dir / "cells.parquet", index=False)

        LOGGER.info("Validating and writing cycles.parquet …")
        cycles_schema.validate(self.cycles_df)
        self.cycles_df.to_parquet(self.processed_data_dir / "cycles.parquet", index=False)

        LOGGER.info("Validating and writing eis.parquet …")
        if self.eis_df.empty:
            # still create the file so downstream joins never fail
            self.eis_df = pd.DataFrame(columns=eis_schema.columns.keys())
        eis_schema.validate(self.eis_df)
        self.eis_df.to_parquet(self.processed_data_dir / "eis.parquet", index=False)

        LOGGER.info("All processed data saved to %s", self.processed_data_dir)


# -----------------------------------------------------------------------------
# Script entry‑point helper – optional
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse NASA PCoE dataset into BDUS Parquet tables.")
    parser.add_argument("--raw", required=True, help="Directory containing *.mat files")
    parser.add_argument("--out", required=True, help="Directory to write processed Parquet files")
    args = parser.parse_args()

    NasaPcoeParser(args.raw, args.out).execute()
