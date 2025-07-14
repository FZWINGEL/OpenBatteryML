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


class CalceParser:
    """Parse CALCE .mat files into the BDUS warehouse."""

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

        cell_id = Path(file_path).stem  # e.g. "CS2_35"

        # CALCE data is structured differently, usually with a single top-level key
        # that contains all the cycle data.
        mat_key = next(iter(mat_data))
        try:
            all_cycles = mat_data[mat_key]["cycle"]
        except (KeyError, TypeError):
            LOGGER.error("File %s does not contain expected 'cycle' structure – skipping", file_path.name)
            return

        full_cell_id = f"calce_{cell_id}"
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

        logical_cycle_num = 0

        for i, cycle_struct in enumerate(all_cycles):
            cycle_type: str = cycle_struct.get("type", "").lower()

            if cycle_type == "charge":
                # CALCE data often has charge and discharge in separate entries
                # We need to find the corresponding discharge cycle to calculate SOH
                # For simplicity, we'll assume a discharge cycle immediately follows a charge cycle
                # or that capacity is available directly in the discharge cycle.
                continue

            if cycle_type != "discharge":
                continue

            logical_cycle_num += 1

            # ----------------------------------------------------------
            # 1. Discharge data – calculate capacity and SOH
            # ----------------------------------------------------------
            discharge_data: dict = cycle_struct["data"]
            # Capacity is usually directly available in CALCE discharge cycles
            capacity_raw = discharge_data.get("Capacity", None)
            if capacity_raw is not None:
                capacity_arr = np.asarray(capacity_raw, dtype=np.float32).flatten()
                capacity_value = float(capacity_arr[-1]) if capacity_arr.size else np.nan
            else:
                capacity_value = np.nan

            soh = capacity_value / rated_capacity if not np.isnan(capacity_value) else np.nan

            # Extract voltage, current, temperature, time from discharge data
            discharge_df = self._mat_data_to_df(discharge_data)
            resampled_df = resample_voltage_current(discharge_df)
            temp_avg_c: float = float(discharge_df["temperature"].mean())

            # CALCE data doesn't typically have a clear CC charge time in discharge cycles
            cc_charge_time_s: int = 0

            # ----------------------------------------------------------
            # 2. Impedance data (optional, usually separate entries)
            # ----------------------------------------------------------
            has_eis_data = False
            # CALCE EIS data is often in separate 'impedance' cycles, not directly linked
            # to charge/discharge. For this parser, we'll assume it's not directly available
            # within the charge/discharge cycle structure.

            # ----------------------------------------------------------
            # 3. Assemble per‑cycle row
            # ----------------------------------------------------------
            if pd.notna(soh):
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
        # CALCE uses different key names for voltage, current, temperature, time
        voltage = np.asarray(data.get("Voltage_measured"), dtype=np.float32).flatten()
        current = np.asarray(data.get("Current_measured"), dtype=np.float32).flatten()
        temperature = np.asarray(data.get("Temperature_measured"), dtype=np.float32).flatten()
        time_vec = np.asarray(data.get("Time"), dtype=np.float32).flatten()

        return pd.DataFrame(
            {
                "voltage": voltage,
                "current": current,
                "temperature": temperature,
                "time": time_vec,
            }
        )

    def _add_cell_metadata(self, full_cell_id: str) -> None:
        """Insert static metadata for one cell if it is not present yet."""
        if full_cell_id in self.cells_df["cell_id"].values:
            return

        # Placeholder values for CALCE, adjust as per actual dataset knowledge
        cell_info = {
            "cell_id": full_cell_id,
            "dataset_source": "calce",
            "chemistry": "LCO",  # Example, verify with actual CALCE data
            "rated_capacity_ah": 1.1,  # Example, verify with actual CALCE data
            "end_of_life_soh": 0.8,  # Example, verify with actual CALCE data
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