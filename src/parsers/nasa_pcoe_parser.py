
import logging
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from typing import Dict, List

from utils.processing_helpers import resample_voltage_current, calculate_rul
from utils.schema import cells_schema, cycles_schema, eis_schema

class NasaPcoeParser:
    """A parser for the NASA PCoE dataset."""

    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        """Initializes the parser with the data directories."""
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.cells_df = pd.DataFrame()
        self.cycles_df = pd.DataFrame()
        self.eis_df = pd.DataFrame()

    def execute(self):
        """Executes the parsing pipeline."""
        self._process_raw_files()
        self._validate_and_save_data()

    def _process_raw_files(self):
        """Processes all raw .mat files in the specified directory."""
        for filename in os.listdir(self.raw_data_dir):
            if filename.endswith(".mat"):
                file_path = os.path.join(self.raw_data_dir, filename)
                self._process_single_file(file_path)

    def _process_single_file(self, file_path: str):
        """Processes a single .mat file."""
        mat_data = loadmat(file_path)
        cell_id = os.path.splitext(os.path.basename(file_path))[0]
        data = mat_data[cell_id]['cycle'][0][0][0]

        # Extract cell metadata
        self._add_cell_metadata(cell_id)

        # Process each cycle
        for i, cycle in enumerate(data):
            self._process_cycle(cycle, cell_id, i + 1)

    def _add_cell_metadata(self, cell_id: str):
        """Adds cell metadata to the cells_df DataFrame."""
        cell_info = {
            "cell_id": f"nasa_pcoe_{cell_id}",
            "dataset_source": "nasa_pcoe",
            "chemistry": "LCO",  # Assuming Li-ion Cobalt Oxide
            "rated_capacity_ah": 2.0,
            "end_of_life_soh": 0.7
        }
        self.cells_df = pd.concat([self.cells_df, pd.DataFrame([cell_info])], ignore_index=True)

    def _process_cycle(self, cycle: np.ndarray, cell_id: str, cycle_number: int):
        """Processes a single cycle from the raw data."""
        cycle_type = cycle['type'][0]
        if cycle_type == 'charge':
            # We are interested in charge cycles for voltage curve analysis
            self._process_charge_cycle(cycle, cell_id, cycle_number)

    def _process_charge_cycle(self, cycle: np.ndarray, cell_id: str, cycle_number: int):
        """Processes a single charge cycle."""
        data = cycle['data'][0][0]
        df = pd.DataFrame({
            'voltage': data['Voltage_measured'].flatten(),
            'current': data['Current_measured'].flatten(),
            'temperature': data['Temperature_measured'].flatten(),
            'time': data['Time'].flatten()
        })

        # Resample voltage and current
        resampled_df = resample_voltage_current(df)

        # Calculate average temperature
        temp_avg_c = df['temperature'].mean()

        # Calculate constant-current charge time
        cc_charge_time_s = self._calculate_cc_charge_time(df)

        # Get SOH from the corresponding discharge cycle
        soh = self._get_soh_from_discharge(cycle_number, cell_id)

        cycle_data = {
            "cell_id": f"nasa_pcoe_{cell_id}",
            "cycle_number": cycle_number,
            "soh": soh,
            "rul": 0,  # Placeholder, will be calculated later
            "voltage_resampled": resampled_df['voltage'].values,
            "temperature_avg_c": temp_avg_c,
            "cc_charge_time_s": cc_charge_time_s,
            "has_eis_data": False  # NASA PCoE dataset does not have EIS data
        }
        self.cycles_df = pd.concat([self.cycles_df, pd.DataFrame([cycle_data])], ignore_index=True)

    def _calculate_cc_charge_time(self, df: pd.DataFrame) -> int:
        """Calculates the constant-current charge time."""
        # Find the time where the current starts to drop (end of CC phase)
        cc_end_time = df[df['current'] < df['current'].max()]['time'].min()
        return int(cc_end_time) if pd.notna(cc_end_time) else 0

    def _get_soh_from_discharge(self, cycle_number: int, cell_id: str) -> float:
        """Retrieves the State of Health (SOH) from the corresponding discharge cycle."""
        # In the NASA dataset, capacity is measured during discharge
        mat_data = loadmat(os.path.join(self.raw_data_dir, f"{cell_id}.mat"))
        data = mat_data[cell_id]['cycle'][0][0][0]
        for cycle in data:
            if cycle['type'][0] == 'discharge' and cycle['ambient_temperature'][0][0] == '24':
                if cycle_number in cycle['data'][0][0]['Cycle']:
                    capacity = cycle['data'][0][0]['Capacity'][0][0]
                    return capacity / 2.0  # Rated capacity is 2.0 Ah
        return np.nan

    def _validate_and_save_data(self):
        """Validates the processed data against the schemas and saves it to Parquet files."""
        # Calculate RUL for all cells
        self.cycles_df = self.cycles_df.groupby("cell_id").apply(
            lambda df: calculate_rul(df, self.cells_df.set_index("cell_id").loc[df.name, "end_of_life_soh"])
        ).reset_index(drop=True)

        # Validate DataFrames
        cells_schema.validate(self.cells_df)
        cycles_schema.validate(self.cycles_df)

        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # Save to Parquet files
        self.cells_df.to_parquet(os.path.join(self.processed_data_dir, "cells.parquet"), index=False)
        self.cycles_df.to_parquet(os.path.join(self.processed_data_dir, "cycles.parquet"), index=False)
        logging.info("Successfully saved processed data to Parquet files.")
