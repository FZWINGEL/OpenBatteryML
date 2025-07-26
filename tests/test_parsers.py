import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
from pathlib import Path

from src.parsers.nasa_pcoe_parser import NasaPcoeParser

@pytest.fixture
def fake_nasa_mat_file(tmp_path: Path) -> Path:
    """
    Creates a temporary directory and a fake NASA .mat file inside it.
    The data is structured to test charge, discharge, and impedance cycles.
    """
    raw_dir = tmp_path / "raw" / "nasa_pcoe"
    raw_dir.mkdir(parents=True)
    file_path = raw_dir / "B0005.mat"

    # --- Create predictable data for two full cycles ---
    
    # Cycle 1: Charge, Discharge, Impedance
    charge_1 = {
        'type': 'charge',
        'data': {
            'Voltage_measured': np.array([3.8, 4.0, 4.2]),
            'Current_measured': np.array([1.5, 1.5, 0.2]),
            'Temperature_measured': np.array([24.0, 24.1, 24.2]),
            'Time': np.array([0, 100, 200]),
        }
    }
    discharge_1 = {
        'type': 'discharge',
        'data': {
            'Capacity': np.array([1.85]),
            'Voltage_measured': np.array([4.2, 3.5, 3.0]),
            'Current_measured': np.array([-2.0, -2.0, -2.0]),
            'Temperature_measured': np.array([25.0, 25.1, 25.2]),
            'Time': np.array([0, 150, 300]),
        }
    }
    impedance_1 = {
        'type': 'impedance',
        'data': {
            'Rectified_impedance': np.array([0.1 + 0.05j])
        }
    }

    # Cycle 2: Charge, Discharge (no impedance)
    charge_2 = {
        'type': 'charge',
        'data': {
            'Voltage_measured': np.array([3.8, 4.0, 4.2]),
            'Current_measured': np.array([1.5, 1.5, 0.3]),
            'Temperature_measured': np.array([26.0, 26.1, 26.2]),
            'Time': np.array([0, 110, 210]),
        }
    }
    discharge_2 = {
        'type': 'discharge',
        'data': {
            'Capacity': np.array([1.80]),
            'Voltage_measured': np.array([4.2, 3.5, 3.0]),
            'Current_measured': np.array([-2.0, -2.0, -2.0]),
            'Temperature_measured': np.array([27.0, 27.1, 27.2]),
            'Time': np.array([0, 160, 310]),
        }
    }

    # Assemble the final .mat structure
    mat_data = {
        'B0005': {
            'cycle': [charge_1, discharge_1, impedance_1, charge_2, discharge_2]
        }
    }
    
    savemat(file_path, mat_data)
    return file_path


def test_nasa_pcoe_parser(fake_nasa_mat_file: Path):
    """
    Tests the NasaPcoeParser by processing a single, controlled .mat file.
    It verifies that the output Parquet files are created correctly and
    that the data within them is accurately parsed and calculated.
    """
    # 1. SETUP: Point the parser to the temporary directories
    raw_dir = fake_nasa_mat_file.parent
    processed_dir = raw_dir.parent.parent / "processed"
    
    parser = NasaPcoeParser(raw_data_dir=raw_dir, processed_data_dir=processed_dir)

    # 2. EXECUTE: Run the parsing logic on the fake file
    parser._process_single_file(fake_nasa_mat_file)
    parser._finalise_and_save_data() # Also triggers RUL calculation

    # 3. ASSERT: Read the output files and verify their contents
    
    # --- Verify cells.parquet ---
    cells_df = pd.read_parquet(processed_dir / "cells.parquet")
    assert len(cells_df) == 1
    cell_row = cells_df.iloc[0]
    assert cell_row['cell_id'] == 'nasa_pcoe_B0005'
    assert cell_row['dataset_source'] == 'nasa_pcoe'
    assert cell_row['rated_capacity_ah'] == 2.0
    assert cell_row['end_of_life_soh'] == 0.7

    # --- Verify cycles.parquet ---
    cycles_df = pd.read_parquet(processed_dir / "cycles.parquet")
    assert len(cycles_df) == 2 # We created two logical discharge cycles

    # Check cycle 1
    cycle_1_row = cycles_df[cycles_df['cycle_number'] == 1].iloc[0]
    assert cycle_1_row['cell_id'] == 'nasa_pcoe_B0005'
    assert pytest.approx(cycle_1_row['soh']) == 1.85 / 2.0 # 0.925
    assert cycle_1_row['rul'] == 1 # EOL is not reached, so RUL is max_cycle - current_cycle (2-1=1)
    assert pytest.approx(cycle_1_row['temperature_avg_c']) == 24.1
    assert cycle_1_row['has_eis_data'] 
    assert len(cycle_1_row['voltage_resampled']) == 1024 # Check if resampling worked

    # Check cycle 2
    cycle_2_row = cycles_df[cycles_df['cycle_number'] == 2].iloc[0]
    assert cycle_2_row['cell_id'] == 'nasa_pcoe_B0005'
    assert pytest.approx(cycle_2_row['soh']) == 1.80 / 2.0 # 0.90
    assert cycle_2_row['rul'] == 0 # It's the last cycle, so 0 cycles remaining
    assert pytest.approx(cycle_2_row['temperature_avg_c']) == 26.1
    assert not cycle_2_row['has_eis_data'] 

    # --- Verify eis.parquet ---
    eis_df = pd.read_parquet(processed_dir / "eis.parquet")
    assert len(eis_df) == 1 # Only the first cycle had impedance data
    eis_row = eis_df.iloc[0]
    assert eis_row['cell_id'] == 'nasa_pcoe_B0005'
    assert eis_row['cycle_number'] == 1
    assert pytest.approx(eis_row['impedance_real_ohm'][0]) == 0.1
    assert pytest.approx(eis_row['impedance_imag_ohm'][0]) == 0.05