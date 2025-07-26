import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.io import savemat

from src.parsers.nasa_pcoe_parser import NasaPcoeParser
from src.utils.schema import cells_schema, cycles_schema, eis_schema

# -----------------------------------------------------------------------------
# Helper builders
# -----------------------------------------------------------------------------

def _charge_cycle(voltage, current, temperature_key, temperature, time_key, time):
    return {
        "type": "charge",
        "data": {
            "Voltage_measured": np.array(voltage),
            "Current_measured": np.array(current),
            temperature_key: np.array(temperature),
            time_key: np.array(time),
        },
    }


def _discharge_cycle(capacity, voltage, current, temperature_key, temperature, time_key, time):
    return {
        "type": "discharge",
        "data": {
            "Capacity": np.array([capacity]),
            "Voltage_measured": np.array(voltage),
            "Current_measured": np.array(current),
            temperature_key: np.array(temperature),
            time_key: np.array(time),
        },
    }


def _impedance_cycle(real_imag_complex):
    return {
        "type": "impedance",
        "data": {
            "Rectified_impedance": np.array(real_imag_complex, dtype=np.complex64)
        },
    }


def _write_mat(path: Path, cell_name: str, cycles: list[dict]):
    savemat(path, {cell_name: {"cycle": cycles}})


# -----------------------------------------------------------------------------
# Pytest fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def tmp_dirs(tmp_path: Path):
    raw = tmp_path / "data" / "raw" / "nasa_pcoe"
    proc = tmp_path / "data" / "processed"
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    return raw, proc


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_execute_end_to_end_multi_file(tmp_dirs):
    """Run execute() on two cells and validate outputs + schema."""
    raw_dir, proc_dir = tmp_dirs

    # Cell 1: two discharge cycles + impedance after first
    file1 = raw_dir / "B0005.mat"
    cycles1 = [
        _charge_cycle([3.8, 4.0, 4.2], [1.5, 1.5, 0.2], "Temperature_measured", [24, 24.1, 24.2], "Time", [0, 100, 200]),
        _discharge_cycle(1.85, [4.2, 3.5, 3.0], [-2, -2, -2], "Temperature_measured", [25, 25.1, 25.2], "Time", [0, 150, 300]),
        _impedance_cycle([0.1 + 0.05j]),
        _charge_cycle([3.8, 4.0, 4.2], [1.5, 1.5, 0.3], "Temperature_measured", [26, 26.1, 26.2], "Time", [0, 110, 210]),
        _discharge_cycle(1.80, [4.2, 3.5, 3.0], [-2, -2, -2], "Temperature_measured", [27, 27.1, 27.2], "Time", [0, 160, 310]),
    ]
    _write_mat(file1, "B0005", cycles1)

    # Cell 2: SOH never crosses EOL (RUL should be max - cycle_number)
    file2 = raw_dir / "B0006.mat"
    cycles2 = [
        _charge_cycle([3.8, 4.0], [1.0, 0.2], "Temperature_measured", [23, 23.2], "Time", [0, 90]),
        _discharge_cycle(1.95, [4.2, 3.2], [-2, -2], "Temperature_measured", [24, 24.3], "Time", [0, 140]),
        _charge_cycle([3.8, 4.0], [1.0, 0.2], "Temperature_measured", [23.5, 23.7], "Time", [0, 95]),
        _discharge_cycle(1.93, [4.2, 3.2], [-2, -2], "Temperature_measured", [24.5, 24.6], "Time", [0, 145]),
    ]
    _write_mat(file2, "B0006", cycles2)

    parser = NasaPcoeParser(raw_dir, proc_dir)
    parser.execute()

    # Read outputs
    cells_df = pd.read_parquet(proc_dir / "cells.parquet")
    cycles_df = pd.read_parquet(proc_dir / "cycles.parquet")
    eis_df = pd.read_parquet(proc_dir / "eis.parquet")

    # Schema validation (will raise if wrong)
    cells_schema.validate(cells_df)
    cycles_schema.validate(cycles_df)
    eis_schema.validate(eis_df)

    # Cells
    assert set(cells_df["cell_id"]) == {"nasa_pcoe_B0005", "nasa_pcoe_B0006"}

    # Cycle counts
    assert len(cycles_df[cycles_df.cell_id == "nasa_pcoe_B0005"]) == 2
    assert len(cycles_df[cycles_df.cell_id == "nasa_pcoe_B0006"]) == 2

    # RUL correctness for cell 1: since SOH < EOL at cycle 2, rul should be [1,0]
    c1 = cycles_df[cycles_df.cell_id == "nasa_pcoe_B0005"].sort_values("cycle_number")
    assert list(c1["rul"]) == [1, 0]

    # RUL for cell 2: never hits EOL (0.7). With 2 cycles, rul = [1,0]
    c2 = cycles_df[cycles_df.cell_id == "nasa_pcoe_B0006"].sort_values("cycle_number")
    assert list(c2["rul"]) == [1, 0]

    # has_eis_data flags
    assert c1.iloc[0]["has_eis_data"] is True
    assert c1.iloc[1]["has_eis_data"] is False

    # Voltage resampled length & dtype
    assert all(len(v) == 1024 for v in cycles_df["voltage_resampled"])
    assert cycles_df["voltage_resampled"].iloc[0].dtype == np.float32

    # CC charge time non-negative
    assert (cycles_df["cc_charge_time_s"] >= 0).all()

    # EIS rows
    assert len(eis_df) == 1
    row = eis_df.iloc[0]
    assert row["cell_id"] == "nasa_pcoe_B0005"
    assert row["cycle_number"] == 1
    assert np.isclose(row["impedance_real_ohm"][0], 0.1)
    assert np.isclose(row["impedance_imag_ohm"][0], 0.05)


def test_temperature_key_fallback_and_time_key(tmp_dirs):
    """Ensure parser uses fallback keys (Temperature_measured_C, Relative_Time)."""
    raw_dir, proc_dir = tmp_dirs
    file1 = raw_dir / "B0007.mat"

    cycles = [
        _charge_cycle([3.7, 4.1], [1.0, 0.1], "Temperature_measured_C", [22.0, 22.2], "Relative_Time", [0, 120]),
        _discharge_cycle(1.9, [4.1, 3.1], [-2, -2], "Temperature_measured_C", [23.0, 23.1], "Relative_Time", [0, 170]),
    ]
    _write_mat(file1, "B0007", cycles)

    parser = NasaPcoeParser(raw_dir, proc_dir)
    parser.execute()

    cycles_df = pd.read_parquet(proc_dir / "cycles.parquet")

    assert "temperature_avg_c" in cycles_df
    # Mean of [22.0, 22.2] is 22.1
    assert pytest.approx(cycles_df.iloc[0]["temperature_avg_c"], rel=1e-4) == 22.1


def test_missing_temperature_and_time_uses_zeros(tmp_dirs, caplog):
    """If no temp/time keys are present, _first_present returns zeros; ensure no crash and zeros present."""
    raw_dir, proc_dir = tmp_dirs
    file1 = raw_dir / "B0008.mat"

    charge = {
        "type": "charge",
        "data": {
            "Voltage_measured": np.array([4.0, 4.2]),
            "Current_measured": np.array([1.5, 0.2]),
            # no temp or time keys
        },
    }
    discharge = {
        "type": "discharge",
        "data": {
            "Capacity": np.array([1.9]),
            "Voltage_measured": np.array([4.2, 3.0]),
            "Current_measured": np.array([-2.0, -2.0]),
        },
    }

    _write_mat(file1, "B0008", [charge, discharge])

    parser = NasaPcoeParser(raw_dir, proc_dir)
    parser.execute()

    cycles_df = pd.read_parquet(proc_dir / "cycles.parquet")
    # Temperature zeros -> avg should be 0.0
    assert cycles_df.iloc[0]["temperature_avg_c"] == 0.0

    # Check a warning about missing charge for discharge? not here, charge exists; instead check logs for first_present zeros
    # (optional) We just ensure no exception and data saved.


def test_discharge_without_charge_is_skipped(tmp_dirs, caplog):
    raw_dir, proc_dir = tmp_dirs
    file1 = raw_dir / "B0009.mat"

    discharge_only = _discharge_cycle(1.8, [4.2, 3.0], [-2, -2], "Temperature_measured", [25, 25.1], "Time", [0, 150])
    _write_mat(file1, "B0009", [discharge_only])

    parser = NasaPcoeParser(raw_dir, proc_dir)
    with caplog.at_level("WARNING"):
        parser.execute()

    cycles_df = pd.read_parquet(proc_dir / "cycles.parquet") if (proc_dir / "cycles.parquet").exists() else pd.DataFrame()
    assert cycles_df.empty
    assert any("no preceding charge" in rec.message for rec in caplog.records)


def test_cc_charge_time_calculation(tmp_dirs):
    """Current drops from 1.0 to 0.5 => CC end ~ first index where < 0.99*max."""
    raw_dir, proc_dir = tmp_dirs
    file1 = raw_dir / "B0010.mat"

    charge = {
        "type": "charge",
        "data": {
            "Voltage_measured": np.array([3.8, 4.0, 4.1, 4.2]),
            "Current_measured": np.array([1.0, 1.0, 0.5, 0.3]),  # CC end at index 2
            "Temperature_measured": np.array([24.0, 24.1, 24.2, 24.3]),
            "Time": np.array([0, 50, 100, 150]),
        },
    }
    discharge = _discharge_cycle(1.9, [4.2, 3.0], [-2, -2], "Temperature_measured", [25, 25.1], "Time", [0, 160])
    _write_mat(file1, "B0010", [charge, discharge])

    parser = NasaPcoeParser(raw_dir, proc_dir)
    parser.execute()

    cycles_df = pd.read_parquet(proc_dir / "cycles.parquet")
    row = cycles_df.iloc[0]
    assert row["cc_charge_time_s"] == 100  # time at index 2


def test_corrupt_mat_file_logs_error_and_continues(tmp_dirs, caplog):
    raw_dir, proc_dir = tmp_dirs
    bad_file = raw_dir / "corrupt.mat"
    bad_file.write_bytes(b"not a mat file")

    # Also add a good file to ensure processing continues
    good_file = raw_dir / "B0011.mat"
    _write_mat(
        good_file,
        "B0011",
        [
            _charge_cycle([3.8, 4.0], [1.0, 0.2], "Temperature_measured", [22, 22.2], "Time", [0, 90]),
            _discharge_cycle(1.9, [4.2, 3.1], [-2, -2], "Temperature_measured", [23, 23.2], "Time", [0, 140]),
        ],
    )

    parser = NasaPcoeParser(raw_dir, proc_dir)
    with caplog.at_level("ERROR"):
        parser.execute()

    # Ensure good file processed
    cycles_df = pd.read_parquet(proc_dir / "cycles.parquet")
    assert not cycles_df.empty
    assert any("Could not load" in rec.message for rec in caplog.records)


def test_voltage_and_current_interpolation_monotonic_length(tmp_dirs):
    """Ensure interpolation returns exactly num_points and preserves bounds."""
    raw_dir, proc_dir = tmp_dirs
    file1 = raw_dir / "B0012.mat"

    charge = _charge_cycle([3.0, 4.2], [1.5, 0.2], "Temperature_measured", [20, 21], "Time", [0, 200])
    discharge = _discharge_cycle(1.7, [4.2, 3.0], [-2, -2], "Temperature_measured", [22, 22.1], "Time", [0, 180])
    _write_mat(file1, "B0012", [charge, discharge])

    parser = NasaPcoeParser(raw_dir, proc_dir)
    parser.execute()

    cycles_df = pd.read_parquet(proc_dir / "cycles.parquet")
    v = cycles_df.iloc[0]["voltage_resampled"]
    assert len(v) == 1024
    assert v[0] == pytest.approx(3.0)
    assert v[-1] == pytest.approx(4.2)


# -----------------------------------------------------------------------------
# Optional: property-like sanity check for RUL non-negativity and monotonicity
# -----------------------------------------------------------------------------

def test_rul_non_negative_and_monotonic(tmp_dirs):
    raw_dir, proc_dir = tmp_dirs
    file1 = raw_dir / "B0013.mat"

    # Build 5 cycles, SOH drops below 0.7 at cycle 4
    cycles = []
    soh_values = [1.0, 0.95, 0.8, 0.69, 0.65]  # capacities given rated 2.0 => multiply by 2
    for i, soh in enumerate(soh_values):
        cycles.append(_charge_cycle([3.8, 4.0], [1.0, 0.2], "Temperature_measured", [25, 25.1], "Time", [0, 100]))
        capacity = soh * 2.0
        cycles.append(_discharge_cycle(capacity, [4.2, 3.0], [-2, -2], "Temperature_measured", [26, 26.1], "Time", [0, 150]))
    _write_mat(file1, "B0013", cycles)

    parser = NasaPcoeParser(raw_dir, proc_dir)
    parser.execute()

    df = pd.read_parquet(proc_dir / "cycles.parquet")
    df = df.sort_values(["cell_id", "cycle_number"])  # single cell anyway

    # Check non-negative
    assert (df["rul"] >= 0).all()

    # After EOL (cycle where soh <= 0.7), RUL must be 0
    eol_cycle = df.loc[df["soh"] <= 0.7, "cycle_number"].min()
    assert (df.loc[df["cycle_number"] >= eol_cycle, "rul"] == 0).all()

    # Before EOL, RUL should strictly decrease by 1 each cycle in this dataset
    pre = df[df["cycle_number"] < eol_cycle]["rul"].to_list()
    assert pre == sorted(pre, reverse=True)
