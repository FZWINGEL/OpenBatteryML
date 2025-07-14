
from pandera import Column, DataFrameSchema, Check, Index
from pandera.typing import Series, String, Float, Int, Bool

# Define the schema for the cells.parquet file, which contains static metadata for each cell.
cells_schema = DataFrameSchema(
    columns={
        "cell_id": Column(String, required=True, unique=True, description="Primary Key. Used to link all data related to a single battery."),
        "dataset_source": Column(String, required=True, description="Used for domain adaptation experiments."),
        "chemistry": Column(String, required=True, description="Can be used as a categorical feature for the model."),
        "rated_capacity_ah": Column(Float, Check.greater_than(0), required=True, description="Crucial for Label Engineering (SOH calculation)."),
        "end_of_life_soh": Column(Float, Check.in_range(0, 1), required=True, description="Crucial for Label Engineering (RUL calculation).")
    },
    index=Index(Int),
    strict=True,  # Ensures no extra columns are present
    coerce=True   # Coerces data types to match the schema
)

# Define the schema for the cycles.parquet file, which contains the main sequential data for model training.
cycles_schema = DataFrameSchema(
    columns={
        "cell_id": Column(String, required=True, description="Foreign Key. Links this cycle event back to its parent cell."),
        "cycle_number": Column(Int, Check.greater_than_or_equal_to(0), required=True, description="A feature representing time."),
        "soh": Column(Float, Check.greater_than_or_equal_to(0), required=True, description="Primary Target Label (State of Health)."),
        "rul": Column(Int, Check.greater_than_or_equal_to(0), required=True, description="Secondary Target Label (Remaining Useful Life)."),
        "voltage_resampled": Column(object, required=True, description="Primary Model Feature. A fixed-length vector representing the voltage curve."),
        "temperature_avg_c": Column(Float, required=True, description="A key scalar feature for degradation."),
        "cc_charge_time_s": Column(Int, Check.greater_than_or_equal_to(0), required=True, description="Engineered Physics Feature."),
        "has_eis_data": Column(Bool, required=True, description="Conditional Flag for EIS data lookup.")
    },
    # Define a composite primary key to uniquely identify each cycle for a given cell.
    index=Index(Int),
    strict=True,
    coerce=True
)

# Define the schema for the eis.parquet file, which contains sparse impedance spectroscopy data.
eis_schema = DataFrameSchema(
    columns={
        "cell_id": Column(String, required=True, description="Composite Key. Used with cycle_number for a unique lookup."),
        "cycle_number": Column(Int, Check.greater_than_or_equal_to(0), required=True, description="Composite Key."),
        "frequency_hz": Column(object, required=True, description="The x-axis of the impedance measurement."),
        "impedance_real_ohm": Column(object, required=True, description="The real component (Z') of the impedance."),
        "impedance_imag_ohm": Column(object, required=True, description="The imaginary component (Z'') of the impedance.")
    },
    # Define a composite primary key to uniquely identify each EIS measurement.
    index=Index(Int),
    strict=True,
    coerce=True
)
