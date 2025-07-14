# Battery Data Unified Schema (BDUS) Framework

The BDUS is not just a collection of files; it's a data-centric framework built on key principles designed to accelerate battery research by solving common data management problems upfront. It provides a standardized, model-ready format for working with diverse battery datasets.

---

## Core Architectural Principles

The framework is built on four key principles to ensure data integrity, speed, and ease of use for machine learning applications.

1.  **Separation of Concerns**: The framework strictly separates raw, immutable data from clean, processed data. The **Raw Data Lake** acts as a permanent, untouched archive for provenance and reproducibility. The **Processed Warehouse** contains the optimized, model-ready data. This ensures you never have to re-download or risk corrupting the original source files.

2.  **Unified Schema**: All data, regardless of its original source (e.g., NASA, CALCE, HNEI), is transformed to fit a single, consistent schema. This is the most crucial principle for enabling multi-dataset modeling. A `voltage_resampled` column means the exact same thing whether the data came from a `.mat` or an `.h5` file, allowing your model to train on all data simultaneously.

3.  **Columnar Storage (Apache Parquet)**: The choice of Parquet is a deliberate performance optimization. Traditional row-based storage (like CSV) forces you to read an entire row even if you only need one column. Parquet stores data in columns, so if you only need the `soh` and `rul` labels for analysis, the system reads only those columns, drastically reducing I/O and speeding up data loading. It also provides excellent compression.

4.  **Data Normalization and Atomicity**: The database is structured similarly to a relational database. Each table (file) has a distinct purpose. `cells.parquet` describes a single entity (a cell). `cycles.parquet` describes a single event (a cycle). This normalization prevents data duplication and keeps the structure clean, efficient, and scalable.

---

## The BDUS Two-Stage Architecture

The framework consists of two distinct stages that represent the data's journey from its raw state to being model-ready.

### Stage 1: The Raw Data Lake

This is the foundation of the framework. It's a simple directory structure where you store the downloaded datasets exactly as you received them.

* **Purpose**: To serve as a permanent, version-controlled archive. If you ever change your processing logic, you can rebuild the warehouse from this pristine source.
* **Structure**:
    ```bash
    data/raw/
    ├── nasa_pcoe/      (contains .mat files)
    ├── calce/          (contains .mat files)
    └── hnei/           (contains .h5 files)
    ```

### Stage 2: The Processed Parquet Warehouse

This is the active database your machine learning code will interact with. It's the output of your parsing and cleaning scripts.

* **Purpose**: To provide high-speed, uniform access to cleaned data for model training and evaluation.
* **Structure & Relationships**:
    ```
    data/processed/
    ├── cells.parquet
    │
    ├── cycles.parquet  <-- (Many-to-one relationship with cells.parquet)
    │
    └── eis.parquet     <-- (One-to-one sparse relationship with cycles.parquet)
    ```
    A `cell_id` in `cells.parquet` links to many rows in `cycles.parquet` (one for each cycle). A row in `cycles.parquet` links to at most one row in `eis.parquet`.

---

## Detailed Data Dictionary & Schemas 📖

This is the blueprint for the processed warehouse files. Each field is chosen to support the machine learning pipeline.

### `cells.parquet` (Static Metadata)

This file is queried once per cell to get its background information.

| Column Name         | Data Type       | Purpose & ML-Relevant Notes                                                                                                          |
| :------------------ | :-------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| **`cell_id`** | `string`        | **Primary Key**. Used to link all data related to a single battery. Essential for all joins.                                           |
| `dataset_source`    | `string`        | Used for domain adaptation experiments (e.g., train on NASA, test on CALCE).                                                         |
| `chemistry`         | `string`        | Can be used as a categorical feature for the model to learn chemistry-specific aging.                                                |
| `rated_capacity_ah` | `float32`       | **Crucial for Label Engineering**. This value is the denominator used to calculate SOH.                                              |
| `end_of_life_soh`   | `float32`       | **Crucial for Label Engineering**. This value defines the failure threshold to calculate RUL.                                        |

### `cycles.parquet` (Dynamic/Sequential Data)

This is the main input to your model. Each row is a complete training example (features + labels).

| Column Name           | Data Type        | Purpose & ML-Relevant Notes                                                                                                        |
| :-------------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **`cell_id`** | `string`         | **Foreign Key**. Links this cycle event back to its parent cell.                                                                   |
| **`cycle_number`** | `int32`          | A feature representing time. Useful for time-series models.                                                                      |
| `soh`                 | `float32`        | **Primary Target Label**. The value your model predicts. Calculated as: `actual_capacity / rated_capacity`.                        |
| `rul`                 | `int32`          | **Secondary Target Label**. Another potential prediction target.                                                                   |
| `voltage_resampled`   | `array[float32]` | **Primary Model Feature**. A fixed-length vector representing the voltage curve. The CNN/Transformer learns patterns from this.      |
| `temperature_avg_c`   | `float32`        | A key scalar feature, as temperature is a major accelerator of degradation.                                                        |
| `cc_charge_time_s`    | `int32`          | **Engineered Physics Feature**. The time in the constant-current phase can be a strong indicator of health.                          |
| `has_eis_data`        | `bool`           | **Conditional Flag**. Tells the data loader whether it needs to perform a lookup in `eis.parquet`.                                   |

### `eis.parquet` (Sparse Physical Measurements)

This file provides a deep physical snapshot, used to augment the main data when available.

| Column Name          | Data Type        | Purpose & ML-Relevant Notes                                                                    |
| :------------------- | :--------------- | :--------------------------------------------------------------------------------------------- |
| **`cell_id`** | `string`         | **Composite Key**. Used with `cycle_number` for a unique lookup.                                 |
| **`cycle_number`** | `int32`          | **Composite Key**.                                                                               |
| `frequency_hz`       | `array[float32]` | The x-axis of the impedance measurement.                                                       |
| `impedance_real_ohm` | `array[float32]` | The real component (Z') of the impedance. Growth is a direct sign of aging.                    |
| `impedance_imag_ohm` | `array[float32]` | The imaginary component (Z'') of the impedance.                                                |

---

## The Data Processing Pipeline: From Raw to Ready ⚙️

Building the BDUS involves a standardized workflow that you apply to each new raw dataset.

1.  **Ingestion**: Your parser script loads a raw data file (e.g., `B0005.mat`) into memory.
2.  **Parsing & Extraction**: The script navigates the specific structure of the raw file to extract the time, voltage, current, and temperature arrays for each cycle, along with the measured capacity.
3.  **Normalization & Resampling**: This is the core data transformation. A neural network needs fixed-size inputs.
    * The script defines a fixed grid (e.g., 1024 points).
    * It uses a numerical interpolation function (like `numpy.interp`) to resample the raw voltage and current arrays onto this fixed grid. This ensures that `voltage_resampled[100]` corresponds to the same relative point in the charge/discharge curve for every cycle in the database.
4.  **Feature & Label Engineering**: The script computes the derived values:
    * **SOH**: It takes the measured capacity for the cycle and divides it by the `rated_capacity_ah` from the cell's metadata.
    * **RUL**: After processing all cycles for a cell, it identifies the cycle where SOH crosses the `end_of_life_soh` threshold. It then iterates backward, calculating the cycles remaining for each prior cycle.
5.  **Writing to Parquet**: The script appends the newly created rows of cleaned data to the `cells.parquet`, `cycles.parquet`, and (if applicable) `eis.parquet` files.

---

## Querying the Database: A Practical Example 🔎

Let's trace how your PyTorch `Dataset` would fetch all data for cycle **100** of cell **`nasa_pcoe_B0005`**.

1.  **Query `cycles.parquet`**: It performs a lookup to get the row where `cell_id == 'nasa_pcoe_B0005'` and `cycle_number == 100`.
    * **Result**: This returns the `soh`, `rul`, `voltage_resampled` array, `temperature_avg_c`, and the flag `has_eis_data = True`.

2.  **Conditional Query of `eis.parquet`**: Because `has_eis_data` was `True`, the data loader now performs a second query. It looks up the row in `eis.parquet` where `cell_id == 'nasa_pcoe_B0005'` and `cycle_number == 100`.
    * **Result**: This returns the `frequency_hz`, `impedance_real_ohm`, and `impedance_imag_ohm` arrays.

3.  **Query `cells.parquet`**: It performs a final, quick lookup in `cells.parquet` for the row where `cell_id == 'nasa_pcoe_B0005'`.
    * **Result**: This returns the static metadata like `chemistry = 'LCO'` and `rated_capacity_ah = 2.0`.

4.  **Assemble**: The data loader combines all this information into a single Python dictionary or object, which is then passed to your model for a training step. This entire process, thanks to Parquet's efficiency, takes milliseconds.

---

## Repository Structure

Here is a description of the key files and directories in this project.
.
├── .gitignore             # Configures Git to ignore transient files (e.g., pycache, datasets).
├── README.md              # Main documentation file for the project (this file).
├── requirements.txt       # Lists all Python package dependencies for easy environment setup.
│
├── data/                  # Parent directory for all project data.
│   ├── processed/         # Stores the clean, model-ready data in Parquet format.
│   └── raw/               # Stores the original, unmodified datasets from various sources.
│
├── notebooks/             # Contains Jupyter notebooks for analysis and experimentation.
│   ├── 1-data-exploration.ipynb  # Notebook for exploratory data analysis (EDA).
│   └── 2-model-training-example.ipynb # Notebook demonstrating how to train a model.
│
├── src/                   # Contains all the source code for the data processing pipeline.
│   ├── main.py            # The main executable script to run the entire processing pipeline.
│   │
│   ├── parsers/           # A package containing modules to parse each raw dataset format.
│   │   ├── nasa_pcoe_parser.py # Specific parser for the NASA PCoE dataset.
│   │   ├── calce_parser.py     # Specific parser for the CALCE dataset.
│   │   └── hnei_parser.py      # Specific parser for the HNEI dataset.
│   │
│   └── utils/             # A package for shared, reusable utility functions.
│       ├── schema.py          # Defines and validates the unified Parquet data schemas.
│       └── processing_helpers.py # Contains common functions for resampling, RUL calculation, etc.
│
└── tests/                 # Contains all tests to ensure code correctness and data integrity.
├── test_parsers.py    # Unit tests for each individual data parser.
└── test_processing.py # Integration tests for the complete data processing pipeline.