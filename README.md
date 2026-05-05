# Sensor-Based Weather Scenario and Cooling Simulation Pipeline

## 1. Overview

This project builds weather scenarios from urban sensor measurements and evaluates their effect on building cooling demand using EUReCA.

The workflow is:

1. Load and filter reliable sensor data.
2. Cluster sensors based on temperature and humidity behaviour.
3. Select representative sensors or representative daily profiles for each cluster.
4. Export representative weather profiles to Excel.
5. Convert representative weather profiles to EPW files.
6. Run EUReCA simulations using the generated EPW files.
7. Post-process simulation outputs to estimate cooling demand and conditioning electricity.
8. Generate summary tables and plots.

The project is organized as a reproducible pipeline. User-editable settings are stored in `config/config.yaml`; source code is stored in `src/`; executable workflow scripts are stored in `scripts/`.

## 2. Project Structure

Recommended folder structure:

```text
project/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── input/
│   │   ├── reliability_table.pkl
│   │   ├── sensors_data_by_name_preprocessed.pickle
│   │   ├── sensors_location_complete.geojson
│   │   ├── epw/
│   │   │   └── ITA_Venezia-Tessera.161050_IGDG.epw
│   │   └── simulation/
│   │       ├── Example_District_Config.json
│   │       ├── Schedules_total.xlsx
│   │       ├── Materials.xlsx
│   │       ├── Example_District - Copia.geojson
│   │       └── systems.xlsx
│   │
│   ├── intermediate/
│   │   └── cluster_exports/
│   │
│   ├── output/
│   │   ├── epw/
│   │   └── simulation_results/
│   │
│   └── figures/
│
├── scripts/
│   ├── 01_cluster_sensors.py
│   ├── 02_export_representatives.py
│   ├── 03_make_epw_files.py
│   ├── 04_run_simulations.py
│   ├── 05_postprocess_simulations.py
│   └── 06_plot_results.py
│
└── src/
    ├── data_loading.py
    ├── clustering.py
    ├── representatives.py
    ├── epw_tools.py
    ├── simulation_postprocess.py
    ├── plotting.py
    └── comparison.py
```

## 3. Required Python Environment

Use Python 3.10 or newer. Python 3.11 is recommended.

Create a conda environment:

```bash
conda create -n sensor_project python=3.11
conda activate sensor_project
```

Install the main dependencies:

```bash
conda install pandas numpy matplotlib openpyxl scikit-learn geopandas pyarrow pyyaml
pip install astral tslearn
```

EUReCA must also be installed and importable from the same environment.

Check the environment:

```bash
python --version
python -c "import pandas, geopandas, sklearn, yaml"
python -c "from eureca_ubem.city import City"
```

If using Spyder, install the Spyder kernel inside the environment:

```bash
conda activate sensor_project
conda install spyder-kernels
```

Then select the environment interpreter in Spyder:

```text
Tools > Preferences > Python interpreter
```

## 4. Required Input Files

### 4.1 Sensor Data

Path:

```text
data/input/sensors_data_by_name_preprocessed.pickle
```

Expected format: a Python pickle containing a dictionary.

```python
{
    "T01": pandas.DataFrame,
    "T02": pandas.DataFrame,
    "T03": pandas.DataFrame,
}
```

Each sensor DataFrame must contain:

```text
Temperature
Humidity
```

It should also contain either:

```text
index_new
```

or have a datetime-like index.

Example table for one sensor:

| index_new | Temperature | Humidity |
|---|---:|---:|
| 2024-06-01 00:00:00 | 24.1 | 62.0 |
| 2024-06-01 00:10:00 | 24.0 | 63.2 |

### 4.2 Reliability Table

Path:

```text
data/input/reliability_table.pkl
```

Expected format: a pandas DataFrame stored as pickle.

Required index:

```text
sensor name
```

Required column:

```text
reliability_index
```

Example:

| sensor | reliability_index |
|---|---:|
| T01 | 3 |
| T02 | 2 |
| T03 | 3 |

The clustering script keeps only sensors that:

1. start with the configured sensor prefix, normally `T`;
2. exist in the reliability table;
3. have `reliability_index` greater than or equal to the configured threshold;
4. are not manually excluded in `drop_sensors`.

### 4.3 Sensor Locations

Path:

```text
data/input/sensors_location_complete.geojson
```

Expected format: GeoJSON.

Required column:

```text
name
```

Recommended columns:

```text
name
Latitude
Longitude
geometry
```

The `name` values must match the sensor names in the sensor dictionary and reliability table.

### 4.4 Base EPW File

Path:

```text
data/input/epw/ITA_Venezia-Tessera.161050_IGDG.epw
```

Expected format: standard EPW file.

The generated cluster EPW files are built by copying the base EPW and replacing:

```text
Dry Bulb Temperature
Relative Humidity
Dew Point Temperature
```

### 4.5 EUReCA Simulation Inputs

Path:

```text
data/input/simulation/
```

Required files:

```text
Example_District_Config.json
Schedules_total.xlsx
Materials.xlsx
Example_District - Copia.geojson
systems.xlsx
```

These are passed directly to EUReCA.

The simulation runner uses the following EUReCA call pattern:

```python
from eureca_building.config import load_config
from eureca_ubem.city import City

load_config(config_path)

city = City(
    city_model=city_model_file,
    epw_weather_file=weather_file,
    end_uses_types_file=schedules_file,
    envelope_types_file=materials_file,
    systems_templates_file=systems_file,
    shading_calculation=shading_calculation,
    building_model=building_model,
    output_folder=output_folder,
)

city.simulate(output_type="csv")
```

## 5. Configuration File

The project is controlled through:

```text
config/config.yaml
```

A typical configuration is:

```yaml
project:
  name: sensor_weather_cooling_project
  timezone: Europe/Rome

paths:
  input_dir: "data/input"
  output_dir: "data/output"
  figures_dir: "data/figures"
  intermediate_dir: "data/intermediate"

  reliability_table: "data/input/reliability_table.pkl"
  sensors_data: "data/input/sensors_data_by_name_preprocessed.pickle"
  sensors_locations: "data/input/sensors_location_complete.geojson"

  cluster_exports_dir: "data/intermediate/cluster_exports"

  epw_base_file: "data/input/epw/ITA_Venezia-Tessera.161050_IGDG.epw"
  epw_input_dir: "data/intermediate/cluster_exports"
  epw_output_dir: "data/output/epw"

  simulation_results_dir: "data/output/simulation_results"
  cooling_summaries_dir: "data/output/simulation_results/Cooling_summaries"

  cooling_summary_excel: "data/output/building_cooling_summary.xlsx"

sensor_filtering:
  sensor_prefix: "T"
  reliability_index_threshold: 3
  drop_sensors:
    - "T91"

clustering:
  city_name: Padua
  country: Italy
  latitude: 45.4064
  longitude: 11.8768
  timezone: Europe/Rome

  raw_resample_frequency: "10min"
  time_resample: "30min"
  max_interpolation_gap: "30min"

  cluster_range:
    min: 2
    max: 9

  final_clusters: 4
  random_state: 42
  sensor_location_name_col: "name"

representatives:
  representative_frequency: "1H"
  internal_frequency: "10min"

  export_daily_representatives: true
  export_full_year_representatives: true
  export_summer_representatives: true

  summer_months:
    - 6
    - 7
    - 8

epw:
  excel_pattern: "cluster_*_representatives_*.xlsx"
  excel_time_col: "time"
  excel_db_col: "db_temp"
  excel_wb_col: "wb_temp"
  excel_rh_col: "rh"

  epw_suffix: ".epw"
  epw_year_override: 2024

  replace_dry_bulb: true
  replace_relative_humidity: true
  recompute_dew_point: true

simulation:
  eureca_config: "data/input/simulation/Example_District_Config.json"
  schedules_file: "data/input/simulation/Schedules_total.xlsx"
  materials_file: "data/input/simulation/Materials.xlsx"
  city_model_file: "data/input/simulation/Example_District - Copia.geojson"
  systems_file: "data/input/simulation/systems.xlsx"

  epw_folder: "data/output/epw"
  simulation_results_dir: "data/output/simulation_results"

  epw_pattern: "cluster_*_representatives_*.epw"

  building_model: "1C"
  shading_calculation: true
  quasi_steady_state: false
  output_type: "csv"
  overwrite: false

simulation_postprocess:
  total_area_m2: 146318.5715

  summer_start: "2005-05-01 00:00"
  summer_end: "2005-10-31 23:00"

  summer_start_hour_index: 2880
  summer_end_hour_index: 7296

  building_pattern: "Results Bd Building*.csv"
  sep: ";"

  design_percentile: 99.0
  design_rounding_kw: 10.0

  cooling:
    sensible_col: "TZ sensible load [kW]"
    latent_col: "TZ latent load [kW]"
    electricity_col: "ConditioningElectricity [kW]"

  eer_model:
    slope: -0.2
    intercept: 11.2
    min_eer: 2.5
    max_eer: 7.0

plotting:
  sep: ";"
  dpi: 300

  months:
    - 5
    - 6
    - 7
    - 8
    - 9
    - 10

  month_labels:
    - May
    - Jun
    - Jul
    - Aug
    - Sep
    - Oct

  scenarios:
    cluster_1: "cluster_1_representatives_hourly_summer_summary.csv"
    cluster_2: "cluster_2_representatives_hourly_summer_summary.csv"
    cluster_3: "cluster_3_representatives_hourly_summer_summary.csv"
    cluster_4: "cluster_4_representatives_hourly_summer_summary.csv"

  target_day: null
```

Adjust filenames in `plotting.scenarios` to match the actual summary filenames produced in `data/output/simulation_results/Cooling_summaries`.

## 6. Workflow

Run the scripts from the project root.

### Step 1: Cluster sensors

```bash
python scripts/01_cluster_sensors.py
```

Main outputs:

```text
data/intermediate/sensor_cluster_labels.csv
data/intermediate/sensor_cluster_features.csv
data/intermediate/cluster_k_evaluation.csv
data/intermediate/dry_bulb_resampled.parquet
data/intermediate/relative_humidity_resampled.parquet
data/intermediate/wet_bulb_resampled.parquet
data/intermediate/sensor_locations_with_clusters.geojson
```

### Step 2: Export representative profiles

```bash
python scripts/02_export_representatives.py
```

Main outputs:

```text
data/intermediate/cluster_exports/cluster_1_representatives_hourly.xlsx
data/intermediate/cluster_exports/cluster_2_representatives_hourly.xlsx
data/intermediate/cluster_exports/cluster_3_representatives_hourly.xlsx
data/intermediate/cluster_exports/cluster_4_representatives_hourly.xlsx
```

Depending on the configuration, it may also write:

```text
cluster_1_representative_all.xlsx
cluster_1_representative_summer.xlsx
representative_temperature_summary.xlsx
```

Only files matching `epw.excel_pattern` are converted to EPW.

### Step 3: Convert representative profiles to EPW

```bash
python scripts/03_make_epw_files.py
```

Main outputs:

```text
data/output/epw/cluster_1_representatives_hourly.epw
data/output/epw/cluster_2_representatives_hourly.epw
data/output/epw/cluster_3_representatives_hourly.epw
data/output/epw/cluster_4_representatives_hourly.epw
```

### Step 4: Run EUReCA simulations

```bash
python scripts/04_run_simulations.py
```

Main outputs:

```text
data/output/simulation_results/cluster_1_representatives_hourly/
data/output/simulation_results/cluster_2_representatives_hourly/
data/output/simulation_results/cluster_3_representatives_hourly/
data/output/simulation_results/cluster_4_representatives_hourly/
```

Each scenario folder should contain EUReCA output CSV files such as:

```text
Results Bd Building 1.csv
Results Bd Building 2.csv
```

### Step 5: Post-process simulations

```bash
python scripts/05_postprocess_simulations.py
```

This step:

1. reads each scenario output folder;
2. matches it to the corresponding EPW file;
3. computes EER from outdoor temperature;
4. computes `ConditioningElectricity [kW]`;
5. writes the updated building CSV files;
6. creates scenario-level summer summary CSVs;
7. creates summary Excel and extra CSV tables.

Main outputs:

```text
data/output/simulation_results/Cooling_summaries/
data/output/building_cooling_summary.xlsx
data/output/peak_cooling_days.csv
data/output/monthly_cooling_summary.csv
```

### Step 6: Plot results

```bash
python scripts/06_plot_results.py
```

Main outputs:

```text
data/figures/monthly_total_cooling.png
data/figures/monthly_sensible_latent_cooling.png
data/figures/monthly_conditioning_electricity.png
data/figures/peak_day_cooling.png
data/figures/cooling_duration_curve.png
```

If `plotting.target_day` is set in `config.yaml`, the script also writes a target-day profile plot.

## 7. Output Interpretation

### 7.1 Cluster Labels

File:

```text
data/intermediate/sensor_cluster_labels.csv
```

Contains one cluster label per sensor.

### 7.2 Representative Weather Profiles

Folder:

```text
data/intermediate/cluster_exports/
```

Files with `representatives` in the name are daily representative profiles. Files with `representative_all` or `representative_summer` use one fixed representative sensor per cluster.

### 7.3 Generated EPW Files

Folder:

```text
data/output/epw/
```

These are the weather files used by EUReCA.

### 7.4 Simulation Outputs

Folder:

```text
data/output/simulation_results/
```

Each subfolder corresponds to one weather scenario.

### 7.5 Cooling Summaries

Folder:

```text
data/output/simulation_results/Cooling_summaries/
```

Each CSV contains hourly summer totals:

```text
Time
Total sensible load [kW]
Total latent load [kW]
Total cooling load [kW]
ConditioningElectricity [kW]
```

### 7.6 Summary Workbook

File:

```text
data/output/building_cooling_summary.xlsx
```

Contains total and specific indicators, including:

```text
Sensible cooling [GWh]
Latent cooling [GWh]
Total cooling [GWh]
Conditioning electricity [GWh]
Total cooling [kWh/m2]
Conditioning electricity [kWh/m2]
Peak cooling load [W/m2]
```

