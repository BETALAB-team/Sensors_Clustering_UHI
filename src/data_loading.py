from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml


def load_config(config_path):
    """
    Load the project YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the YAML file is empty.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def project_path(path_value, project_root):
    """
    Convert a path from the configuration file into an absolute Path.

    Parameters
    ----------
    path_value : str | Path
        Path value from the configuration file.
    project_root : str | Path | None, optional
        Root folder of the project. If None, the current working directory is used.

    Returns
    -------
    Path
        Absolute normalized path.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    root = Path.cwd() if project_root is None else Path(project_root)
    return (root / path).resolve()


def ensure_directory(path_value):
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path_value : str | Path
        Directory path.

    Returns
    -------
    Path
        Created or existing directory path.
    """
    path = Path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_pickle(path_value):
    """
    Load a pickle file.

    Parameters
    ----------
    path_value : str | Path
        Path to the pickle file.

    Returns
    -------
    Any
        Object stored in the pickle file.

    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist.
    """
    path = Path(path_value)

    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    return pd.read_pickle(path)


def load_sensor_data(path_value):
    """
    Load preprocessed sensor data stored as a pickle dictionary.

    Parameters
    ----------
    path_value : str | Path
        Path to the sensor data pickle file.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary where keys are sensor names and values are sensor DataFrames.

    Raises
    ------
    TypeError
        If the loaded object is not a dictionary.
    """
    data = load_pickle(path_value)

    if not isinstance(data, dict):
        raise TypeError("Sensor data must be a dictionary of DataFrames.")

    return data


def load_reliability_table(path_value):
    """
    Load the sensor reliability table.

    Parameters
    ----------
    path_value : str | Path
        Path to the reliability table pickle file.

    Returns
    -------
    pd.DataFrame
        Reliability table indexed by sensor name.

    Raises
    ------
    TypeError
        If the loaded object is not a DataFrame.
    """
    table = load_pickle(path_value)

    if not isinstance(table, pd.DataFrame):
        raise TypeError("Reliability table must be a pandas DataFrame.")

    return table


def load_sensor_locations(path_value):
    """
    Load sensor locations from a geospatial file.

    Parameters
    ----------
    path_value : str | Path
        Path to the sensor locations file.

    Returns
    -------
    gpd.GeoDataFrame
        Sensor locations as a GeoDataFrame.

    Raises
    ------
    FileNotFoundError
        If the geospatial file does not exist.
    """
    path = Path(path_value)

    if not path.exists():
        raise FileNotFoundError(f"Sensor locations file not found: {path}")

    return gpd.read_file(path)


def filter_reliable_sensors(
    sensor_data: dict[str, pd.DataFrame],
    reliability_table: pd.DataFrame,
    sensor_prefix: str = "T",
    reliability_index_threshold: int = 3,
    drop_sensors = None,
) :
    """
    Filter sensors by name prefix, reliability index, and optional exclusion list.

    Parameters
    ----------
    sensor_data : dict[str, pd.DataFrame]
        Dictionary of sensor DataFrames.
    reliability_table : pd.DataFrame
        Reliability table indexed by sensor name.
    sensor_prefix : str, optional
        Prefix required for sensor names.
    reliability_index_threshold : int, optional
        Minimum reliability index required to keep a sensor.
    drop_sensors : list[str] | None, optional
        Sensor names to exclude manually.

    Returns
    -------
    dict[str, pd.DataFrame]
        Filtered dictionary of reliable sensors.

    Raises
    ------
    KeyError
        If the reliability table does not contain the reliability_index column.
    """
    if "reliability_index" not in reliability_table.columns:
        raise KeyError("Reliability table must contain a 'reliability_index' column.")

    drop_set = set(drop_sensors or [])

    return {
        name: df
        for name, df in sensor_data.items()
        if name.startswith(sensor_prefix)
        and name in reliability_table.index
        and reliability_table.loc[name, "reliability_index"] >= reliability_index_threshold
        and name not in drop_set
    }


def load_excel_workbook(path_value) -> dict[str, pd.DataFrame]:
    """
    Load all sheets from an Excel workbook.

    Parameters
    ----------
    path_value : str | Path
        Path to the Excel workbook.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary where keys are sheet names and values are DataFrames.

    Raises
    ------
    FileNotFoundError
        If the Excel workbook does not exist.
    """
    path = Path(path_value)

    if not path.exists():
        raise FileNotFoundError(f"Excel workbook not found: {path}")

    return pd.read_excel(path, sheet_name=None)


def load_csv(path_value, sep: str = ";") -> pd.DataFrame:
    """
    Load a CSV file.

    Parameters
    ----------
    path_value : str | Path
        Path to the CSV file.
    sep : str, optional
        Column separator.

    Returns
    -------
    pd.DataFrame
        Loaded CSV data.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    path = Path(path_value)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path, sep=sep)


def load_epw_data(path_value, skiprows: int = 8) -> pd.DataFrame:
    """
    Load the hourly data section of an EPW file.

    Parameters
    ----------
    path_value : str | Path
        Path to the EPW file.
    skiprows : int, optional
        Number of EPW header rows to skip.

    Returns
    -------
    pd.DataFrame
        EPW hourly data without header lines.

    Raises
    ------
    FileNotFoundError
        If the EPW file does not exist.
    """
    path = Path(path_value)

    if not path.exists():
        raise FileNotFoundError(f"EPW file not found: {path}")

    return pd.read_csv(path, header=None, skiprows=skiprows)


def load_project_inputs(config: dict[str, Any], project_root= None) -> dict[str, Any]:
    """
    Load the standard input files defined in the project configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Project configuration dictionary.
    project_root : str | Path | None, optional
        Root folder used to resolve relative paths.

    Returns
    -------
    dict[str, Any]
        Dictionary containing sensor data, reliability table, and sensor locations.
    """
    paths = config["paths"]

    reliability_path = project_path(paths["reliability_table"], project_root)
    sensor_data_path = project_path(paths["sensors_data"], project_root)
    locations_path = project_path(paths["sensors_locations"], project_root)

    return {
        "sensor_data": load_sensor_data(sensor_data_path),
        "reliability_table": load_reliability_table(reliability_path),
        "sensor_locations": load_sensor_locations(locations_path),
    }
