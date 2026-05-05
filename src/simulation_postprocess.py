from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def eer_from_temperature(
    temperature_c: np.ndarray | pd.Series,
    slope: float = -0.2,
    intercept: float = 11.2,
    min_eer: float = 2.5,
    max_eer: float = 7.0,
) -> np.ndarray:
    """
    Estimate EER from outdoor dry-bulb temperature.

    Parameters
    ----------
    temperature_c : np.ndarray | pd.Series
        Outdoor dry-bulb temperature in Celsius.
    slope : float, optional
        Linear model slope.
    intercept : float, optional
        Linear model intercept.
    min_eer : float, optional
        Minimum allowed EER.
    max_eer : float, optional
        Maximum allowed EER.

    Returns
    -------
    np.ndarray
        EER values.
    """
    return np.clip(slope * np.asarray(temperature_c, dtype=float) + intercept, min_eer, max_eer)


def find_matching_epw(
    folder_name: str,
    epw_map: dict[str, Path],
) -> Path | None:
    """
    Find the EPW file matching a simulation result folder name.

    Parameters
    ----------
    folder_name : str
        Simulation result folder name.
    epw_map : dict[str, Path]
        Dictionary mapping scenario keys to EPW paths.

    Returns
    -------
    Path | None
        Matching EPW path, or None if no match is found.
    """
    name = folder_name.lower()

    if "cluster_1" in name:
        return epw_map.get("cluster_1")

    if "cluster_2" in name:
        return epw_map.get("cluster_2")

    if "cluster_3" in name:
        return epw_map.get("cluster_3") or epw_map.get("nearest_representative")

    if "cluster_4" in name:
        return epw_map.get("cluster_4")

    if "rural" in name:
        return epw_map.get("rural") or epw_map.get("ARPAV_rural")

    if "suburban" in name or "city" in name:
        return epw_map.get("suburban") or epw_map.get("ARPAV_suburban")

    if "tmy" in name:
        return epw_map.get("tmy") or epw_map.get("TMY")

    if "nearest" in name:
        return epw_map.get("nearest_representative") or epw_map.get("cluster_3")

    return None


def build_epw_map(epw_folder: str | Path) -> dict[str, Path]:
    """
    Build a scenario-to-EPW map from an EPW folder.

    Parameters
    ----------
    epw_folder : str | Path
        Folder containing EPW files.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping scenario keys to EPW paths.
    """
    epw_dir = Path(epw_folder)
    epw_map = {}

    for file_path in epw_dir.glob("*.epw"):
        name = file_path.name.lower()

        if "all" in name or "summer" in name:
            continue

        if "cluster_1" in name or "cluster1" in name:
            epw_map["cluster_1"] = file_path
        elif "cluster_2" in name or "cluster2" in name:
            epw_map["cluster_2"] = file_path
        elif "cluster_3" in name or "cluster3" in name:
            epw_map["cluster_3"] = file_path
            epw_map["nearest_representative"] = file_path
        elif "cluster_4" in name or "cluster4" in name:
            epw_map["cluster_4"] = file_path
        elif "rural" in name:
            epw_map["rural"] = file_path
            epw_map["ARPAV_rural"] = file_path
        elif "city" in name or "suburban" in name:
            epw_map["suburban"] = file_path
            epw_map["ARPAV_suburban"] = file_path
        elif "tmy" in name:
            epw_map["tmy"] = file_path
            epw_map["TMY"] = file_path

    return epw_map


def read_epw_dry_bulb(epw_path: str | Path) -> np.ndarray:
    """
    Read dry-bulb temperature from an EPW file.

    Parameters
    ----------
    epw_path : str | Path
        Path to the EPW file.

    Returns
    -------
    np.ndarray
        Dry-bulb temperature array.

    Raises
    ------
    FileNotFoundError
        If the EPW file does not exist.
    """
    path = Path(epw_path)

    if not path.exists():
        raise FileNotFoundError(f"EPW file not found: {path}")

    epw = pd.read_csv(path, skiprows=8, header=None)

    return epw.iloc[:, 6].astype(float).to_numpy()


def read_building_result(csv_path: str | Path, sep: str = ";") -> pd.DataFrame:
    """
    Read one building simulation result CSV.

    Parameters
    ----------
    csv_path : str | Path
        Path to the building result CSV.
    sep : str, optional
        CSV separator.

    Returns
    -------
    pd.DataFrame
        Building result table.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Building result CSV not found: {path}")

    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        return pd.read_csv(path)


def cooling_load_array(
    df: pd.DataFrame,
    sensible_col: str = "TZ sensible load [kW]",
    latent_col: str = "TZ latent load [kW]",
) -> np.ndarray:
    """
    Build total thermal cooling-load array from sensible and latent loads.

    Parameters
    ----------
    df : pd.DataFrame
        Building result table.
    sensible_col : str, optional
        Sensible load column.
    latent_col : str, optional
        Latent load column.

    Returns
    -------
    np.ndarray
        Total load array where negative values represent cooling.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    missing = [c for c in [sensible_col, latent_col] if c not in df.columns]

    if missing:
        raise KeyError(f"Missing cooling columns: {missing}")

    sensible = pd.to_numeric(df[sensible_col], errors="coerce").fillna(0.0)
    latent = pd.to_numeric(df[latent_col], errors="coerce").fillna(0.0)

    return (sensible + latent).to_numpy(dtype=float)


def estimate_conditioning_electricity(
    cooling_load: np.ndarray,
    eer: np.ndarray,
    design_percentile: float = 99.0,
    design_rounding_kw: float = 10.0,
) -> np.ndarray:
    """
    Estimate conditioning electricity from cooling load and EER.

    Parameters
    ----------
    cooling_load : np.ndarray
        Total thermal load where negative values represent cooling.
    eer : np.ndarray
        EER values.
    design_percentile : float, optional
        Percentile used to size the cooling system.
    design_rounding_kw : float, optional
        Rounding interval for design capacity.

    Returns
    -------
    np.ndarray
        Conditioning electricity demand.
    """
    q = np.asarray(cooling_load, dtype=float)
    eer_arr = np.asarray(eer, dtype=float)

    n = min(len(q), len(eer_arr))
    q_head = q[:n]
    eer_head = eer_arr[:n]

    electricity = np.zeros(len(q), dtype=float)
    negative = q_head[q_head < 0.0]

    if negative.size == 0:
        return electricity

    cooling_abs = np.maximum(-q_head, 0.0)
    p_design = np.percentile(cooling_abs[cooling_abs > 0.0], design_percentile)

    if design_rounding_kw > 0:
        design_abs = np.ceil(p_design / design_rounding_kw) * design_rounding_kw
    else:
        design_abs = p_design

    used_abs = np.minimum(cooling_abs, design_abs)
    electricity[:n] = np.where(cooling_abs > 0.0, used_abs / eer_head, 0.0)

    return electricity


def add_conditioning_electricity_to_file(
    csv_path: str | Path,
    dry_bulb_temperature: np.ndarray,
    sensible_col: str = "TZ sensible load [kW]",
    latent_col: str = "TZ latent load [kW]",
    electricity_col: str = "ConditioningElectricity [kW]",
    sep: str = ";",
    eer_slope: float = -0.2,
    eer_intercept: float = 11.2,
    min_eer: float = 2.5,
    max_eer: float = 7.0,
    design_percentile: float = 99.0,
    design_rounding_kw: float = 10.0,
) -> Path:
    """
    Add conditioning electricity to one building simulation result file.

    Parameters
    ----------
    csv_path : str | Path
        Path to the building result CSV.
    dry_bulb_temperature : np.ndarray
        Outdoor dry-bulb temperature array.
    sensible_col : str, optional
        Sensible load column.
    latent_col : str, optional
        Latent load column.
    electricity_col : str, optional
        Output electricity column.
    sep : str, optional
        CSV separator for saving.
    eer_slope : float, optional
        EER linear model slope.
    eer_intercept : float, optional
        EER linear model intercept.
    min_eer : float, optional
        Minimum allowed EER.
    max_eer : float, optional
        Maximum allowed EER.
    design_percentile : float, optional
        Cooling design percentile.
    design_rounding_kw : float, optional
        Cooling design capacity rounding interval.

    Returns
    -------
    Path
        Updated CSV path.
    """
    path = Path(csv_path)
    df = read_building_result(path, sep=sep)

    eer = eer_from_temperature(
        dry_bulb_temperature,
        slope=eer_slope,
        intercept=eer_intercept,
        min_eer=min_eer,
        max_eer=max_eer,
    )

    cooling_load = cooling_load_array(
        df,
        sensible_col=sensible_col,
        latent_col=latent_col,
    )

    electricity = estimate_conditioning_electricity(
        cooling_load=cooling_load,
        eer=eer,
        design_percentile=design_percentile,
        design_rounding_kw=design_rounding_kw,
    )

    df[electricity_col] = electricity
    df.to_csv(path, sep=sep, index=False)

    return path


def add_conditioning_electricity(
    epw_folder: str | Path,
    simulation_results_path: str | Path,
    sensible_col: str = "TZ sensible load [kW]",
    latent_col: str = "TZ latent load [kW]",
    electricity_col: str = "ConditioningElectricity [kW]",
    building_pattern: str = "Results Bd Building*.csv",
    sep: str = ";",
    eer_slope: float = -0.2,
    eer_intercept: float = 11.2,
    min_eer: float = 2.5,
    max_eer: float = 7.0,
    design_percentile: float = 99.0,
    design_rounding_kw: float = 10.0,
) -> list[Path]:
    """
    Add conditioning electricity to all building result files.

    Parameters
    ----------
    epw_folder : str | Path
        Folder containing scenario EPW files.
    simulation_results_path : str | Path
        Folder containing scenario simulation result folders.
    sensible_col : str, optional
        Sensible load column.
    latent_col : str, optional
        Latent load column.
    electricity_col : str, optional
        Output electricity column.
    building_pattern : str, optional
        Glob pattern for building CSV files.
    sep : str, optional
        CSV separator.
    eer_slope : float, optional
        EER linear model slope.
    eer_intercept : float, optional
        EER linear model intercept.
    min_eer : float, optional
        Minimum allowed EER.
    max_eer : float, optional
        Maximum allowed EER.
    design_percentile : float, optional
        Cooling design percentile.
    design_rounding_kw : float, optional
        Cooling design capacity rounding interval.

    Returns
    -------
    list[Path]
        Updated building result paths.
    """
    sim_dir = Path(simulation_results_path)
    epw_map = build_epw_map(epw_folder)
    written = []

    for folder in sim_dir.iterdir():
        if not folder.is_dir():
            continue

        epw_path = find_matching_epw(folder.name, epw_map)

        if epw_path is None:
            continue

        dry_bulb = read_epw_dry_bulb(epw_path)

        for csv_file in folder.glob(building_pattern):
            written.append(
                add_conditioning_electricity_to_file(
                    csv_path=csv_file,
                    dry_bulb_temperature=dry_bulb,
                    sensible_col=sensible_col,
                    latent_col=latent_col,
                    electricity_col=electricity_col,
                    sep=sep,
                    eer_slope=eer_slope,
                    eer_intercept=eer_intercept,
                    min_eer=min_eer,
                    max_eer=max_eer,
                    design_percentile=design_percentile,
                    design_rounding_kw=design_rounding_kw,
                )
            )

    return written


def summer_time_index(
    start: str = "2005-05-01 00:00",
    end: str = "2005-10-31 23:00",
    freq: str = "H",
) -> pd.DatetimeIndex:
    """
    Build the summer time index used for cooling summaries.

    Parameters
    ----------
    start : str, optional
        Start timestamp.
    end : str, optional
        End timestamp.
    freq : str, optional
        Time frequency.

    Returns
    -------
    pd.DatetimeIndex
        Summer time index.
    """
    return pd.date_range(start, end, freq=freq)


def summarize_scenario_cooling(
    scenario_folder: str | Path,
    start_hour_index: int = 120 * 24,
    end_hour_index: int = 304 * 24,
    time_index: pd.DatetimeIndex | None = None,
    sensible_col: str = "TZ sensible load [kW]",
    latent_col: str = "TZ latent load [kW]",
    electricity_col: str = "ConditioningElectricity [kW]",
    building_pattern: str = "Results Bd Building*.csv",
    sep: str = ";",
) -> pd.DataFrame:
    """
    Summarize cooling loads and conditioning electricity for one scenario.

    Parameters
    ----------
    scenario_folder : str | Path
        Scenario result folder.
    start_hour_index : int, optional
        Start hour index in annual simulation result arrays.
    end_hour_index : int, optional
        End hour index in annual simulation result arrays.
    time_index : pd.DatetimeIndex | None, optional
        Output time index.
    sensible_col : str, optional
        Sensible load column.
    latent_col : str, optional
        Latent load column.
    electricity_col : str, optional
        Conditioning electricity column.
    building_pattern : str, optional
        Glob pattern for building CSV files.
    sep : str, optional
        CSV separator.

    Returns
    -------
    pd.DataFrame
        Scenario summer cooling summary.
    """
    folder = Path(scenario_folder)
    n_summer = end_hour_index - start_hour_index

    if time_index is None:
        time_index = summer_time_index()

    sens_sum = np.zeros(n_summer)
    lat_sum = np.zeros(n_summer)
    elec_sum = np.zeros(n_summer)
    has_any = False

    for csv_file in folder.glob(building_pattern):
        df = read_building_result(csv_file, sep=sep)

        missing = [c for c in [sensible_col, latent_col, electricity_col] if c not in df.columns]

        if missing:
            continue

        sensible = pd.to_numeric(df[sensible_col], errors="coerce").fillna(0.0).to_numpy()
        latent = pd.to_numeric(df[latent_col], errors="coerce").fillna(0.0).to_numpy()
        electricity = pd.to_numeric(df[electricity_col], errors="coerce").fillna(0.0).to_numpy()

        n = len(sensible)
        s_start = min(start_hour_index, n)
        s_end = min(end_hour_index, n)

        if s_end <= s_start:
            continue

        length = s_end - s_start
        target_length = min(length, n_summer)

        sens_slice = sensible[s_start:s_start + target_length]
        lat_slice = latent[s_start:s_start + target_length]
        elec_slice = electricity[s_start:s_start + target_length]

        sens_sum[:target_length] += np.where(sens_slice < 0.0, sens_slice, 0.0)
        lat_sum[:target_length] += np.where(lat_slice < 0.0, lat_slice, 0.0)
        elec_sum[:target_length] += elec_slice

        has_any = True

    if not has_any:
        return pd.DataFrame(columns=[
            "Time",
            "Total sensible load [kW]",
            "Total latent load [kW]",
            "Total cooling load [kW]",
            "ConditioningElectricity [kW]",
        ])

    sens_pos = -sens_sum
    lat_pos = -lat_sum
    cooling = sens_pos + lat_pos

    return pd.DataFrame({
        "Time": time_index[:n_summer],
        "Total sensible load [kW]": sens_pos,
        "Total latent load [kW]": lat_pos,
        "Total cooling load [kW]": cooling,
        "ConditioningElectricity [kW]": elec_sum,
    })


def summarize_summer_cooling(
    simulation_results_path: str | Path,
    output_folder_name: str = "Cooling_summaries",
    start_hour_index: int = 120 * 24,
    end_hour_index: int = 304 * 24,
    start_time: str = "2005-05-01 00:00",
    end_time: str = "2005-10-31 23:00",
    sensible_col: str = "TZ sensible load [kW]",
    latent_col: str = "TZ latent load [kW]",
    electricity_col: str = "ConditioningElectricity [kW]",
    building_pattern: str = "Results Bd Building*.csv",
    sep: str = ";",
) -> list[Path]:
    """
    Summarize summer cooling for all simulation scenario folders.

    Parameters
    ----------
    simulation_results_path : str | Path
        Folder containing scenario result folders.
    output_folder_name : str, optional
        Name of the summary output folder.
    start_hour_index : int, optional
        Start hour index in annual result arrays.
    end_hour_index : int, optional
        End hour index in annual result arrays.
    start_time : str, optional
        Output summary start timestamp.
    end_time : str, optional
        Output summary end timestamp.
    sensible_col : str, optional
        Sensible load column.
    latent_col : str, optional
        Latent load column.
    electricity_col : str, optional
        Conditioning electricity column.
    building_pattern : str, optional
        Glob pattern for building CSV files.
    sep : str, optional
        CSV separator.

    Returns
    -------
    list[Path]
        Written summary CSV paths.
    """
    sim_dir = Path(simulation_results_path)
    out_dir = sim_dir / output_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    time_index = summer_time_index(start=start_time, end=end_time)
    written = []

    for folder in sim_dir.iterdir():
        if not folder.is_dir() or folder.name == output_folder_name:
            continue

        summary = summarize_scenario_cooling(
            scenario_folder=folder,
            start_hour_index=start_hour_index,
            end_hour_index=end_hour_index,
            time_index=time_index,
            sensible_col=sensible_col,
            latent_col=latent_col,
            electricity_col=electricity_col,
            building_pattern=building_pattern,
            sep=sep,
        )

        if summary.empty:
            continue

        out_csv = out_dir / f"{folder.name}_summer_summary.csv"
        summary.to_csv(out_csv, sep=sep, index=False)
        written.append(out_csv)

    return written


def scenario_pretty_name(path: str | Path) -> str:
    """
    Convert a summary CSV path to a readable scenario name.

    Parameters
    ----------
    path : str | Path
        Summary CSV path.

    Returns
    -------
    str
        Scenario name.
    """
    name = Path(path).name.lower()

    if "cluster_1" in name or "cluster1" in name:
        return "cluster_1"

    if "cluster_2" in name or "cluster2" in name:
        return "cluster_2"

    if "cluster_3" in name or "cluster3" in name:
        return "cluster_3"

    if "cluster_4" in name or "cluster4" in name:
        return "cluster_4"

    if "nearest" in name:
        return "nearest"

    if "rural" in name:
        return "rural"

    if "suburban" in name or "city" in name:
        return "suburban"

    if "tmy" in name:
        return "TMY"

    return Path(path).stem


def cooling_summary_indicators(
    df: pd.DataFrame,
    scenario: str,
    total_area_m2: float,
) -> tuple[dict[str, float | str], dict[str, float | str]]:
    """
    Compute total and specific cooling summary indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Cooling summary table.
    scenario : str
        Scenario name.
    total_area_m2 : float
        Total net floor area.

    Returns
    -------
    tuple[dict[str, float | str], dict[str, float | str]]
        Total indicators and specific indicators.
    """
    sensible = pd.to_numeric(df["Total sensible load [kW]"], errors="coerce").fillna(0.0)
    latent = pd.to_numeric(df["Total latent load [kW]"], errors="coerce").fillna(0.0)
    cooling = pd.to_numeric(df["Total cooling load [kW]"], errors="coerce").fillna(0.0)
    electricity = pd.to_numeric(df["ConditioningElectricity [kW]"], errors="coerce").fillna(0.0)

    sens_kwh = float(sensible.sum())
    lat_kwh = float(latent.sum())
    cool_kwh = float(cooling.sum())
    elec_kwh = float(electricity.sum())

    total = {
        "Scenario": scenario,
        "Sensible cooling [GWh]": sens_kwh / 1_000_000.0,
        "Latent cooling [GWh]": lat_kwh / 1_000_000.0,
        "Total cooling [GWh]": cool_kwh / 1_000_000.0,
        "Conditioning electricity [GWh]": elec_kwh / 1_000_000.0,
        "Peak cooling load [kW]": float(cooling.max()),
        "Peak conditioning electricity [kW]": float(electricity.max()),
    }

    specific = {
        "Scenario": scenario,
        "Sensible cooling [kWh/m2]": sens_kwh / total_area_m2,
        "Latent cooling [kWh/m2]": lat_kwh / total_area_m2,
        "Total cooling [kWh/m2]": cool_kwh / total_area_m2,
        "Conditioning electricity [kWh/m2]": elec_kwh / total_area_m2,
        "Peak cooling load [W/m2]": float(cooling.max()) * 1000.0 / total_area_m2,
        "Peak conditioning electricity [W/m2]": float(electricity.max()) * 1000.0 / total_area_m2,
    }

    return total, specific


def build_cooling_excel(
    summary_folder_path: str | Path,
    excel_out_path: str | Path = "building_cooling_summary.xlsx",
    total_area_m2: float = 146318.5715,
    sep: str = ";",
) -> Path:
    """
    Build an Excel workbook with total and specific cooling indicators.

    Parameters
    ----------
    summary_folder_path : str | Path
        Folder containing summer cooling summary CSVs.
    excel_out_path : str | Path, optional
        Output Excel path.
    total_area_m2 : float, optional
        Total net floor area.
    sep : str, optional
        CSV separator.

    Returns
    -------
    Path
        Written Excel path.
    """
    summary_dir = Path(summary_folder_path)
    out_path = Path(excel_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_total = []
    rows_specific = []

    for csv_file in sorted(summary_dir.glob("*.csv")):
        df = pd.read_csv(csv_file, sep=sep)
        scenario = scenario_pretty_name(csv_file)
        total, specific = cooling_summary_indicators(
            df=df,
            scenario=scenario,
            total_area_m2=total_area_m2,
        )
        rows_total.append(total)
        rows_specific.append(specific)

    total_df = pd.DataFrame(rows_total)
    specific_df = pd.DataFrame(rows_specific)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        total_df.to_excel(writer, sheet_name="total", index=False)
        specific_df.to_excel(writer, sheet_name="specific", index=False)

    return out_path


def peak_cooling_days(
    summary_folder_path: str | Path,
    cooling_col: str = "Total cooling load [kW]",
    sep: str = ";",
) -> pd.DataFrame:
    """
    Find the peak daily cooling-load day for each scenario.

    Parameters
    ----------
    summary_folder_path : str | Path
        Folder containing summer summary CSV files.
    cooling_col : str, optional
        Cooling load column.
    sep : str, optional
        CSV separator.

    Returns
    -------
    pd.DataFrame
        Peak-day table.
    """
    summary_dir = Path(summary_folder_path)
    rows = []

    for csv_file in sorted(summary_dir.glob("*.csv")):
        df = pd.read_csv(csv_file, sep=sep)

        if "Time" not in df.columns or cooling_col not in df.columns:
            continue

        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df[cooling_col] = pd.to_numeric(df[cooling_col], errors="coerce").fillna(0.0)
        df["Date"] = df["Time"].dt.date

        daily = df.groupby("Date")[cooling_col].sum().reset_index(name="Daily cooling load [kWh]")

        if daily.empty:
            continue

        peak = daily.loc[daily["Daily cooling load [kWh]"].idxmax()]

        rows.append({
            "Scenario": scenario_pretty_name(csv_file),
            "Peak day": peak["Date"],
            "Daily cooling load [MWh]": float(peak["Daily cooling load [kWh]"]) / 1000.0,
        })

    return pd.DataFrame(rows)


def monthly_cooling_summary(
    summary_folder_path: str | Path,
    months: list[int] | None = None,
    sep: str = ";",
) -> pd.DataFrame:
    """
    Compute monthly cooling indicators for all scenarios.

    Parameters
    ----------
    summary_folder_path : str | Path
        Folder containing summer summary CSV files.
    months : list[int] | None, optional
        Optional months to keep.
    sep : str, optional
        CSV separator.

    Returns
    -------
    pd.DataFrame
        Monthly cooling summary table.
    """
    summary_dir = Path(summary_folder_path)
    rows = []

    for csv_file in sorted(summary_dir.glob("*.csv")):
        df = pd.read_csv(csv_file, sep=sep)

        if "Time" not in df.columns:
            continue

        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df["Month"] = df["Time"].dt.month

        if months is not None:
            df = df[df["Month"].isin(months)]

        for col in [
            "Total sensible load [kW]",
            "Total latent load [kW]",
            "Total cooling load [kW]",
            "ConditioningElectricity [kW]",
        ]:
            if col not in df.columns:
                df[col] = 0.0

            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        grouped = df.groupby("Month").agg({
            "Total sensible load [kW]": "sum",
            "Total latent load [kW]": "sum",
            "Total cooling load [kW]": "sum",
            "ConditioningElectricity [kW]": "sum",
        })

        for month, row in grouped.iterrows():
            rows.append({
                "Scenario": scenario_pretty_name(csv_file),
                "Month": int(month),
                "Sensible cooling [MWh]": float(row["Total sensible load [kW]"]) / 1000.0,
                "Latent cooling [MWh]": float(row["Total latent load [kW]"]) / 1000.0,
                "Total cooling [MWh]": float(row["Total cooling load [kW]"]) / 1000.0,
                "Conditioning electricity [MWh]": float(row["ConditioningElectricity [kW]"]) / 1000.0,
            })

    return pd.DataFrame(rows)


def run_postprocess_from_config(
    config: dict[str, Any],
    project_root: str | Path | None = None,
    add_electricity: bool = True,
    summarize: bool = True,
    build_excel: bool = True,
) -> dict[str, Any]:
    """
    Run configured simulation post-processing stages.

    Parameters
    ----------
    config : dict[str, Any]
        Project configuration dictionary.
    project_root : str | Path | None, optional
        Root folder used to resolve relative paths.
    add_electricity : bool, optional
        Whether to add conditioning electricity to building files.
    summarize : bool, optional
        Whether to create summer cooling summaries.
    build_excel : bool, optional
        Whether to create the cooling summary Excel workbook.

    Returns
    -------
    dict[str, Any]
        Paths and outputs produced by the post-processing stages.
    """
    root = Path.cwd() if project_root is None else Path(project_root)

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (root / path).resolve()

    paths = config.get("paths", {})
    post_cfg = config.get("simulation_postprocess", {})
    cooling_cfg = post_cfg.get("cooling", {})
    eer_cfg = post_cfg.get("eer_model", {})

    epw_dir = resolve(paths["epw_output_dir"])
    sim_dir = resolve(paths["simulation_results_dir"])
    summary_dir = resolve(paths.get("cooling_summaries_dir", sim_dir / "Cooling_summaries"))

    sensible_col = cooling_cfg.get("sensible_col", "TZ sensible load [kW]")
    latent_col = cooling_cfg.get("latent_col", "TZ latent load [kW]")
    electricity_col = cooling_cfg.get("electricity_col", "ConditioningElectricity [kW]")

    outputs = {}

    if add_electricity:
        outputs["building_files"] = add_conditioning_electricity(
            epw_folder=epw_dir,
            simulation_results_path=sim_dir,
            sensible_col=sensible_col,
            latent_col=latent_col,
            electricity_col=electricity_col,
            eer_slope=eer_cfg.get("slope", -0.2),
            eer_intercept=eer_cfg.get("intercept", 11.2),
            min_eer=eer_cfg.get("min_eer", 2.5),
            max_eer=eer_cfg.get("max_eer", 7.0),
            design_percentile=post_cfg.get("design_percentile", 99.0),
            design_rounding_kw=post_cfg.get("design_rounding_kw", 10.0),
        )

    if summarize:
        outputs["summary_files"] = summarize_summer_cooling(
            simulation_results_path=sim_dir,
            output_folder_name=summary_dir.name,
            start_hour_index=post_cfg.get("summer_start_hour_index", 120 * 24),
            end_hour_index=post_cfg.get("summer_end_hour_index", 304 * 24),
            start_time=post_cfg.get("summer_start", "2005-05-01 00:00"),
            end_time=post_cfg.get("summer_end", "2005-10-31 23:00"),
            sensible_col=sensible_col,
            latent_col=latent_col,
            electricity_col=electricity_col,
        )

    if build_excel:
        outputs["summary_excel"] = build_cooling_excel(
            summary_folder_path=summary_dir,
            excel_out_path=resolve(paths.get("cooling_summary_excel", "data/output/building_cooling_summary.xlsx")),
            total_area_m2=post_cfg.get("total_area_m2", 146318.5715),
        )

    return outputs