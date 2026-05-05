from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd


EPW_COLS = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Data Source and Uncertainty Flags",
    "Dry Bulb Temperature",
    "Dew Point Temperature",
    "Relative Humidity",
    "Atmospheric Station Pressure",
    "Extraterrestrial Horizontal Radiation",
    "Extraterrestrial Direct Normal Radiation",
    "Horizontal Infrared Radiation Intensity",
    "Global Horizontal Radiation",
    "Direct Normal Radiation",
    "Diffuse Horizontal Radiation",
    "Global Horizontal Illuminance",
    "Direct Normal Illuminance",
    "Diffuse Horizontal Illuminance",
    "Zenith Luminance",
    "Wind Direction",
    "Wind Speed",
    "Total Sky Cover",
    "Opaque Sky Cover",
    "Visibility",
    "Ceiling Height",
    "Present Weather Observation",
    "Present Weather Codes",
    "Precipitable Water",
    "Aerosol Optical Depth",
    "Snow Depth",
    "Days Since Last Snow",
    "Albedo",
    "Liquid Precipitation Depth",
    "Liquid Precipitation Quantity",
]


def read_epw(epw_path: str | Path) -> tuple[list[str], pd.DataFrame]:
    """
    Read an EPW file.

    Parameters
    ----------
    epw_path : str | Path
        Path to the EPW file.

    Returns
    -------
    tuple[list[str], pd.DataFrame]
        EPW header lines and hourly data table.

    Raises
    ------
    FileNotFoundError
        If the EPW file does not exist.
    """
    path = Path(epw_path)

    if not path.exists():
        raise FileNotFoundError(f"EPW file not found: {path}")

    header = []
    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(8):
            header.append(f.readline().rstrip("\n"))

        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split(",")]

            if len(parts) < len(EPW_COLS):
                parts += [""] * (len(EPW_COLS) - len(parts))

            rows.append(parts[:len(EPW_COLS)])

    df = pd.DataFrame(rows, columns=EPW_COLS)

    text_cols = {
        "Data Source and Uncertainty Flags",
        "Present Weather Observation",
        "Present Weather Codes",
    }

    num_cols = [c for c in EPW_COLS if c not in text_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    return header, df


def write_epw(header: list[str], df: pd.DataFrame, out_path: str | Path) -> Path:
    """
    Write an EPW file.

    Parameters
    ----------
    header : list[str]
        EPW header lines.
    df : pd.DataFrame
        EPW hourly data table.
    out_path : str | Path
        Output EPW path.

    Returns
    -------
    Path
        Written EPW path.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for line in header:
            f.write(str(line) + "\n")

        for _, row in df.iterrows():
            values = []

            for col in EPW_COLS:
                value = row[col] if col in df.columns else ""

                if pd.isna(value):
                    values.append("")
                elif isinstance(value, (np.integer, int)):
                    values.append(str(int(value)))
                elif isinstance(value, (np.floating, float)):
                    values.append(f"{float(value):.3f}")
                else:
                    values.append(str(value))

            f.write(",".join(values) + "\n")

    return path


def epw_datetime_index(df: pd.DataFrame, year_override: int | None = None) -> pd.DatetimeIndex:
    """
    Build a DatetimeIndex from EPW date and hour columns.

    Parameters
    ----------
    df : pd.DataFrame
        EPW hourly data table.
    year_override : int | None, optional
        Optional year used instead of the EPW year column.

    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex aligned to EPW hourly records.
    """
    year = year_override if year_override is not None else df["Year"].astype(int)

    dt = pd.to_datetime(
        {
            "year": year,
            "month": df["Month"].astype(int),
            "day": df["Day"].astype(int),
            "hour": (df["Hour"].astype(int) - 1).clip(lower=0) % 24,
            "minute": 0,
        },
        errors="coerce",
    )

    return pd.DatetimeIndex(dt)


def saturation_vapor_pressure_c(temp_c: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    """
    Compute saturation vapor pressure using the Magnus-Tetens approximation.

    Parameters
    ----------
    temp_c : float | np.ndarray | pd.Series
        Temperature in Celsius.

    Returns
    -------
    float | np.ndarray | pd.Series
        Saturation vapor pressure in kPa.
    """
    return 0.61094 * np.exp((17.625 * temp_c) / (temp_c + 243.04))


def dewpoint_from_temp_rh(temp_c: float, rh_pct: float) -> float:
    """
    Compute dew-point temperature from dry-bulb temperature and relative humidity.

    Parameters
    ----------
    temp_c : float
        Dry-bulb temperature in Celsius.
    rh_pct : float
        Relative humidity in percent.

    Returns
    -------
    float
        Dew-point temperature in Celsius.
    """
    rh = max(1e-6, min(100.0, float(rh_pct))) / 100.0
    es = saturation_vapor_pressure_c(float(temp_c))
    e = rh * es
    log_term = math.log(max(1e-12, e / 0.61094))

    return (243.04 * log_term) / (17.625 - log_term)


def dewpoint_series_from_temp_rh(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """
    Compute dew-point temperature from dry-bulb temperature and relative humidity series.

    Parameters
    ----------
    temp_c : pd.Series
        Dry-bulb temperature in Celsius.
    rh_pct : pd.Series
        Relative humidity in percent.

    Returns
    -------
    pd.Series
        Dew-point temperature in Celsius.
    """
    temp, rh = temp_c.align(rh_pct, join="outer")
    rh_frac = rh.clip(lower=1e-6, upper=100.0) / 100.0
    es = saturation_vapor_pressure_c(temp)
    e = rh_frac * es
    log_term = np.log((e / 0.61094).clip(lower=1e-12))

    return (243.04 * log_term) / (17.625 - log_term)


def relative_humidity_from_temp_wetbulb_pressure(
    dry_bulb_c: float,
    wet_bulb_c: float,
    pressure_pa: float,
) -> float:
    """
    Estimate relative humidity from dry-bulb temperature, wet-bulb temperature, and pressure.

    Parameters
    ----------
    dry_bulb_c : float
        Dry-bulb temperature in Celsius.
    wet_bulb_c : float
        Wet-bulb temperature in Celsius.
    pressure_pa : float
        Atmospheric pressure in Pascal.

    Returns
    -------
    float
        Estimated relative humidity in percent.
    """
    rh_est = 100.0 - 5.0 * (dry_bulb_c - wet_bulb_c)
    gamma = 0.00066 * (1 + 0.00115 * wet_bulb_c) * (pressure_pa / 1000.0)
    rh_corr = rh_est - 0.02 * (gamma - 0.66)

    return float(np.clip(rh_corr, 1.0, 100.0))


def normalize_weather_excel_columns(
    df: pd.DataFrame,
    time_col: str = "time",
    db_col: str = "db_temp",
    wb_col: str = "wb_temp",
    rh_col: str = "rh",
) -> pd.DataFrame:
    """
    Normalize representative weather Excel columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input representative weather table.
    time_col : str, optional
        Time column name.
    db_col : str, optional
        Dry-bulb temperature column name.
    wb_col : str, optional
        Wet-bulb temperature column name.
    rh_col : str, optional
        Relative humidity column name.

    Returns
    -------
    pd.DataFrame
        Normalized weather table indexed by time.

    Raises
    ------
    KeyError
        If the time column is missing.
    """
    work = df.copy()
    original_columns = list(work.columns)
    lowered = {str(c).strip().lower(): c for c in original_columns}

    rename_map = {}

    for target in [time_col, db_col, wb_col, rh_col]:
        key = target.lower()
        if key in lowered:
            rename_map[lowered[key]] = target

    work = work.rename(columns=rename_map)

    if time_col not in work.columns:
        raise KeyError(f"Missing time column: {time_col}")

    keep = [c for c in [time_col, db_col, wb_col, rh_col] if c in work.columns]
    work = work[keep].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    work = work.set_index(time_col).sort_index()

    for col in [db_col, wb_col, rh_col]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    return work


def read_weather_excel(
    excel_path: str | Path,
    time_col: str = "time",
    db_col: str = "db_temp",
    wb_col: str = "wb_temp",
    rh_col: str = "rh",
) -> pd.DataFrame:
    """
    Read representative weather data from an Excel file.

    Parameters
    ----------
    excel_path : str | Path
        Path to the Excel file.
    time_col : str, optional
        Time column name.
    db_col : str, optional
        Dry-bulb temperature column name.
    wb_col : str, optional
        Wet-bulb temperature column name.
    rh_col : str, optional
        Relative humidity column name.

    Returns
    -------
    pd.DataFrame
        Normalized weather data indexed by time.

    Raises
    ------
    FileNotFoundError
        If the Excel file does not exist.
    """
    path = Path(excel_path)

    if not path.exists():
        raise FileNotFoundError(f"Weather Excel file not found: {path}")

    df = pd.read_excel(path)

    return normalize_weather_excel_columns(
        df,
        time_col=time_col,
        db_col=db_col,
        wb_col=wb_col,
        rh_col=rh_col,
    )


def align_weather_to_epw(
    weather: pd.DataFrame,
    epw_index: pd.DWatetimeIndex,
    freq: str = "1H",
) -> pd.DataFrame:
    """
    Resample representative weather data and align it to an EPW DatetimeIndex.

    Parameters
    ----------
    weather : pd.DataFrame
        Representative weather table indexed by time.
    epw_index : pd.DatetimeIndex
        EPW DatetimeIndex.
    freq : str, optional
        Resampling frequency.

    Returns
    -------
    pd.DataFrame
        Weather data aligned to the EPW index.
    """
    hourly = weather.resample(freq).mean()

    if getattr(hourly.index, "tz", None) is not None:
        hourly.index = hourly.index.tz_localize(None)

    epw_idx = epw_index

    if getattr(epw_idx, "tz", None) is not None:
        epw_idx = epw_idx.tz_localize(None)

    return hourly.reindex(epw_idx)


def merge_weather_into_epw(
    base_df: pd.DataFrame,
    weather: pd.DataFrame,
    year_override: int | None = None,
    db_col: str = "db_temp",
    wb_col: str = "wb_temp",
    rh_col: str = "rh",
    replace_dry_bulb: bool = True,
    replace_relative_humidity: bool = True,
    recompute_dew_point: bool = True,
) -> pd.DataFrame:
    """
    Merge representative weather variables into an EPW hourly data table.

    Parameters
    ----------
    base_df : pd.DataFrame
        Base EPW hourly data table.
    weather : pd.DataFrame
        Representative weather table indexed by time.
    year_override : int | None, optional
        Optional year used for EPW datetime alignment.
    db_col : str, optional
        Dry-bulb temperature column in representative weather table.
    wb_col : str, optional
        Wet-bulb temperature column in representative weather table.
    rh_col : str, optional
        Relative humidity column in representative weather table.
    replace_dry_bulb : bool, optional
        Whether to replace EPW dry-bulb temperature.
    replace_relative_humidity : bool, optional
        Whether to replace EPW relative humidity.
    recompute_dew_point : bool, optional
        Whether to recompute EPW dew-point temperature.

    Returns
    -------
    pd.DataFrame
        Updated EPW hourly data table.
    """
    out = base_df.copy()
    epw_idx = epw_datetime_index(out, year_override=year_override)
    out.index = epw_idx

    aligned = align_weather_to_epw(weather, epw_idx)

    if replace_dry_bulb and db_col in aligned.columns:
        mask = aligned[db_col].notna()
        out.loc[mask, "Dry Bulb Temperature"] = aligned.loc[mask, db_col]

    rh_new = pd.Series(index=epw_idx, dtype=float)

    if rh_col in aligned.columns:
        rh_new = aligned[rh_col].copy()

    if db_col in aligned.columns and wb_col in aligned.columns:
        need = rh_new.isna() & aligned[db_col].notna() & aligned[wb_col].notna()

        if need.any():
            pressure = out.loc[need, "Atmospheric Station Pressure"]
            rh_new.loc[need] = [
                relative_humidity_from_temp_wetbulb_pressure(t, tw, p)
                for t, tw, p in zip(
                    aligned.loc[need, db_col],
                    aligned.loc[need, wb_col],
                    pressure,
                )
            ]

    if replace_relative_humidity:
        mask_rh = rh_new.notna()
        out.loc[mask_rh, "Relative Humidity"] = rh_new.loc[mask_rh].clip(1.0, 100.0)

    if recompute_dew_point:
        mask_dp = (
            out["Dry Bulb Temperature"].notna()
            & out["Relative Humidity"].notna()
        )

        out.loc[mask_dp, "Dew Point Temperature"] = dewpoint_series_from_temp_rh(
            out.loc[mask_dp, "Dry Bulb Temperature"],
            out.loc[mask_dp, "Relative Humidity"],
        )

    out = out.reset_index(drop=True)

    return out


def convert_weather_excel_to_epw(
    base_epw_path: str | Path,
    excel_path: str | Path,
    out_epw_path: str | Path,
    time_col: str = "time",
    db_col: str = "db_temp",
    wb_col: str = "wb_temp",
    rh_col: str = "rh",
    year_override: int | None = None,
    replace_dry_bulb: bool = True,
    replace_relative_humidity: bool = True,
    recompute_dew_point: bool = True,
) -> Path:
    """
    Convert one representative weather Excel file into an EPW file.

    Parameters
    ----------
    base_epw_path : str | Path
        Path to the base EPW file.
    excel_path : str | Path
        Path to the representative weather Excel file.
    out_epw_path : str | Path
        Output EPW path.
    time_col : str, optional
        Time column name in the Excel file.
    db_col : str, optional
        Dry-bulb temperature column name in the Excel file.
    wb_col : str, optional
        Wet-bulb temperature column name in the Excel file.
    rh_col : str, optional
        Relative humidity column name in the Excel file.
    year_override : int | None, optional
        Optional year used for EPW datetime alignment.
    replace_dry_bulb : bool, optional
        Whether to replace dry-bulb temperature.
    replace_relative_humidity : bool, optional
        Whether to replace relative humidity.
    recompute_dew_point : bool, optional
        Whether to recompute dew-point temperature.

    Returns
    -------
    Path
        Written EPW path.
    """
    header, base_df = read_epw(base_epw_path)

    weather = read_weather_excel(
        excel_path,
        time_col=time_col,
        db_col=db_col,
        wb_col=wb_col,
        rh_col=rh_col,
    )

    updated = merge_weather_into_epw(
        base_df=base_df,
        weather=weather,
        year_override=year_override,
        db_col=db_col,
        wb_col=wb_col,
        rh_col=rh_col,
        replace_dry_bulb=replace_dry_bulb,
        replace_relative_humidity=replace_relative_humidity,
        recompute_dew_point=recompute_dew_point,
    )

    return write_epw(header, updated, out_epw_path)


def batch_convert_weather_excels_to_epw(
    base_epw_path: str | Path,
    in_dir: str | Path,
    out_dir: str | Path,
    pattern: str = "*.xlsx",
    suffix: str = ".epw",
    time_col: str = "time",
    db_col: str = "db_temp",
    wb_col: str = "wb_temp",
    rh_col: str = "rh",
    year_override: int | None = None,
    replace_dry_bulb: bool = True,
    replace_relative_humidity: bool = True,
    recompute_dew_point: bool = True,
) -> list[Path]:
    """
    Convert multiple representative weather Excel files into EPW files.

    Parameters
    ----------
    base_epw_path : str | Path
        Path to the base EPW file.
    in_dir : str | Path
        Folder containing representative weather Excel files.
    out_dir : str | Path
        Folder where EPW files will be written.
    pattern : str, optional
        Glob pattern for Excel files.
    suffix : str, optional
        Output file suffix.
    time_col : str, optional
        Time column name in the Excel files.
    db_col : str, optional
        Dry-bulb temperature column name in the Excel files.
    wb_col : str, optional
        Wet-bulb temperature column name in the Excel files.
    rh_col : str, optional
        Relative humidity column name in the Excel files.
    year_override : int | None, optional
        Optional year used for EPW datetime alignment.
    replace_dry_bulb : bool, optional
        Whether to replace dry-bulb temperature.
    replace_relative_humidity : bool, optional
        Whether to replace relative humidity.
    recompute_dew_point : bool, optional
        Whether to recompute dew-point temperature.

    Returns
    -------
    list[Path]
        Written EPW paths.

    Raises
    ------
    FileNotFoundError
        If no matching Excel files are found.
    """
    input_dir = Path(in_dir)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No weather Excel files found in {input_dir} with pattern {pattern}")

    written = []

    for excel_file in files:
        out_path = output_dir / f"{excel_file.stem}{suffix}"

        written.append(
            convert_weather_excel_to_epw(
                base_epw_path=base_epw_path,
                excel_path=excel_file,
                out_epw_path=out_path,
                time_col=time_col,
                db_col=db_col,
                wb_col=wb_col,
                rh_col=rh_col,
                year_override=year_override,
                replace_dry_bulb=replace_dry_bulb,
                replace_relative_humidity=replace_relative_humidity,
                recompute_dew_point=recompute_dew_point,
            )
        )

    return written


def extract_epw_temperature_humidity(
    epw_path: str | Path,
    year_override: int | None = None,
) -> pd.DataFrame:
    """
    Extract dry-bulb temperature and relative humidity from an EPW file.

    Parameters
    ----------
    epw_path : str | Path
        Path to the EPW file.
    year_override : int | None, optional
        Optional year used for datetime construction.

    Returns
    -------
    pd.DataFrame
        DataFrame with time, dry-bulb temperature, and relative humidity.
    """
    _, df = read_epw(epw_path)
    time = epw_datetime_index(df, year_override=year_override)

    return pd.DataFrame({
        "time": time,
        "dry_bulb": df["Dry Bulb Temperature"].to_numpy(),
        "relative_humidity": df["Relative Humidity"].to_numpy(),
    })


def fill_workbook_temperature_humidity_from_epw(
    excel_path: str | Path,
    epw_folder: str | Path,
    sheet_to_epw: dict[str, str],
    time_col: str = "Time",
    temp_col: str = "External Temperature [°C]",
    rh_col: str = "External Relative Humidity [%]",
    year_override: int | None = None,
) -> Path:
    """
    Fill temperature and relative humidity columns in an Excel workbook using mapped EPW files.

    Parameters
    ----------
    excel_path : str | Path
        Path to the Excel workbook.
    epw_folder : str | Path
        Folder containing EPW files.
    sheet_to_epw : dict[str, str]
        Mapping from Excel sheet name to EPW filename.
    time_col : str, optional
        Time column in workbook sheets.
    temp_col : str, optional
        Temperature column to fill.
    rh_col : str, optional
        Relative humidity column to fill.
    year_override : int | None, optional
        Optional year used for EPW datetime construction.

    Returns
    -------
    Path
        Updated Excel workbook path.

    Raises
    ------
    FileNotFoundError
        If the Excel workbook does not exist.
    """
    workbook_path = Path(excel_path)
    epw_dir = Path(epw_folder)

    if not workbook_path.exists():
        raise FileNotFoundError(f"Excel workbook not found: {workbook_path}")

    sheets = pd.read_excel(workbook_path, sheet_name=None)
    updated = {}

    for sheet_name, df in sheets.items():
        if sheet_name not in sheet_to_epw:
            updated[sheet_name] = df
            continue

        epw_path = epw_dir / sheet_to_epw[sheet_name]

        if not epw_path.exists() or time_col not in df.columns:
            updated[sheet_name] = df
            continue

        epw_df = extract_epw_temperature_humidity(epw_path, year_override=year_override)
        epw_df["_month"] = epw_df["time"].dt.month
        epw_df["_day"] = epw_df["time"].dt.day
        epw_df["_hour"] = epw_df["time"].dt.hour + 1

        work = df.copy()
        ts = pd.to_datetime(work[time_col], errors="coerce", dayfirst=True)
        work["_month"] = ts.dt.month
        work["_day"] = ts.dt.day
        work["_hour"] = ts.dt.hour + 1

        merged = work.merge(
            epw_df[["_month", "_day", "_hour", "dry_bulb", "relative_humidity"]],
            on=["_month", "_day", "_hour"],
            how="left",
        )

        merged[temp_col] = merged["dry_bulb"]
        merged[rh_col] = merged["relative_humidity"]

        merged = merged.drop(
            columns=["_month", "_day", "_hour", "dry_bulb", "relative_humidity"],
            errors="ignore",
        )

        updated[sheet_name] = merged

    with pd.ExcelWriter(workbook_path, engine="openpyxl", mode="w") as writer:
        for sheet_name, df in updated.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return workbook_path


def summarize_epw_monthly_temperature(
    epw_paths: dict[str, str | Path],
    year_override: int | None = None,
) -> pd.DataFrame:
    """
    Summarize monthly dry-bulb temperature from multiple EPW files.

    Parameters
    ----------
    epw_paths : dict[str, str | Path]
        Dictionary mapping scenario name to EPW path.
    year_override : int | None, optional
        Optional year used for datetime construction.

    Returns
    -------
    pd.DataFrame
        Monthly temperature summary table.
    """
    rows = []

    for scenario, path in epw_paths.items():
        df = extract_epw_temperature_humidity(path, year_override=year_override)
        df["month"] = df["time"].dt.month

        monthly = df.groupby("month")["dry_bulb"].agg(["min", "mean", "max"]).reset_index()

        for _, row in monthly.iterrows():
            rows.append({
                "scenario": scenario,
                "month": int(row["month"]),
                "min_dry_bulb": float(row["min"]),
                "mean_dry_bulb": float(row["mean"]),
                "max_dry_bulb": float(row["max"]),
            })

    return pd.DataFrame(rows)


def batch_convert_from_config(config: dict[str, Any], project_root: str | Path | None = None) -> list[Path]:
    """
    Convert representative weather Excel files to EPW files using project configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Project configuration dictionary.
    project_root : str | Path | None, optional
        Root folder used to resolve relative paths.

    Returns
    -------
    list[Path]
        Written EPW paths.
    """
    paths = config.get("paths", {})
    epw_cfg = config.get("epw", {})

    root = Path.cwd() if project_root is None else Path(project_root)

    def resolve(value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (root / p).resolve()

    return batch_convert_weather_excels_to_epw(
        base_epw_path=resolve(paths["epw_base_file"]),
        in_dir=resolve(paths.get("epw_input_dir", paths["cluster_exports_dir"])),
        out_dir=resolve(paths["epw_output_dir"]),
        pattern=epw_cfg.get("excel_pattern", "*.xlsx"),
        time_col=epw_cfg.get("excel_time_col", "time"),
        db_col=epw_cfg.get("excel_db_col", "db_temp"),
        wb_col=epw_cfg.get("excel_wb_col", "wb_temp"),
        rh_col=epw_cfg.get("excel_rh_col", "rh"),
        year_override=epw_cfg.get("epw_year_override"),
        replace_dry_bulb=epw_cfg.get("replace_dry_bulb", True),
        replace_relative_humidity=epw_cfg.get("replace_relative_humidity", True),
        recompute_dew_point=epw_cfg.get("recompute_dew_point", True),
    )