from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def save_or_show(
    fig: plt.Figure,
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Save a figure or show it interactively.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure.
    out_path : str | Path | None, optional
        Output path. If None, the figure is shown.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    if out_path is None:
        plt.show()
        return None

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return path


def make_datetime_index_naive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame index to timezone-naive datetime.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime-like index.

    Returns
    -------
    pd.DataFrame
        DataFrame with timezone-naive DatetimeIndex.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)

    return out


def make_datetime_series_naive(series: pd.Series) -> pd.Series:
    """
    Convert a Series to timezone-naive datetime.

    Parameters
    ----------
    series : pd.Series
        Datetime-like Series.

    Returns
    -------
    pd.Series
        Timezone-naive datetime Series.
    """
    out = pd.to_datetime(series, errors="coerce")

    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert(None)

    return out


def make_timestamp_naive(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """
    Convert a timestamp-like value to timezone-naive Timestamp.

    Parameters
    ----------
    value : str | pd.Timestamp
        Timestamp-like value.

    Returns
    -------
    pd.Timestamp
        Timezone-naive Timestamp.
    """
    ts = pd.Timestamp(value)

    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    return ts


def read_summary_csv(path: PathLike, sep: str = ";") -> pd.DataFrame:
    """
    Read one cooling summary CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the summary CSV.
    sep : str, optional
        CSV separator.

    Returns
    -------
    pd.DataFrame
        Summary table.

    Raises
    ------
    FileNotFoundError
        If the summary CSV does not exist.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {file_path}")

    df = pd.read_csv(file_path, sep=sep)

    if "Time" in df.columns:
        df["Time"] = make_datetime_series_naive(df["Time"])

    return df


def load_summary_folder(
    summary_folder: PathLike,
    scenarios: Optional[Dict[str, str]] = None,
    sep: str = ";",
) -> Dict[str, pd.DataFrame]:
    """
    Load cooling summary CSV files from a folder.

    Parameters
    ----------
    summary_folder : str | Path
        Folder containing summary CSV files.
    scenarios : dict[str, str] | None, optional
        Mapping from scenario name to CSV filename. If None, all CSV files are loaded.
    sep : str, optional
        CSV separator.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping scenario names to summary DataFrames.
    """
    folder = Path(summary_folder)

    if scenarios is None:
        return {
            p.stem: read_summary_csv(p, sep=sep)
            for p in sorted(folder.glob("*.csv"))
        }

    return {
        scenario: read_summary_csv(folder / filename, sep=sep)
        for scenario, filename in scenarios.items()
        if (folder / filename).exists()
    }


def monthly_sum(
    df: pd.DataFrame,
    value_col: str,
    months: Optional[List[int]] = None,
) -> pd.Series:
    """
    Compute monthly sum for one summary variable.

    Parameters
    ----------
    df : pd.DataFrame
        Summary table containing Time and value columns.
    value_col : str
        Column to aggregate.
    months : list[int] | None, optional
        Optional months to keep.

    Returns
    -------
    pd.Series
        Monthly sums indexed by month.
    """
    if "Time" not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=float)

    work = df.copy()
    work["Time"] = make_datetime_series_naive(work["Time"])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work["Month"] = work["Time"].dt.month

    if months is not None:
        work = work[work["Month"].isin(months)]

    return work.groupby("Month")[value_col].sum()


def plot_monthly_total_cooling(
    summaries: Dict[str, pd.DataFrame],
    months: Optional[List[int]] = None,
    month_labels: Optional[List[str]] = None,
    value_col: str = "Total cooling load [kW]",
    unit_divisor: float = 1000.0,
    ylabel: str = "Load [MWh]",
    title: str = "Monthly Total Cooling Load",
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot grouped monthly total cooling load.

    Parameters
    ----------
    summaries : dict[str, pd.DataFrame]
        Scenario summary tables.
    months : list[int] | None, optional
        Month numbers to plot.
    month_labels : list[str] | None, optional
        Labels for plotted months.
    value_col : str, optional
        Column to plot.
    unit_divisor : float, optional
        Divisor used to convert units.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    out_path : str | Path | None, optional
        Output path. If None, the figure is shown.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    if months is None:
        months = sorted({
            int(m)
            for df in summaries.values()
            for m in monthly_sum(df, value_col).index
        })

    if month_labels is None:
        month_labels = [str(m) for m in months]

    scenario_names = list(summaries.keys())
    x = np.arange(len(months))
    width = min(0.8 / max(len(scenario_names), 1), 0.16)

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, scenario in enumerate(scenario_names):
        values = monthly_sum(summaries[scenario], value_col, months=months)
        values = values.reindex(months).fillna(0.0).to_numpy() / unit_divisor
        ax.bar(x + i * width, values, width, label=scenario)

    ax.set_xticks(x + (len(scenario_names) - 1) * width / 2)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Scenario")
    fig.tight_layout()

    return save_or_show(fig, out_path=out_path, dpi=dpi)


def plot_monthly_sensible_latent_stacked(
    summaries: Dict[str, pd.DataFrame],
    months: Optional[List[int]] = None,
    month_labels: Optional[List[str]] = None,
    sensible_col: str = "Total sensible load [kW]",
    latent_col: str = "Total latent load [kW]",
    unit_divisor: float = 1000.0,
    ylabel: str = "Load [MWh]",
    title: str = "Monthly Sensible and Latent Cooling Load",
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot grouped monthly stacked sensible and latent cooling load.

    Parameters
    ----------
    summaries : dict[str, pd.DataFrame]
        Scenario summary tables.
    months : list[int] | None, optional
        Month numbers to plot.
    month_labels : list[str] | None, optional
        Labels for plotted months.
    sensible_col : str, optional
        Sensible load column.
    latent_col : str, optional
        Latent load column.
    unit_divisor : float, optional
        Divisor used to convert units.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    out_path : str | Path | None, optional
        Output path. If None, the figure is shown.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    if months is None:
        months = sorted({
            int(m)
            for df in summaries.values()
            for m in monthly_sum(df, sensible_col).index
        })

    if month_labels is None:
        month_labels = [str(m) for m in months]

    scenario_names = list(summaries.keys())
    x = np.arange(len(months))
    width = min(0.8 / max(len(scenario_names), 1), 0.16)

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, scenario in enumerate(scenario_names):
        sens = monthly_sum(summaries[scenario], sensible_col, months=months)
        lat = monthly_sum(summaries[scenario], latent_col, months=months)

        sens_values = sens.reindex(months).fillna(0.0).to_numpy() / unit_divisor
        lat_values = lat.reindex(months).fillna(0.0).to_numpy() / unit_divisor
        xpos = x + i * width

        ax.bar(xpos, lat_values, width, alpha=0.5)
        ax.bar(xpos, sens_values, width, bottom=lat_values, label=scenario)

    ax.set_xticks(x + (len(scenario_names) - 1) * width / 2)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Scenario")
    fig.tight_layout()

    return save_or_show(fig, out_path=out_path, dpi=dpi)


def peak_day_table(
    summaries: Dict[str, pd.DataFrame],
    value_col: str = "Total cooling load [kW]",
    unit_divisor: float = 1000.0,
) -> pd.DataFrame:
    """
    Find the peak daily load day for each scenario.

    Parameters
    ----------
    summaries : dict[str, pd.DataFrame]
        Scenario summary tables.
    value_col : str, optional
        Load column.
    unit_divisor : float, optional
        Divisor used to convert daily sum units.

    Returns
    -------
    pd.DataFrame
        Peak day table.
    """
    rows = []

    for scenario, df in summaries.items():
        if "Time" not in df.columns or value_col not in df.columns:
            continue

        work = df.copy()
        work["Time"] = make_datetime_series_naive(work["Time"])
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
        work["Date"] = work["Time"].dt.date

        daily = work.groupby("Date")[value_col].sum()

        if daily.empty:
            continue

        peak_day = daily.idxmax()

        rows.append({
            "Scenario": scenario,
            "Peak day": peak_day,
            "Daily load": float(daily.loc[peak_day]) / unit_divisor,
        })

    return pd.DataFrame(rows)


def plot_peak_day_table(
    summaries: Dict[str, pd.DataFrame],
    value_col: str = "Total cooling load [kW]",
    unit_divisor: float = 1000.0,
    ylabel: str = "Daily Load [MWh]",
    title: str = "Peak Cooling Day by Scenario",
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot peak daily load for each scenario.

    Parameters
    ----------
    summaries : dict[str, pd.DataFrame]
        Scenario summary tables.
    value_col : str, optional
        Load column.
    unit_divisor : float, optional
        Divisor used to convert daily sum units.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    out_path : str | Path | None, optional
        Output path. If None, the figure is shown.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    table = peak_day_table(
        summaries=summaries,
        value_col=value_col,
        unit_divisor=unit_divisor,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    if not table.empty:
        ax.bar(table["Scenario"], table["Daily load"])
        ax.set_xticklabels(table["Scenario"], rotation=45, ha="right")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    return save_or_show(fig, out_path=out_path, dpi=dpi)


def plot_target_day_profile(
    summaries: Dict[str, pd.DataFrame],
    target_day: Union[str, pd.Timestamp],
    value_col: str = "ConditioningElectricity [kW]",
    ylabel: str = "Power [kW]",
    title: Optional[str] = None,
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot hourly profiles for a selected day.

    Parameters
    ----------
    summaries : dict[str, pd.DataFrame]
        Scenario summary tables.
    target_day : str | pd.Timestamp
        Target date.
    value_col : str, optional
        Column to plot.
    ylabel : str, optional
        Y-axis label.
    title : str | None, optional
        Plot title.
    out_path : str | Path | None, optional
        Output path. If None, the figure is shown.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    day = make_timestamp_naive(target_day).date()
    fig, ax = plt.subplots(figsize=(12, 5))

    for scenario, df in summaries.items():
        if "Time" not in df.columns or value_col not in df.columns:
            continue

        work = df.copy()
        work["Time"] = make_datetime_series_naive(work["Time"])
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
        selected = work[work["Time"].dt.date == day]

        if selected.empty:
            continue

        ax.plot(selected["Time"].dt.hour, selected[value_col], marker="o", label=scenario)

    ax.set_xlabel("Hour")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{value_col} on {day}")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(title="Scenario")
    fig.tight_layout()

    return save_or_show(fig, out_path=out_path, dpi=dpi)


def plot_annual_or_summer_duration_curve(
    summaries: Dict[str, pd.DataFrame],
    value_col: str = "Total cooling load [kW]",
    ylabel: str = "Load [kW]",
    title: str = "Duration Curve",
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot sorted descending duration curves for all scenarios.

    Parameters
    ----------
    summaries : dict[str, pd.DataFrame]
        Scenario summary tables.
    value_col : str, optional
        Column to plot.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    out_path : str | Path | None, optional
        Output path. If None, the figure is shown.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for scenario, df in summaries.items():
        if value_col not in df.columns:
            continue

        values = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).to_numpy()
        values = np.sort(values)[::-1]
        ax.plot(np.arange(1, len(values) + 1), values, label=scenario)

    ax.set_xlabel("Hour rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Scenario")
    fig.tight_layout()

    return save_or_show(fig, out_path=out_path, dpi=dpi)


def plot_monthly_electricity(
    summaries: Dict[str, pd.DataFrame],
    months: Optional[List[int]] = None,
    month_labels: Optional[List[str]] = None,
    value_col: str = "ConditioningElectricity [kW]",
    unit_divisor: float = 1000.0,
    ylabel: str = "Electricity [MWh]",
    title: str = "Monthly Conditioning Electricity",
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot grouped monthly conditioning electricity.

    Parameters
    ----------
    summaries : dict[str, pd.DataFrame]
        Scenario summary tables.
    months : list[int] | None, optional
        Month numbers to plot.
    month_labels : list[str] | None, optional
        Labels for plotted months.
    value_col : str, optional
        Electricity column.
    unit_divisor : float, optional
        Divisor used to convert units.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Plot title.
    out_path : str | Path | None, optional
        Output path. If None, the figure is shown.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    return plot_monthly_total_cooling(
        summaries=summaries,
        months=months,
        month_labels=month_labels,
        value_col=value_col,
        unit_divisor=unit_divisor,
        ylabel=ylabel,
        title=title,
        out_path=out_path,
        dpi=dpi,
    )


def load_cluster_labels(labels_path: PathLike) -> pd.Series:
    """
    Load sensor cluster labels.

    Parameters
    ----------
    labels_path : str | Path
        Path to sensor_cluster_labels.csv.

    Returns
    -------
    pd.Series
        Cluster labels indexed by sensor name.
    """
    labels = pd.read_csv(labels_path, index_col=0)["cluster"]
    labels.index = labels.index.astype(str)

    return labels


def load_representative_file(path: PathLike) -> pd.DataFrame:
    """
    Load one representative weather Excel file.

    Parameters
    ----------
    path : str | Path
        Path to representative Excel file.

    Returns
    -------
    pd.DataFrame
        Representative weather table.
    """
    df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "time" in df.columns:
        df["time"] = make_datetime_series_naive(df["time"])

    if "db_temp" in df.columns:
        df["db_temp"] = pd.to_numeric(df["db_temp"], errors="coerce")

    return df


def normalize_cluster_id(cluster_id: int, cluster_labels: pd.Series) -> int:
    """
    Normalize cluster id against available labels.

    Parameters
    ----------
    cluster_id : int
        Input cluster id.
    cluster_labels : pd.Series
        Cluster labels.

    Returns
    -------
    int
        Existing cluster id.

    Raises
    ------
    ValueError
        If no matching cluster id exists.
    """
    existing = set([int(x) for x in cluster_labels.dropna().unique()])

    if int(cluster_id) in existing:
        return int(cluster_id)

    if int(cluster_id) - 1 in existing:
        return int(cluster_id) - 1

    if int(cluster_id) + 1 in existing:
        return int(cluster_id) + 1

    raise ValueError(f"Cluster {cluster_id} not found in cluster labels.")


def find_cluster_peak_cooling_day(
    summaries: Dict[str, pd.DataFrame],
    cluster_id: int,
    value_col: str = "Total cooling load [kW]",
) -> pd.Timestamp:
    """
    Find the peak cooling day for one cluster from cooling summary files.

    Parameters
    ----------
    summaries : dict[str, pd.DataFrame]
        Scenario summary tables.
    cluster_id : int
        Cluster id. Accepts zero-based or one-based cluster ids.
    value_col : str, optional
        Cooling load column.

    Returns
    -------
    pd.Timestamp
        Peak cooling day.
    """
    possible_numbers = sorted(set([cluster_id, cluster_id + 1]))

    selected_name = None

    for number in possible_numbers:
        token = f"cluster_{number}"

        for name in summaries:
            if token in str(name).lower():
                selected_name = name
                break

        if selected_name is not None:
            break

    if selected_name is None:
        raise KeyError(f"No cooling summary found for cluster {cluster_id}.")

    df = summaries[selected_name].copy()

    if "Time" not in df.columns:
        raise KeyError(f"Cooling summary for {selected_name} has no Time column.")

    if value_col not in df.columns:
        raise KeyError(f"Cooling summary for {selected_name} has no {value_col} column.")

    df["Time"] = make_datetime_series_naive(df["Time"])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["Time"])
    df["Date"] = df["Time"].dt.date

    daily = df.groupby("Date")[value_col].sum()

    if daily.empty:
        raise ValueError(f"No valid cooling data found for cluster {cluster_id}.")

    return pd.Timestamp(daily.idxmax())


def map_day_to_data_year(day: pd.Timestamp, data_index: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Map a month-day timestamp to the dominant year used by the sensor data.

    Parameters
    ----------
    day : pd.Timestamp
        Source day, usually from cooling summary.
    data_index : pd.DatetimeIndex
        Sensor data datetime index.

    Returns
    -------
    pd.Timestamp
        Same month and day using the sensor data year.
    """
    idx = pd.to_datetime(data_index, errors="coerce")
    idx = idx[~idx.isna()]

    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)

    years = pd.Series(idx.year).dropna()

    if years.empty:
        return pd.Timestamp(day)

    target_year = int(years.mode().iloc[0])
    day = pd.Timestamp(day)

    return pd.Timestamp(year=target_year, month=day.month, day=day.day)


def monday_of_week(day: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """
    Return the Monday of the week containing a selected day.

    Parameters
    ----------
    day : str | pd.Timestamp
        Selected day.

    Returns
    -------
    pd.Timestamp
        Monday timestamp.
    """
    ts = make_timestamp_naive(day).normalize()

    return ts - pd.Timedelta(days=ts.weekday())


def select_representative_for_period(
    representative_df: pd.DataFrame,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    time_col: str = "time",
    value_col: str = "db_temp",
) -> pd.DataFrame:
    """
    Select representative profile values for a time period.

    Parameters
    ----------
    representative_df : pd.DataFrame
        Representative weather table.
    start : str | pd.Timestamp
        Start timestamp.
    end : str | pd.Timestamp
        End timestamp.
    time_col : str, optional
        Time column.
    value_col : str, optional
        Value column.

    Returns
    -------
    pd.DataFrame
        Representative data for the selected period.
    """
    if representative_df is None or representative_df.empty:
        return pd.DataFrame(columns=[time_col, value_col])

    if time_col not in representative_df.columns or value_col not in representative_df.columns:
        return pd.DataFrame(columns=[time_col, value_col])

    start_ts = make_timestamp_naive(start)
    end_ts = make_timestamp_naive(end)

    rep = representative_df.copy()
    rep[time_col] = make_datetime_series_naive(rep[time_col])
    rep[value_col] = pd.to_numeric(rep[value_col], errors="coerce")
    rep = rep.dropna(subset=[time_col, value_col])

    return rep.loc[
        (rep[time_col] >= start_ts) & (rep[time_col] <= end_ts),
        [time_col, value_col],
    ].copy()


def plot_cluster_sensor_timeseries(
    dry_bulb: pd.DataFrame,
    cluster_labels: pd.Series,
    cluster_id: int,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    representative_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    ylabel: str = "Dry-bulb temperature [°C]",
    sensor_alpha: float = 0.12,
    sensor_lw: float = 0.7,
    representative_lw: float = 2.8,
    mean_lw: float = 2.2,
    show_cluster_mean: bool = True,
    fail_on_empty: bool = True,
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot sensor time series for one cluster and overlay representative and mean profiles.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table with sensors as columns.
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    cluster_id : int
        Cluster id to plot.
    start : str | pd.Timestamp
        Start timestamp.
    end : str | pd.Timestamp
        End timestamp.
    representative_df : pd.DataFrame | None, optional
        Representative profile table with time and db_temp columns.
    title : str | None, optional
        Figure title.
    ylabel : str, optional
        Y-axis label.
    sensor_alpha : float, optional
        Transparency of individual sensor lines.
    sensor_lw : float, optional
        Line width of individual sensor lines.
    representative_lw : float, optional
        Line width of representative profile.
    mean_lw : float, optional
        Line width of cluster mean.
    show_cluster_mean : bool, optional
        Whether to plot the cluster mean.
    fail_on_empty : bool, optional
        Whether to raise an error when the selected period has no data.
    out_path : str | Path | None, optional
        Output path. If None, the figure is shown.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    start_ts = make_timestamp_naive(start)
    end_ts = make_timestamp_naive(end)

    labels = cluster_labels.copy()
    labels.index = labels.index.astype(str)

    normalized_cluster_id = normalize_cluster_id(cluster_id, labels)
    members = labels[labels == normalized_cluster_id].index

    db = make_datetime_index_naive(dry_bulb)
    db.columns = db.columns.astype(str)

    members = members.intersection(db.columns)

    if len(members) == 0:
        raise ValueError(f"No sensors found for cluster {cluster_id}.")

    selected = db.loc[(db.index >= start_ts) & (db.index <= end_ts), members]

    if selected.empty and fail_on_empty:
        raise ValueError(
            f"No sensor data found between {start_ts} and {end_ts}. "
            f"Available sensor range is {db.index.min()} to {db.index.max()}."
        )

    fig, ax = plt.subplots(figsize=(13, 5))

    for sensor in members:
        if sensor in selected.columns:
            ax.plot(selected.index, selected[sensor], alpha=sensor_alpha, lw=sensor_lw)

    if show_cluster_mean and not selected.empty:
        ax.plot(selected.index, selected.mean(axis=1), lw=mean_lw, label="Cluster mean")

    if representative_df is not None and not representative_df.empty:
        rep = select_representative_for_period(
            representative_df=representative_df,
            start=start_ts,
            end=end_ts,
            time_col="time",
            value_col="db_temp",
        )

        if not rep.empty:
            ax.plot(rep["time"], rep["db_temp"], lw=representative_lw, label="Representative")

    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Cluster {normalized_cluster_id + 1} sensor time series")
    ax.legend()
    fig.tight_layout()

    return save_or_show(fig, out_path=out_path, dpi=dpi)


def plot_cluster_daily_peak_timeseries(
    dry_bulb: pd.DataFrame,
    cluster_labels: pd.Series,
    summaries: Dict[str, pd.DataFrame],
    cluster_id: int,
    representative_df: Optional[pd.DataFrame] = None,
    value_col: str = "Total cooling load [kW]",
    out_dir: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot sensor and representative time series for the peak cooling day of one cluster.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    summaries : dict[str, pd.DataFrame]
        Cooling summary tables.
    cluster_id : int
        Cluster id.
    representative_df : pd.DataFrame | None, optional
        Representative profile table.
    value_col : str, optional
        Cooling load column.
    out_dir : str | Path | None, optional
        Output directory.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    normalized_cluster_id = normalize_cluster_id(cluster_id, cluster_labels)
    peak_day = find_cluster_peak_cooling_day(
        summaries=summaries,
        cluster_id=normalized_cluster_id,
        value_col=value_col,
    )

    mapped_day = map_day_to_data_year(peak_day, dry_bulb.index)

    start = mapped_day.normalize()
    end = start + pd.Timedelta(hours=23, minutes=59)

    out_path = None

    if out_dir is not None:
        out_path = Path(out_dir) / f"cluster_{normalized_cluster_id + 1}_daily_peak_timeseries.png"

    return plot_cluster_sensor_timeseries(
        dry_bulb=dry_bulb,
        cluster_labels=cluster_labels,
        cluster_id=normalized_cluster_id,
        start=start,
        end=end,
        representative_df=representative_df,
        title=f"Cluster {normalized_cluster_id + 1}: peak cooling day sensor time series ({mapped_day.date()})",
        out_path=out_path,
        dpi=dpi,
    )


def plot_cluster_weekly_peak_timeseries(
    dry_bulb: pd.DataFrame,
    cluster_labels: pd.Series,
    summaries: Dict[str, pd.DataFrame],
    cluster_id: int,
    representative_df: Optional[pd.DataFrame] = None,
    value_col: str = "Total cooling load [kW]",
    out_dir: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot sensor and representative time series for the week containing the peak cooling day.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    summaries : dict[str, pd.DataFrame]
        Cooling summary tables.
    cluster_id : int
        Cluster id.
    representative_df : pd.DataFrame | None, optional
        Representative profile table.
    value_col : str, optional
        Cooling load column.
    out_dir : str | Path | None, optional
        Output directory.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    normalized_cluster_id = normalize_cluster_id(cluster_id, cluster_labels)
    peak_day = find_cluster_peak_cooling_day(
        summaries=summaries,
        cluster_id=normalized_cluster_id,
        value_col=value_col,
    )

    mapped_day = map_day_to_data_year(peak_day, dry_bulb.index)

    start = monday_of_week(mapped_day)
    end = start + pd.Timedelta(days=7) - pd.Timedelta(minutes=1)

    out_path = None

    if out_dir is not None:
        out_path = Path(out_dir) / f"cluster_{normalized_cluster_id + 1}_weekly_peak_timeseries.png"

    return plot_cluster_sensor_timeseries(
        dry_bulb=dry_bulb,
        cluster_labels=cluster_labels,
        cluster_id=normalized_cluster_id,
        start=start,
        end=end,
        representative_df=representative_df,
        title=f"Cluster {normalized_cluster_id + 1}: week containing peak cooling day ({start.date()} to {end.date()})",
        out_path=out_path,
        dpi=dpi,
    )


def plot_cluster_summer_timeseries(
    dry_bulb: pd.DataFrame,
    cluster_labels: pd.Series,
    cluster_id: int,
    representative_df: Optional[pd.DataFrame] = None,
    summer_months: Optional[List[int]] = None,
    out_dir: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot sensor, representative, and mean time series for the whole summer period.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    cluster_id : int
        Cluster id.
    representative_df : pd.DataFrame | None, optional
        Representative profile table.
    summer_months : list[int] | None, optional
        Months included in summer.
    out_dir : str | Path | None, optional
        Output directory.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    if summer_months is None:
        summer_months = [5, 6, 7, 8, 9, 10]

    normalized_cluster_id = normalize_cluster_id(cluster_id, cluster_labels)

    db = make_datetime_index_naive(dry_bulb)
    db = db[db.index.month.isin(summer_months)]

    if db.empty:
        raise ValueError("No dry-bulb data found for the selected summer months.")

    start = db.index.min()
    end = db.index.max()

    out_path = None

    if out_dir is not None:
        out_path = Path(out_dir) / f"cluster_{normalized_cluster_id + 1}_summer_timeseries.png"

    return plot_cluster_sensor_timeseries(
        dry_bulb=db,
        cluster_labels=cluster_labels,
        cluster_id=normalized_cluster_id,
        start=start,
        end=end,
        representative_df=representative_df,
        title=f"Cluster {normalized_cluster_id + 1}: summer sensor time series",
        sensor_alpha=0.05,
        sensor_lw=0.5,
        representative_lw=2.5,
        mean_lw=2.0,
        show_cluster_mean=True,
        out_path=out_path,
        dpi=dpi,
    )


def find_representative_for_cluster(
    representative_folder: PathLike,
    cluster_id: int,
    pattern_template: str = "cluster_{cluster_number}_representatives_*.xlsx",
) -> Optional[pd.DataFrame]:
    """
    Find and load representative Excel file for one cluster.

    Parameters
    ----------
    representative_folder : str | Path
        Folder containing representative Excel files.
    cluster_id : int
        Zero-based cluster id.
    pattern_template : str, optional
        Representative file pattern.

    Returns
    -------
    pd.DataFrame | None
        Representative DataFrame, or None if not found.
    """
    rep_dir = Path(representative_folder)
    cluster_number = int(cluster_id) + 1
    pattern = pattern_template.format(cluster_number=cluster_number)
    files = sorted(rep_dir.glob(pattern))

    if not files:
        return None

    return load_representative_file(files[0])


def plot_all_cluster_sensor_timeseries(
    dry_bulb: pd.DataFrame,
    cluster_labels: pd.Series,
    summaries: Dict[str, pd.DataFrame],
    representative_folder: PathLike,
    out_dir: PathLike,
    representative_pattern: str = "cluster_{cluster_number}_representatives_*.xlsx",
    value_col: str = "Total cooling load [kW]",
    summer_months: Optional[List[int]] = None,
    dpi: int = 300,
) -> List[Optional[Path]]:
    """
    Plot daily, weekly, and summer sensor time series for all clusters.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    summaries : dict[str, pd.DataFrame]
        Cooling summary tables.
    representative_folder : str | Path
        Folder containing representative Excel files.
    out_dir : str | Path
        Output figure folder.
    representative_pattern : str, optional
        Representative file pattern.
    value_col : str, optional
        Cooling load column.
    summer_months : list[int] | None, optional
        Months included in the summer plot.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    list[Path | None]
        Written figure paths.
    """
    fig_dir = Path(out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    written = []
    labels = cluster_labels.dropna()
    cluster_ids = sorted([int(c) for c in labels.unique()])

    for cluster_id in cluster_ids:
        representative_df = find_representative_for_cluster(
            representative_folder=representative_folder,
            cluster_id=cluster_id,
            pattern_template=representative_pattern,
        )

        written.append(
            plot_cluster_daily_peak_timeseries(
                dry_bulb=dry_bulb,
                cluster_labels=cluster_labels,
                summaries=summaries,
                cluster_id=cluster_id,
                representative_df=representative_df,
                value_col=value_col,
                out_dir=fig_dir,
                dpi=dpi,
            )
        )

        written.append(
            plot_cluster_weekly_peak_timeseries(
                dry_bulb=dry_bulb,
                cluster_labels=cluster_labels,
                summaries=summaries,
                cluster_id=cluster_id,
                representative_df=representative_df,
                value_col=value_col,
                out_dir=fig_dir,
                dpi=dpi,
            )
        )

        written.append(
            plot_cluster_summer_timeseries(
                dry_bulb=dry_bulb,
                cluster_labels=cluster_labels,
                cluster_id=cluster_id,
                representative_df=representative_df,
                summer_months=summer_months,
                out_dir=fig_dir,
                dpi=dpi,
            )
        )

    return written


def plot_temperature_boxplot_by_month(
    temperature_profiles: Dict[str, pd.DataFrame],
    time_col: str = "time",
    value_col: str = "db_temp",
    months: Optional[List[int]] = None,
    title: str = "Monthly Temperature Distribution",
    ylabel: str = "Temperature [°C]",
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot monthly temperature boxplots for representative profiles.

    Parameters
    ----------
    temperature_profiles : dict[str, pd.DataFrame]
        Dictionary of representative weather profile tables.
    time_col : str, optional
        Time column.
    value_col : str, optional
        Temperature column.
    months : list[int] | None, optional
        Months to keep.
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.
    out_path : str | Path | None, optional
        Output path. If None, the figure is shown.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    Path | None
        Saved figure path, or None if shown interactively.
    """
    data = []
    labels = []

    for profile, df in temperature_profiles.items():
        if time_col not in df.columns or value_col not in df.columns:
            continue

        work = df.copy()
        work[time_col] = make_datetime_series_naive(work[time_col])
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[time_col, value_col])
        work["Month"] = work[time_col].dt.month

        selected_months = months or sorted(work["Month"].dropna().unique())

        for month in selected_months:
            values = work.loc[work["Month"] == month, value_col].dropna()

            if values.empty:
                continue

            data.append(values.to_numpy())
            labels.append(f"{month}\n{profile}")

    fig, ax = plt.subplots(figsize=(max(12, len(data) * 0.45), 6))

    if data:
        ax.boxplot(data, labels=labels, showfliers=False)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()

    return save_or_show(fig, out_path=out_path, dpi=dpi)


def plot_cluster_daily_profiles(
    profiles: Dict[int, Dict[str, pd.DataFrame]],
    out_dir: Optional[PathLike] = None,
    dpi: int = 300,
) -> List[Optional[Path]]:
    """
    Plot dry-bulb and wet-bulb mean daily profiles for each cluster.

    Parameters
    ----------
    profiles : dict[int, dict[str, pd.DataFrame]]
        Cluster profile dictionary.
    out_dir : str | Path | None, optional
        Output folder. If None, figures are shown.
    dpi : int, optional
        Figure resolution.

    Returns
    -------
    list[Path | None]
        Saved figure paths, or None entries for shown figures.
    """
    written = []

    for cluster_id, data in profiles.items():
        db = data.get("dry_bulb")
        wb = data.get("wet_bulb")

        if db is None or wb is None:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))

        for sensor in db.columns:
            ax.plot(db.index, db[sensor], alpha=0.35, lw=0.7)

        for sensor in wb.columns:
            ax.plot(wb.index, wb[sensor], alpha=0.35, lw=0.7)

        if len(db.columns) > 0:
            ax.plot(db.index, db.mean(axis=1), lw=2, label="DB mean")

        if len(wb.columns) > 0:
            ax.plot(wb.index, wb.mean(axis=1), lw=2, label="WB mean")

        ax.set_title(f"Cluster {int(cluster_id) + 1}")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Temperature [°C]")
        ax.set_xticks(np.arange(0, 25, 4))
        ax.legend()
        fig.tight_layout()

        if out_dir is None:
            written.append(save_or_show(fig, None, dpi=dpi))
        else:
            out_path = Path(out_dir) / f"cluster_{int(cluster_id) + 1}_daily_profile.png"
            written.append(save_or_show(fig, out_path=out_path, dpi=dpi))

    return written


def run_plots_from_config(
    config: Dict[str, Any],
    project_root: Optional[PathLike] = None,
) -> Dict[str, Optional[Path]]:
    """
    Run standard plots using project configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Project configuration dictionary.
    project_root : str | Path | None, optional
        Root folder used to resolve relative paths.

    Returns
    -------
    dict[str, Path | None]
        Written plot paths.
    """
    root = Path.cwd() if project_root is None else Path(project_root)

    def resolve(value: PathLike) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (root / path).resolve()

    paths = config.get("paths", {})
    plot_cfg = config.get("plotting", {})

    summary_folder = resolve(paths["cooling_summaries_dir"])
    figures_dir = resolve(paths.get("figures_dir", "data/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenarios = plot_cfg.get("scenarios")
    months = plot_cfg.get("months", [5, 6, 7, 8, 9, 10])
    month_labels = plot_cfg.get("month_labels", [str(m) for m in months])

    summaries = load_summary_folder(
        summary_folder=summary_folder,
        scenarios=scenarios,
        sep=plot_cfg.get("sep", ";"),
    )

    outputs = {}

    outputs["monthly_total_cooling"] = plot_monthly_total_cooling(
        summaries=summaries,
        months=months,
        month_labels=month_labels,
        out_path=figures_dir / "monthly_total_cooling.png",
    )

    outputs["monthly_sensible_latent"] = plot_monthly_sensible_latent_stacked(
        summaries=summaries,
        months=months,
        month_labels=month_labels,
        out_path=figures_dir / "monthly_sensible_latent_cooling.png",
    )

    outputs["monthly_electricity"] = plot_monthly_electricity(
        summaries=summaries,
        months=months,
        month_labels=month_labels,
        out_path=figures_dir / "monthly_conditioning_electricity.png",
    )

    outputs["duration_curve"] = plot_annual_or_summer_duration_curve(
        summaries=summaries,
        out_path=figures_dir / "cooling_duration_curve.png",
    )

    target_day = plot_cfg.get("target_day")

    if target_day is not None:
        outputs["target_day_electricity"] = plot_target_day_profile(
            summaries=summaries,
            target_day=target_day,
            out_path=figures_dir / f"conditioning_electricity_{target_day}.png",
        )

    outputs["peak_day"] = plot_peak_day_table(
        summaries=summaries,
        out_path=figures_dir / "peak_day_cooling.png",
    )

    return outputs