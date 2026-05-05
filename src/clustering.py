from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


ClusteringMethod = Literal["kmeans_features", "euclidean_ts", "softdtw", "kmeans_flat"]


@dataclass
class ClusteringResult:
    """
    Store clustering outputs.

    Parameters
    ----------
    labels : pd.Series
        Cluster label for each sensor.
    features : pd.DataFrame | None
        Feature matrix used for feature-based clustering.
    scaled_features : np.ndarray | None
        Scaled feature matrix.
    time_series_array : np.ndarray | None
        Time-series array with shape (n_sensors, n_timesteps, 1).
    sensor_names : list[str]
        Sensor names used in the clustering.
    model : object
        Fitted clustering model.
    kpis : dict[str, float]
        Clustering quality indicators.
    """
    labels: pd.Series
    features: pd.DataFrame | None
    scaled_features: np.ndarray | None
    time_series_array: np.ndarray | None
    sensor_names: list[str]
    model: object
    kpis: dict[str, float]


def build_wide_tables(sensor_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build wide temperature and humidity tables from a dictionary of sensor DataFrames.

    Parameters
    ----------
    sensor_data : dict[str, pd.DataFrame]
        Dictionary where keys are sensor names and values are DataFrames.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Temperature and humidity wide tables with timestamps as index and sensors as columns.
    """
    temp_series = []
    hum_series = []

    for name, df in sensor_data.items():
        work = df.copy()

        if "index_new" in work.columns:
            idx = pd.to_datetime(work["index_new"], errors="coerce")
        else:
            idx = pd.to_datetime(work.index, errors="coerce")

        work.index = idx
        work = work[~work.index.isna()]
        work = work.sort_index()
        work = work[~work.index.duplicated(keep="last")]

        if "Temperature" in work.columns:
            temp_series.append(pd.to_numeric(work["Temperature"], errors="coerce").rename(name))

        if "Humidity" in work.columns:
            hum_series.append(pd.to_numeric(work["Humidity"], errors="coerce").rename(name))

    df_temperature = pd.concat(temp_series, axis=1, join="outer").sort_index() if temp_series else pd.DataFrame()
    df_humidity = pd.concat(hum_series, axis=1, join="outer").sort_index() if hum_series else pd.DataFrame()

    return df_temperature, df_humidity


def resample_with_time_interpolation(
    df: pd.DataFrame,
    freq: str = "10min",
    max_gap: str = "30min",
) -> pd.DataFrame:
    """
    Resample a wide sensor table using time interpolation with a maximum allowed source gap.

    Parameters
    ----------
    df : pd.DataFrame
        Wide table with DatetimeIndex and sensors as columns.
    freq : str, optional
        Target resampling frequency.
    max_gap : str, optional
        Maximum distance to a real observation allowed for interpolation.

    Returns
    -------
    pd.DataFrame
        Resampled and interpolated table.
    """
    if df.empty:
        return df

    start = df.index.min().floor(freq)
    end = df.index.max().ceil(freq)
    target_idx = pd.date_range(start, end, freq=freq, tz=getattr(df.index, "tz", None))
    out = pd.DataFrame(index=target_idx)
    max_gap_ns = pd.Timedelta(max_gap).value

    for col in df.columns:
        s = df[col].astype(float)

        if s.dropna().empty:
            out[col] = np.nan
            continue

        valid = s.dropna()

        union_idx = valid.index.union(target_idx)
        s_union = valid.reindex(union_idx).interpolate(method="time", limit_direction="both")
        s_interp = s_union.reindex(target_idx)

        t_valid = valid.index.view("int64")
        t_target = target_idx.view("int64")

        pos = np.searchsorted(t_valid, t_target, side="left")
        left = np.clip(pos - 1, 0, len(t_valid) - 1)
        right = np.clip(pos, 0, len(t_valid) - 1)

        d_left = np.abs(t_target - t_valid[left])
        d_right = np.abs(t_valid[right] - t_target)
        d_min = np.minimum(d_left, d_right)

        s_interp[d_min > max_gap_ns] = np.nan
        out[col] = s_interp

    return out


def wet_bulb_temperature(dry_bulb: pd.DataFrame, relative_humidity: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate wet-bulb temperature using the Stull approximation.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table in Celsius.
    relative_humidity : pd.DataFrame
        Relative humidity table in percent.

    Returns
    -------
    pd.DataFrame
        Wet-bulb temperature table in Celsius.
    """
    dry_bulb, relative_humidity = dry_bulb.align(relative_humidity, join="outer")
    t = dry_bulb.astype(float)
    rh = relative_humidity.astype(float)

    tw = (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )

    return tw.where(~(t.isna() | rh.isna()))


def to_timezone_index(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    """
    Convert or localize a DatetimeIndex to a target timezone.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Input datetime index.
    timezone : str
        IANA timezone name.

    Returns
    -------
    pd.DatetimeIndex
        Timezone-aware datetime index.
    """
    if index.tz is not None:
        return index.tz_convert(timezone)

    try:
        return index.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        try:
            return index.tz_localize(timezone, ambiguous="infer", nonexistent="shift_backward")
        except Exception:
            return index.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")


def compute_sun_times(
    dates: np.ndarray,
    city_name: str,
    country: str,
    timezone: str,
    latitude: float,
    longitude: float,
) -> dict:
    """
    Compute sunrise and sunset times for a list of dates.

    Parameters
    ----------
    dates : np.ndarray
        Array of dates.
    city_name : str
        City name.
    country : str
        Country name.
    timezone : str
        IANA timezone name.
    latitude : float
        Latitude.
    longitude : float
        Longitude.

    Returns
    -------
    dict
        Dictionary mapping each date to sunrise and sunset timestamps.
    """
    city = LocationInfo(city_name, country, timezone, latitude, longitude)

    return {
        d: (sun(city.observer, date=d, tzinfo=city.timezone)["sunrise"],
            sun(city.observer, date=d, tzinfo=city.timezone)["sunset"])
        for d in dates
    }


def isolate_daylight(
    temperature: pd.DataFrame,
    humidity: pd.DataFrame | None = None,
    city_name: str = "Padua",
    country: str = "Italy",
    timezone: str = "Europe/Rome",
    latitude: float = 45.4064,
    longitude: float = 11.8768,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Keep only timestamps between sunrise and sunset.

    Parameters
    ----------
    temperature : pd.DataFrame
        Temperature table with DatetimeIndex.
    humidity : pd.DataFrame | None, optional
        Humidity table with DatetimeIndex.
    city_name : str, optional
        City name.
    country : str, optional
        Country name.
    timezone : str, optional
        IANA timezone name.
    latitude : float, optional
        Latitude.
    longitude : float, optional
        Longitude.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        Daylight-filtered temperature and humidity tables.
    """
    temp = temperature.copy()

    if not isinstance(temp.index, pd.DatetimeIndex):
        temp.index = pd.to_datetime(temp.index, errors="coerce")

    temp = temp[~temp.index.isna()]
    temp.index = to_timezone_index(temp.index, timezone)
    temp = temp[~temp.index.isna()]

    dates = np.unique(temp.index.date)
    sun_times = compute_sun_times(dates, city_name, country, timezone, latitude, longitude)

    mask = pd.Series(False, index=temp.index)
    for d in dates:
        start, end = sun_times[d]
        mask |= (temp.index >= start) & (temp.index <= end)

    temp_day = temp.loc[mask]

    if humidity is None:
        return temp_day, None

    hum = humidity.copy()

    if not isinstance(hum.index, pd.DatetimeIndex):
        hum.index = pd.to_datetime(hum.index, errors="coerce")

    hum = hum[~hum.index.isna()]
    hum.index = to_timezone_index(hum.index, timezone)
    hum = hum[~hum.index.isna()]
    hum_day = hum.reindex(temp_day.index).dropna(axis=0, how="all")

    return temp_day, hum_day


def isolate_period(
    df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Keep rows inside a given time period.

    Parameters
    ----------
    df : pd.DataFrame
        Input table with DatetimeIndex.
    start : str | pd.Timestamp
        Start timestamp.
    end : str | pd.Timestamp
        End timestamp.

    Returns
    -------
    pd.DataFrame
        Filtered table.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if getattr(df.index, "tz", None) is not None and start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(df.index.tz)
        end_ts = end_ts.tz_localize(df.index.tz)

    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


def isolate_peak_windows(
    temperature_daylight: pd.DataFrame,
    window_hours: int = 2,
) -> pd.DataFrame:
    """
    Extract daily windows around the mean-field temperature peak.

    Parameters
    ----------
    temperature_daylight : pd.DataFrame
        Daylight-filtered temperature table.
    window_hours : int, optional
        Number of hours before and after the daily peak.

    Returns
    -------
    pd.DataFrame
        Concatenated peak-window temperature table.

    Raises
    ------
    ValueError
        If no peak windows are found.
    """
    frames = []

    for d in np.unique(temperature_daylight.index.date):
        df_d = temperature_daylight.loc[temperature_daylight.index.date == d]

        if df_d.empty:
            continue

        mean_series = df_d.mean(axis=1)
        peak_time = mean_series.idxmax()
        start = peak_time - pd.Timedelta(hours=window_hours)
        end = peak_time + pd.Timedelta(hours=window_hours)

        win = df_d.loc[(df_d.index >= start) & (df_d.index <= end)].copy()

        if not win.empty:
            win["_day"] = pd.to_datetime(d)
            frames.append(win)

    if not frames:
        raise ValueError("No peak windows found.")

    return pd.concat(frames)


def build_concatenated_peak_series(temperature_peak_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Build one concatenated peak-window time series per sensor.

    Parameters
    ----------
    temperature_peak_windows : pd.DataFrame
        Peak-window table containing a '_day' column.

    Returns
    -------
    pd.DataFrame
        Table with sensors as rows and concatenated time positions as columns.
    """
    df_reset = temperature_peak_windows.reset_index()
    time_col = df_reset.columns[0]
    df_sorted = df_reset.sort_values(["_day", time_col])
    values = df_sorted.drop(columns=["_day", time_col])
    values = values.reset_index(drop=True)

    return values.T


def prepare_time_series_array(
    series_per_sensor: pd.DataFrame,
    preserve_magnitude: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert a sensor-by-time table to a tslearn-compatible array.

    Parameters
    ----------
    series_per_sensor : pd.DataFrame
        Table with sensors as rows and time positions as columns.
    preserve_magnitude : bool, optional
        If True, use global min-max scaling. If False, z-score each time series.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Time-series array and sensor names.
    """
    clean = series_per_sensor.dropna(axis=0, how="any")
    sensor_names = list(clean.index)
    arr = clean.values.astype(float)

    if preserve_magnitude:
        global_min = np.nanmin(arr)
        global_max = np.nanmax(arr)

        if global_max - global_min > 0:
            arr_scaled = (arr - global_min) / (global_max - global_min)
        else:
            arr_scaled = arr.copy()
    else:
        scaler = TimeSeriesScalerMeanVariance()
        arr_scaled = scaler.fit_transform(arr[:, :, np.newaxis])[:, :, 0]

    return arr_scaled[:, :, np.newaxis], sensor_names


def daily_amplitude(values: np.ndarray, steps_per_day: int) -> float:
    """
    Compute the mean daily amplitude of a one-dimensional time series.

    Parameters
    ----------
    values : np.ndarray
        Input values.
    steps_per_day : int
        Number of timesteps per day.

    Returns
    -------
    float
        Mean daily amplitude.
    """
    series = pd.Series(values)
    grouped = series.groupby(np.arange(len(series)) // steps_per_day)

    return float((grouped.max() - grouped.min()).mean())


def frequency_steps_per_day(freq: str) -> int:
    """
    Convert a frequency string to the number of timesteps per day.

    Parameters
    ----------
    freq : str
        Frequency string.

    Returns
    -------
    int
        Timesteps per day.

    Raises
    ------
    ValueError
        If the frequency is unsupported.
    """
    mapping = {
        "10min": 144,
        "15min": 96,
        "20min": 72,
        "30min": 48,
        "60min": 24,
        "1H": 24,
        "h": 24,
    }

    if freq not in mapping:
        raise ValueError(f"Unsupported frequency: {freq}")

    return mapping[freq]


def prepare_db_wb_tables(
    dry_bulb: pd.DataFrame,
    wet_bulb: pd.DataFrame,
    freq: str = "30min",
    drop_sensors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """
    Align, resample, interpolate, and clean dry-bulb and wet-bulb tables.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    wet_bulb : pd.DataFrame
        Wet-bulb temperature table.
    freq : str, optional
        Resampling frequency.
    drop_sensors : list[str] | None, optional
        Sensor names to remove.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Index]
        Prepared dry-bulb table, wet-bulb table, and common sensor names.
    """
    drop_sensors = drop_sensors or []

    db = dry_bulb.drop(columns=drop_sensors, errors="ignore")
    wb = wet_bulb.drop(columns=drop_sensors, errors="ignore")

    idx = db.index.intersection(wb.index)
    db = db.loc[idx].astype("float32").resample(freq).mean()
    wb = wb.loc[idx].astype("float32").resample(freq).mean()

    sensors = db.columns.intersection(wb.columns)
    db = db[sensors].interpolate(limit_direction="both")
    wb = wb[sensors].interpolate(limit_direction="both")

    return db, wb, sensors


def build_feature_matrix(
    dry_bulb: pd.DataFrame,
    wet_bulb: pd.DataFrame,
    freq: str = "30min",
) -> pd.DataFrame:
    """
    Build a feature matrix from dry-bulb and wet-bulb sensor time series.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    wet_bulb : pd.DataFrame
        Wet-bulb temperature table.
    freq : str, optional
        Frequency used to estimate daily amplitudes.

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by sensor name.
    """
    steps = frequency_steps_per_day(freq)
    rows = []

    for sensor in dry_bulb.columns:
        db = dry_bulb[sensor].to_numpy(dtype=float)
        wb = wet_bulb[sensor].to_numpy(dtype=float)
        delta = db - wb

        ac1 = np.corrcoef(db[:-1], db[1:])[0, 1] if len(db) > 1 else 0.0
        x = np.vstack([db, wb]).T
        x = x - np.mean(x, axis=0, keepdims=True)
        spec = np.fft.rfft(x, axis=0)
        mag = np.abs(spec)

        h1_db = mag[1, 0] / (mag[:, 0].sum() + 1e-9) if mag.shape[0] > 1 else 0.0
        h1_wb = mag[1, 1] / (mag[:, 1].sum() + 1e-9) if mag.shape[0] > 1 else 0.0

        rows.append([
            np.nanmean(db),
            np.nanstd(db),
            np.nanmean(wb),
            np.nanstd(wb),
            np.nanmean(delta),
            np.nanstd(delta),
            np.nanpercentile(db, 5),
            np.nanpercentile(db, 95),
            ac1,
            daily_amplitude(db, steps),
            daily_amplitude(wb, steps),
            h1_db,
            h1_wb,
        ])

    columns = [
        "m_db",
        "sd_db",
        "m_wb",
        "sd_wb",
        "m_delta",
        "sd_delta",
        "p5_db",
        "p95_db",
        "ac1_db",
        "amp_db",
        "amp_wb",
        "h1_db",
        "h1_wb",
    ]

    return pd.DataFrame(rows, index=dry_bulb.columns, columns=columns)


def compute_clustering_kpis(
    x: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute basic clustering quality indicators.

    Parameters
    ----------
    x : np.ndarray
        Two-dimensional feature matrix.
    labels : np.ndarray
        Cluster labels.

    Returns
    -------
    dict[str, float]
        Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
    """
    kpis = {}

    try:
        kpis["silhouette"] = float(silhouette_score(x, labels))
    except Exception:
        kpis["silhouette"] = np.nan

    try:
        kpis["davies_bouldin"] = float(davies_bouldin_score(x, labels))
    except Exception:
        kpis["davies_bouldin"] = np.nan

    try:
        kpis["calinski_harabasz"] = float(calinski_harabasz_score(x, labels))
    except Exception:
        kpis["calinski_harabasz"] = np.nan

    return kpis


def run_feature_kmeans(
    dry_bulb: pd.DataFrame,
    wet_bulb: pd.DataFrame,
    n_clusters: int = 4,
    freq: str = "30min",
    random_state: int = 42,
) -> ClusteringResult:
    """
    Cluster sensors using engineered dry-bulb and wet-bulb features.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Prepared dry-bulb table.
    wet_bulb : pd.DataFrame
        Prepared wet-bulb table.
    n_clusters : int, optional
        Number of clusters.
    freq : str, optional
        Frequency used in feature construction.
    random_state : int, optional
        Random state.

    Returns
    -------
    ClusteringResult
        Clustering result object.
    """
    features = build_feature_matrix(dry_bulb, wet_bulb, freq=freq)
    scaled = StandardScaler().fit_transform(features.values)

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = model.fit_predict(scaled)

    label_series = pd.Series(labels, index=features.index, name="cluster")
    kpis = compute_clustering_kpis(scaled, labels)

    return ClusteringResult(
        labels=label_series,
        features=features,
        scaled_features=scaled,
        time_series_array=None,
        sensor_names=list(features.index),
        model=model,
        kpis=kpis,
    )


def run_time_series_clustering(
    arr: np.ndarray,
    sensor_names: list[str],
    n_clusters: int,
    method: Literal["euclidean_ts", "softdtw", "kmeans_flat"] = "euclidean_ts",
    random_state: int = 42,
    dtw_gamma: float = 0.01,
) -> ClusteringResult:
    """
    Cluster sensors using time-series clustering.

    Parameters
    ----------
    arr : np.ndarray
        Time-series array with shape (n_sensors, n_timesteps, 1).
    sensor_names : list[str]
        Sensor names matching the array order.
    n_clusters : int
        Number of clusters.
    method : {"euclidean_ts", "softdtw", "kmeans_flat"}, optional
        Clustering method.
    random_state : int, optional
        Random state.
    dtw_gamma : float, optional
        Soft-DTW gamma parameter.

    Returns
    -------
    ClusteringResult
        Clustering result object.
    """
    if method == "euclidean_ts":
        model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="euclidean",
            random_state=random_state,
        )
        labels = model.fit_predict(arr)
        x = arr.reshape(arr.shape[0], -1)

    elif method == "softdtw":
        model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="softdtw",
            metric_params={"gamma": dtw_gamma},
            random_state=random_state,
        )
        labels = model.fit_predict(arr)
        x = arr.reshape(arr.shape[0], -1)

    elif method == "kmeans_flat":
        x = arr.reshape(arr.shape[0], -1)
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = model.fit_predict(x)

    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    label_series = pd.Series(labels, index=sensor_names, name="cluster")
    kpis = compute_clustering_kpis(x, labels)

    return ClusteringResult(
        labels=label_series,
        features=None,
        scaled_features=x,
        time_series_array=arr,
        sensor_names=sensor_names,
        model=model,
        kpis=kpis,
    )


def evaluate_k_range(
    x: np.ndarray,
    k_min: int = 2,
    k_max: int = 9,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate KMeans clustering quality across a range of cluster numbers.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix.
    k_min : int, optional
        Minimum number of clusters.
    k_max : int, optional
        Maximum number of clusters.
    random_state : int, optional
        Random state.

    Returns
    -------
    pd.DataFrame
        Table with inertia and clustering quality metrics for each k.
    """
    rows = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(x)
        kpis = compute_clustering_kpis(x, labels)

        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            **kpis,
        })

    return pd.DataFrame(rows)


def mean_daily_profile(df: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    """
    Compute the mean daily profile of a wide sensor table.

    Parameters
    ----------
    df : pd.DataFrame
        Wide sensor table.
    freq : str, optional
        Frequency of the table.

    Returns
    -------
    pd.DataFrame
        Mean daily profile indexed by hour of day.
    """
    work = df.copy()
    step_sec = int(pd.to_timedelta(freq).total_seconds())
    seconds_in_day = 24 * 3600

    day_pos = (
        work.index.hour * 3600
        + work.index.minute * 60
        + work.index.second
    )

    bins = np.arange(0, seconds_in_day + step_sec, step_sec)
    labels = bins[:-1] / 3600

    work["__bin__"] = pd.cut(day_pos, bins=bins, labels=labels, include_lowest=True)
    out = work.groupby("__bin__").mean(numeric_only=True)
    out.index = out.index.astype(float)

    return out


def cluster_daily_profiles(
    dry_bulb: pd.DataFrame,
    wet_bulb: pd.DataFrame,
    cluster_labels: pd.Series,
    freq: str = "30min",
) -> dict[int, dict[str, pd.DataFrame]]:
    """
    Prepare daily mean dry-bulb and wet-bulb profiles for each cluster.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb table.
    wet_bulb : pd.DataFrame
        Wet-bulb table.
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    freq : str, optional
        Frequency of the tables.

    Returns
    -------
    dict[int, dict[str, pd.DataFrame]]
        Dictionary with dry-bulb and wet-bulb profiles for each cluster.
    """
    profiles = {}

    for cluster_id in sorted(cluster_labels.unique()):
        members = cluster_labels[cluster_labels == cluster_id].index
        members = members.intersection(dry_bulb.columns).intersection(wet_bulb.columns)

        profiles[int(cluster_id)] = {
            "dry_bulb": mean_daily_profile(dry_bulb[members], freq=freq),
            "wet_bulb": mean_daily_profile(wet_bulb[members], freq=freq),
        }

    return profiles


def add_cluster_labels_to_locations(
    locations: pd.DataFrame,
    cluster_labels: pd.Series,
    sensor_column: str = "name",
    output_column: str = "cluster",
) -> pd.DataFrame:
    """
    Add cluster labels to a sensor-location table.

    Parameters
    ----------
    locations : pd.DataFrame
        Sensor location table.
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    sensor_column : str, optional
        Column containing sensor names.
    output_column : str, optional
        Output cluster column name.

    Returns
    -------
    pd.DataFrame
        Location table with cluster labels.
    """
    out = locations.copy()
    out[output_column] = out[sensor_column].map(cluster_labels)

    return out


def compute_spatial_kpi(
    cluster_labels: pd.Series,
    locations: pd.DataFrame,
    sensor_column: str = "name",
    latitude_column: str = "Latitude",
    longitude_column: str = "Longitude",
) -> float:
    """
    Compute mean within-cluster pairwise spatial distance.

    Parameters
    ----------
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    locations : pd.DataFrame
        Sensor location table.
    sensor_column : str, optional
        Sensor name column.
    latitude_column : str, optional
        Latitude column.
    longitude_column : str, optional
        Longitude column.

    Returns
    -------
    float
        Mean within-cluster pairwise distance.
    """
    loc = locations.set_index(sensor_column)
    sensors = [s for s in cluster_labels.index if s in loc.index]

    if len(sensors) < 2:
        return np.nan

    coords = loc.loc[sensors, [latitude_column, longitude_column]].astype(float).values
    dist_mat = squareform(pdist(coords))
    idx = {s: i for i, s in enumerate(sensors)}

    distances = []

    for i, s1 in enumerate(cluster_labels.index):
        if s1 not in idx:
            continue

        for s2 in cluster_labels.index[i + 1:]:
            if s2 not in idx:
                continue

            if cluster_labels.loc[s1] == cluster_labels.loc[s2]:
                distances.append(dist_mat[idx[s1], idx[s2]])

    return float(np.nanmean(distances)) if distances else np.nan


def run_clustering_from_config(
    sensor_data: dict[str, pd.DataFrame],
    config: dict,
) -> ClusteringResult:
    """
    Run the configured feature-based clustering pipeline.

    Parameters
    ----------
    sensor_data : dict[str, pd.DataFrame]
        Dictionary of filtered sensor DataFrames.
    config : dict
        Project configuration dictionary.

    Returns
    -------
    ClusteringResult
        Clustering result object.
    """
    sensor_cfg = config.get("sensor_filtering", {})
    clustering_cfg = config.get("clustering", {})

    drop_sensors = sensor_cfg.get("drop_sensors", [])
    freq = clustering_cfg.get("time_resample", "30min")
    n_clusters = clustering_cfg.get("final_clusters", 4)
    random_state = clustering_cfg.get("random_state", 42)

    temperature, humidity = build_wide_tables(sensor_data)
    temperature = temperature.drop(columns=drop_sensors, errors="ignore")
    humidity = humidity.drop(columns=drop_sensors, errors="ignore")

    temperature = resample_with_time_interpolation(temperature, freq="10min")
    humidity = resample_with_time_interpolation(humidity, freq="10min")
    wet_bulb = wet_bulb_temperature(temperature, humidity)

    db, wb, _ = prepare_db_wb_tables(
        dry_bulb=temperature,
        wet_bulb=wet_bulb,
        freq=freq,
        drop_sensors=drop_sensors,
    )

    return run_feature_kmeans(
        dry_bulb=db,
        wet_bulb=wb,
        n_clusters=n_clusters,
        freq=freq,
        random_state=random_state,
    )

