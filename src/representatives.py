from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_naive_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove timezone information from a DataFrame index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with a timezone-naive DatetimeIndex.
    """
    out = df.copy()

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    return out


def align_weather_tables(
    dry_bulb: pd.DataFrame,
    wet_bulb: pd.DataFrame,
    relative_humidity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index]:
    """
    Align dry-bulb, wet-bulb, and relative humidity tables by common sensors.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    wet_bulb : pd.DataFrame
        Wet-bulb temperature table.
    relative_humidity : pd.DataFrame
        Relative humidity table.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index]
        Aligned dry-bulb, wet-bulb, relative humidity tables, and common sensor names.
    """
    db = to_naive_datetime_index(dry_bulb)
    wb = to_naive_datetime_index(wet_bulb)
    rh = to_naive_datetime_index(relative_humidity)

    common = db.columns.intersection(wb.columns).intersection(rh.columns)

    return db[common], wb[common], rh[common], common


def mask_months(df: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """
    Keep rows whose timestamp month is in a selected list.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    months : list[int]
        Month numbers to keep.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    return df[df.index.month.isin(months)]


def representative_per_cluster(
    dry_bulb: pd.DataFrame,
    wet_bulb: pd.DataFrame,
    cluster_labels: pd.Series,
    freq: str = "10min",
    months: list[int] | None = None,
) -> dict[int, str | None]:
    """
    Select one representative sensor per cluster.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    wet_bulb : pd.DataFrame
        Wet-bulb temperature table.
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    freq : str, optional
        Resampling frequency used before distance calculation.
    months : list[int] | None, optional
        Optional months used to select representatives.

    Returns
    -------
    dict[int, str | None]
        Dictionary mapping cluster id to representative sensor name.
    """
    db = to_naive_datetime_index(dry_bulb)
    wb = to_naive_datetime_index(wet_bulb)

    if months is not None:
        db = mask_months(db, months)
        wb = mask_months(wb, months)

    db = db.resample(freq).mean()
    wb = wb.resample(freq).mean()

    reps = {}

    for cluster_id in sorted(cluster_labels.dropna().unique()):
        members = cluster_labels[cluster_labels == cluster_id].index
        members = members.intersection(db.columns).intersection(wb.columns)

        if len(members) == 0:
            reps[int(cluster_id)] = None
            continue

        db_c = db[members]
        wb_c = wb[members]

        mean_db = db_c.mean(axis=1)
        mean_wb = wb_c.mean(axis=1)

        scores = {}

        for sensor in members:
            d = np.sqrt((db_c[sensor] - mean_db) ** 2 + (wb_c[sensor] - mean_wb) ** 2)
            scores[sensor] = float(d.dropna().sum())

        reps[int(cluster_id)] = min(scores, key=scores.get) if scores else None

    return reps


def daily_representatives(
    dry_bulb: pd.DataFrame,
    wet_bulb: pd.DataFrame,
    cluster_labels: pd.Series,
    freq: str = "10min",
) -> dict[int, pd.DataFrame]:
    """
    Select the closest representative sensor for each cluster and each day.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    wet_bulb : pd.DataFrame
        Wet-bulb temperature table.
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    freq : str, optional
        Resampling frequency used before distance calculation.

    Returns
    -------
    dict[int, pd.DataFrame]
        Dictionary mapping cluster id to a daily representative table.
    """
    db = to_naive_datetime_index(dry_bulb).resample(freq).mean()
    wb = to_naive_datetime_index(wet_bulb).resample(freq).mean()

    reps = {}

    for cluster_id in sorted(cluster_labels.dropna().unique()):
        members = cluster_labels[cluster_labels == cluster_id].index
        members = members.intersection(db.columns).intersection(wb.columns)

        if len(members) == 0:
            reps[int(cluster_id)] = pd.DataFrame(columns=["day", "sensor", "distance"])
            continue

        db_c = db[members]
        wb_c = wb[members]

        mean_db = db_c.mean(axis=1)
        mean_wb = wb_c.mean(axis=1)

        dist_df = pd.DataFrame(index=db.index, columns=members, dtype=float)

        for sensor in members:
            dist_df[sensor] = np.sqrt((db_c[sensor] - mean_db) ** 2 + (wb_c[sensor] - mean_wb) ** 2)

        daily_sum = dist_df.groupby(dist_df.index.date).sum()
        best_sensors = daily_sum.idxmin(axis=1)
        best_distances = daily_sum.min(axis=1)

        reps[int(cluster_id)] = pd.DataFrame({
            "day": daily_sum.index,
            "sensor": best_sensors.values,
            "distance": best_distances.values,
        })

    return reps


def export_daily_representatives(
    dry_bulb: pd.DataFrame,
    wet_bulb: pd.DataFrame,
    relative_humidity: pd.DataFrame,
    representatives: dict[int, pd.DataFrame],
    out_dir: str | Path,
    out_freq: str = "1H",
    filename_template: str = "cluster_{cluster_id}_representatives_hourly.xlsx",
) -> list[Path]:
    """
    Export hourly weather profiles built from daily representative sensors.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    wet_bulb : pd.DataFrame
        Wet-bulb temperature table.
    relative_humidity : pd.DataFrame
        Relative humidity table.
    representatives : dict[int, pd.DataFrame]
        Daily representatives by cluster.
    out_dir : str | Path
        Output directory.
    out_freq : str, optional
        Output frequency.
    filename_template : str, optional
        Excel filename template.

    Returns
    -------
    list[Path]
        Paths of exported Excel files.
    """
    db, wb, rh, common = align_weather_tables(dry_bulb, wet_bulb, relative_humidity)

    db_h = db.resample(out_freq).mean()
    wb_h = wb.resample(out_freq).mean()
    rh_h = rh.resample(out_freq).mean()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written = []

    for cluster_id, rep in representatives.items():
        rows = []

        if rep is not None and len(rep) > 0:
            for _, r in rep.iterrows():
                sensor = r["sensor"]

                if pd.isna(sensor) or sensor not in common:
                    continue

                day = pd.to_datetime(r["day"]).date()
                start = pd.Timestamp(day)
                end = start + pd.Timedelta(days=1)

                idx = db_h.loc[start:end - pd.Timedelta(seconds=1)].index

                if len(idx) == 0:
                    continue

                rows.append(pd.DataFrame({
                    "sensor": sensor,
                    "time": idx,
                    "db_temp": db_h.loc[idx, sensor].to_numpy(),
                    "wb_temp": wb_h.loc[idx, sensor].to_numpy(),
                    "rh": rh_h.loc[idx, sensor].to_numpy(),
                    "day": day,
                }))

        out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["sensor", "time", "db_temp", "wb_temp", "rh", "day"]
        )

        file_path = out_path / filename_template.format(cluster_id=int(cluster_id) + 1)

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="hourly")

        written.append(file_path)

    return written


def export_fixed_representatives(
    dry_bulb: pd.DataFrame,
    wet_bulb: pd.DataFrame,
    relative_humidity: pd.DataFrame,
    representatives: dict[int, str | None],
    out_dir: str | Path,
    out_freq: str = "1H",
    tag: str = "all",
    filename_template: str = "cluster_{cluster_id}_representative_{tag}.xlsx",
) -> list[Path]:
    """
    Export full-period weather profiles using one fixed representative sensor per cluster.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    wet_bulb : pd.DataFrame
        Wet-bulb temperature table.
    relative_humidity : pd.DataFrame
        Relative humidity table.
    representatives : dict[int, str | None]
        Fixed representative sensor by cluster.
    out_dir : str | Path
        Output directory.
    out_freq : str, optional
        Output frequency.
    tag : str, optional
        Tag inserted into output filenames.
    filename_template : str, optional
        Excel filename template.

    Returns
    -------
    list[Path]
        Paths of exported Excel files.
    """
    db, wb, rh, common = align_weather_tables(dry_bulb, wet_bulb, relative_humidity)

    db_h = db.resample(out_freq).mean()
    wb_h = wb.resample(out_freq).mean()
    rh_h = rh.resample(out_freq).mean()

    idx = db_h.index.intersection(wb_h.index).intersection(rh_h.index)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written = []

    for cluster_id, sensor in representatives.items():
        if sensor is None or sensor not in common:
            out_df = pd.DataFrame(columns=["sensor", "time", "db_temp", "wb_temp", "rh"])
        else:
            out_df = pd.DataFrame({
                "sensor": sensor,
                "time": idx,
                "db_temp": db_h.loc[idx, sensor].to_numpy(),
                "wb_temp": wb_h.loc[idx, sensor].to_numpy(),
                "rh": rh_h.loc[idx, sensor].to_numpy(),
            })

        file_path = out_path / filename_template.format(
            cluster_id=int(cluster_id) + 1,
            tag=tag,
        )

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="hourly")

        written.append(file_path)

    return written


def load_representative_excel(path: str | Path) -> pd.DataFrame:
    """
    Load one representative weather Excel file.

    Parameters
    ----------
    path : str | Path
        Path to a representative Excel file.

    Returns
    -------
    pd.DataFrame
        Representative weather table.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Representative Excel not found: {file_path}")

    df = pd.read_excel(file_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    return df


def load_representative_excels(
    folder: str | Path,
    pattern: str = "cluster_*_representative*.xlsx",
) -> dict[str, pd.DataFrame]:
    """
    Load multiple representative weather Excel files from a folder.

    Parameters
    ----------
    folder : str | Path
        Folder containing representative Excel files.
    pattern : str, optional
        Glob pattern used to select files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary where keys are file stems and values are DataFrames.
    """
    folder_path = Path(folder)

    return {
        p.stem: load_representative_excel(p)
        for p in sorted(folder_path.glob(pattern))
    }


def summarize_representative_temperatures(
    representatives: dict[str, pd.DataFrame],
    months: list[int] | None = None,
) -> pd.DataFrame:
    """
    Summarize representative dry-bulb temperature by file and month.

    Parameters
    ----------
    representatives : dict[str, pd.DataFrame]
        Dictionary of representative weather tables.
    months : list[int] | None, optional
        Optional months to keep.

    Returns
    -------
    pd.DataFrame
        Monthly temperature summary table.
    """
    rows = []

    for name, df in representatives.items():
        if "time" not in df.columns or "db_temp" not in df.columns:
            continue

        work = df.copy()
        work["time"] = pd.to_datetime(work["time"], errors="coerce")
        work["db_temp"] = pd.to_numeric(work["db_temp"], errors="coerce")
        work = work.dropna(subset=["time", "db_temp"])
        work["month"] = work["time"].dt.month

        if months is not None:
            work = work[work["month"].isin(months)]

        grouped = work.groupby("month")["db_temp"].agg(["min", "mean", "max"]).reset_index()

        for _, r in grouped.iterrows():
            rows.append({
                "profile": name,
                "month": int(r["month"]),
                "min_db_temp": float(r["min"]),
                "mean_db_temp": float(r["mean"]),
                "max_db_temp": float(r["max"]),
            })

    return pd.DataFrame(rows)


def export_representative_temperature_summary(
    representatives: dict[str, pd.DataFrame],
    out_path: str | Path,
    months: list[int] | None = None,
) -> Path:
    """
    Export monthly dry-bulb temperature summaries for representative profiles.

    Parameters
    ----------
    representatives : dict[str, pd.DataFrame]
        Dictionary of representative weather tables.
    out_path : str | Path
        Output Excel path.
    months : list[int] | None, optional
        Optional months to keep.

    Returns
    -------
    Path
        Written Excel path.
    """
    summary = summarize_representative_temperatures(representatives, months=months)
    file_path = Path(out_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_excel(file_path, index=False)

    return file_path


def build_representatives_from_config(
    dry_bulb: pd.DataFrame,
    wet_bulb: pd.DataFrame,
    relative_humidity: pd.DataFrame,
    cluster_labels: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Build configured daily, full-year, and summer representative profiles.

    Parameters
    ----------
    dry_bulb : pd.DataFrame
        Dry-bulb temperature table.
    wet_bulb : pd.DataFrame
        Wet-bulb temperature table.
    relative_humidity : pd.DataFrame
        Relative humidity table.
    cluster_labels : pd.Series
        Cluster labels indexed by sensor name.
    config : dict[str, Any]
        Project configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary containing representative selections and exported file paths.
    """
    rep_cfg = config.get("representatives", {})
    path_cfg = config.get("paths", {})

    internal_freq = rep_cfg.get("internal_frequency", "10min")
    output_freq = rep_cfg.get("representative_frequency", "1H")
    summer_months = rep_cfg.get("summer_months", [6, 7, 8])
    out_dir = path_cfg.get("cluster_exports_dir", "data/intermediate/cluster_exports")

    results = {}

    if rep_cfg.get("export_daily_representatives", True):
        daily = daily_representatives(
            dry_bulb=dry_bulb,
            wet_bulb=wet_bulb,
            cluster_labels=cluster_labels,
            freq=internal_freq,
        )
        daily_files = export_daily_representatives(
            dry_bulb=dry_bulb,
            wet_bulb=wet_bulb,
            relative_humidity=relative_humidity,
            representatives=daily,
            out_dir=out_dir,
            out_freq=output_freq,
        )
        results["daily_representatives"] = daily
        results["daily_files"] = daily_files

    if rep_cfg.get("export_full_year_representatives", True):
        full_year = representative_per_cluster(
            dry_bulb=dry_bulb,
            wet_bulb=wet_bulb,
            cluster_labels=cluster_labels,
            freq=internal_freq,
            months=None,
        )
        full_year_files = export_fixed_representatives(
            dry_bulb=dry_bulb,
            wet_bulb=wet_bulb,
            relative_humidity=relative_humidity,
            representatives=full_year,
            out_dir=out_dir,
            out_freq=output_freq,
            tag="all",
        )
        results["full_year_representatives"] = full_year
        results["full_year_files"] = full_year_files

    if rep_cfg.get("export_summer_representatives", True):
        summer = representative_per_cluster(
            dry_bulb=dry_bulb,
            wet_bulb=wet_bulb,
            cluster_labels=cluster_labels,
            freq=internal_freq,
            months=summer_months,
        )
        summer_files = export_fixed_representatives(
            dry_bulb=dry_bulb,
            wet_bulb=wet_bulb,
            relative_humidity=relative_humidity,
            representatives=summer,
            out_dir=out_dir,
            out_freq=output_freq,
            tag="summer",
        )
        results["summer_representatives"] = summer
        results["summer_files"] = summer_files

    return results