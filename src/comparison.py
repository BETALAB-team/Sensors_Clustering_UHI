from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_naive_datetime(series: pd.Series) -> pd.Series:
    """
    Convert a Series to timezone-naive datetime values.

    Parameters
    ----------
    series : pd.Series
        Input datetime-like Series.

    Returns
    -------
    pd.Series
        Timezone-naive datetime Series.
    """
    dt = pd.to_datetime(series, errors="coerce")

    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)

    return dt


def infer_version_from_filename(path: str | Path, default_version: int = 1) -> int:
    """
    Infer version number from a filename ending with v<number>.

    Parameters
    ----------
    path : str | Path
        Input file path.
    default_version : int, optional
        Version used when no explicit version is found.

    Returns
    -------
    int
        Inferred version number.
    """
    name = Path(path).name
    match = re.search(r"v(\d+)\.xlsx$", name, flags=re.IGNORECASE)

    if match:
        return int(match.group(1))

    return default_version


def find_versioned_excel_files(
    folder: str | Path,
    pattern: str = "T_urban Heat island*.xlsx",
    min_version: int = 1,
    max_version: int = 5,
) -> list[tuple[int, Path]]:
    """
    Find versioned Excel files in a folder.

    Parameters
    ----------
    folder : str | Path
        Folder containing Excel files.
    pattern : str, optional
        Glob pattern.
    min_version : int, optional
        Minimum accepted version.
    max_version : int, optional
        Maximum accepted version.

    Returns
    -------
    list[tuple[int, Path]]
        Sorted list of version number and file path pairs.

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    """
    folder_path = Path(folder)

    files = [
        (infer_version_from_filename(p), p)
        for p in folder_path.glob(pattern)
    ]

    files = [
        item
        for item in files
        if min_version <= item[0] <= max_version
    ]

    files = sorted(files, key=lambda x: x[0])

    if not files:
        raise FileNotFoundError(f"No matching versioned Excel files found in {folder_path}")

    return files


def load_temperature_version_sheet(
    path: str | Path,
    sheet_name: str = "Foglio3",
    rural_col: str = "T_rural (A station)",
    eureca_col: str = "T_EUReCA",
    date_col: str | None = None,
    version: int | None = None,
) -> pd.DataFrame:
    """
    Load one versioned temperature-comparison Excel sheet.

    Parameters
    ----------
    path : str | Path
        Excel file path.
    sheet_name : str, optional
        Sheet name.
    rural_col : str, optional
        Rural temperature column.
    eureca_col : str, optional
        EUReCA temperature column.
    date_col : str | None, optional
        Datetime column. If None, the first column is used.
    version : int | None, optional
        Version number used to rename the EUReCA column.

    Returns
    -------
    pd.DataFrame
        Loaded comparison table.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    dt_col = df.columns[0] if date_col is None else date_col
    df = df.rename(columns={dt_col: "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", dayfirst=True)

    cols = ["Datetime"]

    if rural_col in df.columns:
        cols.append(rural_col)

    if eureca_col in df.columns:
        if version is None:
            cols.append(eureca_col)
        else:
            new_col = f"T_EUReCA_v{version}"
            df = df.rename(columns={eureca_col: new_col})
            cols.append(new_col)

    return df[cols].dropna(subset=["Datetime"])


def merge_temperature_versions(
    folder: str | Path,
    pattern: str = "T_urban Heat island*.xlsx",
    sheet_name: str = "Foglio3",
    rural_col: str = "T_rural (A station)",
    eureca_col: str = "T_EUReCA",
    date_col: str | None = None,
    min_version: int = 1,
    max_version: int = 5,
) -> pd.DataFrame:
    """
    Merge multiple EUReCA temperature versions against one rural reference.

    Parameters
    ----------
    folder : str | Path
        Folder containing versioned Excel files.
    pattern : str, optional
        Glob pattern.
    sheet_name : str, optional
        Sheet name.
    rural_col : str, optional
        Rural temperature column.
    eureca_col : str, optional
        EUReCA temperature column.
    date_col : str | None, optional
        Datetime column. If None, the first column is used.
    min_version : int, optional
        Minimum accepted version.
    max_version : int, optional
        Maximum accepted version.

    Returns
    -------
    pd.DataFrame
        Merged temperature comparison table.

    Raises
    ------
    FileNotFoundError
        If version 1 is missing.
    """
    files = find_versioned_excel_files(
        folder=folder,
        pattern=pattern,
        min_version=min_version,
        max_version=max_version,
    )

    v1_path = next((p for v, p in files if v == 1), None)

    if v1_path is None:
        raise FileNotFoundError("Version 1 file not found.")

    base = load_temperature_version_sheet(
        v1_path,
        sheet_name=sheet_name,
        rural_col=rural_col,
        eureca_col=eureca_col,
        date_col=date_col,
        version=1,
    )

    base = base.sort_values("Datetime").drop_duplicates(subset=["Datetime"], keep="first")

    if rural_col in base.columns:
        out = base[["Datetime", rural_col]].copy()
    else:
        out = base[["Datetime"]].copy()

    for version, path in files:
        dfv = load_temperature_version_sheet(
            path,
            sheet_name=sheet_name,
            rural_col=rural_col,
            eureca_col=eureca_col,
            date_col=date_col,
            version=version,
        )

        col = f"T_EUReCA_v{version}"

        if col in dfv.columns:
            out = out.merge(dfv[["Datetime", col]], on="Datetime", how="outer")

    eureca_cols = sorted(
        [c for c in out.columns if c.startswith("T_EUReCA_v")],
        key=lambda c: int(c.split("_v")[-1]),
    )

    ordered = ["Datetime"]

    if rural_col in out.columns:
        ordered.append(rural_col)

    ordered.extend(eureca_cols)

    return out[ordered].sort_values("Datetime").reset_index(drop=True)


def temperature_version_differences(
    merged: pd.DataFrame,
    rural_col: str = "T_rural (A station)",
) -> pd.DataFrame:
    """
    Convert merged version table into long-format EUReCA-rural differences.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged temperature comparison table.
    rural_col : str, optional
        Rural temperature column.

    Returns
    -------
    pd.DataFrame
        Long-format difference table.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    if "Datetime" not in merged.columns:
        raise KeyError("Merged table must contain 'Datetime'.")

    if rural_col not in merged.columns:
        raise KeyError(f"Merged table must contain rural column: {rural_col}")

    work = merged.copy()
    work["Datetime"] = pd.to_datetime(work["Datetime"], errors="coerce")
    work["Month"] = work["Datetime"].dt.month

    rows = []

    for col in [c for c in work.columns if c.startswith("T_EUReCA_v")]:
        version = col.split("_v")[-1]
        tmp = work[["Datetime", "Month", rural_col, col]].copy()
        tmp["Difference"] = pd.to_numeric(tmp[col], errors="coerce") - pd.to_numeric(tmp[rural_col], errors="coerce")
        tmp["Version"] = version
        rows.append(tmp[["Datetime", "Month", "Difference", "Version"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Datetime", "Month", "Difference", "Version"]
    )


def read_epw_temperature(
    epw_path: str | Path,
    year_override: int | None = 2024,
) -> pd.DataFrame:
    """
    Read dry-bulb temperature from an EPW file.

    Parameters
    ----------
    epw_path : str | Path
        EPW file path.
    year_override : int | None, optional
        Year used for timestamp construction.

    Returns
    -------
    pd.DataFrame
        DataFrame with Datetime and T_rural columns.

    Raises
    ------
    FileNotFoundError
        If the EPW file does not exist.
    """
    path = Path(epw_path)

    if not path.exists():
        raise FileNotFoundError(f"EPW file not found: {path}")

    epw = pd.read_csv(path, header=None, skiprows=8)
    epw = epw[[0, 1, 2, 3, 4, 6]].copy()
    epw.columns = ["Year", "Month", "Day", "Hour", "Minute", "Temperature"]

    if year_override is not None:
        epw["Year"] = year_override

    dt = pd.to_datetime(
        epw[["Year", "Month", "Day", "Hour", "Minute"]],
        errors="coerce",
    ) - pd.to_timedelta(1, unit="h")

    return pd.DataFrame({
        "Datetime": dt,
        "Temperature": pd.to_numeric(epw["Temperature"], errors="coerce"),
    }).dropna(subset=["Datetime"]).sort_values("Datetime")


def load_cluster_excel(
    path: str | Path,
    time_col: str = "time",
    db_col: str = "db_temp",
) -> tuple[str, pd.DataFrame]:
    """
    Load one cluster representative Excel file.

    Parameters
    ----------
    path : str | Path
        Cluster representative Excel path.
    time_col : str, optional
        Time column.
    db_col : str, optional
        Dry-bulb temperature column.

    Returns
    -------
    tuple[str, pd.DataFrame]
        Cluster name and normalized DataFrame.
    """
    file_path = Path(path)
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if time_col.lower() not in df.columns:
        raise KeyError(f"{file_path.name}: missing '{time_col}' column.")

    if db_col.lower() not in df.columns:
        raise KeyError(f"{file_path.name}: missing '{db_col}' column.")

    out = df[[time_col.lower(), db_col.lower()]].copy()
    out.columns = ["Datetime", "T_cluster"]
    out["Datetime"] = to_naive_datetime(out["Datetime"])
    out["T_cluster"] = pd.to_numeric(out["T_cluster"], errors="coerce")
    out = out.dropna(subset=["Datetime"]).sort_values("Datetime").drop_duplicates("Datetime")

    match = re.search(r"cluster_(\d+)", file_path.name, flags=re.IGNORECASE)
    cluster_name = f"cluster_{match.group(1)}" if match else file_path.stem

    return cluster_name, out


def merge_clusters_with_reference_epw(
    cluster_folder: str | Path,
    reference_epw_path: str | Path,
    cluster_pattern: str = "cluster_*_representative*.xlsx",
    year_override: int | None = 2024,
    time_col: str = "time",
    db_col: str = "db_temp",
    freq: str = "H",
) -> pd.DataFrame:
    """
    Merge cluster representative temperatures with a reference EPW temperature.

    Parameters
    ----------
    cluster_folder : str | Path
        Folder containing cluster representative Excel files.
    reference_epw_path : str | Path
        Reference EPW file path.
    cluster_pattern : str, optional
        Glob pattern for cluster representative Excel files.
    year_override : int | None, optional
        Year used for reference EPW timestamps.
    time_col : str, optional
        Time column in representative Excel files.
    db_col : str, optional
        Temperature column in representative Excel files.
    freq : str, optional
        Frequency used for regularizing output.

    Returns
    -------
    pd.DataFrame
        Wide table with Datetime, T_reference, and cluster columns.
    """
    folder = Path(cluster_folder)
    ref = read_epw_temperature(reference_epw_path, year_override=year_override)
    ref = ref.rename(columns={"Temperature": "T_reference"})

    wide = ref.copy()

    for file_path in sorted(folder.glob(cluster_pattern)):
        cluster_name, dfc = load_cluster_excel(file_path, time_col=time_col, db_col=db_col)
        one = dfc.rename(columns={"T_cluster": cluster_name})
        wide = wide.merge(one[["Datetime", cluster_name]], on="Datetime", how="outer")

    wide = wide.sort_values("Datetime").drop_duplicates("Datetime")

    if freq is not None:
        wide = wide.set_index("Datetime").asfreq(freq).reset_index()

    cols = ["Datetime", "T_reference"] + sorted(
        [c for c in wide.columns if c not in ["Datetime", "T_reference"]]
    )

    return wide[cols]


def cluster_reference_differences(
    merged: pd.DataFrame,
    reference_col: str = "T_reference",
) -> pd.DataFrame:
    """
    Convert a cluster-reference table into long-format temperature differences.

    Parameters
    ----------
    merged : pd.DataFrame
        Wide table with reference and cluster temperatures.
    reference_col : str, optional
        Reference temperature column.

    Returns
    -------
    pd.DataFrame
        Long-format table with cluster-reference differences.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    if "Datetime" not in merged.columns:
        raise KeyError("Merged table must contain 'Datetime'.")

    if reference_col not in merged.columns:
        raise KeyError(f"Merged table must contain reference column: {reference_col}")

    work = merged.copy()
    work["Datetime"] = pd.to_datetime(work["Datetime"], errors="coerce")
    work["Month"] = work["Datetime"].dt.month

    cluster_cols = [
        c for c in work.columns
        if c not in ["Datetime", reference_col, "Month"]
    ]

    rows = []

    for col in cluster_cols:
        tmp = work[["Datetime", "Month", reference_col, col]].copy()
        tmp["Cluster"] = col
        tmp["Difference"] = pd.to_numeric(tmp[col], errors="coerce") - pd.to_numeric(tmp[reference_col], errors="coerce")
        rows.append(tmp[["Datetime", "Month", "Cluster", "Difference"]])

    return pd.concat(rows, ignore_index=True).dropna(subset=["Difference", "Month"]) if rows else pd.DataFrame(
        columns=["Datetime", "Month", "Cluster", "Difference"]
    )


def monthly_difference_summary(
    differences: pd.DataFrame,
    group_col: str,
    value_col: str = "Difference",
) -> pd.DataFrame:
    """
    Summarize monthly temperature differences.

    Parameters
    ----------
    differences : pd.DataFrame
        Long-format difference table.
    group_col : str
        Grouping column, such as Version or Cluster.
    value_col : str, optional
        Difference value column.

    Returns
    -------
    pd.DataFrame
        Monthly summary table.
    """
    if differences.empty:
        return pd.DataFrame()

    work = differences.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    return (
        work
        .groupby([group_col, "Month"])[value_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )


def save_comparison_outputs(
    output_folder: str | Path,
    merged_versions: pd.DataFrame | None = None,
    version_differences: pd.DataFrame | None = None,
    merged_clusters: pd.DataFrame | None = None,
    cluster_differences: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    Save comparison tables to Excel files.

    Parameters
    ----------
    output_folder : str | Path
        Output folder.
    merged_versions : pd.DataFrame | None, optional
        Merged version table.
    version_differences : pd.DataFrame | None, optional
        Version difference table.
    merged_clusters : pd.DataFrame | None, optional
        Merged cluster-reference table.
    cluster_differences : pd.DataFrame | None, optional
        Cluster-reference difference table.

    Returns
    -------
    dict[str, Path]
        Written output paths.
    """
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}

    if merged_versions is not None:
        path = out_dir / "temperature_versions_merged.xlsx"
        merged_versions.to_excel(path, index=False)
        written["merged_versions"] = path

    if version_differences is not None:
        path = out_dir / "temperature_version_differences.xlsx"
        version_differences.to_excel(path, index=False)
        written["version_differences"] = path

        summary = monthly_difference_summary(version_differences, group_col="Version")
        summary_path = out_dir / "temperature_version_differences_monthly_summary.xlsx"
        summary.to_excel(summary_path, index=False)
        written["version_differences_monthly_summary"] = summary_path

    if merged_clusters is not None:
        path = out_dir / "clusters_vs_reference_merged.xlsx"
        merged_clusters.to_excel(path, index=False)
        written["merged_clusters"] = path

    if cluster_differences is not None:
        path = out_dir / "cluster_reference_differences.xlsx"
        cluster_differences.to_excel(path, index=False)
        written["cluster_differences"] = path

        summary = monthly_difference_summary(cluster_differences, group_col="Cluster")
        summary_path = out_dir / "cluster_reference_differences_monthly_summary.xlsx"
        summary.to_excel(summary_path, index=False)
        written["cluster_differences_monthly_summary"] = summary_path

    return written


def compare_temperature_versions_from_config(
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compare EUReCA temperature versions using project configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Project configuration dictionary.
    project_root : str | Path | None, optional
        Root folder used to resolve relative paths.

    Returns
    -------
    dict[str, Any]
        Comparison tables and written paths.
    """
    root = Path.cwd() if project_root is None else Path(project_root)

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (root / path).resolve()

    cfg = config.get("comparison", {})
    paths = config.get("paths", {})

    folder = resolve(cfg.get("version_folder", paths.get("comparison_input_dir", "data/input/comparison")))
    output_folder = resolve(paths.get("comparison_output_dir", "data/output/comparison"))

    merged = merge_temperature_versions(
        folder=folder,
        pattern=cfg.get("version_pattern", "T_urban Heat island*.xlsx"),
        sheet_name=cfg.get("version_sheet_name", "Foglio3"),
        rural_col=cfg.get("rural_col", "T_rural (A station)"),
        eureca_col=cfg.get("eureca_col", "T_EUReCA"),
        date_col=cfg.get("date_col"),
        min_version=cfg.get("min_version", 1),
        max_version=cfg.get("max_version", 5),
    )

    differences = temperature_version_differences(
        merged=merged,
        rural_col=cfg.get("rural_col", "T_rural (A station)"),
    )

    written = save_comparison_outputs(
        output_folder=output_folder,
        merged_versions=merged,
        version_differences=differences,
    )

    return {
        "merged_versions": merged,
        "version_differences": differences,
        "written": written,
    }


def compare_clusters_with_reference_from_config(
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compare cluster representative temperatures against a reference EPW using project configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Project configuration dictionary.
    project_root : str | Path | None, optional
        Root folder used to resolve relative paths.

    Returns
    -------
    dict[str, Any]
        Comparison tables and written paths.
    """
    root = Path.cwd() if project_root is None else Path(project_root)

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (root / path).resolve()

    cfg = config.get("comparison", {})
    paths = config.get("paths", {})

    cluster_folder = resolve(cfg.get("cluster_folder", paths.get("cluster_exports_dir", "data/intermediate/cluster_exports")))
    reference_epw = resolve(cfg.get("reference_epw", paths.get("epw_base_file")))
    output_folder = resolve(paths.get("comparison_output_dir", "data/output/comparison"))

    merged = merge_clusters_with_reference_epw(
        cluster_folder=cluster_folder,
        reference_epw_path=reference_epw,
        cluster_pattern=cfg.get("cluster_pattern", "cluster_*_representative*.xlsx"),
        year_override=cfg.get("year_override", 2024),
        time_col=cfg.get("time_col", "time"),
        db_col=cfg.get("db_col", "db_temp"),
        freq=cfg.get("freq", "H"),
    )

    differences = cluster_reference_differences(
        merged=merged,
        reference_col=cfg.get("reference_col", "T_reference"),
    )

    written = save_comparison_outputs(
        output_folder=output_folder,
        merged_clusters=merged,
        cluster_differences=differences,
    )

    return {
        "merged_clusters": merged,
        "cluster_differences": differences,
        "written": written,
    }


def run_comparison_from_config(
    config: dict[str, Any],
    project_root: str | Path | None = None,
    compare_versions: bool = True,
    compare_clusters: bool = True,
) -> dict[str, Any]:
    """
    Run configured comparison analyses.

    Parameters
    ----------
    config : dict[str, Any]
        Project configuration dictionary.
    project_root : str | Path | None, optional
        Root folder used to resolve relative paths.
    compare_versions : bool, optional
        Whether to compare EUReCA temperature versions.
    compare_clusters : bool, optional
        Whether to compare cluster representatives with a reference EPW.

    Returns
    -------
    dict[str, Any]
        Comparison outputs.
    """
    outputs = {}

    if compare_versions:
        outputs["versions"] = compare_temperature_versions_from_config(
            config=config,
            project_root=project_root,
        )

    if compare_clusters:
        outputs["clusters"] = compare_clusters_with_reference_from_config(
            config=config,
            project_root=project_root,
        )

    return outputs