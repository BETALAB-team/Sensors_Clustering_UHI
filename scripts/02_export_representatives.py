from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_loading import load_config, project_path, ensure_directory
from representatives import (
    daily_representatives,
    representative_per_cluster,
    export_daily_representatives,
    export_fixed_representatives,
    export_representative_temperature_summary,
    load_representative_excels,
)


def main(config_path = PROJECT_ROOT / "config" / "config.yaml") -> dict:
    """
    Export representative weather profiles for each sensor cluster.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to the project configuration file.

    Returns
    -------
    dict
        Dictionary containing selected representatives and written file paths.
    """
    config = load_config(config_path)

    paths = config["paths"]
    rep_cfg = config.get("representatives", {})

    intermediate_dir = project_path(paths["intermediate_dir"], PROJECT_ROOT)
    cluster_exports_dir = ensure_directory(project_path(paths["cluster_exports_dir"], PROJECT_ROOT))

    dry_bulb_path = intermediate_dir / "dry_bulb_resampled.parquet"
    wet_bulb_path = intermediate_dir / "wet_bulb_resampled.parquet"
    humidity_path = intermediate_dir / "relative_humidity_resampled.parquet"
    labels_path = intermediate_dir / "sensor_cluster_labels.csv"

    dry_bulb = pd.read_parquet(dry_bulb_path)
    wet_bulb = pd.read_parquet(wet_bulb_path)
    relative_humidity = pd.read_parquet(humidity_path)

    cluster_labels = pd.read_csv(labels_path, index_col=0)["cluster"]
    cluster_labels.index = cluster_labels.index.astype(str)

    internal_freq = rep_cfg.get("internal_frequency", "10min")
    output_freq = rep_cfg.get("representative_frequency", "1H")
    summer_months = rep_cfg.get("summer_months", [6, 7, 8])

    outputs = {
        "daily_representatives": None,
        "full_year_representatives": None,
        "summer_representatives": None,
        "written": {},
    }

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
            out_dir=cluster_exports_dir,
            out_freq=output_freq,
            filename_template="cluster_{cluster_id}_representatives_hourly.xlsx",
        )

        outputs["daily_representatives"] = daily
        outputs["written"]["daily_files"] = daily_files

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
            out_dir=cluster_exports_dir,
            out_freq=output_freq,
            tag="all",
            filename_template="cluster_{cluster_id}_representative_{tag}.xlsx",
        )

        outputs["full_year_representatives"] = full_year
        outputs["written"]["full_year_files"] = full_year_files

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
            out_dir=cluster_exports_dir,
            out_freq=output_freq,
            tag="summer",
            filename_template="cluster_{cluster_id}_representative_{tag}.xlsx",
        )

        outputs["summer_representatives"] = summer
        outputs["written"]["summer_files"] = summer_files

    summaries = load_representative_excels(
        folder=cluster_exports_dir,
        pattern="cluster_*_representative*.xlsx",
    )

    summary_path = project_path(
        rep_cfg.get(
            "temperature_summary_path",
            "data/intermediate/cluster_exports/representative_temperature_summary.xlsx",
        ),
        PROJECT_ROOT,
    )

    outputs["written"]["temperature_summary"] = export_representative_temperature_summary(
        representatives=summaries,
        out_path=summary_path,
        months=rep_cfg.get("summary_months"),
    )

    return outputs


if __name__ == "__main__":
    outputs = main(config_path="../config/config.yaml")

    print("Representative export completed.")

    for key, value in outputs["written"].items():
        print(f"{key}: {value}")