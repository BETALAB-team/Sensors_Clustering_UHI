from pathlib import Path
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_loading import load_config, project_path, ensure_directory
from plotting import (
    load_summary_folder,
    plot_monthly_total_cooling,
    plot_monthly_sensible_latent_stacked,
    plot_monthly_electricity,
    plot_peak_day_table,
    plot_target_day_profile,
    plot_annual_or_summer_duration_curve,
    load_summary_folder,
    plot_monthly_total_cooling,
    plot_monthly_sensible_latent_stacked,
    plot_monthly_electricity,
    plot_peak_day_table,
    plot_target_day_profile,
    plot_annual_or_summer_duration_curve,
    load_cluster_labels,
    plot_all_cluster_sensor_timeseries,
)


def main(config_path = PROJECT_ROOT / "config" / "config.yaml") -> dict:
    """
    Plot simulation result figures.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to the project configuration file.

    Returns
    -------
    dict
        Dictionary containing written figure paths.
    """
    config = load_config(config_path)

    paths = config["paths"]
    plot_cfg = config.get("plotting", {})

    summary_folder = project_path(paths["cooling_summaries_dir"], PROJECT_ROOT)
    figures_dir = ensure_directory(project_path(paths["figures_dir"], PROJECT_ROOT))

    scenarios = plot_cfg.get("scenarios")
    months = plot_cfg.get("months", [5, 6, 7, 8, 9, 10])
    month_labels = plot_cfg.get("month_labels", [str(m) for m in months])
    sep = plot_cfg.get("sep", ";")

    summaries = load_summary_folder(
        summary_folder=summary_folder,
        scenarios=scenarios,
        sep=sep,
    )

    outputs = {
        "config": config,
        "summaries": summaries,
        "written": {},
    }

    outputs["written"]["monthly_total_cooling"] = plot_monthly_total_cooling(
        summaries=summaries,
        months=months,
        month_labels=month_labels,
        value_col=plot_cfg.get("total_cooling_col", "Total cooling load [kW]"),
        unit_divisor=plot_cfg.get("load_unit_divisor", 1000.0),
        ylabel=plot_cfg.get("monthly_total_ylabel", "Load [MWh]"),
        title=plot_cfg.get("monthly_total_title", "Monthly Total Cooling Load"),
        out_path=figures_dir / "monthly_total_cooling.png",
        dpi=plot_cfg.get("dpi", 300),
    )

    outputs["written"]["monthly_sensible_latent"] = plot_monthly_sensible_latent_stacked(
        summaries=summaries,
        months=months,
        month_labels=month_labels,
        sensible_col=plot_cfg.get("sensible_col", "Total sensible load [kW]"),
        latent_col=plot_cfg.get("latent_col", "Total latent load [kW]"),
        unit_divisor=plot_cfg.get("load_unit_divisor", 1000.0),
        ylabel=plot_cfg.get("monthly_stack_ylabel", "Load [MWh]"),
        title=plot_cfg.get("monthly_stack_title", "Monthly Sensible and Latent Cooling Load"),
        out_path=figures_dir / "monthly_sensible_latent_cooling.png",
        dpi=plot_cfg.get("dpi", 300),
    )

    outputs["written"]["monthly_electricity"] = plot_monthly_electricity(
        summaries=summaries,
        months=months,
        month_labels=month_labels,
        value_col=plot_cfg.get("electricity_col", "ConditioningElectricity [kW]"),
        unit_divisor=plot_cfg.get("electricity_unit_divisor", 1000.0),
        ylabel=plot_cfg.get("monthly_electricity_ylabel", "Electricity [MWh]"),
        title=plot_cfg.get("monthly_electricity_title", "Monthly Conditioning Electricity"),
        out_path=figures_dir / "monthly_conditioning_electricity.png",
        dpi=plot_cfg.get("dpi", 300),
    )

    outputs["written"]["peak_day"] = plot_peak_day_table(
        summaries=summaries,
        value_col=plot_cfg.get("peak_day_col", "Total cooling load [kW]"),
        unit_divisor=plot_cfg.get("peak_day_unit_divisor", 1000.0),
        ylabel=plot_cfg.get("peak_day_ylabel", "Daily Load [MWh]"),
        title=plot_cfg.get("peak_day_title", "Peak Cooling Day by Scenario"),
        out_path=figures_dir / "peak_day_cooling.png",
        dpi=plot_cfg.get("dpi", 300),
    )

    outputs["written"]["duration_curve"] = plot_annual_or_summer_duration_curve(
        summaries=summaries,
        value_col=plot_cfg.get("duration_curve_col", "Total cooling load [kW]"),
        ylabel=plot_cfg.get("duration_curve_ylabel", "Load [kW]"),
        title=plot_cfg.get("duration_curve_title", "Cooling Duration Curve"),
        out_path=figures_dir / "cooling_duration_curve.png",
        dpi=plot_cfg.get("dpi", 300),
    )

    target_day = plot_cfg.get("target_day")

    if target_day is not None:
        outputs["written"]["target_day_electricity"] = plot_target_day_profile(
            summaries=summaries,
            target_day=target_day,
            value_col=plot_cfg.get("target_day_col", "ConditioningElectricity [kW]"),
            ylabel=plot_cfg.get("target_day_ylabel", "Power [kW]"),
            title=plot_cfg.get("target_day_title"),
            out_path=figures_dir / f"target_day_{target_day}.png",
            dpi=plot_cfg.get("dpi", 300),
        )
    
    
    if plot_cfg.get("plot_cluster_sensor_timeseries", True):
        intermediate_dir = project_path(paths["intermediate_dir"], PROJECT_ROOT)
        cluster_exports_dir = project_path(paths["cluster_exports_dir"], PROJECT_ROOT)
    
        dry_bulb_path = intermediate_dir / "dry_bulb_resampled.parquet"
        labels_path = intermediate_dir / "sensor_cluster_labels.csv"
    
        dry_bulb = pd.read_parquet(dry_bulb_path)
        cluster_labels = load_cluster_labels(labels_path)
    
        cluster_ts_dir = ensure_directory(figures_dir / "cluster_sensor_timeseries")
    
        outputs["written"]["cluster_sensor_timeseries"] = plot_all_cluster_sensor_timeseries(
            dry_bulb=dry_bulb,
            cluster_labels=cluster_labels,
            summaries=summaries,
            representative_folder=cluster_exports_dir,
            out_dir=cluster_ts_dir,
            representative_pattern=plot_cfg.get(
                "representative_pattern",
                "cluster_{cluster_number}_representatives_*.xlsx",
            ),
            value_col=plot_cfg.get("total_cooling_col", "Total cooling load [kW]"),
            summer_months=plot_cfg.get("summer_months", [5, 6, 7, 8, 9, 10]),
            dpi=plot_cfg.get("dpi", 300),
        )
    return outputs


if __name__ == "__main__":
    outputs = main()

    print("Plotting completed.")

    for key, value in outputs["written"].items():
        print(f"{key}: {value}")