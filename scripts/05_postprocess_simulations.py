from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_loading import load_config, project_path, ensure_directory
from simulation_postprocess import (
    add_conditioning_electricity,
    summarize_summer_cooling,
    build_cooling_excel,
    peak_cooling_days,
    monthly_cooling_summary,
)


def main(
    config_path= PROJECT_ROOT / "config" / "config.yaml",
    add_electricity: bool = True,
    summarize: bool = True,
    build_excel: bool = True,
    build_extra_tables: bool = True,
) -> dict:
    """
    Run simulation post-processing.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to the project configuration file.
    add_electricity : bool, optional
        Whether to add conditioning electricity to building result files.
    summarize : bool, optional
        Whether to create summer cooling summary CSV files.
    build_excel : bool, optional
        Whether to create the cooling summary Excel workbook.
    build_extra_tables : bool, optional
        Whether to create peak-day and monthly summary tables.

    Returns
    -------
    dict
        Dictionary containing generated output paths.
    """
    config = load_config(config_path)

    paths = config["paths"]
    post_cfg = config.get("simulation_postprocess", {})
    cooling_cfg = post_cfg.get("cooling", {})
    eer_cfg = post_cfg.get("eer_model", {})

    epw_dir = project_path(paths["epw_output_dir"], PROJECT_ROOT)
    sim_dir = project_path(paths["simulation_results_dir"], PROJECT_ROOT)
    summary_dir = ensure_directory(project_path(paths["cooling_summaries_dir"], PROJECT_ROOT))
    output_dir = ensure_directory(project_path(paths["output_dir"], PROJECT_ROOT))

    sensible_col = cooling_cfg.get("sensible_col", "TZ sensible load [kW]")
    latent_col = cooling_cfg.get("latent_col", "TZ latent load [kW]")
    electricity_col = cooling_cfg.get("electricity_col", "ConditioningElectricity [kW]")

    outputs = {
        "config": config,
        "written": {},
    }
    print(1)
    if add_electricity:
        building_files = add_conditioning_electricity(
            epw_folder=epw_dir,
            simulation_results_path=sim_dir,
            sensible_col=sensible_col,
            latent_col=latent_col,
            electricity_col=electricity_col,
            building_pattern=post_cfg.get("building_pattern", "Results Bd Building*.csv"),
            sep=post_cfg.get("sep", ";"),
            eer_slope=eer_cfg.get("slope", -0.2),
            eer_intercept=eer_cfg.get("intercept", 11.2),
            min_eer=eer_cfg.get("min_eer", 2.5),
            max_eer=eer_cfg.get("max_eer", 7.0),
            design_percentile=post_cfg.get("design_percentile", 99.0),
            design_rounding_kw=post_cfg.get("design_rounding_kw", 10.0),
        )

        outputs["written"]["building_files"] = building_files
    print(1)

    if summarize:
        summary_files = summarize_summer_cooling(
            simulation_results_path=sim_dir,
            output_folder_name=summary_dir.name,
            start_hour_index=post_cfg.get("summer_start_hour_index", 120 * 24),
            end_hour_index=post_cfg.get("summer_end_hour_index", 304 * 24),
            start_time=post_cfg.get("summer_start", "2005-05-01 00:00"),
            end_time=post_cfg.get("summer_end", "2005-10-31 23:00"),
            sensible_col=sensible_col,
            latent_col=latent_col,
            electricity_col=electricity_col,
            building_pattern=post_cfg.get("building_pattern", "Results Bd Building*.csv"),
            sep=post_cfg.get("sep", ";"),
        )

        outputs["written"]["summary_files"] = summary_files
    print(1)

    if build_excel:
        excel_path = project_path(
            paths.get("cooling_summary_excel", "data/output/building_cooling_summary.xlsx"),
            PROJECT_ROOT,
        )

        outputs["written"]["summary_excel"] = build_cooling_excel(
            summary_folder_path=summary_dir,
            excel_out_path=excel_path,
            total_area_m2=post_cfg.get("total_area_m2", 146318.5715),
            sep=post_cfg.get("sep", ";"),
        )
    print(1)

    if build_extra_tables:
        peak_days = peak_cooling_days(
            summary_folder_path=summary_dir,
            cooling_col=post_cfg.get("peak_day_col", "Total cooling load [kW]"),
            sep=post_cfg.get("sep", ";"),
        )

        monthly = monthly_cooling_summary(
            summary_folder_path=summary_dir,
            months=post_cfg.get("months", [5, 6, 7, 8, 9, 10]),
            sep=post_cfg.get("sep", ";"),
        )

        peak_path = output_dir / "peak_cooling_days.csv"
        monthly_path = output_dir / "monthly_cooling_summary.csv"

        peak_days.to_csv(peak_path, index=False)
        monthly.to_csv(monthly_path, index=False)

        outputs["written"]["peak_days"] = peak_path
        outputs["written"]["monthly_summary"] = monthly_path
    print(1)

    return outputs


if __name__ == "__main__":
    outputs = main()

    print("Post-processing completed.")

    for key, value in outputs["written"].items():
        print(f"{key}: {value}")