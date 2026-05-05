from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_loading import load_config, project_path, ensure_directory
from epw_tools import batch_convert_weather_excels_to_epw


def main(config_path = PROJECT_ROOT / "config" / "config.yaml") -> dict:
    """
    Convert representative weather Excel files to EPW files.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to the project configuration file.

    Returns
    -------
    dict
        Dictionary containing written EPW paths.
    """
    config = load_config(config_path)

    paths = config["paths"]
    epw_cfg = config.get("epw", {})

    base_epw_path = project_path(paths["epw_base_file"], PROJECT_ROOT)
    input_dir = project_path(paths.get("epw_input_dir", paths["cluster_exports_dir"]), PROJECT_ROOT)
    output_dir = ensure_directory(project_path(paths["epw_output_dir"], PROJECT_ROOT))

    written = batch_convert_weather_excels_to_epw(
        base_epw_path=base_epw_path,
        in_dir=input_dir,
        out_dir=output_dir,
        pattern=epw_cfg.get("excel_pattern", "*.xlsx"),
        suffix=epw_cfg.get("epw_suffix", ".epw"),
        time_col=epw_cfg.get("excel_time_col", "time"),
        db_col=epw_cfg.get("excel_db_col", "db_temp"),
        wb_col=epw_cfg.get("excel_wb_col", "wb_temp"),
        rh_col=epw_cfg.get("excel_rh_col", "rh"),
        year_override=epw_cfg.get("epw_year_override"),
        replace_dry_bulb=epw_cfg.get("replace_dry_bulb", True),
        replace_relative_humidity=epw_cfg.get("replace_relative_humidity", True),
        recompute_dew_point=epw_cfg.get("recompute_dew_point", True),
    )

    return {
        "config": config,
        "written": {
            "epw_files": written,
        },
    }


if __name__ == "__main__":
    outputs = main()

    print("EPW conversion completed.")

    for path in outputs["written"]["epw_files"]:
        print(path)