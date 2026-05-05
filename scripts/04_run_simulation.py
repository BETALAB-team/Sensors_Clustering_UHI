from pathlib import Path
import sys
import shutil
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_loading import load_config, project_path, ensure_directory

from eureca_building.config import load_config as load_eureca_config
from eureca_ubem.city import City


def run_one_simulation(
    config_path,
    weather_file,
    schedules_file,
    materials_file,
    city_model_file,
    systems_file,
    output_folder,
    building_model="1C",
    shading_calculation=True,
    quasi_steady_state=False,
    output_type="csv",
):
    """
    Run one EUReCA simulation using one EPW weather file.

    Parameters
    ----------
    config_path : str | Path
        Path to the EUReCA JSON configuration file.
    weather_file : str | Path
        Path to the EPW weather file.
    schedules_file : str | Path
        Path to the schedules Excel file.
    materials_file : str | Path
        Path to the materials Excel file.
    city_model_file : str | Path
        Path to the city GeoJSON file.
    systems_file : str | Path
        Path to the systems Excel file.
    output_folder : str | Path
        Folder where EUReCA outputs will be written.
    building_model : str, optional
        EUReCA building model type.
    shading_calculation : bool, optional
        Whether to run shading calculation.
    quasi_steady_state : bool, optional
        Whether to run quasi-steady-state simulation.
    output_type : str, optional
        EUReCA output type.

    Returns
    -------
    Path
        Simulation output folder.
    """
    config_path = Path(config_path)
    weather_file = Path(weather_file)
    schedules_file = Path(schedules_file)
    materials_file = Path(materials_file)
    city_model_file = Path(city_model_file)
    systems_file = Path(systems_file)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    load_eureca_config(str(config_path))


    city = City(
        city_model=str(city_model_file),
        epw_weather_file=str(weather_file),
        end_uses_types_file=str(schedules_file),
        envelope_types_file=str(materials_file),
        systems_templates_file=str(systems_file),
        shading_calculation=shading_calculation,
        building_model=building_model,
        output_folder=str(output_folder),
    )

    if quasi_steady_state:
        city.simulate_quasi_steady_state()
    else:
        city.simulate(output_type=output_type)

    return output_folder


def main(config_path=PROJECT_ROOT / "config" / "config.yaml") -> dict:
    """
    Run EUReCA simulations for all generated EPW files.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to the project configuration file.

    Returns
    -------
    dict
        Dictionary containing simulation output folders.
    """
    config = load_config(config_path)

    paths = config["paths"]
    sim_cfg = config.get("simulation", {})

    epw_folder = project_path(sim_cfg.get("epw_folder", paths["epw_output_dir"]), PROJECT_ROOT)
    output_root = ensure_directory(
        project_path(sim_cfg.get("simulation_results_dir", paths["simulation_results_dir"]), PROJECT_ROOT)
    )

    eureca_config = project_path(sim_cfg["eureca_config"], PROJECT_ROOT)
    schedules_file = project_path(sim_cfg["schedules_file"], PROJECT_ROOT)
    materials_file = project_path(sim_cfg["materials_file"], PROJECT_ROOT)
    city_model_file = project_path(sim_cfg["city_model_file"], PROJECT_ROOT)
    systems_file = project_path(sim_cfg["systems_file"], PROJECT_ROOT)

    epw_pattern = sim_cfg.get("epw_pattern", "cluster_*_representative*.epw")
    overwrite = sim_cfg.get("overwrite", False)

    building_model = sim_cfg.get("building_model", "1C")
    shading_calculation = sim_cfg.get("shading_calculation", True)
    quasi_steady_state = sim_cfg.get("quasi_steady_state", False)
    output_type = sim_cfg.get("output_type", "csv")

    epw_files = sorted(epw_folder.glob(epw_pattern))

    if not epw_files:
        raise FileNotFoundError(f"No EPW files found in {epw_folder} with pattern {epw_pattern}")

    written = []

    for epw_file in epw_files:
        scenario_name = epw_file.stem
        scenario_output = output_root / scenario_name
        if scenario_output.exists() and overwrite:
            shutil.rmtree(scenario_output)

        if scenario_output.exists() and any(scenario_output.glob("Results Bd Building*.csv")):
            written.append(scenario_output)
            continue

        out = run_one_simulation(
            config_path=eureca_config,
            weather_file=epw_file,
            schedules_file=schedules_file,
            materials_file=materials_file,
            city_model_file=city_model_file,
            systems_file=systems_file,
            output_folder=scenario_output,
            building_model=building_model,
            shading_calculation=shading_calculation,
            quasi_steady_state=quasi_steady_state,
            output_type=output_type,
        )

        written.append(out)

    return {
        "config": config,
        "written": {
            "simulation_folders": written,
        },
    }


if __name__ == "__main__":
    outputs = main()

    print("Simulations completed.")

    for path in outputs["written"]["simulation_folders"]:
        print(path)