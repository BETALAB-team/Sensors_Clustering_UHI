from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_loading import (
    load_config,
    project_path,
    ensure_directory,
    load_project_inputs,
    filter_reliable_sensors,
)
from clustering import (
    build_wide_tables,
    resample_with_time_interpolation,
    wet_bulb_temperature,
    prepare_db_wb_tables,
    run_feature_kmeans,
    evaluate_k_range,
    add_cluster_labels_to_locations,
)


def main(config_path = PROJECT_ROOT / "config" / "config.yaml"):
    """
    Run sensor clustering.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to the project configuration file.

    Returns
    -------
    dict
        Dictionary containing clustering outputs and written file paths.
    """
    config = load_config(config_path)

    paths = config["paths"]
    sensor_cfg = config.get("sensor_filtering", {})
    clustering_cfg = config.get("clustering", {})

    output_dir = ensure_directory(project_path(paths["intermediate_dir"], PROJECT_ROOT))
    figures_dir = ensure_directory(project_path(paths["figures_dir"], PROJECT_ROOT))

    inputs = load_project_inputs(config, project_root=PROJECT_ROOT)

    sensor_data = inputs["sensor_data"]
    reliability_table = inputs["reliability_table"]
    sensor_locations = inputs["sensor_locations"]

    filtered_sensors = filter_reliable_sensors(
        sensor_data=sensor_data,
        reliability_table=reliability_table,
        sensor_prefix=sensor_cfg.get("sensor_prefix", "T"),
        reliability_index_threshold=sensor_cfg.get("reliability_index_threshold", 3),
        drop_sensors=sensor_cfg.get("drop_sensors", []),
    )

    temperature, humidity = build_wide_tables(filtered_sensors)

    raw_freq = clustering_cfg.get("raw_resample_frequency", "10min")
    cluster_freq = clustering_cfg.get("time_resample", "30min")
    n_clusters = clustering_cfg.get("final_clusters", 4)
    random_state = clustering_cfg.get("random_state", 42)
    drop_sensors = sensor_cfg.get("drop_sensors", [])

    temperature = temperature.drop(columns=drop_sensors, errors="ignore")
    humidity = humidity.drop(columns=drop_sensors, errors="ignore")

    temperature_resampled = resample_with_time_interpolation(
        temperature,
        freq=raw_freq,
        max_gap=clustering_cfg.get("max_interpolation_gap", "30min"),
    )

    humidity_resampled = resample_with_time_interpolation(
        humidity,
        freq=raw_freq,
        max_gap=clustering_cfg.get("max_interpolation_gap", "30min"),
    )

    wet_bulb = wet_bulb_temperature(
        dry_bulb=temperature_resampled,
        relative_humidity=humidity_resampled,
    )

    db, wb, sensors = prepare_db_wb_tables(
        dry_bulb=temperature_resampled,
        wet_bulb=wet_bulb,
        freq=cluster_freq,
        drop_sensors=drop_sensors,
    )

    result = run_feature_kmeans(
        dry_bulb=db,
        wet_bulb=wb,
        n_clusters=n_clusters,
        freq=cluster_freq,
        random_state=random_state,
    )

    k_min = clustering_cfg.get("cluster_range", {}).get("min", 2)
    k_max = clustering_cfg.get("cluster_range", {}).get("max", 9)

    k_eval = evaluate_k_range(
        x=result.scaled_features,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )

    cluster_labels = result.labels

    clustered_locations = add_cluster_labels_to_locations(
        locations=sensor_locations,
        cluster_labels=cluster_labels,
        sensor_column=clustering_cfg.get("sensor_location_name_col", "name"),
        output_column="cluster",
    )

    labels_path = output_dir / "sensor_cluster_labels.csv"
    features_path = output_dir / "sensor_cluster_features.csv"
    k_eval_path = output_dir / "cluster_k_evaluation.csv"
    temperature_path = output_dir / "dry_bulb_resampled.parquet"
    humidity_path = output_dir / "relative_humidity_resampled.parquet"
    wet_bulb_path = output_dir / "wet_bulb_resampled.parquet"
    db_cluster_path = output_dir / "dry_bulb_for_clustering.parquet"
    wb_cluster_path = output_dir / "wet_bulb_for_clustering.parquet"
    locations_path = output_dir / "sensor_locations_with_clusters.geojson"

    cluster_labels.to_frame().to_csv(labels_path)
    result.features.to_csv(features_path)
    k_eval.to_csv(k_eval_path, index=False)

    temperature_resampled.to_parquet(temperature_path)
    humidity_resampled.to_parquet(humidity_path)
    wet_bulb.to_parquet(wet_bulb_path)
    db.to_parquet(db_cluster_path)
    wb.to_parquet(wb_cluster_path)

    clustered_locations.to_file(locations_path, driver="GeoJSON")

    outputs = {
        "config": config,
        "filtered_sensors": filtered_sensors,
        "dry_bulb": temperature_resampled,
        "relative_humidity": humidity_resampled,
        "wet_bulb": wet_bulb,
        "dry_bulb_for_clustering": db,
        "wet_bulb_for_clustering": wb,
        "cluster_result": result,
        "cluster_labels": cluster_labels,
        "k_evaluation": k_eval,
        "clustered_locations": clustered_locations,
        "written": {
            "labels": labels_path,
            "features": features_path,
            "k_evaluation": k_eval_path,
            "dry_bulb": temperature_path,
            "relative_humidity": humidity_path,
            "wet_bulb": wet_bulb_path,
            "dry_bulb_for_clustering": db_cluster_path,
            "wet_bulb_for_clustering": wb_cluster_path,
            "clustered_locations": locations_path,
        },
    }

    return outputs


if __name__ == "__main__":
    outputs = main(config_path="../config/config.yaml")

    print("Clustering completed.")
    print(f"Labels: {outputs['written']['labels']}")
    print(f"Features: {outputs['written']['features']}")
    print(f"K evaluation: {outputs['written']['k_evaluation']}")
    print(f"Clustered locations: {outputs['written']['clustered_locations']}")