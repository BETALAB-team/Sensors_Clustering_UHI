# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 07:45:22 2025

@author: khajmoh18975
"""

import pandas as pd
from pathlib import Path
import numpy as np

TOTAL_AREA = 146318.5715

def conditioning_electricity(cluster_epw_path: str, sim_results_path: str) -> None:
    epw_dir = Path(cluster_epw_path)
    sim_dir = Path(sim_results_path)

    epw_map = {}
    for f in epw_dir.glob("*.epw"):
        name = f.name.lower()
        if "all" in name or "summer" in name:
            continue
        if "cluster_1" in name:
            epw_map["cluster_1"] = f
        elif "cluster_2" in name:
            epw_map["cluster_2"] = f
        elif "cluster_3" in name:
            epw_map["nearest_representative"] = f
        elif "cluster_4" in name:
            epw_map["cluster_4"] = f
        elif "rural" in name:
            epw_map["ARPAV_rural"] = f
        elif "city" in name or "suburban" in name:
            epw_map["ARPAV_suburban"] = f
        elif "tmy" in name:
            epw_map["TMY"] = f

    def eer_from_temp(T):
        return np.clip(-0.2 * T + 11.2, 2.5, 7.0)

    for folder in sim_dir.iterdir():
        if not folder.is_dir():
            continue

        fname = folder.name.lower()
        if "cluster_1" in fname:
            epw = epw_map.get("cluster_1")
        elif "cluster_2" in fname:
            epw = epw_map.get("cluster_2")
        elif "cluster_3" in fname:
            epw = epw_map.get("nearest_representative")
        elif "cluster_4" in fname:
            epw = epw_map.get("cluster_4")
        elif "rural" in fname:
            epw = epw_map.get("ARPAV_rural")
        elif "suburban" in fname:
            epw = epw_map.get("ARPAV_suburban")
        elif "tmy" in fname:
            epw = epw_map.get("TMY")
        elif "nearest" in fname:
            epw = epw_map.get("nearest_representative")
        else:
            continue

        epw_df = pd.read_csv(epw, skiprows=8, header=None)
        temps = epw_df.iloc[:, 6].astype(float).values
        EER_all = eer_from_temp(temps)

        for csv_file in folder.glob("Results Bd Building*.csv"):
            df = pd.read_csv(csv_file)

            s = pd.to_numeric(df["TZ sensible load [kW]"], errors="coerce").fillna(0.0)
            l = pd.to_numeric(df["TZ latent load [kW]"], errors="coerce").fillna(0.0)
            Q = s + l  

            n = min(len(Q), len(EER_all))
            Q = Q.iloc[:n].to_numpy()
            EER = EER_all[:n]

           
            neg = Q[Q < 0]
            if neg.size == 0:
                elec = np.zeros(len(df))
                df["ConditioningElectricity [kW]"] = elec
                df.to_csv(csv_file, sep=";", index=False)
                continue

            
            abs_neg = -neg
            p99 = np.percentile(abs_neg, 99)
            design_abs = np.ceil(p99 / 10.0) * 10.0 
            design = -design_abs

            
            Q_cool = Q.copy()
            Q_cool[Q_cool > 0] = 0.0
            cool_abs = -Q_cool  

            
            used_abs = np.minimum(cool_abs, design_abs)

            
            elec_head = np.where(cool_abs > 0, used_abs / EER, 0.0)

            
            elec = np.zeros(len(df))
            elec[:n] = elec_head

            df["ConditioningElectricity [kW]"] = elec
            df.to_csv(csv_file, sep=";", index=False)
            
            
def summarize_summer_cooling(sim_results_path: str,
                             output_folder_name: str = "Cooling_summaries") -> None:
    sim_dir = Path(sim_results_path)
    out_dir = sim_dir / output_folder_name
    out_dir.mkdir(exist_ok=True)

    start_idx = 120 * 24
    end_idx = 304 * 24
    n_summer = end_idx - start_idx
    time_index = pd.date_range("2005-05-01 00:00", "2005-10-31 23:00", freq="H")

    for folder in sim_dir.iterdir():
        if not folder.is_dir():
            continue
        if folder.name == output_folder_name:
            continue

        sens_sum = np.zeros(n_summer)
        lat_sum = np.zeros(n_summer)
        elec_sum = np.zeros(n_summer)
        has_any = False

        for csv_file in folder.glob("Results Bd Building*.csv"):
            df = pd.read_csv(csv_file, sep=";")
            s = pd.to_numeric(df["TZ sensible load [kW]"], errors="coerce").fillna(0.0).to_numpy()
            l = pd.to_numeric(df["TZ latent load [kW]"], errors="coerce").fillna(0.0).to_numpy()
            e = pd.to_numeric(df["ConditioningElectricity [kW]"], errors="coerce").fillna(0.0).to_numpy()

            n = len(s)
            s_start = min(start_idx, n)
            s_end = min(end_idx, n)
            if s_end <= s_start:
                continue

            length = s_end - s_start
            if length != n_summer:
                L = min(length, n_summer)
                sens_slice = s[s_start:s_start + L]
                lat_slice = l[s_start:s_start + L]
                elec_slice = e[s_start:s_start + L]
                sens_sum[:L] += np.where(sens_slice < 0, sens_slice, 0.0)
                lat_sum[:L] += np.where(lat_slice < 0, lat_slice, 0.0)
                elec_sum[:L] += elec_slice
            else:
                sens_slice = s[s_start:s_end]
                lat_slice = l[s_start:s_end]
                elec_slice = e[s_start:s_end]
                sens_sum += np.where(sens_slice < 0, sens_slice, 0.0)
                lat_sum += np.where(lat_slice < 0, lat_slice, 0.0)
                elec_sum += elec_slice

            has_any = True

        if not has_any:
            continue

        sens_pos = -sens_sum
        lat_pos = -lat_sum
        cooling = sens_pos + lat_pos

        summary_df = pd.DataFrame({
            "Time": time_index,
            "Total sensible load [kW]": sens_pos,
            "Total latent load [kW]": lat_pos,
            "Total cooling load [kW]": cooling,
            "ConditioningElectricity [kW]": elec_sum
        })

        out_csv = out_dir / f"{folder.name}_summer_summary.csv"
        summary_df.to_csv(out_csv, sep=";", index=False)
        


def build_cooling_excel(summary_folder_path: str,
                        excel_out_path: str = "building_cooling_summary.xlsx") -> None:
    summary_dir = Path(summary_folder_path)

    def pretty_name(p: Path) -> str:
        n = p.name.lower()
        if "cluster_1" in n:
            return "building_cooling_summary_cluster1"
        if "cluster_2" in n:
            return "building_cooling_summary_cluster2"
        if "cluster_3" in n:
            return "building_cooling_summary_cluster3"
        if "cluster_4" in n:
            return "building_cooling_summary_cluster4"
        if "nearest" in n:
            return "building_cooling_summary_nearests"
        if "rural" in n:
            return "building_cooling_summary_rural"
        if "suburban" in n or "city" in n:
            return "building_cooling_summary_suburban"
        if "tmy" in n:
            return "building_cooling_summary_TMY"
        return p.stem

    rows_total = []
    rows_spec = []

    for csv_file in sorted(summary_dir.glob("*.csv")):
        df = pd.read_csv(csv_file, sep=";")

        sens = pd.to_numeric(df["Total sensible load [kW]"], errors="coerce").fillna(0.0)
        lat = pd.to_numeric(df["Total latent load [kW]"], errors="coerce").fillna(0.0)
        cool = pd.to_numeric(df["Total cooling load [kW]"], errors="coerce").fillna(0.0)
        elec = pd.to_numeric(df["ConditioningElectricity [kW]"], errors="coerce").fillna(0.0)

        sens_kWh = sens.sum()
        lat_kWh = lat.sum()
        cool_kWh = cool.sum()
        elec_kWh = elec.sum()

        sens_GWh = sens_kWh / 1_000_000.0
        lat_GWh = lat_kWh / 1_000_000.0
        cool_GWh = cool_kWh / 1_000_000.0
        elec_GWh = elec_kWh / 1_000_000.0

        max_cool_kW = cool.max()
        max_elec_kW = elec.max()

        sens_kWh_m2 = sens_kWh / TOTAL_AREA
        lat_kWh_m2 = lat_kWh / TOTAL_AREA
        cool_kWh_m2 = cool_kWh / TOTAL_AREA
        elec_W_m2 = elec_kWh * 1000.0 / TOTAL_AREA
        max_cool_W_m2 = max_cool_kW * 1000.0 / TOTAL_AREA
        max_elec_W_m2 = max_elec_kW * 1000.0 / TOTAL_AREA

        name = pretty_name(csv_file)

        rows_total.append(
            [name, sens_GWh, lat_GWh, cool_GWh, elec_GWh, max_cool_kW, max_elec_kW]
        )
        rows_spec.append(
            [name, sens_kWh_m2, lat_kWh_m2, cool_kWh_m2, elec_W_m2, max_cool_W_m2, max_elec_W_m2]
        )

    cols_total = [
        "File",
        "Total Sensible Demand [GWh]",
        "Total Latent Demand [GWh]",
        "Total Cooling Demand [GWh]",
        "Total Electric Load [GWh]",
        "Max Cooling Load [kW]",
        "Max Electric Load [kW]",
    ]

    cols_spec = [
        "File",
        "Total Sensible Demand [kWh/m2]",
        "Total Latent Demand [kWh/m2]",
        "Total Cooling Demand [kWh/m2]",
        "Total Electric Load [W/m2]",
        "Max Cooling Load [W/m2]",
        "Max Electric Load [W/m2]",
    ]

    df_total = pd.DataFrame(rows_total, columns=cols_total)
    df_spec = pd.DataFrame(rows_spec, columns=cols_spec)

    with pd.ExcelWriter(excel_out_path) as writer:
        df_total.to_excel(writer, sheet_name="Cooling", index=False, startrow=1)
        df_spec.to_excel(writer, sheet_name="Cooling", index=False, startrow=len(df_total) + 4)

        ws = writer.sheets["Cooling"]
        ws.cell(row=1, column=1, value="")
        ws.cell(row=1, column=2, value="Total")
        ws.cell(row=len(df_total) + 3, column=1, value="Specific")
        
        
        
import pandas as pd
from pathlib import Path

coeffs = {
    "cluster_1_representatives_hourly_fromexcel": (47.85/41.28, 20.55/7.12),
    "cluster_2_representatives_hourly_fromexcel": (47.85/39.11, 20.56/7.13),
    "cluster_3_representatives_hourly_fromexcel": (48.31/38.81, 20.44/7.07),
    "cluster_4_representatives_hourly_fromexcel": (47.97/37.91, 20.53/7.08),
    "nearest_representative": (47.91/39.10, 20.54/7.12),
    "ARPAV_rural": (47.87/34.23, 20.54/9.91),
    "ARPAV_suburban": (49.13/35.91, 22.55/8.15),
    "ITA_VN_Padova.AP.160950_TMYx.2009-2023": (44.65/29.76, 10.99/7.84),
}

def undo_adjust_negative_loads(sim_results_root: str) -> None:
    root = Path(sim_results_root)
    i=0
    for folder, (sens_coeff, lat_coeff) in coeffs.items():
        folder_path = root / folder
        if not folder_path.exists():
            continue

        for csv_file in folder_path.glob("Results Bd Building*.csv"):
            df = pd.read_csv(csv_file, sep= ";")

            if "TZ sensible load [kW]" in df.columns:
                if i == 0:
                    print(1)
                    i=i+1
                sens = pd.to_numeric(df["TZ sensible load [kW]"], errors="coerce")
                mask_sens = sens < 0
                df.loc[mask_sens, "TZ sensible load [kW]"] = sens[mask_sens] / sens_coeff

            if "TZ latent load [kW]" in df.columns:
                lat = pd.to_numeric(df["TZ latent load [kW]"], errors="coerce")
                mask_lat = lat < 0
                df.loc[mask_lat, "TZ latent load [kW]"] = lat[mask_lat] / lat_coeff

            df.to_csv(csv_file, index=False)