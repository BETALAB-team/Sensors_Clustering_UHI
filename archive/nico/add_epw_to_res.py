import pandas as pd
from pathlib import Path

# ===== Manual mapping between sheet names and EPW filenames =====
SHEET_TO_EPW = {
    "EnergyPlus TMY Venezia Tessara": "ITA_Venezia-Tessera.161050_IGDG.epw",
    "Padova TMY 2009 - 2023": "ITA_VN_Padova.AP.160950_TMYx.2009-2023.epw",
    "ARPAV CITY 2024": "ITA_Venezia-Tessera.161050_IGDG__arpav_city.epw",
    "ARPAV_RURAL_2024 ": "ITA_Venezia-Tessera.161050_IGDG__arpav_rural.epw",
    "CLUSTER_00": "ITA_Venezia-Tessera.161050_IGDG___cluster_0_tmy.epw",
    "CLUSTER_01": "ITA_Venezia-Tessera.161050_IGDG___cluster_1_tmy.epw",
    "CLUSTER_02": "ITA_Venezia-Tessera.161050_IGDG___cluster_2_tmy.epw"
}

# EPW column names
EPW_COLS = [
    "Year","Month","Day","Hour","Minute","DataSource",
    "DryBulb","DewPoint","RelHum","AtmPressure",
    "ETR","ETRN","HIR","GHI","DNI","DHI",
    "GHI_Ill","DNI_Ill","DHI_Ill","ZenLum",
    "WindDir","WindSpd","TotSkyCvr","OpaqSkyCvr",
    "Visibility","CeilingHgt","PresWeathObs","PresWeathCodes",
    "PrecipWtr","AOD","SnowDepth","DaysSinceLastSnow",
    "Albedo","LiqPrecDepth","LiqPrecQty"
]

def read_epw(epw_path: Path) -> pd.DataFrame:
    """
    Read an EPW file into a DataFrame with only Month, Day, Hour, DryBulb, RelHum.
    """
    df = pd.read_csv(epw_path, header=None, skiprows=8)
    n = min(len(df.columns), len(EPW_COLS))
    df = df.iloc[:, :n]
    df.columns = EPW_COLS[:n]
    return df[["Month", "Day", "Hour", "DryBulb", "RelHum"]].copy()

def fill_temp_hum_from_epw(
    excel_path: str | Path,
    epw_folder: str | Path,
    time_col: str = "Time",
    temp_col: str = "External Temperature [°C]",
    rh_col: str = "External Relative Humidity [%]"
) -> None:
    """
    Fill temperature and humidity columns in all sheets based on EPW files.
    """
    excel_path = Path(excel_path)
    epw_folder = Path(epw_folder)

    # Read all sheets
    sheets = pd.read_excel(excel_path, sheet_name=None)

    updated_sheets = {}

    for sheet_name, df in sheets.items():
        if sheet_name not in SHEET_TO_EPW:
            print(f"[WARN] No EPW mapping for sheet '{sheet_name}'. Skipping.")
            updated_sheets[sheet_name] = df
            continue

        epw_path = epw_folder / SHEET_TO_EPW[sheet_name]
        if not epw_path.exists():
            print(f"[WARN] EPW file missing for sheet '{sheet_name}': {epw_path.name}")
            updated_sheets[sheet_name] = df
            continue

        # Read EPW
        epw_df = read_epw(epw_path)

        if time_col not in df.columns:
            print(f"[WARN] Sheet '{sheet_name}' missing '{time_col}' column.")
            updated_sheets[sheet_name] = df
            continue

        # Convert times in Excel to datetime
        ts = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
        if ts.isna().all():
            print(f"[WARN] Couldn't parse '{time_col}' in '{sheet_name}'. Skipping.")
            updated_sheets[sheet_name] = df
            continue

        # Add temporary merge keys
        work = df.copy()
        work["_Month"] = ts.dt.month
        work["_Day"] = ts.dt.day
        work["_Hour"] = ts.dt.hour + 1  # EPW hours are 1-24

        # Merge EPW data into sheet
        merged = work.merge(
            epw_df,
            left_on=["_Month", "_Day", "_Hour"],
            right_on=["Month", "Day", "Hour"],
            how="left"
        )

        # Fill the temperature and humidity columns
        merged[temp_col] = merged["DryBulb"]
        merged[rh_col] = merged["RelHum"]

        # Drop helper columns
        merged.drop(columns=["_Month","_Day","_Hour","Month","Day","Hour","DryBulb","RelHum"], inplace=True, errors="ignore")

        updated_sheets[sheet_name] = merged
        print(f"[OK] Filled '{sheet_name}' using '{epw_path.name}'.")

    # Save results
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        for name, out_df in updated_sheets.items():
            out_df.to_excel(writer, sheet_name=name, index=False)

    print(f"[DONE] Updated workbook saved: {excel_path}")

# --- Example usage ------------------------------------------------------------
fill_temp_hum_from_epw(
    excel_path=r"C:/Works/Sensors/Sensors/Simulation_Results_hourly.xlsx",
    epw_folder=r"C:/Works/Sensors/Sensors/epw"
)
#%%
import pandas as pd
from pathlib import Path

def summarize_monthly_temp(
    excel_path: str | Path,
    time_col: str = "Time",
    temp_col: str = "External Temperature [°C]"
) -> pd.DataFrame:
    """
    For every sheet in the workbook, compute monthly min/mean/max temperature.
    Prints the summary and returns a DataFrame with a MultiIndex (Sheet, Month).
    """
    excel_path = Path(excel_path)
    sheets = pd.read_excel(excel_path, sheet_name=None)

    results = []

    for sname, df in sheets.items():
        if time_col not in df.columns or temp_col not in df.columns:
            print(f"[WARN] '{sname}' missing required columns. Skipping.")
            continue

        # parse datetime and drop rows with invalid time or temp
        t = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
        temp = pd.to_numeric(df[temp_col], errors="coerce")

        valid = (~t.isna()) & (~temp.isna())
        if not valid.any():
            print(f"[INFO] '{sname}' has no valid {time_col}/{temp_col} rows.")
            continue

        work = pd.DataFrame({
            "Month": t[valid].dt.month,
            "Temp": temp[valid]
        })

        agg = (work
               .groupby("Month")["Temp"]
               .agg(Min="min", Mean="mean", Max="max")
               .round(3))

        # Pretty print
        print(f"\n=== {sname} ===")
        # add month names for print
        printable = agg.copy()
        printable.index = printable.index.map(lambda m: f"{m:02d} - {pd.Timestamp(2000, m, 1).strftime('%B')}")
        print(printable.to_string())

        # store for return with sheet label
        agg.insert(0, "Sheet", sname)
        agg.reset_index(inplace=True)
        results.append(agg)

    if not results:
        return pd.DataFrame(columns=["Sheet", "Month", "Min", "Mean", "Max"])

    out = pd.concat(results, ignore_index=True)
    # Optional: add month name column
    out["MonthName"] = out["Month"].apply(lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))
    # order columns
    out = out[["Sheet", "Month", "MonthName", "Min", "Mean", "Max"]]
    return out

# Example usage:
summary_df = summarize_monthly_temp(r"C:/Works/Sensors/Sensors/Simulation_Results_hourly.xlsx")
# summary_df.to_excel(r"C:\path\to\monthly_temp_summary.xlsx", index=False)  # optional
