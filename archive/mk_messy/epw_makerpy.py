# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 13:31:01 2025

@author: khajmoh18975
"""

#!/usr/bin/env python3
# excel_to_epw_batch.py
import argparse
import math
from pathlib import Path
import pandas as pd
import numpy as np

# ---- EPW helpers ------------------------------------------------------------
EPW_COLS = [
    "Year","Month","Day","Hour","Minute","Data Source and Uncertainty Flags",
    "Dry Bulb Temperature","Dew Point Temperature","Relative Humidity",
    "Atmospheric Station Pressure","Extraterrestrial Horizontal Radiation",
    "Extraterrestrial Direct Normal Radiation","Horizontal Infrared Radiation Intensity",
    "Global Horizontal Radiation","Direct Normal Radiation","Diffuse Horizontal Radiation",
    "Global Horizontal Illuminance","Direct Normal Illuminance","Diffuse Horizontal Illuminance",
    "Zenith Luminance","Wind Direction","Wind Speed","Total Sky Cover",
    "Opaque Sky Cover","Visibility","Ceiling Height","Present Weather Observation",
    "Present Weather Codes","Precipitable Water","Aerosol Optical Depth",
    "Snow Depth","Days Since Last Snow","Albedo","Liquid Precipitation Depth",
    "Liquid Precipitation Quantity"
]

def read_epw(epw_path: Path):
    """Return (header_lines:list[str], df:DataFrame with EPW_COLS)."""
    header = []
    data_rows = []
    with epw_path.open("r", encoding="utf-8", errors="ignore") as f:
        # EPW has 8 header lines
        for _ in range(8):
            header.append(f.readline().rstrip("\n"))
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            # pad if short
            if len(parts) < len(EPW_COLS):
                parts += [""] * (len(EPW_COLS) - len(parts))
            data_rows.append(parts[:len(EPW_COLS)])
    df = pd.DataFrame(data_rows, columns=EPW_COLS)
    # Numeric conversion where appropriate
    num_cols = [c for c in EPW_COLS if c not in ("Data Source and Uncertainty Flags","Present Weather Observation","Present Weather Codes")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return header, df

def write_epw(header, df, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for h in header:
            f.write(h + "\n")
        for _, r in df.iterrows():
            vals = [r[c] if c in df.columns else "" for c in EPW_COLS]
            # ensure types/format
            vals_fmt = []
            for c, v in zip(EPW_COLS, vals):
                if isinstance(v, float):
                    vals_fmt.append(f"{v:.3f}")
                else:
                    vals_fmt.append(str(int(v)) if isinstance(v, (np.integer, int)) else str(v))
            f.write(",".join(vals_fmt) + "\n")

# ---- Psychrometrics (SI) ----------------------------------------------------
# Magnus-Tetens over water (0–50C): es (kPa)
def sat_vapor_pressure_C(T_C: float) -> float:
    return 0.61094 * math.exp((17.625 * T_C) / (T_C + 243.04))  # kPa

def dewpoint_from_T_RH(T_C: float, RH_pct: float) -> float:
    RH = max(1e-6, min(100.0, RH_pct)) / 100.0
    es = sat_vapor_pressure_C(T_C)
    e = RH * es
    # inverse Magnus
    ln = math.log(max(1e-12, e / 0.61094))
    return (243.04 * ln) / (17.625 - ln)

def RH_from_T_Tw_P(T_C: float, Tw_C: float, P_Pa: float) -> float:
    """
    Estimate RH (%) from dry-bulb T, wet-bulb T, and pressure (Pa).
    Stull (2011) psychrometric approximation for Tw; then back to RH via dewpoint relation.
    Approach:
    1) Estimate RH from T and Tw using an empirical fit that weakly depends on P.
       We adjust via psychrometric constant gamma ~ Cp*P / (lambda*epsilon).
    2) Clamp to [1,100].
    """
    # Stull 2011 approximation of RH from T and Tw (pressure-free empirical):
    # RH ≈ 100 - 5*(T - Tw)
    # We'll refine with a small pressure correction via psychrometric constant.
    RH_est = 100.0 - 5.0 * (T_C - Tw_C)
    # pressure correction (small): higher P -> slightly lower RH needed for same Tw depression
    gamma = 0.00066 * (1 + 0.00115 * Tw_C) * (P_Pa / 1000.0)  # ~kPa/°C scaled
    RH_corr = RH_est - 0.02 * (gamma - 0.66)  # tiny tweak toward sea-level baseline
    return float(np.clip(RH_corr, 1.0, 100.0))

# ---- Time alignment ----------------------------------------------------------
def epw_to_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    # EPW hour = 1..24 meaning end of hour; Minute often 60.
    # Build naive timestamps as Year-Month-Day Hour:Minute with hour-1 as start; we’ll keep end-of-hour alignment.
    dt = pd.to_datetime({
        "year": df["Year"].astype(int),
        "month": df["Month"].astype(int),
        "day": df["Day"].astype(int),
        # convert EPW hour (1-24 end-of-hour) to clock hour by subtracting 1, set minute to 0
        "hour": (df["Hour"].astype(int) - 1).clip(lower=0) % 24,
        "minute": 0
    }, errors="coerce")
    return pd.DatetimeIndex(dt)

def excel_read(path: Path) -> pd.DataFrame:
    # Expect columns: time, db_temp, wb_temp, rh (case-insensitive)
    df = pd.read_excel(path)
    # normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "time" not in df.columns:
        raise ValueError(f"{path.name}: missing 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    # keep known fields if present
    keep = [c for c in ["db_temp","wb_temp","rh"] if c in df.columns]
    return df[keep]

# ---- Main merge logic --------------------------------------------------------
def merge_excel_into_epw(base_header, base_df, excel_df, excel_name: str):
    epw_idx = epw_to_datetime_index(base_df)
    base_df = base_df.copy()
    base_df.index = epw_idx

    # Standardize excel to hourly end-of-hour like EPW: assume timestamps mark the END of the hour
    # If user data marks the start, uncomment the shift(1).
    excel_hourly = excel_df.resample("1H").mean()

    # Align to EPW index
    aligned = excel_hourly.reindex(epw_idx)

    # We will replace columns 7 (Dry Bulb), 9 (RH), and recompute 8 (Dew Point)
    # Start with dry-bulb
    if "db_temp" in aligned.columns:
        mask_db = aligned["db_temp"].notna()
        base_df.loc[mask_db, "Dry Bulb Temperature"] = aligned.loc[mask_db, "db_temp"]

    # Relative Humidity: prefer provided rh; else compute from T & Tw if both present
    RH_new = pd.Series(index=epw_idx, dtype=float)
    if "rh" in aligned.columns:
        RH_new = aligned["rh"].copy()

    # Estimate RH from T & Tw where RH missing and both temps present
    if "db_temp" in aligned.columns and "wb_temp" in aligned.columns:
        need = RH_new.isna() & aligned["db_temp"].notna() & aligned["wb_temp"].notna()
        if need.any():
            # pressure from base EPW (Pa)
            P = base_df.loc[need, "Atmospheric Station Pressure"].astype(float)
            RH_est = [
                RH_from_T_Tw_P(T, Tw, p)
                for T, Tw, p in zip(aligned.loc[need,"db_temp"], aligned.loc[need,"wb_temp"], P)
            ]
            RH_new.loc[need] = RH_est

    # Apply RH where we have it
    mask_rh = RH_new.notna()
    base_df.loc[mask_rh, "Relative Humidity"] = np.clip(RH_new.loc[mask_rh], 1.0, 100.0)

    # Dew point from (T, RH). Use whichever is now in base_df.
    T_used = base_df["Dry Bulb Temperature"].astype(float)
    RH_used = base_df["Relative Humidity"].astype(float)

    dew = pd.Series(index=epw_idx, dtype=float)
    ok = T_used.notna() & RH_used.notna()
    dew.loc[ok] = [dewpoint_from_T_RH(t, rh) for t, rh in zip(T_used.loc[ok], RH_used.loc[ok])]
    base_df.loc[ok, "Dew Point Temperature"] = dew.loc[ok]

    # Return header unchanged and merged df
    return base_header, base_df

# ---- CLI --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Batch-generate EPWs from Excel sensor files using a base EPW.")
    ap.add_argument("--base_epw", required=True, type=Path)
    ap.add_argument("--in_dir", required=True, type=Path, help="Folder with Excel files")
    ap.add_argument("--out_dir", required=True, type=Path, help="Output EPW folder")
    ap.add_argument("--suffix", default="_fromexcel", help="Suffix to append to base name")
    args = ap.parse_args()

    header, base_df = read_epw(args.base_epw)

    excels = sorted(list(args.in_dir.glob("*.xlsx")) + list(args.in_dir.glob("*.xls")))
    if not excels:
        raise SystemExit(f"No Excel files found in {args.in_dir}")

    for x in excels:
        try:
            exdf = excel_read(x)
            h, merged = merge_excel_into_epw(header, base_df, exdf, x.name)
            out = args.out_dir / f"{x.stem}{args.suffix}.epw"
            write_epw(h, merged, out)
            print(f"✅ Wrote {out}")
        except Exception as e:
            print(f"⚠️  Skipped {x.name}: {e}")

if __name__ == "__main__":
    main()
