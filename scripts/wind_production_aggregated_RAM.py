#!/usr/bin/env python
# coding: utf-8

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from pathlib import Path
from scipy.stats import pearsonr
import importlib
import functions
from tqdm import tqdm

importlib.reload(functions)
from functions import estimate_wind_power

print("Starting wind production script...", flush=True)

YEAR = "2024"
PROD_YEAR = int(YEAR)
MONTH = "sep"
month_number = "09"
# MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
# month_numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]


print(f"Loading dataset for {YEAR}-{MONTH}...", flush=True)

fn = f"/Data/gfi/vindenergi/nab015/CERRA_multi_level/{YEAR}/cerra_{YEAR}_multi_level_{MONTH}.nc"
ds = xr.open_dataset(fn, engine="netcdf4")

actual_generation_file = pd.read_csv(
    f"/Data/gfi/vindenergi/nab015/Actual_Generation/AggregatedGenerationPerType/{YEAR}/{YEAR}_{month_number}_AggregatedGenerationPerType_16.1.B_C_r3.csv",
    sep="\t",
)

print("Loading dataset into memory...", flush=True)
ds.load()

df = pd.read_csv(
    "/Data/gfi/vindenergi/nab015/Wind_data/Global-Wind-Power-Tracker-February-2025.csv",
    sep=";",
    decimal=",",
)

countries_tracker = [
    "Austria", "Bosnia and Herzegovina", "Belgium", "Bulgaria",
    "Switzerland", "Cyprus", "Czech Republic", "Germany",
    "Denmark", "Denmark", "Estonia", "Spain",
    "Finland", "France", "United Kingdom", "Georgia", "Greece", "Croatia", "Hungary",
    "Ireland", "Italy",
    "Lithuania", "Luxembourg", "Latvia", "Moldova",
    "Montenegro", "North Macedonia", "Netherlands",
    "Norway", "Norway", "Norway", "Norway", "Norway", "Poland", "Portugal",
    "Romania", "Serbia", "Sweden", "Sweden", "Sweden", "Sweden",
    "Slovenia", "Slovakia", "Kosovo",
]

countries = [
    "Austria (AT)", "Bosnia and Herz. (BA)", "Belgium (BE)", "Bulgaria (BG)",
    "Switzerland (CH)", "Cyprus (CY)", "Czech Republic (CZ)", "Germany (DE)",
    "DK1", "DK2", "Estonia (EE)", "Spain (ES)",
    "Finland (FI)", "France (FR)", "United Kingdom (UK)", "Georgia (GE)", "Greece (GR)", "Croatia (HR)", "Hungary (HU)",
    "Ireland (IE)", "Italy (IT)",
    "Lithuania (LT)", "Luxembourg (LU)", "Latvia (LV)", "Moldova (MD)",
    "Montenegro (ME)", "North Macedonia (MK)", "Netherlands (NL)",
    "NO1", "NO2", "NO3", "NO4", "NO5", "Poland (PL)", "Portugal (PT)",
    "Romania (RO)", "Serbia (RS)", "SE1", "SE2", "SE3", "SE4",
    "Slovenia (SI)", "Slovakia (SK)", "Kosovo (XK)",
]

zones = ["NO1", "NO2", "NO3", "NO4", "NO5", "DK1", "DK2", "SE1", "SE2", "SE3", "SE4"]
location_cols = ["City", "State/Province", "Local area (taluk, county)", "Subregion", "Region"]

timezone_mapping = functions.get_timezone_mapping()

# Output: country time series results
results_by_area = []

# Precompute a reference year for missing start years (global median like your original)
ref_startyear = int(pd.to_numeric(df["Start year"], errors="coerce").median())

# For faster nearest-cell lookup, extract lat/lon arrays once
lat2d = ds["latitude"].values
lon2d = ds["longitude"].values

for COUNTRY, COUNTRY_CODE in zip(countries_tracker, countries):
    print(f"Processing country: {COUNTRY} ({COUNTRY_CODE})...", flush=True)

    timezone = timezone_mapping.get(COUNTRY_CODE, "Europe/Copenhagen")

    df_country = df[df["Country/Area"] == COUNTRY].copy()

    # Zone filter for multi-zone countries
    if COUNTRY_CODE in zones:
        zone_cities = functions.get_bidding_zone_mapping(COUNTRY_CODE)
        mask = pd.Series(False, index=df_country.index)
        for col in location_cols:
            if col in df_country.columns:
                mask |= df_country[col].isin(zone_cities)
        df_country = df_country[mask].copy()

    if len(df_country) == 0:
        continue

    # Keep ONLY operating farms (your requirement #1)
    operating_statuses = set(functions.operating_farms(COUNTRY, "wind"))
    df_country = df_country[df_country["Status"].isin(operating_statuses)].copy()

    if len(df_country) == 0:
        print(f"No operating wind farms for {COUNTRY}.", flush=True)
        continue

    # Actual wind (onshore + offshore)
    actual_generation_country = actual_generation_file[actual_generation_file["AreaDisplayName"] == COUNTRY_CODE].copy()

    if "DateTime(UTC)" in actual_generation_country.columns:
        time_stamps = pd.to_datetime(actual_generation_country["DateTime(UTC)"])
        time_stamps = time_stamps.dt.tz_localize("UTC").dt.tz_convert(timezone)
        actual_generation_country.index = time_stamps
    elif "MTU" in actual_generation_country.columns:
        naive_datetime = pd.to_datetime(
            actual_generation_country["MTU"].str.split(" - ").str[0],
            format="%d.%m.%Y %H:%M",
        )
        actual_generation_country.index = naive_datetime.dt.tz_localize(
            timezone, nonexistent="shift_forward", ambiguous="NaT"
        )
        actual_generation_country = actual_generation_country[actual_generation_country.index.notna()]

    wind_data_onshore = actual_generation_country[actual_generation_country["ProductionType"] == "Wind Onshore"]
    wind_data_offshore = actual_generation_country[actual_generation_country["ProductionType"] == "Wind Offshore"]
    wind_data = pd.concat([wind_data_onshore, wind_data_offshore])

    wind_actual = pd.to_numeric(wind_data["ActualGenerationOutput[MW]"], errors="coerce").fillna(0)
    wind_actual = wind_actual.groupby(wind_actual.index).sum().resample("h").mean()

    # Scenario accumulators (MW) – we compute farm series once (as MW) then add into scenarios
    country_mw_2025 = np.zeros(ds.dims["valid_time"], dtype=np.float64)
    country_mw_asbuilt = np.zeros(ds.dims["valid_time"], dtype=np.float64)

    for _, farm in tqdm(df_country.iterrows(), total=df_country.shape[0], desc="Calculating Wind Farms"):
        lat = float(farm["Latitude"])
        lon = float(farm["Longitude"])

        # Nearest grid point (same method as your original)
        diff_lon = np.abs(lon2d - lon)
        diff_lon = np.minimum(diff_lon, 360 - diff_lon)
        distance_sq = (lat2d - lat) ** 2 + diff_lon ** 2
        min_dist_idx_flat = int(np.argmin(distance_sq.reshape(-1)))
        y_idx, x_idx = np.unravel_index(min_dist_idx_flat, lat2d.shape)

        startyear = farm["Start year"]
        if pd.isna(startyear):
            startyear = ref_startyear

        # Compute ONCE per farm for this weather year (ignore start year here)
        power_timeseries_mw = estimate_wind_power(
            country=COUNTRY,
            lat=lat,
            lon=lon,
            capacity=farm["Capacity (MW)"],
            startyear=startyear,
            prod_year=PROD_YEAR,
            status=farm["Status"],
            installation_type=farm["Installation Type"],
            xrds=ds,
            y_idx=y_idx,
            x_idx=x_idx,
            wts_smoothing=True,
            wake_loss_factor=0.95,
            spatial_interpolation=True,
            verbose=False,
            single_turb_curve=False,
            enforce_start_year=False, 
        )

        if power_timeseries_mw is None:
            continue

        # 2025 fleet: all operating farms
        country_mw_2025 += power_timeseries_mw

        # as-built: only farms started by weather year
        if isinstance(startyear, (int, float)) and int(startyear) <= PROD_YEAR:
            country_mw_asbuilt += power_timeseries_mw

    # Calibration factor from AS-BUILT only; apply to BOTH
    def calc_factor_from_asbuilt(model_mw: np.ndarray) -> float:
        if float(np.nansum(model_mw)) <= 0:
            return 0.0
        calc_series_raw = pd.Series(model_mw, index=pd.to_datetime(ds["valid_time"].values).tz_localize("UTC"))
        if wind_actual.index.tz is not None:
            calc_series_raw.index = calc_series_raw.index.tz_convert(wind_actual.index.tz)
        common_idx = wind_actual.index.intersection(calc_series_raw.index)
        if len(common_idx) == 0:
            return 1.0
        return functions.get_correction_factor(calc_series_raw.loc[common_idx], wind_actual.loc[common_idx])

    factor = calc_factor_from_asbuilt(country_mw_asbuilt)
    print(f"Calibration factor (as-built) for {COUNTRY}: {factor:.4f}", flush=True)

    country_mw_asbuilt_cal = country_mw_asbuilt * factor
    country_mw_2025_cal = country_mw_2025 * factor

    # Plot still compares actual vs AS-BUILT (recommended)
    calc_series = pd.Series(country_mw_asbuilt_cal, index=ds["valid_time"].values)
    if calc_series.index.tz is None:
        calc_series.index = calc_series.index.tz_localize("UTC")
    if wind_actual.index.tz is not None:
        calc_series.index = calc_series.index.tz_convert(wind_actual.index.tz)

    common_index = calc_series.index.intersection(wind_actual.index)
    calc_aligned = calc_series.loc[common_index]
    actual_aligned = wind_actual.loc[common_index]
    mask = ~np.isnan(calc_aligned.values) & ~np.isnan(actual_aligned.values)
    calc_aligned = calc_aligned.iloc[mask]
    actual_aligned = actual_aligned.iloc[mask]
    r_squared = pearsonr(actual_aligned, calc_aligned)[0] ** 2 if len(calc_aligned) > 1 else np.nan

    fig_dir = Path(f"/Data/gfi/vindenergi/nab015/figures/wind_power_comparison/{YEAR}/{month_number}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(calc_series.index, calc_series.values, "x-", label="Estimated Wind Power (MW) as-built", color="red")
    ax.plot(wind_actual.index, wind_actual.values, label="Actual Wind Power (MW)", alpha=0.7, linestyle="--", color="black")
    ax.text(0.15, 0.95, fr"$\alpha = $ {factor:.4f}", transform=ax.transAxes, fontsize=16, va="top")
    if not pd.isna(r_squared):
        ax.text(0.15, 0.90, fr"$R^2 = $ {r_squared:.4f}", transform=ax.transAxes, fontsize=16, va="top")

    ax.set_xlabel("Time")
    ax.set_ylabel("Power (MW)")
    ax.set_title(f"Wind Power Generation for {COUNTRY_CODE} - {month_number}-{YEAR}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True)
    ax.legend(loc="upper right")

    out_svg = fig_dir / f"wind_power_comparison_{month_number}_{YEAR}_{COUNTRY_CODE}.svg"
    plt.savefig(out_svg, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure saved to {out_svg}", flush=True)

    results_by_area.append(
        (COUNTRY_CODE, country_mw_asbuilt_cal.astype(np.float32), country_mw_2025_cal.astype(np.float32))
    )

# Save (time, area) dataset
time_vals = ds["valid_time"].values
areas = [a for (a, _, _) in results_by_area]

wind_power_mw = np.stack([ts for (_, ts, _) in results_by_area], axis=1)       # (time, area)
wind_power_mw_2025 = np.stack([ts for (_, _, ts) in results_by_area], axis=1)  # (time, area)

wind_dataset = xr.Dataset(
    data_vars={
        "wind_power_mw": (("time", "area"), wind_power_mw),
        "wind_power_mw_2025": (("time", "area"), wind_power_mw_2025),
    },
    coords={
        "time": time_vals,
        "area": np.array(areas, dtype="U"),
    },
)

outputfile_dir = Path(f"/Data/gfi/vindenergi/nab015/wind_production/{YEAR}")
outputfile_dir.mkdir(parents=True, exist_ok=True)

output_file = outputfile_dir / f"{month_number}_{YEAR}_wind_production_country_timeseries.nc"
if output_file.exists():
    output_file.unlink()

wind_dataset.to_netcdf(output_file, engine="netcdf4")
print(f"Wind country time series saved to {output_file}", flush=True)

ds.close()