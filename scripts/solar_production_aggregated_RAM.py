#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
import xarray as xr
from pathlib import Path
import importlib
import functions
from tqdm import tqdm
import seaborn as sns
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

importlib.reload(functions)
from functions import estimate_power_final

sns.set_style("darkgrid")

print("Starting solar production script...", flush=True)

YEAR = "2023"
MONTH = "aug"
month_number = "08"

fn1 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{YEAR}/reanalysis-cerra-single-levels-{MONTH}-{YEAR}-time1.nc"
fn2 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{YEAR}/reanalysis-cerra-single-levels-{MONTH}-{YEAR}-time2.nc"
fn3 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{YEAR}/reanalysis-cerra-single-levels-{MONTH}-{YEAR}-time3.nc"
fn4 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{YEAR}/reanalysis-cerra-single-levels-{MONTH}-{YEAR}-time4.nc"

print("Loading datasets eagerly (speed-first)...", flush=True)
# Speed-first: open + load into RAM (no dask chunking)
ds_list = [xr.open_dataset(f, engine="h5netcdf") for f in (fn1, fn2, fn3, fn4)]
raw_ds = xr.concat(ds_list, dim="valid_time", combine_attrs="drop_conflicts").sortby("valid_time")
raw_ds = raw_ds.sel(valid_time=~raw_ds.get_index("valid_time").duplicated())
raw_ds = raw_ds.rename({"valid_time": "time"})

for ds in ds_list:
    ds.close()

def deaccumulate_robust(da: xr.DataArray) -> xr.DataArray:
    da_diff = da.diff("time", label="upper")
    hourly = xr.concat([da.isel(time=0), da_diff], dim="time")
    # reset handling
    mask = hourly < 0
    hourly = hourly.where(~mask, other=da)
    return hourly

print("Building combined dataset...", flush=True)
combined_ds = xr.Dataset(
    data_vars={
        "irradiance": deaccumulate_robust(raw_ds["ssrd"]) / 3600,
        "wind_speed": raw_ds["si10"],
        "temperature": raw_ds["t2m"],
        "albedo": raw_ds["al"],
    }
).assign_coords(
    latitude=(("y", "x"), raw_ds["latitude"].values),
    longitude=(("y", "x"), raw_ds["longitude"].values),
)

combined_ds = combined_ds.sortby("time")

print("Loading met fields into memory (this can take a while)...", flush=True)
combined_ds.load()

T = combined_ds.sizes["time"]
Y = combined_ds.sizes["y"]
X = combined_ds.sizes["x"]

lats2d = combined_ds["latitude"].values
lons2d = combined_ds["longitude"].values
grid_shape = lats2d.shape

grid_points_0 = np.column_stack((lats2d.ravel(), lons2d.ravel()))
grid_points_p360 = np.column_stack((lats2d.ravel(), (lons2d.ravel() + 360.0)))

tree_0 = cKDTree(grid_points_0)
tree_p360 = cKDTree(grid_points_p360)

df_20MW_plus = pd.read_csv(
    "/Data/gfi/vindenergi/nab015/Solar_data/Global-Solar-Power-Tracker-February-2025-20MW+.csv",
    sep=";",
    decimal=",",
)
df_1MW_20MW = pd.read_csv(
    "/Data/gfi/vindenergi/nab015/Solar_data/Global-Solar-Power-Tracker-February-2025-1MW-20MW.csv",
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
location_cols = ["City", "State/Province", "Local area (taluk, county)", "Subregion", "Region", "Project Name"]

timezone_mapping = functions.get_timezone_mapping()
actual_generation_file = pd.read_csv(
    f"/Data/gfi/vindenergi/nab015/Actual_Generation/AggregatedGenerationPerType/{YEAR}/{YEAR}_{month_number}_AggregatedGenerationPerType_16.1.B_C_r3.csv",
    sep="\t",
)

global_power_watts = np.zeros((T, Y, X), dtype=np.float32)

N_JOBS = 8

def map_farms_to_grid(lat_arr, lon_arr):
    """Vectorized mapping of farm coords -> (y_idx, x_idx)."""
    pts0 = np.column_stack((lat_arr, lon_arr))
    pts1 = np.column_stack((lat_arr, lon_arr + 360.0))
    d0, idx0 = tree_0.query(pts0, k=1)
    d1, idx1 = tree_p360.query(pts1, k=1)
    idx = np.where(d0 <= d1, idx0, idx1).astype(np.int64)
    y_idx, x_idx = np.unravel_index(idx, grid_shape)
    return y_idx.astype(np.int32), x_idx.astype(np.int32)

for COUNTRY, COUNTRY_CODE in zip(countries_tracker, countries):
    print(f"\nProcessing solar for {COUNTRY} ({COUNTRY_CODE})...", flush=True)

    df_country = pd.concat(
        [
            df_20MW_plus[df_20MW_plus["Country/Area"] == COUNTRY],
            df_1MW_20MW[df_1MW_20MW["Country/Area"] == COUNTRY],
        ],
        ignore_index=True,
    )

    if COUNTRY_CODE in zones:
        zone_cities = functions.get_bidding_zone_mapping(COUNTRY_CODE)
        mask = pd.Series(False, index=df_country.index)
        for col in location_cols:
            if col in df_country.columns:
                mask |= df_country[col].isin(zone_cities)
        df_country = df_country[mask].copy()
    else:
        df_country = df_country.copy()

    if len(df_country) == 0:
        continue

    actual_generation_country = actual_generation_file[actual_generation_file["AreaDisplayName"] == COUNTRY_CODE].copy()
    timezone = timezone_mapping.get(COUNTRY_CODE, "Europe/Copenhagen")

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

    solar_data = actual_generation_country[actual_generation_country["ProductionType"] == "Solar"]
    solar_actual = pd.to_numeric(solar_data["ActualGenerationOutput[MW]"], errors="coerce").fillna(0).resample("h").mean()

    # Speed-first: precompute grid indices for all farms (vectorized)
    farm_lat = df_country["Latitude"].astype(float).to_numpy()
    farm_lon = df_country["Longitude"].astype(float).to_numpy()
    y_idx_all, x_idx_all = map_farms_to_grid(farm_lat, farm_lon)

    # Dense per-country accumulation
    country_power_watts = np.zeros((T, Y, X), dtype=np.float32)

    def process_farm(i):
        row = df_country.iloc[i]
        y_idx = int(y_idx_all[i])
        x_idx = int(x_idx_all[i])

        power_ts = estimate_power_final(
            country=COUNTRY,
            lat=float(row["Latitude"]),
            lon=float(row["Longitude"]),
            status=row["Status"],
            capacity_mw=row["Capacity (MW)"],
            capacity_rating=row["Capacity Rating"],
            tech_type=row["Technology Type"],
            xrds=combined_ds,
            y_idx=y_idx,
            x_idx=x_idx,
            Spatial_interpolation=True,
            min_irr=10,
            twilight_zenith_limit=80,
            smoothing_window_hours=3,
            performance_ratio=0.9,
            start_year=row["Start year"],
            prod_year=int(YEAR),
            enforce_start_year=False,
            mounting_type="default",
        )
        if power_ts is None:
            return None
        return (y_idx, x_idx, np.asarray(power_ts.values, dtype=np.float32))

    print(f"Processing {len(df_country)} farms with n_jobs={N_JOBS} (speed-first)...", flush=True)

    # Try threads first (no data duplication); switch to loky if you see no speedup
    results = Parallel(n_jobs=N_JOBS, backend="threading", verbose=10)(
        delayed(process_farm)(i) for i in range(len(df_country))
    )

    # Fast dense add
    for r in results:
        if r is None:
            continue
        y_idx, x_idx, ts = r
        country_power_watts[:, y_idx, x_idx] += ts

    total_pv_power_mw = country_power_watts.sum(axis=(1, 2), dtype=np.float64) / 1_000_000

    # Calibration factor (same logic)
    if float(total_pv_power_mw.sum()) > 0:
        calc_series_raw = pd.Series(
            total_pv_power_mw,
            index=pd.to_datetime(combined_ds["time"].values).tz_localize("UTC"),
        )
        if solar_actual.index.tz is not None:
            calc_series_raw.index = calc_series_raw.index.tz_convert(solar_actual.index.tz)

        common_idx = solar_actual.index.intersection(calc_series_raw.index)
        factor = functions.get_correction_factor(calc_series_raw.loc[common_idx], solar_actual.loc[common_idx]) if len(common_idx) else 1.0
    else:
        factor = 0.0

    print(f"Calibration factor for {COUNTRY}: {factor:.4f}", flush=True)

    global_power_watts += country_power_watts * np.float32(factor)

    # Plot (unchanged)
    calc_series = pd.Series(
        (total_pv_power_mw * float(factor)),
        index=combined_ds["time"].values,
    )
    if calc_series.index.tz is None:
        calc_series.index = calc_series.index.tz_localize("UTC")
    if solar_actual.index.tz is not None:
        calc_series.index = calc_series.index.tz_convert(solar_actual.index.tz)

    common_index = calc_series.index.intersection(solar_actual.index)
    calc_aligned = calc_series.loc[common_index]
    actual_aligned = solar_actual.loc[common_index]
    mask = ~np.isnan(calc_aligned.values) & ~np.isnan(actual_aligned.values)
    calc_aligned = calc_aligned.iloc[mask]
    actual_aligned = actual_aligned.iloc[mask]

    if len(calc_aligned) > 1:
        corr, _ = pearsonr(actual_aligned, calc_aligned)
        r_squared = corr**2
    else:
        r_squared = np.nan

    fig_dir = Path(f"/Data/gfi/vindenergi/nab015/figures/pv_power_comparison/{YEAR}/{month_number}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(calc_series.index, calc_series.values, "x-", label="Estimated PV Power (MW)", color="red")
    ax.plot(solar_actual.index, solar_actual.values, label="Actual PV Power (MW)", alpha=0.7, linestyle="--", color="black")
    ax.text(0.15, 0.95, fr"$\alpha = $ {factor:.4f}", transform=ax.transAxes, fontsize=16, va="top")
    if not pd.isna(r_squared):
        ax.text(0.15, 0.90, fr"$R^2 = $ {r_squared:.4f}", transform=ax.transAxes, fontsize=16, va="top")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (MW)")
    ax.set_title(f"Aggregated PV Power Generation for {COUNTRY_CODE if COUNTRY_CODE in zones else COUNTRY} - {month_number}-{YEAR}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True)
    ax.legend(loc="upper right")
    out_name = f"pv_power_comparison_{month_number}_{YEAR}_{COUNTRY_CODE if COUNTRY_CODE in zones else COUNTRY}.svg"
    plt.savefig(fig_dir / out_name, bbox_inches="tight")
    plt.close(fig)

# Output dataset
pv_dataset = xr.Dataset(
    {"pv_power_mw": (("time", "y", "x"), global_power_watts / 1_000_000)},
    coords={
        "time": combined_ds["time"].values,
        "y": combined_ds["y"].values,
        "x": combined_ds["x"].values,
        "latitude": (("y", "x"), lats2d),
        "longitude": (("y", "x"), lons2d),
    },
)

output_dir = Path(f"/Data/gfi/vindenergi/nab015/pv_production/{YEAR}")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"{month_number}_{YEAR}_pv_production_aggregated.nc"
if output_file.exists():
    output_file.unlink()

pv_dataset.to_netcdf(output_file, engine="h5netcdf")
print(f"Aggregated PV power dataset saved to {output_file}", flush=True)

combined_ds.close()
raw_ds.close()