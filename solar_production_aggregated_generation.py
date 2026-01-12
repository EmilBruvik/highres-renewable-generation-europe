#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
import pvlib
import netCDF4 as nc
import xarray as xr
from pathlib import Path
import importlib
import functions
from collections import Counter, defaultdict
from tqdm import tqdm
import pytz
import seaborn as sns
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
importlib.reload(functions)
from functions import estimate_power_final, process_data

sns.set_style("darkgrid")

print("Starting solar production script...", flush=True)

YEAR = "2023"
MONTH = "aug"
MONTH_NUM = 8
month_number = "08"

print(f"Loading datasets for {YEAR}-{MONTH}...", flush=True)

fn1 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{YEAR}/reanalysis-cerra-single-levels-{MONTH}-{YEAR}-time1.nc"
fn2 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{YEAR}/reanalysis-cerra-single-levels-{MONTH}-{YEAR}-time2.nc"
fn3 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{YEAR}/reanalysis-cerra-single-levels-{MONTH}-{YEAR}-time3.nc"
fn4 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{YEAR}/reanalysis-cerra-single-levels-{MONTH}-{YEAR}-time4.nc"

# Keep datasets lazy and chunked
ds1 = xr.open_dataset(fn1, engine="h5netcdf", chunks={"valid_time": 47})
print(ds1["ssrd"].encoding.get("chunksizes"))
ds2 = xr.open_dataset(fn2, engine="h5netcdf", chunks={"valid_time": 47})
ds3 = xr.open_dataset(fn3, engine="h5netcdf", chunks={"valid_time": 47})
ds4 = xr.open_dataset(fn4, engine="h5netcdf", chunks={"valid_time": 47})

print("Processing datasets (concatenating and deaccumulating)...", flush=True)

ds_list = [ds1, ds2, ds3, ds4]
raw_ds = xr.concat(ds_list, dim="valid_time", combine_attrs="drop_conflicts").chunk({"valid_time": 100})

raw_ds = raw_ds.sortby("valid_time")
raw_ds = raw_ds.sel(valid_time=~raw_ds.get_index("valid_time").duplicated())
raw_ds = raw_ds.rename({"valid_time": "time"})

# Change E: close source datasets once concatenated (frees file handles/resources)
ds1.close()
ds2.close()
ds3.close()
ds4.close()

def deaccumulate_robust(da: xr.DataArray) -> xr.DataArray:
    """
    De-accumulates data by calculating diffs, but handles resets automatically.
    If x[t] < x[t-1], it's a reset, so we take x[t] as the value.
    """
    da_diff = da.diff("time", label="upper")
    hourly = xr.concat([da.isel(time=0), da_diff], dim="time")
    mask = hourly < 0
    hourly = hourly.where(~mask, other=da)
    return hourly

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

# Change D (fix 3): hoist lat/lon arrays out of per-farm function (avoid repeated .values)
lats2d = combined_ds["latitude"].values
lons2d = combined_ds["longitude"].values

grid_points_0 = np.column_stack((lats2d.ravel(), lons2d.ravel()))
grid_points_p360 = np.column_stack((lats2d.ravel(), (lons2d.ravel() + 360.0)))

tree_0 = cKDTree(grid_points_0)
tree_p360 = cKDTree(grid_points_p360)
grid_shape = lats2d.shape  # (Y, X)

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

timezone_mapping = functions.get_timezone_mapping()
actual_generation_file = pd.read_csv(
    f"/Data/gfi/vindenergi/nab015/Actual_Generation/AggregatedGenerationPerType/{YEAR}/{YEAR}_{month_number}_AggregatedGenerationPerType_16.1.B_C_r3.csv",
    sep="\t",
)

T = combined_ds.sizes["time"]
Y = combined_ds.sizes["y"]
X = combined_ds.sizes["x"]

# Change C (fix 2): sparse accumulation of only touched grid cells
global_cells: dict[tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(T, dtype=np.float32))

location_cols = ["City", "State/Province", "Local area (taluk, county)", "Subregion", "Region", "Project Name"]

# Change D (fix 1): cap workers; -1 is dangerous on shared node + multiplies memory
N_JOBS = 2

for COUNTRY, COUNTRY_CODE in zip(countries_tracker, countries):
    print(f"Processing solar for {COUNTRY} ({COUNTRY_CODE})...", flush=True)

    # sparse per-country cells too (avoid allocating (T,Y,X))
    country_cells: dict[tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(T, dtype=np.float32))

    df_20MW_plus_country = df_20MW_plus[df_20MW_plus["Country/Area"] == COUNTRY].copy()
    df_1MW_20MW_country = df_1MW_20MW[df_1MW_20MW["Country/Area"] == COUNTRY].copy()
    df_country = pd.concat([df_20MW_plus_country, df_1MW_20MW_country], ignore_index=True)

    if COUNTRY_CODE in zones:
        zone_cities = functions.get_bidding_zone_mapping(COUNTRY_CODE)
        mask = pd.Series(False, index=df_country.index)
        for col in location_cols:
            if col in df_country.columns:
                mask |= df_country[col].isin(zone_cities)
        df_country = df_country[mask]
    else:
        df_country = df_country[(df_country["Country/Area"] == COUNTRY)].copy()

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
    solar_actual = pd.to_numeric(solar_data["ActualGenerationOutput[MW]"], errors="coerce").fillna(0)
    solar_actual = solar_actual.resample("h").mean()

    def process_farm(farm_row: pd.Series):
        """
        Returns (y_idx, x_idx, power_values_float32) or None.
        Note: uses hoisted lats2d/lons2d and shared combined_ds.
        """
        lat = float(farm_row["Latitude"])
        lon = float(farm_row["Longitude"])

        diff_lon = np.abs(lons2d - lon)
        diff_lon = np.minimum(diff_lon, 360 - diff_lon)
        dist_sq = (lats2d - lat) ** 2 + diff_lon ** 2

        min_flat = int(np.argmin(dist_sq))
        d0, idx0 = tree_0.query([lat, lon], k=1)
        d1, idx1 = tree_p360.query([lat, lon + 360.0], k=1)
        idx = int(idx0 if d0 <= d1 else idx1)
        y_idx, x_idx = np.unravel_index(idx, grid_shape)
        # y_idx, x_idx = np.unravel_index(min_flat, grid_shape)

        power_ts = estimate_power_final(
            country=COUNTRY,
            lat=lat,
            lon=lon,
            status=farm_row["Status"],
            capacity_mw=farm_row["Capacity (MW)"],
            capacity_rating=farm_row["Capacity Rating"],
            tech_type=farm_row["Technology Type"],
            xrds=combined_ds,
            y_idx=y_idx,
            x_idx=x_idx,
            Spatial_interpolation=True,
            min_irr=10,
            twilight_zenith_limit=80,
            smoothing_window_hours=3,
            performance_ratio=0.9,
            start_year=farm_row["Start year"],
            prod_year=int(YEAR),
            mounting_type="default",
        )

        if power_ts is None:
            return None

        # Change D (fix 2): ensure we return float32 to cut result memory
        return (y_idx, x_idx, np.asarray(power_ts.values, dtype=np.float32))

    print(f"Processing {len(df_country)} farms with n_jobs={N_JOBS}...", flush=True)

    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        delayed(process_farm)(farm) for _, farm in df_country.iterrows()
    )

    # Aggregate into sparse per-country cells
    for r in results:
        if r is None:
            continue
        y_idx, x_idx, power_values = r
        country_cells[(y_idx, x_idx)] += power_values

    # total PV power time series for the country (MW)
    if len(country_cells) == 0:
        total_pv_power_mw = np.zeros(T, dtype=np.float32)
    else:
        total_watts = np.zeros(T, dtype=np.float32)
        for ts in country_cells.values():
            total_watts += ts
        total_pv_power_mw = total_watts / 1_000_000

    # calibration factor
    if float(total_pv_power_mw.sum()) > 0:
        calc_series_raw = pd.Series(
            total_pv_power_mw.astype(np.float64),
            index=pd.to_datetime(combined_ds["time"].values).tz_localize("UTC"),
        )
        if solar_actual.index.tz is not None:
            calc_series_raw.index = calc_series_raw.index.tz_convert(solar_actual.index.tz)

        common_idx = solar_actual.index.intersection(calc_series_raw.index)
        if len(common_idx) > 0:
            factor = functions.get_correction_factor(
                calc_series_raw.loc[common_idx], solar_actual.loc[common_idx]
            )
        else:
            factor = 1.0
    else:
        factor = 1.0

    print(f"Calibration factor for {COUNTRY}: {factor:.4f}")

    # Apply factor and add to global sparse grid
    for (y_idx, x_idx), ts in country_cells.items():
        global_cells[(y_idx, x_idx)] += ts * np.float32(factor)

    calc_series = pd.Series(
        (total_pv_power_mw * np.float32(factor)).astype(np.float64),
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

    print(f"PV power calculated for {COUNTRY}.")

    fig_dir = Path(f"/Data/gfi/vindenergi/nab015/figures/pv_power_comparison/{YEAR}/{month_number}")
    # fig_dir = Path(f"/Data/gfi/vindenergi/nab015/figures/pv_power_comparison/robust_CF2/{YEAR}/{month_number}")

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(calc_series.index, calc_series.values, "x-", label="Estimated PV Power (MW)", color="red")
    ax.plot(solar_actual.index, solar_actual.values, label="Actual PV Power (MW)", alpha=0.7, linestyle="--", color="black")
    ax.text(0.15, 0.95, fr"$\alpha = $ {factor:.4f}", transform=ax.transAxes, fontsize=16, va="top")
    if not pd.isna(r_squared):
        ax.text(0.15, 0.90, fr"$R^2 = $ {r_squared:.4f}", transform=ax.transAxes, fontsize=16, va="top")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (MW)")
    if COUNTRY_CODE in zones:
        ax.set_title(f"Aggregated PV Power Generation for {COUNTRY_CODE} - {month_number}-{YEAR}")
    else:
        ax.set_title(f"Aggregated PV Power Generation for {COUNTRY} - {month_number}-{YEAR}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True)
    ax.legend(loc="upper right")
    out_name = f"pv_power_comparison_{month_number}_{YEAR}_{COUNTRY_CODE if COUNTRY_CODE in zones else COUNTRY}.svg"
    plt.savefig(fig_dir / out_name, bbox_inches="tight")
    plt.close(fig)

# Materialize final full grid once (still potentially large, but only once at end)
global_grid = np.zeros((T, Y, X), dtype=np.float32)
for (y_idx, x_idx), ts in global_cells.items():
    global_grid[:, y_idx, x_idx] = ts

pv_dataset = xr.Dataset(
    {"pv_power_mw": (("time", "y", "x"), global_grid / 1_000_000)},
    coords={
        "time": combined_ds["time"].values,
        "y": combined_ds["y"].values,
        "x": combined_ds["x"].values,
        "latitude": (("y", "x"), combined_ds["latitude"].values),
        "longitude": (("y", "x"), combined_ds["longitude"].values),
    },
)

output_dir = Path(f"/Data/gfi/vindenergi/nab015/pv_production/{YEAR}")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"{month_number}_{YEAR}_pv_production_aggregated.nc"
if output_file.exists():
    output_file.unlink()
pv_dataset.to_netcdf(output_file, engine="h5netcdf")
print(f"Aggregated PV power dataset saved to {output_file}")

# cleanup
combined_ds.close()
raw_ds.close()