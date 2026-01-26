#!/usr/bin/env python3
# coding: utf-8

#-----------Run commands----------------#
# python -u scripts/weather_energy_monthly.py --year 2023 --month 02 --n-jobs-pv 2
# for m in $(seq -w 1 12); do   python -u scripts/weather_energy_monthly.py --year 2024 --month "$m" --n-jobs-pv 2; done
# for m in $(seq -w 1 12); do python -u scripts/weather_energy_monthly.py --year 2024 --month "$m" --n-jobs-pv 2 --write-farm-timeseries; done

# nohup bash -lc 'for m in $(seq -w 1 12); do python -u scripts/weather_energy_monthly.py --year 2024 --month "$m" --n-jobs-pv 2; done' > run_2024.log 2>&1 &
# disown
# set -e; for m in $(seq -w 1 12); do python -u scripts/weather_energy_monthly.py --year 2024 --month "$m" --n-jobs-pv 2; done
#---------------------------------------#

from __future__ import annotations

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import gc
import tempfile
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
if "functions" in sys.modules:
    del sys.modules["functions"]

import functions

MONTHS = [
    ("01", "jan"), ("02", "feb"), ("03", "mar"), ("04", "apr"),
    ("05", "may"), ("06", "jun"), ("07", "jul"), ("08", "aug"),
    ("09", "sep"), ("10", "oct"), ("11", "nov"), ("12", "dec"),
]

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

countries_codes = [
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

ZONES = ["NO1", "NO2", "NO3", "NO4", "NO5", "DK1", "DK2", "SE1", "SE2", "SE3", "SE4"]
LOCATION_COLS_PV = ["City", "State/Province", "Local area (taluk, county)", "Subregion", "Region", "Project Name"]
LOCATION_COLS_WIND = ["City", "State/Province", "Local area (taluk, county)", "Subregion", "Region"]


@dataclass(frozen=True)
class MonthSpec:
    year: str
    month_number: str   # "01"
    month_name: str     # "jan"

    @property
    def prod_year(self) -> int:
        return int(self.year)


class GridIndexer:
    """
    Robust KDTree indexer handling longitude wrapping (0-360 vs -180-180).
    Builds a tree with 3 copies of the grid (L-360, L, L+360) to ensure
    Euclidean nearest neighbor works correctly across the dateline/seam.
    """
    def __init__(self, lat2d: np.ndarray, lon2d: np.ndarray):
        self.shape = lat2d.shape
        self.n_points = lat2d.size
        
        lon_norm = lon2d.ravel() % 360.0
        lat_flat = lat2d.ravel()
        
        p_main = np.column_stack((lat_flat, lon_norm))
        p_left = np.column_stack((lat_flat, lon_norm - 360.0))
        p_right = np.column_stack((lat_flat, lon_norm + 360.0))
        
        self.tree = cKDTree(np.vstack((p_main, p_left, p_right)))

    def map_points(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Normalize query longitude to 0-360
        lon_q = lon % 360.0
        pts = np.column_stack((lat, lon_q))
        
        _, idx = self.tree.query(pts, k=1)
        
        idx_orig = idx % self.n_points
        y, x = np.unravel_index(idx_orig, self.shape)
        return y.astype(int), x.astype(int)


class ActualGenerationLoader:
    def __init__(self, timezone_mapping: dict):
        self.tzmap = timezone_mapping

    def load_month_file(self, year: str, month_number: str) -> pd.DataFrame:
        return pd.read_csv(
            f"/Data/gfi/vindenergi/nab015/Actual_Generation/AggregatedGenerationPerType/{year}/"
            f"{year}_{month_number}_AggregatedGenerationPerType_16.1.B_C_r3.csv",
            sep="\t",
        )

    def _index_to_local_tz(self, df_area: pd.DataFrame, tz: str) -> pd.DataFrame:
        if "DateTime(UTC)" in df_area.columns:
            ts = pd.to_datetime(df_area["DateTime(UTC)"]).dt.tz_localize("UTC").dt.tz_convert(tz)
            df_area = df_area.copy()
            df_area.index = ts
            return df_area

        if "MTU" in df_area.columns:
            naive = pd.to_datetime(df_area["MTU"].str.split(" - ").str[0], format="%d.%m.%Y %H:%M")
            df_area = df_area.copy()
            df_area.index = naive.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
            return df_area[df_area.index.notna()]

        raise ValueError("Actual generation file format not recognized (missing DateTime(UTC) and MTU).")

    def solar_series_mw(self, actual_file: pd.DataFrame, area_code: str) -> pd.Series:
        tz = self.tzmap.get(area_code, "Europe/Copenhagen")
        df_area = actual_file[actual_file["AreaDisplayName"] == area_code].copy()
        df_area = self._index_to_local_tz(df_area, tz)

        solar = df_area[df_area["ProductionType"] == "Solar"]
        s = pd.to_numeric(solar["ActualGenerationOutput[MW]"], errors="coerce").fillna(0)
        return s.resample("h").mean()

    def wind_series_mw(self, actual_file: pd.DataFrame, area_code: str) -> pd.Series:
        tz = self.tzmap.get(area_code, "Europe/Copenhagen")
        df_area = actual_file[actual_file["AreaDisplayName"] == area_code].copy()
        df_area = self._index_to_local_tz(df_area, tz)

        on = df_area[df_area["ProductionType"] == "Wind Onshore"]
        off = df_area[df_area["ProductionType"] == "Wind Offshore"]
        w = pd.concat([on, off])
        s = pd.to_numeric(w["ActualGenerationOutput[MW]"], errors="coerce").fillna(0)
        return s.groupby(s.index).sum().resample("h").mean()


class PVCalculator:
    def __init__(self, n_jobs: int = 8):
        self.n_jobs = n_jobs
        self.df_20 = pd.read_csv(
            "/Data/gfi/vindenergi/nab015/Solar_data/Global-Solar-Power-Tracker-February-2025-20MW+.csv",
            sep=";",
            decimal=",",
        )
        self.df_1_20 = pd.read_csv(
            "/Data/gfi/vindenergi/nab015/Solar_data/Global-Solar-Power-Tracker-February-2025-1MW-20MW.csv",
            sep=";",
            decimal=",",
        )

    @staticmethod
    def _deaccumulate_robust(da: xr.DataArray) -> xr.DataArray:
        vals = da.values
        diffs = np.diff(vals, axis=0)
        hourly_vals = np.vstack([vals[0:1], diffs])
        mask = hourly_vals < 0
        if np.any(mask):
            hourly_vals[mask] = vals[mask]
        return hourly_vals

    def open_weather(self, ms: MonthSpec) -> xr.Dataset:
        fn1 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{ms.year}/reanalysis-cerra-single-levels-{ms.month_name}-{ms.year}-time1.nc"
        fn2 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{ms.year}/reanalysis-cerra-single-levels-{ms.month_name}-{ms.year}-time2.nc"
        fn3 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{ms.year}/reanalysis-cerra-single-levels-{ms.month_name}-{ms.year}-time3.nc"
        fn4 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{ms.year}/reanalysis-cerra-single-levels-{ms.month_name}-{ms.year}-time4.nc"

        ds_list = [xr.open_dataset(f, engine="h5netcdf") for f in (fn1, fn2, fn3, fn4)]
        raw = xr.concat(ds_list, dim="valid_time").sortby("valid_time")
        raw = raw.sel(valid_time=~raw.get_index("valid_time").duplicated()).rename({"valid_time": "time"})
        for d in ds_list:
            d.close()

        combined = xr.Dataset(
            data_vars={
                "irradiance": (("time", "y", "x"), self._deaccumulate_robust(raw["ssrd"]) / 3600),
                "wind_speed": raw["si10"],
                "temperature": raw["t2m"],
                "albedo": raw["al"],
            }
        ).assign_coords(
            latitude=(("y", "x"), raw["latitude"].values),
            longitude=(("y", "x"), raw["longitude"].values),
        ).sortby("time")

        combined.load()
        raw.close()
        return combined

    def country_timeseries(
        self,
        ms: MonthSpec,
        xrds: xr.Dataset,
        indexer: GridIndexer,
        actual_loader: ActualGenerationLoader,
        actual_file: pd.DataFrame,
        country: str,
        area_code: str,
        return_farm_data: bool = True,
    ):
        df_country = pd.concat(
            [
                self.df_20[self.df_20["Country/Area"] == country],
                self.df_1_20[self.df_1_20["Country/Area"] == country],
            ],
            ignore_index=True,
        )

        if area_code in ZONES:
            if area_code.startswith('DK'):
                # Robust Longitude split for Denmark
                lon_vals = pd.to_numeric(df_country['Longitude'], errors='coerce')
                if area_code == 'DK1':
                    df_country = df_country[lon_vals < 10.9].copy()
                else: # DK2
                    df_country = df_country[lon_vals >= 10.9].copy()
            else:
                # Standard city-mapping for NO/SE zones
                zone_cities = functions.get_bidding_zone_mapping(area_code)
                mask = pd.Series(False, index=df_country.index)
                # Use specific columns based on whether this is PV or Wind loop
                cols_to_check = LOCATION_COLS_PV if "PV" in str(type(self)) else LOCATION_COLS_WIND
                for col in cols_to_check:
                    if col in df_country.columns:
                        mask |= df_country[col].isin(zone_cities)
                df_country = df_country[mask].copy()
        
        T = xrds.sizes["time"]
        empty_ret = (np.zeros(T, dtype=np.float64), np.zeros(T, dtype=np.float64), 0.0)
        
        if len(df_country) == 0:
            return (*empty_ret, pd.DataFrame(), []) if return_farm_data else empty_ret

        op_status = set(functions.operating_farms(country, "solar"))
        df_country = df_country[df_country["Status"].isin(op_status)].copy()
        
        df_country["Latitude"] = pd.to_numeric(df_country["Latitude"], errors="coerce")
        df_country["Longitude"] = pd.to_numeric(df_country["Longitude"], errors="coerce")
        df_country = df_country.dropna(subset=["Latitude", "Longitude"])

        if len(df_country) == 0:
            return (*empty_ret, pd.DataFrame(), []) if return_farm_data else empty_ret

        solar_actual = actual_loader.solar_series_mw(actual_file, area_code)

        if solar_actual.sum() <= 1.0:
            print(f"Skipping {country} (PV): No actual production data found.", flush=True)
            return (*empty_ret, pd.DataFrame(), []) if return_farm_data else empty_ret

        lat = df_country["Latitude"].to_numpy()
        lon = df_country["Longitude"].to_numpy()
        y_idx, x_idx = indexer.map_points(lat, lon)
        
        df_country['y_idx'] = y_idx
        df_country['x_idx'] = x_idx

        start_year = pd.to_numeric(df_country["Start year"], errors="coerce").to_numpy()
        asbuilt_mask = np.isfinite(start_year) & (start_year <= ms.prod_year)

        def compute_farm(i: int):
            row = df_country.iloc[i]
            ts = functions.estimate_power_final(
                country=country,
                lat=float(row["Latitude"]),
                lon=float(row["Longitude"]),
                status=row["Status"],
                capacity_mw=row["Capacity (MW)"],
                capacity_rating=row["Capacity Rating"],
                tech_type=row["Technology Type"],
                xrds=xrds,
                y_idx=int(y_idx[i]),
                x_idx=int(x_idx[i]),
                Spatial_interpolation=True,
                min_irr=10,
                twilight_zenith_limit=80,
                smoothing_window_hours=3,
                performance_ratio=0.9,
                start_year=row["Start year"],
                prod_year=ms.prod_year,
                enforce_start_year=False, 
                mounting_type="default",
            )
            if ts is None:
                return None
            return (i, np.asarray(ts.values, dtype=np.float64))

        farms = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=0)(
            delayed(compute_farm)(i) for i in tqdm(range(len(df_country)), desc=f"Estimating solar power for {country}", unit="farm")
        )

        watts_2025 = np.zeros(T, dtype=np.float64)
        watts_asbuilt = np.zeros(T, dtype=np.float64)
        farm_results = []

        for r in farms:
            if r is None:
                continue
            i, ts_w = r
            
            watts_2025 += ts_w
            if asbuilt_mask[i]:
                watts_asbuilt += ts_w
            
            if return_farm_data:
                farm_results.append((i, ts_w))

        mw_2025 = watts_2025 / 1_000_000.0
        mw_asbuilt = watts_asbuilt / 1_000_000.0

        shifted_index = pd.to_datetime(xrds["time"].values) - pd.Timedelta(hours=1)
        factor = self._factor_from_asbuilt(mw_asbuilt, shifted_index, solar_actual)

        mw_asbuilt_series = pd.Series(mw_asbuilt * factor, index=shifted_index).astype(np.float64)
        mw_2025_series = pd.Series(mw_2025 * factor, index=shifted_index).astype(np.float64)

        # Ensure the factor calculation also uses the shifted index
        res = (
            mw_asbuilt_series.values,
            mw_2025_series.values,
            float(factor)
        )
        
        if return_farm_data:
            return (*res, df_country, farm_results)
        return res

    @staticmethod
    def _factor_from_asbuilt(model_mw: np.ndarray, time_vals: np.ndarray, actual_mw: pd.Series) -> float:
        if float(np.nansum(model_mw)) <= 0:
            return 0.0
        calc = pd.Series(model_mw, index=pd.to_datetime(time_vals).tz_localize("UTC"))
        if actual_mw.index.tz is not None:
            calc.index = calc.index.tz_convert(actual_mw.index.tz)
        common = actual_mw.index.intersection(calc.index)
        if len(common) == 0:
            return 1.0
        return functions.get_correction_factor(calc.loc[common], actual_mw.loc[common])


class WindCalculator:
    def __init__(self):
        self.df = pd.read_csv(
            "/Data/gfi/vindenergi/nab015/Wind_data/Global-Wind-Power-Tracker-February-2025.csv",
            sep=";",
            decimal=",",
        )
        self.ref_startyear = int(pd.to_numeric(self.df["Start year"], errors="coerce").median())

    def open_weather(self, ms: MonthSpec) -> xr.Dataset:
        fn = f"/Data/gfi/vindenergi/nab015/CERRA_multi_level/{ms.year}/cerra_{ms.year}_multi_level_{ms.month_name}.nc"
        ds = xr.open_dataset(fn, engine="h5netcdf")
        ds.load()
        return ds

    def country_timeseries(
        self,
        ms: MonthSpec,
        xrds: xr.Dataset,
        indexer: GridIndexer,
        actual_loader: ActualGenerationLoader,
        actual_file: pd.DataFrame,
        country: str,
        area_code: str,
        return_farm_data: bool = True,
    ):
        T = xrds.sizes["valid_time"]
        empty_ret = (np.zeros(T, dtype=np.float64), np.zeros(T, dtype=np.float64), 0.0)

        df_country = self.df[self.df["Country/Area"] == country].copy()
        if len(df_country) == 0:
             return (*empty_ret, pd.DataFrame(), []) if return_farm_data else empty_ret

        if area_code in ZONES:
            if area_code.startswith('DK'):
                lon_vals = pd.to_numeric(df_country['Longitude'], errors='coerce')
                if area_code == 'DK1':
                    df_country = df_country[lon_vals < 10.9].copy()
                else: # DK2
                    df_country = df_country[lon_vals >= 10.9].copy()
            else:
                zone_cities = functions.get_bidding_zone_mapping(area_code)
                mask = pd.Series(False, index=df_country.index)
                cols_to_check = LOCATION_COLS_PV if "PV" in str(type(self)) else LOCATION_COLS_WIND
                for col in cols_to_check:
                    if col in df_country.columns:
                        mask |= df_country[col].isin(zone_cities)
                df_country = df_country[mask].copy()

        if len(df_country) == 0:
             return (*empty_ret, pd.DataFrame(), []) if return_farm_data else empty_ret

        wind_actual = actual_loader.wind_series_mw(actual_file, area_code)

        if wind_actual.sum() <= 1.0:
            print(f"Skipping {country} (Wind): No actual production data found.", flush=True)
            return (*empty_ret, pd.DataFrame(), []) if return_farm_data else empty_ret

        lat = df_country["Latitude"].to_numpy()
        lon = df_country["Longitude"].to_numpy()
        y_idx, x_idx = indexer.map_points(lat, lon)
        
        df_country['y_idx'] = y_idx
        df_country['x_idx'] = x_idx

        start_year_f = pd.to_numeric(df_country["Start year"], errors="coerce").to_numpy(dtype=float)
        asbuilt_mask = (
            np.isfinite(start_year_f)
            & (start_year_f >= 1800)
            & (start_year_f <= ms.prod_year)
        )
        
        ref = float(self.ref_startyear) if getattr(self, "ref_startyear", None) is not None else float(ms.prod_year)
        start_year_model = np.where(np.isfinite(start_year_f), np.floor(start_year_f), ref).astype(int)

        mw_2025 = np.zeros(T, dtype=np.float64)
        mw_asbuilt = np.zeros(T, dtype=np.float64)
        farm_results = []

        for i in tqdm(range(len(df_country)), desc=f"Estimating wind power for {country}", unit="farm"):
            row = df_country.iloc[i]
            ts_mw = functions.estimate_wind_power(
                country=country,
                lat=float(row["Latitude"]),
                lon=float(row["Longitude"]),
                capacity=row["Capacity (MW)"],
                startyear=int(start_year_model[i]),
                prod_year=ms.prod_year,
                status=row["Status"],
                installation_type=row["Installation Type"],
                xrds=xrds,
                y_idx=int(y_idx[i]),
                x_idx=int(x_idx[i]),
                wts_smoothing=False,
                power_smoothing=False,
                wake_loss_factor=0.99,
                spatial_interpolation=True,
                verbose=False,
                single_turb_curve=False,
                enforce_start_year=False,
            )
            if ts_mw is None:
                continue

            mw_2025 += ts_mw
            if asbuilt_mask[i]:
                mw_asbuilt += ts_mw
            
            if return_farm_data:
                farm_results.append((i, ts_mw))

        # FIX: Align Wind Timestamps similarly to PV (End of Hour -> Start of Hour labeling)
        # Assuming ENTSO-E actuals are interval-starting.
        shifted_index_wind = pd.to_datetime(xrds["valid_time"].values) - pd.Timedelta(hours=1)

        factor = PVCalculator._factor_from_asbuilt(
            mw_asbuilt,
            shifted_index_wind,
            wind_actual,
        )

        res = (
            (mw_asbuilt * factor).astype(np.float64),
            (mw_2025 * factor).astype(np.float64),
            float(factor)
        )
        
        if return_farm_data:
             return (*res, df_country, farm_results)
        return res


class MonthlyRunner:
    def __init__(self, out_dir_aggregated: Path, out_dir_farms: Path, n_jobs_pv: int = 8):
        self.out_dir_aggregated = out_dir_aggregated
        self.out_dir_farms = out_dir_farms
        
        self.out_dir_aggregated.mkdir(parents=True, exist_ok=True)
        self.out_dir_farms.mkdir(parents=True, exist_ok=True)

        self.tzmap = functions.get_timezone_mapping()
        self.actual_loader = ActualGenerationLoader(self.tzmap)

        self.pv = PVCalculator(n_jobs=n_jobs_pv)
        self.wind = WindCalculator()

    @staticmethod
    def _write_atomic(ds: xr.Dataset, out_path: Path):
        """Write NetCDF with atomic write and COMPRESSION."""
        
        # Define compression settings for every data variable
        comp = dict(zlib=True, complevel=4)
        encoding = {var: comp for var in ds.data_vars}

        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=out_path.parent,
            prefix=".tmp_",
            suffix=".nc",
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            ds.to_netcdf(tmp_path, engine="h5netcdf", encoding=encoding)
            os.replace(tmp_path, out_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _align_series(self, model_vals: np.ndarray, time_vals: np.ndarray, actual: pd.Series) -> tuple[pd.Series, pd.Series]:
        model = pd.Series(model_vals, index=pd.to_datetime(time_vals).tz_localize("UTC"))
        if len(actual) == 0:
            return model, actual
        if actual.index.tz is None:
            actual = actual.copy()
            actual.index = actual.index.tz_localize("UTC")
        else:
            model.index = model.index.tz_convert(actual.index.tz)
        common = actual.index.intersection(model.index)
        return model.loc[common], actual.loc[common]

    def plotting_timeseries(
        self,
        ms: MonthSpec,
        area_code: str,
        country: str,
        pv_model: np.ndarray,
        pv_time: np.ndarray,
        wind_model: np.ndarray,
        wind_time: np.ndarray,
        pv_factor: float,
        wind_factor: float,
        actual_file: pd.DataFrame,
    ) -> None:
        label = area_code if area_code in ZONES else country

        # PV plot
        solar_actual = self.actual_loader.solar_series_mw(actual_file, area_code)
        pv_model_series, solar_actual_series = self._align_series(pv_model, pv_time, solar_actual)

        pv_dir = Path(f"/Data/gfi/vindenergi/nab015/figures/pv_power_comparison/{ms.year}/{ms.month_number}")
        pv_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(pv_model_series.index, pv_model_series.values, "x-", color="red", label="Estimated PV (MW)")
        if len(solar_actual_series) > 0:
            ax.plot(solar_actual_series.index, solar_actual_series.values, "--", color="black", label="Actual PV (MW)")
        ax.text(0.02, 0.95, f"α={pv_factor:.4f}", transform=ax.transAxes, va="top")
        if len(solar_actual_series) > 1:
            corr = solar_actual_series.corr(pv_model_series)
            if pd.notna(corr):
                ax.text(0.02, 0.88, f"R²={corr**2:.4f}", transform=ax.transAxes, va="top")
        ax.set_title(f"PV power {label} {ms.month_number}-{ms.year}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power (MW)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.legend(loc="upper right")
        ax.grid(True)
        pv_out = pv_dir / f"pv_power_comparison_{ms.month_number}_{ms.year}_{label}.svg"
        fig.savefig(pv_out, bbox_inches="tight")
        plt.close(fig)

        # Wind plot
        wind_actual = self.actual_loader.wind_series_mw(actual_file, area_code)
        wind_model_series, wind_actual_series = self._align_series(wind_model, wind_time, wind_actual)

        wind_dir = Path(f"/Data/gfi/vindenergi/nab015/figures/wind_power_comparison/{ms.year}/{ms.month_number}")
        wind_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(wind_model_series.index, wind_model_series.values, "x-", color="red", label="Estimated Wind (MW)")
        if len(wind_actual_series) > 0:
            ax.plot(wind_actual_series.index, wind_actual_series.values, "--", color="black", label="Actual Wind (MW)")
        ax.text(0.02, 0.95, f"α={wind_factor:.4f}", transform=ax.transAxes, va="top")
        if len(wind_actual_series) > 1:
            corr = wind_actual_series.corr(wind_model_series)
            if pd.notna(corr):
                ax.text(0.02, 0.88, f"R²={corr**2:.4f}", transform=ax.transAxes, va="top")
        ax.set_title(f"Wind power {label} {ms.month_number}-{ms.year}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power (MW)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.legend(loc="upper right")
        ax.grid(True)
        wind_out = wind_dir / f"wind_power_comparison_{ms.month_number}_{ms.year}_{label}.svg"
        fig.savefig(wind_out, bbox_inches="tight")
        plt.close(fig)

    def run_month(self, ms: MonthSpec) -> Path:
        print(f"\n=== Running {ms.year}-{ms.month_number} ({ms.month_name}) ===", flush=True)

        actual_file = self.actual_loader.load_month_file(ms.year, ms.month_number)

        # 1. Open Weather Data
        # PV Dataset
        pv_ds = self.pv.open_weather(ms)
        pv_indexer = GridIndexer(pv_ds["latitude"].values, pv_ds["longitude"].values)
        
        # Wind Dataset
        wind_ds = self.wind.open_weather(ms)
        wind_indexer = GridIndexer(wind_ds["latitude"].values, wind_ds["longitude"].values)

        # 2. Initialize Grids (Float32 to save memory)
        # Dimensions: [Time, Y, X]
        print("Initializing global grids...", flush=True)
        T_pv = pv_ds.sizes["time"]
        Y_dim = pv_ds.sizes["y"]
        X_dim = pv_ds.sizes["x"]
        
        # We will use PV coordinates for the final output grid
        global_pv_grid = np.zeros((T_pv, Y_dim, X_dim), dtype=np.float32)
        global_wind_grid = np.zeros((T_pv, Y_dim, X_dim), dtype=np.float32)

        # 3. Aggregation Lists
        pv_asbuilt_all = []
        pv_2025_all = []
        wind_asbuilt_all = []
        wind_2025_all = []
        areas = []
        
        # --- NEW: List to collect correction factors ---
        factor_records = [] 
        # -----------------------------------------------

        # 4. Iterate Countries
        for country, code in zip(countries_tracker, countries_codes):
            # --- PV Calculation ---
            pv_ret = self.pv.country_timeseries(
                ms, pv_ds, pv_indexer, self.actual_loader, actual_file, country, code, return_farm_data=True
            )
            pv_as, pv_25, pv_factor, pv_df_country, pv_farms = pv_ret
            
            # Aggregate stats
            pv_asbuilt_all.append(pv_as)
            pv_2025_all.append(pv_25)

            # Accumulate to global grid (Apply factor, convert Watts to MW)
            if pv_farms:
                for i, ts_watts in pv_farms:
                    y = pv_df_country.iloc[i]["y_idx"]
                    x = pv_df_country.iloc[i]["x_idx"]
                    
                    # Convert W -> MW and apply factor
                    ts_mw = (ts_watts / 1_000_000.0) * pv_factor
                    global_pv_grid[:, int(y), int(x)] += ts_mw

            # --- Wind Calculation ---
            wind_ret = self.wind.country_timeseries(
                ms, wind_ds, wind_indexer, self.actual_loader, actual_file, country, code, return_farm_data=True
            )
            w_as, w_25, w_factor, w_df_country, w_farms = wind_ret

            # Aggregate stats
            wind_asbuilt_all.append(w_as)
            wind_2025_all.append(w_25)

            # Accumulate to global grid (Apply factor)
            if w_farms:
                for i, ts_mw_unc in w_farms:
                    y = w_df_country.iloc[i]["y_idx"]
                    x = w_df_country.iloc[i]["x_idx"]
                    
                    # Apply factor
                    ts_mw = ts_mw_unc * w_factor
                    global_wind_grid[:, int(y), int(x)] += ts_mw

            areas.append(code)

            # --- NEW: Append factors to list ---
            factor_records.append({
                "Area": code,
                "PV_Factor": pv_factor,
                "Wind_Factor": w_factor
            })
            
            # Save incrementally (optional, but good practice) or save after loop.
            # Here we follow the requested style of saving inside the loop to ensure progress is captured.
            alpha_out = self.out_dir_aggregated / f"correction_factors/{ms.month_number}_{ms.year}_pv_wind_country_factors.csv"
            alpha_out.parent.mkdir(parents=True, exist_ok=True)
            alpha_dataframe = pd.DataFrame(factor_records, columns=["Area", "PV_Factor", "Wind_Factor"])
            alpha_dataframe.to_csv(alpha_out, index=False)
            print(f"Wrote correction factors file: {alpha_out}", flush=True)
            # -----------------------------------

            # Plots
            self.plotting_timeseries(
                ms=ms,
                area_code=code,
                country=country,
                pv_model=pv_as,
                pv_time=pv_ds["time"].values,
                wind_model=w_as,
                wind_time=wind_ds["valid_time"].values,
                pv_factor=pv_factor,
                wind_factor=w_factor,
                actual_file=actual_file,
            )

        # 5. Save Aggregated Country Output
        pv_asbuilt = np.stack(pv_asbuilt_all, axis=1)
        pv_2025 = np.stack(pv_2025_all, axis=1)
        wind_asbuilt = np.stack(wind_asbuilt_all, axis=1)
        wind_2025 = np.stack(wind_2025_all, axis=1)

        out_agg = xr.Dataset(
            data_vars={
                "pv_power_mw": (("time", "area"), pv_asbuilt),
                "pv_power_mw_2025": (("time", "area"), pv_2025),
                "wind_power_mw": (("time", "area"), wind_asbuilt),
                "wind_power_mw_2025": (("time", "area"), wind_2025),
            },
            coords={
                "time": pv_ds["time"].values,
                "area": np.array(areas, dtype="U"),
            },
            attrs={
                "year": ms.year,
                "month_number": ms.month_number,
                "month_name": ms.month_name,
                "note": "as-built calibrated factor applied to both as-built and 2025 fleet scenarios per area",
            },
        )

        out_file_agg = self.out_dir_aggregated / f"{ms.month_number}_{ms.year}_pv_wind_country_timeseries.nc"
        if out_file_agg.exists():
            out_file_agg.unlink()
        self._write_atomic(out_agg, out_file_agg)
        print(f"Wrote aggregated file: {out_file_agg}", flush=True)

        # 6. Save Gridded Per-Farm Output
        print("Saving gridded per-farm output...", flush=True)
        # Using dimensions and coords from pv_ds as reference
        grid_out = xr.Dataset(
            {
                'wind_power_mw': (('time', 'y', 'x'), global_wind_power_grid := global_wind_grid),
                'pv_power_mw': (('time', 'y', 'x'), global_pv_power_grid := global_pv_grid),
            },
            coords={
                'time': (('time',), pv_ds['time'].values),
                'y': (('y',), pv_ds['y'].values),
                'x': (('x',), pv_ds['x'].values),
                'latitude': (('y', 'x'), pv_ds['latitude'].values),
                'longitude': (('y', 'x'), pv_ds['longitude'].values)
            }
        )
        
        # Free memory
        del global_wind_grid, global_pv_grid

        out_file_grid = self.out_dir_farms / f"{ms.month_number}_{ms.year}_pv_wind_grid.nc"
        if out_file_grid.exists():
            out_file_grid.unlink()
        self._write_atomic(grid_out, out_file_grid)
        print(f"Wrote gridded file: {out_file_grid}", flush=True)

        # Cleanup
        pv_ds.close()
        wind_ds.close()
        del pv_ds, wind_ds, pv_indexer, wind_indexer, out_agg, grid_out
        gc.collect()

        return out_file_agg


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--year", required=True)
    p.add_argument("--month", required=True, help="Month number: 01..12")
    p.add_argument("--n-jobs-pv", type=int, default=8)
    
    # Aggregated output defaults to original path
    p.add_argument("--out-dir", default="/Data/gfi/vindenergi/nab015/highres-renewable-dataset/country-aggregated-production")
    # Per-farm output defaults to new request path
    p.add_argument("--out-dir-farm", default="/Data/gfi/vindenergi/nab015/highres-renewable-dataset/per-farm-production")
    
    args = p.parse_args()

    month_number = args.month.zfill(2)
    month_name = dict(MONTHS).get(month_number)
    if month_name is None:
        raise ValueError(f"Invalid month {args.month}; expected 01..12")

    runner = MonthlyRunner(
        out_dir_aggregated=Path(args.out_dir) / args.year,
        out_dir_farms=Path(args.out_dir_farm) / args.year,
        n_jobs_pv=args.n_jobs_pv
    )
    runner.run_month(MonthSpec(year=args.year, month_number=month_number, month_name=month_name))


if __name__ == "__main__":
    main()