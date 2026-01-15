#!/usr/bin/env python3
# coding: utf-8

#-----------Run commands----------------#
# python -u scripts/weather_energy_monthly.py --year 2024 --month 09
# for m in $(seq -w 1 12); do   python -u scripts/weather_energy_monthly.py --year 2024 --month "$m" --n-jobs-pv 2; done
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
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
if "functions" in sys.modules:
    del sys.modules["functions"]

import functions
import inspect
print("functions module file:", functions.__file__, flush=True)
print("estimate_power_final sig:", inspect.signature(functions.estimate_power_final), flush=True)
print("weather_energy_monthly.py file:", __file__, flush=True)

# python -u scripts/weather_energy_monthly.py --year 2024 --month 09 --n-jobs-pv 2

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
    Build KDTree for nearest (lat, lon) lookup with dateline wrap.
    Reused for PV and wind mapping.
    """
    def __init__(self, lat2d: np.ndarray, lon2d: np.ndarray):
        self.lat2d = lat2d
        self.lon2d = lon2d
        self.shape = lat2d.shape

        pts0 = np.column_stack((lat2d.ravel(), lon2d.ravel()))
        pts360 = np.column_stack((lat2d.ravel(), lon2d.ravel() + 360.0))
        self.tree0 = cKDTree(pts0)
        self.tree360 = cKDTree(pts360)

    def map_points(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts0 = np.column_stack((lat, lon))
        pts1 = np.column_stack((lat, lon + 360.0))
        d0, idx0 = self.tree0.query(pts0, k=1)
        d1, idx1 = self.tree360.query(pts1, k=1)
        idx = np.where(d0 <= d1, idx0, idx1).astype(int)
        y, x = np.unravel_index(idx, self.shape)
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
        da_diff = da.diff("time", label="upper")
        hourly = xr.concat([da.isel(time=0), da_diff], dim="time")
        return hourly.where(hourly >= 0, other=da)

    def open_weather(self, ms: MonthSpec) -> xr.Dataset:
        fn1 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{ms.year}/reanalysis-cerra-single-levels-{ms.month_name}-{ms.year}-time1.nc"
        fn2 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{ms.year}/reanalysis-cerra-single-levels-{ms.month_name}-{ms.year}-time2.nc"
        fn3 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{ms.year}/reanalysis-cerra-single-levels-{ms.month_name}-{ms.year}-time3.nc"
        fn4 = f"/Data/gfi/vindenergi/nab015/CERRA_single_level/{ms.year}/reanalysis-cerra-single-levels-{ms.month_name}-{ms.year}-time4.nc"

        ds_list = [xr.open_dataset(f, engine="h5netcdf") for f in (fn1, fn2, fn3, fn4)]
        raw = xr.concat(ds_list, dim="valid_time", combine_attrs="drop_conflicts").sortby("valid_time")
        raw = raw.sel(valid_time=~raw.get_index("valid_time").duplicated()).rename({"valid_time": "time"})
        for d in ds_list:
            d.close()

        combined = xr.Dataset(
            data_vars={
                "irradiance": self._deaccumulate_robust(raw["ssrd"]) / 3600,
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
        return_farm_data: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, pd.DataFrame, list]:
        # farm list
        df_country = pd.concat(
            [
                self.df_20[self.df_20["Country/Area"] == country],
                self.df_1_20[self.df_1_20["Country/Area"] == country],
            ],
            ignore_index=True,
        )

        if area_code in ZONES:
            zone_cities = functions.get_bidding_zone_mapping(area_code)
            mask = pd.Series(False, index=df_country.index)
            for col in LOCATION_COLS_PV:
                if col in df_country.columns:
                    mask |= df_country[col].isin(zone_cities)
            df_country = df_country[mask].copy()
        else:
            df_country = df_country.copy()

        if len(df_country) == 0:
            if return_farm_data:
                return (np.zeros(xrds.sizes["time"], dtype=np.float64), 
                       np.zeros(xrds.sizes["time"], dtype=np.float64), 
                       0.0, pd.DataFrame(), [])
            return np.zeros(xrds.sizes["time"], dtype=np.float64), np.zeros(xrds.sizes["time"], dtype=np.float64), 0.0

        #operating only (both scenarios)
        op_status = set(functions.operating_farms(country, "solar"))
        df_country = df_country[df_country["Status"].isin(op_status)].copy()
        if len(df_country) == 0:
            if return_farm_data:
                return (np.zeros(xrds.sizes["time"], dtype=np.float64), 
                       np.zeros(xrds.sizes["time"], dtype=np.float64), 
                       0.0, pd.DataFrame(), [])
            return np.zeros(xrds.sizes["time"], dtype=np.float64), np.zeros(xrds.sizes["time"], dtype=np.float64), 0.0

        #actual
        solar_actual = actual_loader.solar_series_mw(actual_file, area_code)

        #indices
        lat = df_country["Latitude"].astype(float).to_numpy()
        lon = df_country["Longitude"].astype(float).to_numpy()
        y_idx, x_idx = indexer.map_points(lat, lon)

        start_year = pd.to_numeric(df_country["Start year"], errors="coerce").to_numpy()
        asbuilt_mask = np.isfinite(start_year) & (start_year <= ms.prod_year)

        T = xrds.sizes["time"]

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
            delayed(compute_farm)(i) for i in range(len(df_country))
        )

        watts_2025 = np.zeros(T, dtype=np.float64)
        watts_asbuilt = np.zeros(T, dtype=np.float64)
        farm_timeseries = []  # Store individual farm timeseries if needed
        
        for r in farms:
            if r is None:
                continue
            i, ts_w = r
            watts_2025 += ts_w
            if asbuilt_mask[i]:
                watts_asbuilt += ts_w
            if return_farm_data:
                farm_timeseries.append((i, ts_w))

        mw_2025 = watts_2025 / 1_000_000.0
        mw_asbuilt = watts_asbuilt / 1_000_000.0

        # factor from as-built; apply to both
        factor = self._factor_from_asbuilt(mw_asbuilt, xrds["time"].values, solar_actual)
        
        if return_farm_data:
            # Prepare farm metadata
            df_meta = df_country.copy()
            df_meta['y_idx'] = y_idx
            df_meta['x_idx'] = x_idx
            df_meta['start_year_parsed'] = start_year
            df_meta['asbuilt_mask'] = asbuilt_mask
            return ((mw_asbuilt * factor).astype(np.float64), 
                   (mw_2025 * factor).astype(np.float64), 
                   float(factor), 
                   df_meta, 
                   farm_timeseries)
        
        return (mw_asbuilt * factor).astype(np.float64), (mw_2025 * factor).astype(np.float64), float(factor)

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
        return_farm_data: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, pd.DataFrame, list]:
        df_country = self.df[self.df["Country/Area"] == country].copy()
        if len(df_country) == 0:
            T = xrds.sizes["valid_time"]
            if return_farm_data:
                return (np.zeros(T, dtype=np.float64), 
                       np.zeros(T, dtype=np.float64), 
                       0.0, pd.DataFrame(), [])
            return np.zeros(T, dtype=np.float64), np.zeros(T, dtype=np.float64), 0.0

        if area_code in ZONES:
            zone_cities = functions.get_bidding_zone_mapping(area_code)
            mask = pd.Series(False, index=df_country.index)
            for col in LOCATION_COLS_WIND:
                if col in df_country.columns:
                    mask |= df_country[col].isin(zone_cities)
            df_country = df_country[mask].copy()

        # operating only (both scenarios)
        op_status = set(functions.operating_farms(country, "wind"))
        df_country = df_country[df_country["Status"].isin(op_status)].copy()
        if len(df_country) == 0:
            T = xrds.sizes["valid_time"]
            if return_farm_data:
                return (np.zeros(T, dtype=np.float64), 
                       np.zeros(T, dtype=np.float64), 
                       0.0, pd.DataFrame(), [])
            return np.zeros(T, dtype=np.float64), np.zeros(T, dtype=np.float64), 0.0

        wind_actual = actual_loader.wind_series_mw(actual_file, area_code)

        lat = df_country["Latitude"].astype(float).to_numpy()
        lon = df_country["Longitude"].astype(float).to_numpy()
        y_idx, x_idx = indexer.map_points(lat, lon)

        # Parse start year as float with NaNs for unknown values
        start_year_f = pd.to_numeric(df_country["Start year"], errors="coerce").to_numpy(dtype=float)

        # "As-built" scenario should only include farms where the start year is known and <= prod_year.
        # Add a plausible range guard to avoid garbage sentinel values.
        asbuilt_mask = (
            np.isfinite(start_year_f)
            & (start_year_f >= 1800)
            & (start_year_f <= ms.prod_year)
        )

        # For modeling, fill missing/invalid start years with a reference year (do NOT use this for asbuilt_mask)
        ref = float(self.ref_startyear) if getattr(self, "ref_startyear", None) is not None else float(ms.prod_year)
        start_year_model = np.where(np.isfinite(start_year_f), np.floor(start_year_f), ref).astype(int)

        T = xrds.sizes["valid_time"]
        mw_2025 = np.zeros(T, dtype=np.float64)
        mw_asbuilt = np.zeros(T, dtype=np.float64)
        farm_timeseries = []  # Store individual farm timeseries if needed

        for i in range(len(df_country)):
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
                wts_smoothing=True,
                wake_loss_factor=0.95,
                spatial_interpolation=True,
                verbose=False,
                single_turb_curve=False,
                enforce_start_year=False,  # compute once, scenario decides
            )
            if ts_mw is None:
                continue

            mw_2025 += ts_mw
            if asbuilt_mask[i]:
                mw_asbuilt += ts_mw
            if return_farm_data:
                farm_timeseries.append((i, ts_mw))

        factor = PVCalculator._factor_from_asbuilt(
            mw_asbuilt,
            xrds["valid_time"].values,
            wind_actual,
        )
        
        if return_farm_data:
            # Prepare farm metadata
            df_meta = df_country.copy()
            df_meta['y_idx'] = y_idx
            df_meta['x_idx'] = x_idx
            df_meta['start_year_parsed'] = start_year_f
            df_meta['asbuilt_mask'] = asbuilt_mask
            return ((mw_asbuilt * factor).astype(np.float64), 
                   (mw_2025 * factor).astype(np.float64), 
                   float(factor), 
                   df_meta, 
                   farm_timeseries)
        
        return (mw_asbuilt * factor).astype(np.float64), (mw_2025 * factor).astype(np.float64), float(factor)


class MonthlyRunner:
    def __init__(self, out_dir: Path, n_jobs_pv: int = 8, write_farm_timeseries: bool = False):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.write_farm_timeseries = write_farm_timeseries

        self.tzmap = functions.get_timezone_mapping()
        self.actual_loader = ActualGenerationLoader(self.tzmap)

        self.pv = PVCalculator(n_jobs=n_jobs_pv)
        self.wind = WindCalculator()

    @staticmethod
    def _write_farm_netcdf_atomic(ds: xr.Dataset, out_path: Path):
        """Write NetCDF with atomic write and h5netcdf engine."""
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=out_path.parent,
            prefix=".tmp_",
            suffix=".nc",
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            ds.to_netcdf(tmp_path, engine="h5netcdf")
            os.replace(tmp_path, out_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _write_pv_farm_timeseries(
        self,
        ms: MonthSpec,
        time_coords: np.ndarray,
        all_meta: list[pd.DataFrame],
        all_farms: list[list],
        correction_factors: list[float],
    ):
        """Write per-farm PV timeseries to NetCDF."""
        # Collect all farms across areas
        farm_list = []
        for area_idx, (df_meta, farm_ts_list) in enumerate(zip(all_meta, all_farms)):
            if len(df_meta) == 0:
                continue
            area_code = countries_codes[area_idx]
            country = countries_tracker[area_idx]
            factor = correction_factors[area_idx]
            
            for i, ts_w in farm_ts_list:
                row = df_meta.iloc[i]
                # Convert watts to MW and apply correction factor
                ts_mw = (ts_w / 1_000_000.0) * factor
                
                farm_id = f"{country}_{area_code}_{row.name}"
                farm_list.append({
                    "farm_id": farm_id,
                    "country": country,
                    "area": area_code,
                    "latitude": float(row["Latitude"]),
                    "longitude": float(row["Longitude"]),
                    "y_idx": int(row["y_idx"]),
                    "x_idx": int(row["x_idx"]),
                    "capacity_mw": float(row["Capacity (MW)"]),
                    "status": str(row["Status"]),
                    "start_year": float(row["start_year_parsed"]) if np.isfinite(row["start_year_parsed"]) else np.nan,
                    "is_asbuilt": bool(row["asbuilt_mask"]),
                    "timeseries_mw": ts_mw,
                })
        
        if not farm_list:
            print("No PV farms to write.", flush=True)
            return
        
        # Build dataset
        n_farms = len(farm_list)
        n_time = len(time_coords)
        
        # Stack timeseries into 2D array (farm, time)
        ts_array = np.zeros((n_farms, n_time), dtype=np.float32)
        for i, farm in enumerate(farm_list):
            ts_array[i, :] = farm["timeseries_mw"]
        
        # Create dataset
        ds = xr.Dataset(
            data_vars={
                "power_mw_2025": (("farm", "time"), ts_array, {"long_name": "PV power output MW (2025 scenario)"}),
                "latitude": (("farm",), np.array([f["latitude"] for f in farm_list], dtype=np.float32)),
                "longitude": (("farm",), np.array([f["longitude"] for f in farm_list], dtype=np.float32)),
                "y_idx": (("farm",), np.array([f["y_idx"] for f in farm_list], dtype=np.int32)),
                "x_idx": (("farm",), np.array([f["x_idx"] for f in farm_list], dtype=np.int32)),
                "capacity_mw": (("farm",), np.array([f["capacity_mw"] for f in farm_list], dtype=np.float32)),
                "start_year": (("farm",), np.array([f["start_year"] for f in farm_list], dtype=np.float32)),
                "is_asbuilt": (("farm",), np.array([f["is_asbuilt"] for f in farm_list], dtype=bool)),
            },
            coords={
                "farm": np.array([f["farm_id"] for f in farm_list], dtype="U"),
                "time": time_coords,
                "country": (("farm",), np.array([f["country"] for f in farm_list], dtype="U")),
                "area": (("farm",), np.array([f["area"] for f in farm_list], dtype="U")),
                "status": (("farm",), np.array([f["status"] for f in farm_list], dtype="U")),
            },
            attrs={
                "year": ms.year,
                "month_number": ms.month_number,
                "month_name": ms.month_name,
                "description": "Per-farm PV generation timeseries with metadata",
                "note": "power_mw_2025 includes all farms with correction factor applied; is_asbuilt indicates farms in as-built scenario",
            },
        )
        
        out_file = self.out_dir / f"{ms.month_number}_{ms.year}_pv_farm_timeseries.nc"
        if out_file.exists():
            out_file.unlink()
        self._write_farm_netcdf_atomic(ds, out_file)
        print(f"Wrote {out_file} ({n_farms} farms)", flush=True)

    def _write_wind_farm_timeseries(
        self,
        ms: MonthSpec,
        time_coords: np.ndarray,
        all_meta: list[pd.DataFrame],
        all_farms: list[list],
        correction_factors: list[float],
    ):
        """Write per-farm wind timeseries to NetCDF."""
        # Collect all farms across areas
        farm_list = []
        for area_idx, (df_meta, farm_ts_list) in enumerate(zip(all_meta, all_farms)):
            if len(df_meta) == 0:
                continue
            area_code = countries_codes[area_idx]
            country = countries_tracker[area_idx]
            factor = correction_factors[area_idx]
            
            for i, ts_mw in farm_ts_list:
                row = df_meta.iloc[i]
                # Apply correction factor
                ts_mw_corrected = ts_mw * factor
                
                farm_id = f"{country}_{area_code}_{row.name}"
                farm_list.append({
                    "farm_id": farm_id,
                    "country": country,
                    "area": area_code,
                    "latitude": float(row["Latitude"]),
                    "longitude": float(row["Longitude"]),
                    "y_idx": int(row["y_idx"]),
                    "x_idx": int(row["x_idx"]),
                    "capacity_mw": float(row["Capacity (MW)"]),
                    "status": str(row["Status"]),
                    "start_year": float(row["start_year_parsed"]) if np.isfinite(row["start_year_parsed"]) else np.nan,
                    "is_asbuilt": bool(row["asbuilt_mask"]),
                    "timeseries_mw": ts_mw_corrected,
                })
        
        if not farm_list:
            print("No wind farms to write.", flush=True)
            return
        
        # Build dataset
        n_farms = len(farm_list)
        n_time = len(time_coords)
        
        # Stack timeseries into 2D array (farm, time)
        ts_array = np.zeros((n_farms, n_time), dtype=np.float32)
        for i, farm in enumerate(farm_list):
            ts_array[i, :] = farm["timeseries_mw"]
        
        # Create dataset
        ds = xr.Dataset(
            data_vars={
                "power_mw_2025": (("farm", "time"), ts_array, {"long_name": "Wind power output MW (2025 scenario)"}),
                "latitude": (("farm",), np.array([f["latitude"] for f in farm_list], dtype=np.float32)),
                "longitude": (("farm",), np.array([f["longitude"] for f in farm_list], dtype=np.float32)),
                "y_idx": (("farm",), np.array([f["y_idx"] for f in farm_list], dtype=np.int32)),
                "x_idx": (("farm",), np.array([f["x_idx"] for f in farm_list], dtype=np.int32)),
                "capacity_mw": (("farm",), np.array([f["capacity_mw"] for f in farm_list], dtype=np.float32)),
                "start_year": (("farm",), np.array([f["start_year"] for f in farm_list], dtype=np.float32)),
                "is_asbuilt": (("farm",), np.array([f["is_asbuilt"] for f in farm_list], dtype=bool)),
            },
            coords={
                "farm": np.array([f["farm_id"] for f in farm_list], dtype="U"),
                "time": time_coords,
                "country": (("farm",), np.array([f["country"] for f in farm_list], dtype="U")),
                "area": (("farm",), np.array([f["area"] for f in farm_list], dtype="U")),
                "status": (("farm",), np.array([f["status"] for f in farm_list], dtype="U")),
            },
            attrs={
                "year": ms.year,
                "month_number": ms.month_number,
                "month_name": ms.month_name,
                "description": "Per-farm wind generation timeseries with metadata",
                "note": "power_mw_2025 includes all farms with correction factor applied; is_asbuilt indicates farms in as-built scenario",
            },
        )
        
        out_file = self.out_dir / f"{ms.month_number}_{ms.year}_wind_farm_timeseries.nc"
        if out_file.exists():
            out_file.unlink()
        self._write_farm_netcdf_atomic(ds, out_file)
        print(f"Wrote {out_file} ({n_farms} farms)", flush=True)

    def run_month(self, ms: MonthSpec) -> Path:
        print(f"\n=== Running {ms.year}-{ms.month_number} ({ms.month_name}) ===", flush=True)

        actual_file = self.actual_loader.load_month_file(ms.year, ms.month_number)

        # PV weather/grid
        pv_ds = self.pv.open_weather(ms)
        pv_indexer = GridIndexer(pv_ds["latitude"].values, pv_ds["longitude"].values)

        # Wind weather/grid (identical grid, but different dataset)
        wind_ds = self.wind.open_weather(ms)
        wind_indexer = GridIndexer(wind_ds["latitude"].values, wind_ds["longitude"].values)

        # Compute country series
        pv_asbuilt_all = []
        pv_2025_all = []
        wind_asbuilt_all = []
        wind_2025_all = []
        areas = []
        
        # For per-farm outputs
        pv_meta_all = []
        pv_farms_all = []
        pv_factors_all = []
        wind_meta_all = []
        wind_farms_all = []
        wind_factors_all = []

        for country, code in zip(countries_tracker, countries_codes):
            if self.write_farm_timeseries:
                pv_as, pv_25, pv_factor, pv_meta, pv_farms = self.pv.country_timeseries(
                    ms, pv_ds, pv_indexer, self.actual_loader, actual_file, country, code, return_farm_data=True
                )
                w_as, w_25, w_factor, w_meta, w_farms = self.wind.country_timeseries(
                    ms, wind_ds, wind_indexer, self.actual_loader, actual_file, country, code, return_farm_data=True
                )
                pv_meta_all.append(pv_meta)
                pv_farms_all.append(pv_farms)
                pv_factors_all.append(pv_factor)
                wind_meta_all.append(w_meta)
                wind_farms_all.append(w_farms)
                wind_factors_all.append(w_factor)
            else:
                pv_as, pv_25, pv_factor = self.pv.country_timeseries(
                    ms, pv_ds, pv_indexer, self.actual_loader, actual_file, country, code
                )
                w_as, w_25, w_factor = self.wind.country_timeseries(
                    ms, wind_ds, wind_indexer, self.actual_loader, actual_file, country, code
                )

            # include areas even if empty so dimensions are consistent
            areas.append(code)
            pv_asbuilt_all.append(pv_as)
            pv_2025_all.append(pv_25)
            wind_asbuilt_all.append(w_as)
            wind_2025_all.append(w_25)

        # Stack to (time, area)
        pv_asbuilt = np.stack(pv_asbuilt_all, axis=1)
        pv_2025 = np.stack(pv_2025_all, axis=1)
        wind_asbuilt = np.stack(wind_asbuilt_all, axis=1)
        wind_2025 = np.stack(wind_2025_all, axis=1)

        # Build output dataset
        # Use PV time coordinate as canonical for the month
        out = xr.Dataset(
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

        out_file = self.out_dir / f"{ms.month_number}_{ms.year}_pv_wind_country_timeseries.nc"
        if out_file.exists():
            out_file.unlink()
        out.to_netcdf(out_file, engine="h5netcdf")
        print(f"Wrote {out_file}", flush=True)

        # Write per-farm outputs if requested
        if self.write_farm_timeseries:
            self._write_pv_farm_timeseries(ms, pv_ds["time"].values, pv_meta_all, pv_farms_all, pv_factors_all)
            self._write_wind_farm_timeseries(ms, wind_ds["valid_time"].values, wind_meta_all, wind_farms_all, wind_factors_all)

        # Cleanup
        pv_ds.close()
        wind_ds.close()
        del pv_ds, wind_ds, pv_indexer, wind_indexer, out
        gc.collect()

        return out_file


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--year", required=True)
    p.add_argument("--month", required=True, help="Month number: 01..12")
    p.add_argument("--n-jobs-pv", type=int, default=8)
    p.add_argument("--out-dir", default="/Data/gfi/vindenergi/nab015/energy_country_timeseries")
    p.add_argument("--write-farm-timeseries", action="store_true", 
                   help="Write per-farm timeseries outputs in addition to aggregated country/area outputs")
    args = p.parse_args()

    month_number = args.month.zfill(2)
    month_name = dict(MONTHS).get(month_number)
    if month_name is None:
        raise ValueError(f"Invalid month {args.month}; expected 01..12")

    runner = MonthlyRunner(
        out_dir=Path(args.out_dir) / args.year, 
        n_jobs_pv=args.n_jobs_pv,
        write_farm_timeseries=args.write_farm_timeseries
    )
    runner.run_month(MonthSpec(year=args.year, month_number=month_number, month_name=month_name))


if __name__ == "__main__":
    main()