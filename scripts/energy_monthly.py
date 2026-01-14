#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

import functions

# python -u scripts/energy_monthly.py --year 2024 --month 09 --n-jobs-pv 2

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
        idx = np.where(d0 <= d1, idx0, idx1).astype(np.int64)
        y, x = np.unravel_index(idx, self.shape)
        return y.astype(np.int32), x.astype(np.int32)


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
    ) -> tuple[np.ndarray, np.ndarray, float]:
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
            return np.zeros(xrds.sizes["time"], dtype=np.float32), np.zeros(xrds.sizes["time"], dtype=np.float32), 0.0

        #operating only (both scenarios)
        op_status = set(functions.operating_farms(country, "solar"))
        df_country = df_country[df_country["Status"].isin(op_status)].copy()
        if len(df_country) == 0:
            return np.zeros(xrds.sizes["time"], dtype=np.float32), np.zeros(xrds.sizes["time"], dtype=np.float32), 0.0

        #actual
        solar_actual = actual_loader.solar_series_mw(actual_file, area_code)

        #indices
        lat = df_country["Latitude"].astype(float).to_numpy()
        lon = df_country["Longitude"].astype(float).to_numpy()
        y_idx, x_idx = indexer.map_points(lat, lon)

        start_year = pd.to_numeric(df_country["Start year"], errors="coerce").to_numpy()
        asbuilt_mask = np.isfinite(start_year) & (start_year.astype(np.int32) <= ms.prod_year)

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
        for r in farms:
            if r is None:
                continue
            i, ts_w = r
            watts_2025 += ts_w
            if asbuilt_mask[i]:
                watts_asbuilt += ts_w

        mw_2025 = watts_2025 / 1_000_000.0
        mw_asbuilt = watts_asbuilt / 1_000_000.0

        # factor from as-built; apply to both
        factor = self._factor_from_asbuilt(mw_asbuilt, xrds["time"].values, solar_actual)
        return (mw_asbuilt * factor).astype(np.float32), (mw_2025 * factor).astype(np.float32), float(factor)

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
    ) -> tuple[np.ndarray, np.ndarray, float]:
        df_country = self.df[self.df["Country/Area"] == country].copy()
        if len(df_country) == 0:
            T = xrds.sizes["valid_time"]
            return np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32), 0.0

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
            return np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32), 0.0

        wind_actual = actual_loader.wind_series_mw(actual_file, area_code)

        lat = df_country["Latitude"].astype(float).to_numpy()
        lon = df_country["Longitude"].astype(float).to_numpy()
        y_idx, x_idx = indexer.map_points(lat, lon)

        start_year = pd.to_numeric(df_country["Start year"], errors="coerce").to_numpy()
        start_year = np.where(np.isfinite(start_year), start_year, self.ref_startyear).astype(np.int32)
        asbuilt_mask = start_year <= ms.prod_year

        T = xrds.sizes["valid_time"]
        mw_2025 = np.zeros(T, dtype=np.float64)
        mw_asbuilt = np.zeros(T, dtype=np.float64)

        for i in range(len(df_country)):
            row = df_country.iloc[i]
            ts_mw = functions.estimate_wind_power(
                country=country,
                lat=float(row["Latitude"]),
                lon=float(row["Longitude"]),
                capacity=row["Capacity (MW)"],
                startyear=int(start_year[i]),
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

        factor = PVCalculator._factor_from_asbuilt(mw_asbuilt, xrds["valid_time"].values, wind_actual)
        return (mw_asbuilt * factor).astype(np.float32), (mw_2025 * factor).astype(np.float32), float(factor)


class MonthlyRunner:
    def __init__(self, out_dir: Path, n_jobs_pv: int = 8):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.tzmap = functions.get_timezone_mapping()
        self.actual_loader = ActualGenerationLoader(self.tzmap)

        self.pv = PVCalculator(n_jobs=n_jobs_pv)
        self.wind = WindCalculator()

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

        for country, code in zip(countries_tracker, countries_codes):
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
    args = p.parse_args()

    month_number = args.month.zfill(2)
    month_name = dict(MONTHS).get(month_number)
    if month_name is None:
        raise ValueError(f"Invalid month {args.month}; expected 01..12")

    runner = MonthlyRunner(out_dir=Path(args.out_dir) / args.year, n_jobs_pv=args.n_jobs_pv)
    runner.run_month(MonthSpec(year=args.year, month_number=month_number, month_name=month_name))


if __name__ == "__main__":
    main()