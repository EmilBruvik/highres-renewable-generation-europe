#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
# Disable HDF5 file locking - MUST BE DONE BEFORE IMPORTING XARRAY/NETCDF4
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import netCDF4 as nc
import xarray as xr
from pathlib import Path

from scipy.stats import pearsonr
import importlib
import functions as functions
from collections import Counter
from tqdm import tqdm
importlib.reload(functions)
from functions import estimate_wind_power

print("Starting wind production script...", flush=True)

# In[ ]:


YEAR = '2024'
MONTH = 'sep'
MONTH_NUM = 9
month_number = '09'

print(f"Loading dataset for {YEAR}-{MONTH}...", flush=True)

fn = f"/Data/gfi/vindenergi/nab015/CERRA_multi_level/{YEAR}/cerra_{YEAR}_multi_level_{MONTH}.nc"
ds = xr.open_dataset(fn, engine='netcdf4')
actual_generation_file = pd.read_csv(f'/Data/gfi/vindenergi/nab015/Actual_Generation/AggregatedGenerationPerType/{YEAR}/{YEAR}_{month_number}_AggregatedGenerationPerType_16.1.B_C_r3.csv', sep='\t')


# In[ ]:

print("Loading dataset into memory...", flush=True)

ds.load() 



# In[ ]:


df = pd.read_csv('/Data/gfi/vindenergi/nab015/Wind_data/Global-Wind-Power-Tracker-February-2025.csv', sep=';', decimal=',')
countries_tracker = ['Austria', 'Bosnia and Herzegovina', 'Belgium', 'Bulgaria',
 'Switzerland', 'Cyprus', 'Czech Republic', 'Germany',
 'Denmark', 'Denmark', 'Estonia',  'Spain',
 'Finland', 'France', 'United Kingdom', 'Georgia', 'Greece', 'Croatia', 'Hungary',
 'Ireland', 'Italy',
 'Lithuania', 'Luxembourg', 'Latvia', 'Moldova',
 'Montenegro', 'North Macedonia', 'Netherlands',
 'Norway', 'Norway', 'Norway', 'Norway', 'Norway', 'Poland', 'Portugal',
 'Romania', 'Serbia', 'Sweden', 'Sweden', 'Sweden', 'Sweden',
 'Slovenia', 'Slovakia', 'Kosovo']

countries = ['Austria (AT)', 'Bosnia and Herz. (BA)', 'Belgium (BE)', 'Bulgaria (BG)',
 'Switzerland (CH)', 'Cyprus (CY)', 'Czech Republic (CZ)', 'Germany (DE)',
 'DK1', 'DK2', 'Estonia (EE)', 'Spain (ES)',
 'Finland (FI)', 'France (FR)', 'United Kingdom (UK)', 'Georgia (GE)', 'Greece (GR)', 'Croatia (HR)', 'Hungary (HU)',
 'Ireland (IE)', 'Italy (IT)',
 'Lithuania (LT)', 'Luxembourg (LU)', 'Latvia (LV)', 'Moldova (MD)',
 'Montenegro (ME)', 'North Macedonia (MK)', 'Netherlands (NL)',
 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'Poland (PL)', 'Portugal (PT)',
 'Romania (RO)', 'Serbia (RS)', 'SE1', 'SE2', 'SE3', 'SE4',
 'Slovenia (SI)', 'Slovakia (SK)', 'Kosovo (XK)']

zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK1', 'DK2', 'SE1', 'SE2', 'SE3', 'SE4']

# In[ ]:


timezone_mapping = functions.get_timezone_mapping()
global_wind_power_grid = np.zeros((ds.dims['valid_time'], ds.dims['y'], ds.dims['x']))

zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK1', 'DK2', 'SE1', 'SE2', 'SE3', 'SE4']
location_cols = ['City', 'State/Province', 'Local area (taluk, county)', 'Subregion', 'Region']
actual_generation_file = pd.read_csv(f'/Data/gfi/vindenergi/nab015/Actual_Generation/AggregatedGenerationPerType/{YEAR}/{YEAR}_{month_number}_AggregatedGenerationPerType_16.1.B_C_r3.csv', sep='\t')

for COUNTRY, COUNTRY_CODE in zip(countries_tracker, countries):
    print(f"Processing country: {COUNTRY} ({COUNTRY_CODE})...", flush=True)

    country_wind_power_grid = np.zeros((ds.dims['valid_time'], ds.dims['y'], ds.dims['x']))

    timezone = timezone_mapping.get(COUNTRY, 'Europe/Copenhagen') 

    df_country = df[df['Country/Area'] == COUNTRY].copy()
    if COUNTRY_CODE in zones:
        zone_cities = functions.get_bidding_zone_mapping(COUNTRY_CODE)
        mask = pd.Series(False, index=df_country.index)
        for col in location_cols:
            if col in df_country.columns:
                mask |= df_country[col].isin(zone_cities)
        df_country = df_country[mask]
    else:          
        df_country = df_country[(df_country['Country/Area'] == COUNTRY)].copy()
    df['Capacity (MW)'].mean()
    ref_startyear = int(df['Start year'].median())
    Counter(df_country['Installation Type']).most_common()
    actual_generation_country = actual_generation_file[actual_generation_file['AreaDisplayName'] == COUNTRY_CODE].copy()
    
    if 'DateTime(UTC)' in actual_generation_country.columns:
        time_stamps = pd.to_datetime(actual_generation_country['DateTime(UTC)'])
        time_stamps = time_stamps.dt.tz_localize('UTC').dt.tz_convert(timezone)
        actual_generation_country.index = time_stamps
    elif 'MTU' in actual_generation_country.columns:
        naive_datetime = pd.to_datetime(
            actual_generation_country['MTU'].str.split(' - ').str[0], 
            format='%d.%m.%Y %H:%M'
        )
        actual_generation_country.index = naive_datetime.dt.tz_localize(timezone, nonexistent='shift_forward', ambiguous='NaT')
        actual_generation_country = actual_generation_country[actual_generation_country.index.notna()]        
    wind_data_onshore = actual_generation_country[actual_generation_country['ProductionType'] == 'Wind Onshore']
    wind_data_offshore = actual_generation_country[actual_generation_country['ProductionType'] == 'Wind Offshore']
    wind_data = pd.concat([wind_data_onshore, wind_data_offshore])
    wind_actual = pd.to_numeric(wind_data['ActualGenerationOutput[MW]'], errors='coerce').fillna(0)
    wind_actual = wind_actual.groupby(wind_actual.index).sum().resample('h').mean()

    for index, farm in tqdm(df_country.iterrows(), total=df_country.shape[0], desc="Calculating Farm Energy"):
        lat = float(farm['Latitude'])
        lon = float(farm['Longitude'])
        
        # Handle longitude wrapping (0-360 vs -180 to 180)
        diff_lon = np.abs(ds.longitude - lon)
        diff_lon = np.minimum(diff_lon, 360 - diff_lon)
        
        distance_sq = (ds.latitude - lat)**2 + diff_lon**2
        min_dist_idx_flat = np.argmin(distance_sq.values.flatten())
        y_idx, x_idx = np.unravel_index(min_dist_idx_flat, ds.latitude.shape)
        startyear = farm['Start year']
        if pd.isna(startyear):
            startyear = ref_startyear

        power_timeseries = estimate_wind_power(
                country=COUNTRY,
                lat=float(farm['Latitude']),
                lon=float(farm['Longitude']),
                capacity=farm['Capacity (MW)'],
                startyear=startyear,
                prod_year=int(YEAR),
                status=farm['Status'],
                installation_type=farm['Installation Type'],
                xrds=ds,
                y_idx=y_idx,
                x_idx=x_idx,
                wts_smoothing=True,
                wake_loss_factor=0.95,
                spatial_interpolation=True,
                verbose=False,
                single_turb_curve=False
            )

        if power_timeseries is not None:
            country_wind_power_grid[:, y_idx, x_idx] += power_timeseries

    total_wind_power = country_wind_power_grid.sum(axis=(1, 2))

    if total_wind_power.sum() > 0:
        # Create Series for calculated power to ensure alignment
        calc_series_raw = pd.Series(total_wind_power, index=pd.to_datetime(ds['valid_time'].values).tz_localize('UTC'))
        
        # Align timezones if needed
        if wind_actual.index.tz is not None:
             calc_series_raw.index = calc_series_raw.index.tz_convert(wind_actual.index.tz)
             
        common_idx = wind_actual.index.intersection(calc_series_raw.index)
        
        if len(common_idx) > 0:
            # Pass arguments as (calculated, actual)
            factor = functions.get_correction_factor(calc_series_raw.loc[common_idx], wind_actual.loc[common_idx])
        else:
            factor = 1.0
    else:
        factor = 1.0
    print(f'Calibration factor for {COUNTRY}: {factor:.4f}')
    
    # if total_wind_power.sum() > 0:
    #     factor = wind_actual.sum() / total_wind_power.sum()
    # else:
    #     factor = 1.0
    # print(f'Calibration factor for {COUNTRY}: {factor:.4f}')
        
    
    global_wind_power_grid += country_wind_power_grid * factor
    
    calc_series = pd.Series(total_wind_power * factor, index=ds['valid_time'].values)

    if calc_series.index.tz is None:
        calc_series.index = calc_series.index.tz_localize('UTC')

    if  wind_actual.index.tz is not None:
        calc_series.index = calc_series.index.tz_convert(wind_actual.index.tz)

    common_index = calc_series.index.intersection(wind_actual.index)
    calc_aligned = calc_series.loc[common_index]
    actual_aligned = wind_actual.loc[common_index]
    mask = ~np.isnan(calc_aligned.values) & ~np.isnan(actual_aligned.values)
    calc_aligned = calc_aligned.iloc[mask]
    actual_aligned = actual_aligned.iloc[mask]

    if len(calc_aligned) > 1:
        corr, _ = pearsonr(actual_aligned, calc_aligned)
        r_squared = corr**2
    else:
        r_squared = np.nan

    print(f'Wind power calculated for {COUNTRY}.')
    
    fig_dir = Path(f'/Data/gfi/vindenergi/nab015/figures/wind_power_comparison/{YEAR}/{month_number}')
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    time_hours = pd.to_datetime(ds['valid_time'].values)
    ax.plot(calc_series.index, calc_series.values, 'x-', label='Estimated Wind Power (MW)', color='red')
    ax.plot(wind_actual.index, wind_actual.values, label='Actual Wind Power (MW)', alpha=0.7, linestyle='--', color='black')
    ax.text(0.15, 0.95, fr'$\alpha = $ {factor:.4f}', transform=ax.transAxes, fontsize=16, verticalalignment='top')
    if pd.isna(r_squared) == False:
        ax.text(0.15, 0.90, fr'$R^2 = $ {r_squared:.4f}', transform=ax.transAxes, fontsize=16, verticalalignment='top')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power (MW)')
    if COUNTRY_CODE in zones:
        ax.set_title(f'Wind Power Generation for {COUNTRY_CODE} - {month_number}-{YEAR}')
    else:
        ax.set_title(f'Wind Power Generation for {COUNTRY} - {month_number}-{YEAR}')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.grid(True)
    ax.legend(loc='upper right')
    if COUNTRY_CODE in zones:
        plt.savefig(fig_dir / f'wind_power_comparison_{month_number}_{YEAR}_{COUNTRY_CODE}.svg', bbox_inches='tight', dpi=300) 
    else:
        plt.savefig(fig_dir / f'wind_power_comparison_{month_number}_{YEAR}_{COUNTRY}.svg', bbox_inches='tight', dpi=300)
    print(f'Figure saved to {fig_dir / f"wind_power_comparison_{month_number}_{YEAR}_{COUNTRY}.svg"}, dpi=300')

# Save global dataset
wind_dataset = xr.Dataset(
    {
        'wind_power_mw': (('time', 'y', 'x'), global_wind_power_grid)
    },
    coords={
        'time': (('time',), ds['valid_time'].values),
        'y': (('y',), ds['y'].values),
        'x': (('x',), ds['x'].values),
        'latitude': (('y', 'x'), ds['latitude'].values),
        'longitude': (('y', 'x'), ds['longitude'].values)
    }
)

outputfile_dir = Path(f'/Data/gfi/vindenergi/nab015/wind_production/{YEAR}')
outputfile_dir.mkdir(parents=True, exist_ok=True)
output_file = outputfile_dir / f'{month_number}_{YEAR}_wind_production_aggregated.nc'

if output_file.exists():
    try:
        output_file.unlink()
    except PermissionError:
        print(f"Warning: Could not delete existing file {output_file}. It might be in use.")
        # Try writing to a temporary file instead
        output_file = outputfile_dir / f'{month_number}_{YEAR}_wind_production_aggregated_new.nc'
        print(f"Attempting to save to new filename: {output_file}")

try:
    wind_dataset.to_netcdf(output_file, engine='netcdf4')
    print(f'Aggregated Wind power dataset saved to {output_file}')
except PermissionError:
    print(f"Error: Permission denied when writing to {output_file}. Please check if the file is open in another program.")
except Exception as e:
    print(f"Error saving dataset: {e}")

# In[ ]:


ds.close()

