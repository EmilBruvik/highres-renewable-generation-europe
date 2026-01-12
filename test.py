import pandas as pd
import xarray as xr

YEAR = '2024'
MONTH = 'sep'
MONTH_NUM = 9
month_number = '09'

print(f"Loading dataset for {YEAR}-{MONTH}...", flush=True)

fn = f"/Data/gfi/vindenergi/nab015/CERRA_multi_level/{YEAR}/cerra_{YEAR}_multi_level_{MONTH}.nc"
ds = xr.open_dataset(fn, engine='netcdf4')
actual_generation_file = pd.read_csv(f'/Data/gfi/vindenergi/nab015/Actual_Generation/AggregatedGenerationPerType/{YEAR}/{YEAR}_{month_number}_AggregatedGenerationPerType_16.1.B_C_r3.csv', sep='\t')

print("Dataset loaded successfully.", flush=True)