import pvlib
import numpy as np
import pandas as pd
import xarray as xr
import turbine_models
from turbine_models.parser import Turbines
from scipy.stats import norm
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator

def map_turbine_model(start_year: int, installation_type: str) -> str:
    """
    Map a wind farm's start year to an appropriate turbine model
    based on representative technology level for either onshore or offshore.
    Uses models available in the provided turbine library.
    """
    if installation_type == "Onshore" or installation_type == "Unknown" or pd.isna(installation_type):
        if start_year <= 2005:
            return "VestasV47_660kW_47"  # Early commercial model
        elif start_year <= 2010:
            return "DOE_GE_1.5MW_77"  # Represents the highly common 1.5MW class
        elif start_year <= 2015:
            return "2017COE_Market_Average_2.3MW_113"  # Market average proxy
        elif start_year <= 2018:
            return "IEA_Reference_3.4MW_130"  # More modern reference for this period
        elif start_year <= 2021:
            return "2020ATB_NREL_Reference_4MW_150"  # NREL reference for ~4MW class
        else: # for 2022 and later
            return "2023NREL_Bespoke_6MW_170"  # A modern, large onshore turbine
    elif installation_type == "Offshore floating":
        if start_year <= 2018:
            return "IEA_Reference_6MW_100" 
        elif start_year <= 2021:
            return '2023NREL_Bespoke_8.3MW_196' # Mid-generation floating model
        else:
            return 'DTU_Reference_v1_10MW_178'
    else:  # Offshore models
        if start_year <= 2010:
            return "NREL_Reference_5MW_126"  # Foundational reference model
        elif start_year <= 2014:
            return "LEANWIND_Reference_8MW_164"  # Representative of the ~8MW class
        elif start_year <= 2017:
            return "DTU_Reference_v1_10MW_178"  # 10MW models became reference standard
        elif start_year <= 2020:
            return "2019ORCost_NREL_Reference_12MW_222"  # Moving into the 12MW class
        else: # for 2021 and later
            return "IEA_Reference_15MW_240"  # 15MW class as a benchmark for modern turbines

def multi_turb_curve(turb_name, num_turbs, wind_speed_std_dev=1.5):
    turbs = Turbines()
    specs = turbs.specs(turb_name)

    multi_wind_speeds = np.arange(0, 40, 0.25)
    multi_power = np.zeros_like(multi_wind_speeds)
    
    power_curve = turbs.table(turb_name)
    def gaussian_filter(n, sigma):
        return (1 / np.sqrt(2*np.pi*sigma))*np.exp(-(n**2)/(2*sigma))
    
    single_turbine_ws = power_curve['wind_speed_ms']
    single_turbine_power = power_curve['power_kw']

    interp_ws = np.linspace(min(single_turbine_ws), max(single_turbine_ws), 500)
    pchip = PchipInterpolator(single_turbine_ws, single_turbine_power)
    interp_power_pchip = pchip(interp_ws)

    for i, ws_mean in enumerate(multi_wind_speeds):
        # Create a probability distribution for wind speeds around the mean
        probabilities = gaussian_filter(interp_ws - ws_mean, wind_speed_std_dev**2)
        
        # Calculate the expected power by integrating power curve * probability
        # The integral is approximated by a sum over the high-resolution interpolated curve
        bin_width = interp_ws[1] - interp_ws[0]
        expected_power = np.sum(interp_power_pchip * probabilities) * bin_width
        
        multi_power[i] = expected_power

    return {'wind_speed_ms': multi_wind_speeds, 'power_kw': multi_power * num_turbs}

def generate_farm_power_curve(turb_name, n_turbines):
    """
    Generates a smoothed, farm-level power curve from a single-turbine curve.

    This function convolves the single-turbine power curve with a Gaussian
    distribution of wind speeds to represent the spatial variation of wind
    across a large farm. This results in a smoother power curve that is
    more representative of a wind farm's aggregate output.

    Args:
        turbine_power_curve (dict): A dictionary with 'wind_speed_ms' and 'power_kw' keys.
        n_turbines (int): The number of turbines in the farm.
        wind_speed_std_dev (float): The standard deviation of the wind speed
                                    across the farm. This controls the amount
                                    of "smearing" or smoothing. A larger value
                                    results in a smoother curve. Defaults to 1.5 m/s.

    Returns:
        dict: A new power curve dictionary with 'wind_speed_ms' and 'power_kw' for the entire farm.
    """
    turbs = Turbines()
    specs = turbs.specs(turb_name)

    min_ws = 0.0
    max_ws = 40.0
    ws_step = 0.25
    farm_wind_speeds = np.arange(min_ws, max_ws + ws_step, ws_step)
    farm_power = np.zeros_like(farm_wind_speeds)
    power_curve = turbs.table(turb_name)

    single_turbine_ws = power_curve['wind_speed_ms']
    single_turbine_power = power_curve['power_kw']
    #Interpolation
    interp_ws = np.linspace(min(single_turbine_ws), max(single_turbine_ws), 500)
    interp_power = np.interp(interp_ws, single_turbine_ws, single_turbine_power)

    for i, ws_mean in enumerate(farm_wind_speeds):
        std_dev = 0.6 + 0.2 * farm_wind_speeds[i] #Increasing std dev with wind speed

        wind_dist = norm(loc=ws_mean, scale=std_dev)

        bin_width = interp_ws[1] - interp_ws[0]
        probabilities = wind_dist.pdf(interp_ws) * bin_width

        expected_power_single_turbine = np.sum(interp_power * probabilities)
        farm_power[i] = expected_power_single_turbine * n_turbines

    return {'wind_speed_ms': farm_wind_speeds, 'power_kw': farm_power}

def interpolate_idw(xrds, lat, lon, var, y_idx, x_idx, ref_height_idx, neighbors=4):
    """
    Performs Inverse Distance Weighting (IDW) interpolation for a variable 
    at a specific lat/lon using a window around the nearest grid point.
    """
    # Define a search window around the nearest point (3x3 covers the 4 closest in a regular grid)
    y_min = max(0, y_idx - 1)
    y_max = min(len(xrds['y']), y_idx + 2)
    x_min = max(0, x_idx - 1)
    x_max = min(len(xrds['x']), x_idx + 2)

    # Extract coordinates of the window
    sub_lats = xrds['latitude'].isel(y=slice(y_min, y_max), x=slice(x_min, x_max)).values
    sub_lons = xrds['longitude'].isel(y=slice(y_min, y_max), x=slice(x_min, x_max)).values

    # Calculate squared distances
    dists_sq = (sub_lats - lat)**2 + (sub_lons - lon)**2
    flat_dists = dists_sq.flatten()

    # Find k nearest neighbors
    k = min(neighbors, len(flat_dists))
    idx_nearest = np.argsort(flat_dists)[:k]
    
    # Calculate weights
    dists_nearest = flat_dists[idx_nearest]
    weights = 1 / (dists_nearest + 1e-8) # Avoid division by zero
    weights /= weights.sum()

    # Interpolate
    interpolated_series = 0
    window_shape = sub_lats.shape
    
    for i, flat_idx in enumerate(idx_nearest):
        wy, wx = np.unravel_index(flat_idx, window_shape)
        # Map back to global indices
        gy = y_min + wy
        gx = x_min + wx
        if ref_height_idx is not None:
            interpolated_series += xrds[var].isel(heightAboveGround=ref_height_idx, y=gy, x=gx) * weights[i]
        else:
            interpolated_series += xrds[var].isel(y=gy, x=gx) * weights[i]

    return interpolated_series

def estimate_wind_power(country, lat, lon, capacity, startyear, prod_year, status, installation_type, xrds, 
                        y_idx, x_idx, wts_smoothing=False, spatial_interpolation=False, wake_loss_factor=None, single_turb_curve = False, verbose=True): 
    if status not in operating_farms(country, "wind") or (isinstance(startyear, (int, float)) and startyear > prod_year):
        return None
    
    try:
        turbs = Turbines()
        turbine_model = map_turbine_model(startyear, installation_type)
        specs = turbs.specs(turbine_model)
        hub_height = specs['hub_height']
        rated_power_kw = specs['rated_power']

        #Find the index of the height level closest to the hub_height
        available_heights = xrds['heightAboveGround'].values

        ref_height = 100.0
        ref_height_idx = np.abs(available_heights - ref_height).argmin()
        if spatial_interpolation == True:
            wind_ts_ref = interpolate_idw(xrds, lat, lon, 'ws', y_idx, x_idx, ref_height_idx=ref_height_idx, neighbors=4).values
        else:
            wind_ts_ref = xrds['ws'].isel(heightAboveGround=ref_height_idx, y=y_idx, x=x_idx).values

        alpha = 1/7 
        wind_ts_hub_height = wind_ts_ref * (hub_height / ref_height)**alpha

        # 3. Smooth the calculated hub-height wind time series
        wind_ts_series = pd.Series(wind_ts_hub_height)
        if wts_smoothing:
            wind_ts = wind_ts_series.rolling(window=3, center=True, min_periods=1).mean().values
        else:
            wind_ts = wind_ts_series.values
        num_turbines = capacity / (rated_power_kw / 1000)

        farm_power_curve = generate_farm_power_curve(turbine_model, num_turbines)

        #Interpolate the farm's power from the smoothed curve
        total_farm_power_mw = np.interp(
            wind_ts, 
            farm_power_curve['wind_speed_ms'], 
            farm_power_curve['power_kw'] / 1e3
        )

        if single_turb_curve:
            power_curve = turbs.table(turbine_model)
            single_turbine_kw = np.interp(
                wind_ts, 
                power_curve['wind_speed_ms'], 
                power_curve['power_kw']
            )
            total_farm_power_mw = (single_turbine_kw * num_turbines)# Return power in watts

        if wake_loss_factor is not None:
            total_farm_power_mw *= wake_loss_factor

        return total_farm_power_mw
    except Exception as e:
        if verbose:
            print(f"Could not process farm at ({lat}, {lon}). Error: {e}")
        return None

def get_correction_factor(calc_series, act_series):
    """
    Calculates a robust correction factor to scale estimated power to actual power,
    filtering out anomalous periods where the relationship deviates significantly.

    The function calculates the hourly ratio (Actual / Estimated), finds the median
    ratio to represent the "typical" scaling, and then computes the final factor
    using only hours where the ratio is within +/- 20% of this median. This prevents
    outliers (e.g., extreme weather events or data errors) from skewing the calibration.

    Parameters:
    -----------
    calc_series : pd.Series
        Time series of estimated/calculated power generation.
    act_series : pd.Series
        Time series of actual power generation.

    Returns:
    --------
    float
        The calculated correction factor.
    """
    valid_hours = (calc_series > 1) & (act_series > 1)
    ratios = act_series[valid_hours] / calc_series[valid_hours]

    median_ratio = ratios.median()

    factor_mask = (ratios > median_ratio * 0.9) & (ratios < median_ratio * 1.1)
    if factor_mask.sum() > 10:
        clean_calc = calc_series[valid_hours][factor_mask]
        clean_actual = act_series[valid_hours][factor_mask]
        
        factor = clean_actual.sum() / clean_calc.sum()
    else:
        factor = act_series.sum() / calc_series.sum()

    return factor


def estimate_power_final(country, lat, lon, capacity_mw, capacity_rating, status, tech_type, xrds,
                         y_idx, x_idx, Spatial_interpolation, min_irr, twilight_zenith_limit, 
                         smoothing_window_hours=None, performance_ratio=0.85, 
                         start_year=None, prod_year=None, mounting_type='default'):
    """
    Estimates solar farm power production by dispatching to the correct model based on technology type.

    Args:
        lat (float): Latitude of the solar farm.
        lon (float): Longitude of the solar farm.
        capacity_mw (float): Total capacity of the farm in MW.
        capacity_rating (str): The type of capacity rating ('MWac', 'MWp/dc', 'unknown').
        status (str): The operational status of the farm.
        tech_type (str): The technology type ('PV', 'Solar Thermal', etc.).
        ssrd (xarray.DataArray): Time series of surface solar radiation downards.
        ws (xarray.DataArray): Time series of 10m wind speed.
        temp (xarray.DataArray): Time series of 2m temperature.
        al (xarray.DataArray): Time series of albedo.
        derate_factor (float, optional): The overall system derate factor for PV. Defaults to 0.95.
        start_year (int, optional): The year the farm started operation. Used for tracking heuristic.
        mounting_type (str, optional): 'fixed', 'single_axis', or 'default'. 
                                       'default' infers based on capacity and year.

    Returns:
        pandas.Series: Time series of estimated AC power in watts, or None if not operational.
    """
    # if status in ['canceled', 'pre-construction', 'announced', 'construction']: #'shelved - inferred 2 y', 'shelved']:
    #     return None

    if status not in operating_farms(country, "solar") or (isinstance(start_year, (int, float)) and start_year > prod_year):
        return None
    
    if Spatial_interpolation == True:
        ssrd = interpolate_idw(xrds, lat, lon, 'irradiance', y_idx, x_idx, ref_height_idx=None, neighbors=4)
        ws = interpolate_idw(xrds, lat, lon, 'wind_speed', y_idx, x_idx, ref_height_idx=None, neighbors=4)  
        temp = interpolate_idw(xrds, lat, lon, 'temperature', y_idx, x_idx, ref_height_idx=None, neighbors=4)
    else:
        ssrd = xrds['irradiance'].isel(y=y_idx, x=x_idx)
        ws = xrds['wind_speed'].isel(y=y_idx, x=x_idx)
        temp = xrds['temperature'].isel(y=y_idx, x=x_idx)
    
    al = xrds['albedo'].isel(y=y_idx, x=x_idx)

    if tech_type == 'Solar Thermal':
        return estimate_csp_power(
            lat=lat,
            lon=lon,
            capacity_mw=capacity_mw,
            status=status,
            ssrd=ssrd,
            ws=ws,
            temp=temp,
            al=al
        )
    
    # Default to PV calculation for 'PV' and 'Assumed PV'
    try:
        # 1. Determine DC and AC capacity based on rating
        dc_ac_ratio = 1.25  # Assumed DC/AC ratio for MWac and unknown capacities
        if capacity_rating == 'MWac':
            ac_capacity_mw = capacity_mw
            dc_capacity_mw = ac_capacity_mw * dc_ac_ratio
        elif capacity_rating == 'MWp/dc':
            dc_capacity_mw = capacity_mw
            ac_capacity_mw = dc_capacity_mw / dc_ac_ratio # Estimate AC capacity
        else:  # 'unknown' or other
            dc_capacity_mw = capacity_mw * dc_ac_ratio # Assume it's AC or a conservative estimate
            ac_capacity_mw = capacity_mw

        # 2. Prepare weather DataFrame from xarray DataArrays
        time_idx = ssrd['time'].values
        weather_df = pd.DataFrame(index=pd.to_datetime(time_idx).tz_localize('UTC'))
        weather_df['ghi'] = ssrd.values
        weather_df['temp_air'] = temp.values - 273.15
        weather_df['wind_speed'] = ws.values
        weather_df['albedo'] = al.values
        weather_df.fillna(0, inplace=True)
        if min_irr:
            low_mask = (weather_df['ghi'] > 0) & (weather_df['ghi'] < min_irr)
            weather_df.loc[low_mask, 'ghi'] = min_irr

        # 3. Get solar position and decompose GHI
        location = pvlib.location.Location(latitude=lat, longitude=lon, tz='UTC')
        solar_position = location.get_solarposition(times=weather_df.index)
        if twilight_zenith_limit:
            solar_position['zenith'] = solar_position['zenith'].clip(upper=twilight_zenith_limit)
            solar_position['apparent_zenith'] = solar_position['apparent_zenith'].clip(upper=twilight_zenith_limit)
            
        erbs_model = pvlib.irradiance.erbs(weather_df['ghi'], solar_position['zenith'], weather_df.index)
        weather_df['dni'] = erbs_model['dni']
        weather_df['dhi'] = erbs_model['dhi']
        weather_df.fillna(0, inplace=True)

        # 4. Determine Mounting Configuration (Fixed vs Tracking)
        use_tracking = False
        
        if mounting_type == 'single_axis':
            use_tracking = True
        elif mounting_type == 'default':
            # Heuristic: Assume tracking for large, modern farms if not specified
            # This is a rough guess to improve accuracy where metadata is missing
            is_large = capacity_mw > 50
            is_modern = start_year is not None and isinstance(start_year, (int, float)) and start_year >= 2019
            if is_large and is_modern:
                use_tracking = True
            # Also check if tech_type implies tracking
            if 'track' in str(tech_type).lower():
                use_tracking = True

        if use_tracking:
            # Single-axis tracking calculation
            tracker_data = pvlib.tracking.singleaxis(
                apparent_zenith=solar_position['apparent_zenith'],
                apparent_azimuth=solar_position['azimuth'],
                axis_tilt=0,  # Horizontal axis
                axis_azimuth=180,  # N-S axis
                max_angle=60,
                backtrack=True,
                gcr=0.4  # Ground Coverage Ratio
            )
            surface_tilt = tracker_data['surface_tilt']
            surface_azimuth = tracker_data['surface_azimuth']
            # Replace NaNs (night time) with 0 or default to avoid errors in get_total_irradiance
            surface_tilt = surface_tilt.fillna(0)
            surface_azimuth = surface_azimuth.fillna(180)
        else:
            # Fixed tilt calculation
            surface_tilt = abs(lat)-5
            surface_azimuth = 180 if lat > 0 else 0

        dni_extra = pvlib.irradiance.get_extra_radiation(weather_df.index)

        poa_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth'],
            dni=weather_df['dni'],
            ghi=weather_df['ghi'],
            dhi=weather_df['dhi'],
            dni_extra=dni_extra,
            albedo=weather_df['albedo'],
            model='haydavies'  #HAYDAVIES
        )

        # 5. Apply Incidence Angle Modifier (IAM)
        # Calculate Angle of Incidence (AOI)
        aoi = pvlib.irradiance.aoi(
            surface_tilt, 
            surface_azimuth, 
            solar_position['apparent_zenith'], 
            solar_position['azimuth']
        )
        
        # Calculate IAM using ASHRAE model (b=0.05 for standard glass)
        iam = pvlib.iam.ashrae(aoi, b=0.05)
        
        # Effective IAM for diffuse (approximate as IAM at 60 degrees)
        iam_diffuse = pvlib.iam.ashrae(60, b=0.05)
        
        # Calculate Effective Irradiance (irradiance reaching the cell)
        effective_irradiance = (
            poa_irradiance['poa_direct'] * iam + 
            poa_irradiance['poa_diffuse'] * iam_diffuse
        )
        
        # 6. Model cell temperature
        # Choose model based on capacity (proxy for installation type)
        # < 1 MW likely rooftop (close mount), > 1 MW likely ground mount (open rack)
        if capacity_mw < 1.0:
             temp_model_name = 'close_mount_glass_glass'
        else:
             temp_model_name = 'open_rack_glass_glass'

        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][temp_model_name]
        weather_df['cell_temperature'] = pvlib.temperature.sapm_cell(
            poa_irradiance['poa_global'],
            weather_df['temp_air'],
            weather_df['wind_speed'],
            **temperature_model_parameters,
        )

        # 7. Calculate DC power
        dc_capacity_watts = dc_capacity_mw * 1_000_000
        power_dc = pvlib.pvsystem.pvwatts_dc(
            effective_irradiance=effective_irradiance,
            temp_cell=weather_df['cell_temperature'],
            pdc0=dc_capacity_watts,
            gamma_pdc=-0.004,
            temp_ref=25.0
        )

        # 8. Calculate AC power with inverter clipping
        ac_capacity_watts = ac_capacity_mw * 1_000_000
        power_ac = pvlib.inverter.pvwatts(
            pdc=power_dc,
            pdc0=ac_capacity_watts, # In pvwatts inverter model, pdc0 is the AC rating
            eta_inv_nom=0.96
        )

        # Apply derate factor for other losses
        power_ac = power_ac * performance_ratio

        if smoothing_window_hours is not None and smoothing_window_hours > 0:
  
            power_ac = power_ac.rolling(window=smoothing_window_hours, center=True, win_type='gaussian').mean(std=smoothing_window_hours/3)
        
        return power_ac.fillna(0)

    except Exception as e:
        print(f"Could not process PV farm at ({lat}, {lon}). Error: {e}")
        return None


def estimate_csp_power(lat, lon, capacity_mw, status, ssrd, ws, temp, al, dni_ref=900):
    """
    Estimates Concentrated Solar Power (CSP) farm power production using a simplified model.

    This model scales power output based on Direct Normal Irradiance (DNI) relative
    to a reference DNI for rated capacity.

    Args:
        lat (float): Latitude of the solar farm.
        lon (float): Longitude of the solar farm.
        capacity_mw (float): Total AC capacity of the farm in MW.
        status (str): The operational status of the farm.
        ssrd (xarray.DataArray): Time series of surface solar radiation downards (GHI).
        ws (xarray.DataArray): Time series of 10m wind speed (not used in this simple model).
        temp (xarray.DataArray): Time series of 2m temperature (not used in this simple model).
        al (xarray.DataArray): Time series of albedo (not used in this simple model).
        dni_ref (float, optional): Reference DNI at which the plant reaches its rated capacity (W/m^2).
                                 Defaults to 900 W/m^2.

    Returns:
        pandas.Series: Time series of estimated AC power in watts, or None if not operational.
    """
    try:
        # 1. Prepare weather DataFrame
        time_idx = ssrd.time.values
        weather_df = pd.DataFrame(index=pd.to_datetime(time_idx).tz_localize('UTC'))
        weather_df['ghi'] = ssrd.values
        weather_df.fillna(0, inplace=True)

        # 2. Get solar position and DNI
        location = pvlib.location.Location(latitude=lat, longitude=lon, tz='UTC')
        solar_position = location.get_solarposition(times=weather_df.index)
        
        # Use ERBS model to estimate DNI from GHI
        erbs_model = pvlib.irradiance.erbs(weather_df['ghi'], solar_position['zenith'], weather_df.index)
        dni = erbs_model['dni'].fillna(0)

        # 3. Calculate power based on DNI
        # Simple linear scaling of power with DNI up to the reference level
        ac_capacity_watts = capacity_mw * 1_000_000
        
        # Power is proportional to DNI, capped at the plant's AC capacity
        power_ac = (dni / dni_ref) * ac_capacity_watts
        power_ac[power_ac > ac_capacity_watts] = ac_capacity_watts
        power_ac[dni <= 0] = 0 # No power if there's no DNI

        return power_ac.fillna(0)

    except Exception as e:
        print(f"Could not process CSP farm at ({lat}, {lon}). Error: {e}")
        return None
    
def operating_farms(country, power_type):
    """
    Returns a list of statuses that indicate a farm is operational.

    Args:
        country (str): The country code (e.g., 'US', 'CA').

    Returns:
        list: A list of operational statuses.
    """

    if power_type == 'wind':
        operating_dict = {
            "Sweden":["operating", 'construction'],
            "Norway": ["operating"],
            "Finland": ["operating", 'shelved - inferred 2 y'],
            "Denmark": ['operating', 'shelved', 'pre-construction', 'announced', 'shelved - inferred 2 y'],
            "Netherlands": ["operating", "construction"],
            "Germany": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "United Kingdom": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Belgium": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "France": ["operating"],
            "Austria": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Switzerland": ["operating"],
            "Italy": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Spain": ["operating", "construction"],
            "Portugal": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Greece": ["operating"],
            "Ireland": ["operating"],
            "Croatia": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Slovenia": ["operating"],
            "Czech Republic": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Slovakia": ["operating"],
            "Moldova": ["operating"],
            "Romania": ["operating"],
            "Bulgaria": ["operating"],
            "Hungary": ["operating"],
            "Poland": ["operating"],
            "Lithuania": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Latvia": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Estonia": ["operating"],
            "Luxembourg": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Iceland": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Bosnia and Herzegovina": ["operating"],
            "Cyprus": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Montenegro": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "North Macedonia": ["operating"],
            "Kosovo": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
        }
    elif power_type == 'solar':
        operating_dict = {
            "Sweden":["operating", 'construction', "shelved"],
            "Norway": ["operating"],
            "Finland": ["operating", "construction", 'shelved - inferred 2 y', "shelved"],
            "Denmark": ["operating", "construction", "shelved", "shelved - inferred 2 y", "pre-construction"],
            "Netherlands": ["operating", "construction"],
            "Germany": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "United Kingdom": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Belgium": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "France": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Austria": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Switzerland": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Italy": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Spain": ["operating", "construction"],
            "Portugal": ["operating", "construction", "shelved", "shelved - inferred 2 y"],
            "Greece": ["operating", "construction"],
            "Ireland": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Croatia": ["operating"],
            "Slovenia": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Czech Republic": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Slovakia": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Moldova": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Romania": ["operating"],
            "Bulgaria": ["operating", "construction"],
            "Hungary": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Poland": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Lithuania": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Latvia": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Estonia": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Luxembourg": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Iceland": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Bosnia and Herzegovina": ["operating"],
            "Cyprus": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Montenegro": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "North Macedonia": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
            "Kosovo": ["operating", "construction", "pre-construction", "shelved", "shelved - inferred 2 y"],
        }
    else:
        raise ValueError("power_type must be either 'wind' or 'solar'")

    return operating_dict.get(country, [])

def process_data(data_array):
    """
    Processes the ssrd data array to calculate differential values,
    replicating the MATLAB script's logic.
    """
    datadiff = data_array.diff(dim='valid_time')
    

    time_coords = data_array.coords['valid_time']
    
    max_leadtime = 6
    results = []
    for i in range(0, len(time_coords), max_leadtime):
        chunk = data_array.isel(valid_time=slice(i, i + max_leadtime))
        # First value is as-is
        first_step = chunk.isel(valid_time=0)
        # Subsequent values are differences
        diff_steps = chunk.diff(dim='valid_time')
        
        # Combine them
        processed_chunk = xr.concat([first_step.expand_dims('valid_time'), diff_steps], dim='valid_time')
        results.append(processed_chunk)

    final_data_array = xr.concat(results, dim='valid_time')
    
    datadiff_values = final_data_array.values
    
    num_times = data_array.shape[2]
    datadiff_np = np.zeros_like(data_array.values)
    ser_time_diff_np = []
    
    count = 1
    count2 = 0
    for i in range(num_times):
        if count2 >= len(data_array['valid_time']):
            break # Avoid index error
            
        if count == 1:
            datadiff_np[i, :, :] = data_array.isel(valid_time=count2).values
        else:
            # Ensure we don't go out of bounds on the previous step
            if count2 > 0:
                datadiff_np[i, :, :] = data_array.isel(valid_time=count2).values - data_array.isel(valid_time=count2 - 1).values
        
        ser_time_diff_np.append(data_array.coords['valid_time'].values.flatten()[count2])
        
        count += 1
        if count > max_leadtime:
            count = 1
        count2 += 1
        
    return np.array(ser_time_diff_np), datadiff_np

def get_bidding_zone_mapping(zone):
    """
    Returns a list of regions for a given bidding zone.
    Used for filtering wind farms in countries with multiple bidding zones.
    """
    zone_dict = {
        'NO1': [ 'Viken', 'Innland', 'Innlandet', 'Oslo', 'Vestfold', 'Østfold', 'Akershus', 'Buskerud', 'Hedmark', 'Oppland'],
        'NO2': [ 'Agder', 'Adger', 'Rogaland', 'Telemark', 'Vest-Agder', 'Aust-Agder', 'Stavanger'],
        'NO3': [ 'Møre og Romsdal', 'More og Romsdal', 'Trøndelag', 'Trondelag', 'Sør-Trøndelag', 'Nord-Trøndelag', 'Selbu', 'Osen'],
        'NO4': [ 'Nordland', 'Troms og Finnmark', 'Troms', 'Finnmark', 'Lebesby', 'Bo'],
        'NO5': [ 'Vestland', 'Hordaland', 'Sogn og Fjordane'],
        'SE1': [ 'Norrbotten County', 'Norrbottens län', 'Norbotten', 'Piteå', 'Boden', 'Luleå', 'Gällivare', 'Kiruna', 'Jokkmokk', 'Haparanda', 'Kalix', 'Älvsbyn', 'Arjeplog', 'Arvidsjaur', 'Överkalix', 'Övertorneå', 'Pajala'],
        'SE2': [ 'Västerbotten County', 'Västerbottens län', 'Jämtland County', 'Jämtlands län', 'Västernorrland County', 'Västernorrland',
                 'Skellefteå', 'Lycksele', 'Umeå', 'Sundsvall', 'Härnösand', 'Ånge', 'Östersund', 'Åre', 'Strömsund', 'Västernorrlands län', 'Krokom', 'Berg', 'Bräcke', 'Ragunda', 'Sollefteå', 'Kramfors', 'Timrå', 'Åsele', 'Dorotea', 'Vilhelmina', 'Storuman', 'Sorsele', 'Malå', 'Norsjö', 'Robertsfors', 'Vindeln', 'Vännäs', 'Bjurholm', 'Nordmaling'],
        'SE3': ['Gävleborg County', 'Gävleborg', 'Dalarna County', 'Dalecarlia', 'Uppsala County', 'Västmanland County',
                'Värmland County', 'Örebro County', 'Stockholm County', 'Södermanland County', 'Gotland County', 'Gotland',
                'Västra Götaland County', 'Västra Götaland', 'Västra Götalands län', 'Östergötland County', 'Östergötlands län', 'Jönköping County', 'Jönköping',
                'Gävle', 'Sandviken', 'Hudiksvall', 'Falun', 'Borlänge', 'Karlstad', 'Västerås', 'Örebro', 'Uppsala', 'Stockholm', 'Norrtälje', 'Göteborg', 'Mariestad',
                'Södermanlands län', 'Uppsala län', 'Värmlands län', 'Örebro län', 'Västmanlands län', 'Dalarnas län', 'Gävleborgs län', 'Stockholms län', 'Södertälje', 'Nacka', 'Eskilstuna', 'Linköping', 'Norrköping', 'Motala', 'Mjölby', 'Finspång', 'Valdemarsvik', 'Söderköping', 'Åtvidaberg', 'Kinda', 'Ydre', 'Boxholm', 'Ödeshög', 'Vadstena'],
        'SE4': ['Halland County', 'Halland', 'Kronoberg County', 'Kalmar County', 'Kalmar', 'Blekinge County', 'Blekinge', 'Skåne County', 'Skane',
                'Malmö', 'Helsingborg', 'Lund', 'Karlskrona', 'Växjö', 'Halmstad', 'Falkenberg', 'Varberg',
                'Skåne län', 'Blekinge län', 'Kronobergs län', 'Hallands län', 'Kalmar län', 'Kristianstad', 'Landskrona', 'Trelleborg', 'Ängelholm', 'Hässleholm', 'Eslöv', 'Ystad', 'Simrishamn', 'Tomelilla', 'Sjöbo', 'Skurup', 'Svedala', 'Vellinge', 'Lomma', 'Burlöv', 'Staffanstorp', 'Kävlinge', 'Höör', 'Hörby', 'Bromölla', 'Osby', 'Östra Göinge', 'Perstorp', 'Klippan', 'Åstorp', 'Bjuv', 'Svalöv', 'Båstad', 'Laholm', 'Hylte', 'Ljungby', 'Markaryd', 'Älmhult', 'Alvesta', 'Lessebo', 'Tingsryd', 'Uppvidinge', 'Nybro', 'Emmaboda', 'Torsås', 'Mörbylånga', 'Borgholm', 'Oskarshamn', 'Mönsterås', 'Högsby', 'Hultsfred', 'Vimmerby', 'Västervik'],
        'DK1': ['North Denmark Region', 'Central Denmark Region', 'Region of Southern Denmark', 'Kattegat', 'Ringkobing', 'Samsø Municipality', 'Aabenraa', 'Aarhus', 'Assens', 'Herning', 'Hjørring', 'Horsens', 'Ikast-Brande', 'Kolding', 'Lemvig', 'Læsø', 'Mariagerfjord', 'Middelfart', 'Norddjurs', 'Nordfyns', 'Odense', 'Randers', 'Ringkøbing-Skjern', 'Silkeborg', 'Svendborg', 'Sønderborg', 'Varde', 'Vejle', 'Viborg', 'Frederikshavn', 'Brønderslev', 'Skive', 'Holstebro', 'Thisted', 'North Jutland', 'North Denmark', 'Aalborg CSP-Brønderslev CSP with ORC project', 'Dronninglund A solar project', 'Agersted solar farm', 'Arla solar farm'],
        'DK2': ['Region Zealand', 'Capital Region of Denmark', 'Zealand', 'Bornholm', 'Rødby', 'Sjælland', 'Hovedstaden', 'Lolland', 'Falster', 'Møn', 'Ballerup', 'Faxe', 'Guldborgsund', 'Holbæk', 'Hvidovre', 'Ishøj', 'Tårnby', 'Vordingborg', 'Kalundborg'],
    }
    if zone not in zone_dict:
        raise ValueError(f"Bidding zone '{zone}' not found.")
    return zone_dict[zone]

def get_timezone_mapping():
    """
    Returns a dictionary mapping country names/codes to their respective timezones.
    """
    # Default to CET/CEST for most of Europe
    cet_timezone = 'Europe/Copenhagen' # Representative for CET
    eet_timezone = 'Europe/Helsinki'   # Representative for EET
    wet_timezone = 'Europe/Lisbon'     # Representative for WET (Portugal)
    uk_timezone = 'Europe/London'      # UK/Ireland

    mapping = {
        # Western European Time (WET) / UTC
        'Portugal (PT)': wet_timezone,
        'Ireland (IE)': uk_timezone,
        'IE(SEM)': uk_timezone,
        'NIE': uk_timezone,
        'United Kingdom (UK)': uk_timezone,
        
        # Eastern European Time (EET) / UTC+2
        'Bulgaria (BG)': eet_timezone,
        'Cyprus (CY)': eet_timezone,
        'Estonia (EE)': eet_timezone,
        'Finland (FI)': eet_timezone,
        'Greece (GR)': eet_timezone,
        'Lithuania (LT)': eet_timezone,
        'Latvia (LV)': eet_timezone,
        'Moldova (MD)': eet_timezone,
        'Romania (RO)': eet_timezone,
        'Ukraine (UA)': eet_timezone,
        
        # Central European Time (CET) / UTC+1
        # (Explicitly listing some, but could be default)
        'Austria (AT)': cet_timezone,
        'Belgium (BE)': cet_timezone,
        'Switzerland (CH)': cet_timezone,
        'Czech Republic (CZ)': cet_timezone,
        'Germany (DE)': cet_timezone,
        'Denmark (DK)': cet_timezone,
        'Spain (ES)': cet_timezone,
        'France (FR)': cet_timezone,
        'Croatia (HR)': cet_timezone,
        'Hungary (HU)': cet_timezone,
        'Italy (IT)': cet_timezone,
        'Luxembourg (LU)': cet_timezone,
        'Netherlands (NL)': cet_timezone,
        'Norway (NO)': cet_timezone,
        'Poland (PL)': cet_timezone,
        'Sweden (SE)': cet_timezone,
        'Slovenia (SI)': cet_timezone,
        'Slovakia (SK)': cet_timezone,
        
        # Bidding Zones (inherit from country)
        'NO1': cet_timezone, 'NO2': cet_timezone, 'NO3': cet_timezone, 'NO4': cet_timezone, 'NO5': cet_timezone,
        'SE1': cet_timezone, 'SE2': cet_timezone, 'SE3': cet_timezone, 'SE4': cet_timezone,
        'DK1': cet_timezone, 'DK2': cet_timezone,
    }
    return mapping

def get_correction_factor(calc_series, act_series):
    """
    Calculates a robust correction factor to scale estimated power to actual power,
    filtering out anomalous periods where the relationship deviates significantly.

    The function calculates the hourly ratio (Actual / Estimated), finds the median
    ratio to represent the "typical" scaling, and then computes the final factor
    using only hours where the ratio is within +/- 20% of this median. This prevents
    outliers (e.g., extreme weather events or data errors) from skewing the calibration.

    Parameters:
    -----------
    calc_series : pd.Series
        Time series of estimated/calculated power generation.
    act_series : pd.Series
        Time series of actual power generation.

    Returns:
    --------
    float
        The calculated correction factor.
    """
    valid_hours = (calc_series > 1) & (act_series > 1)
    ratios = act_series[valid_hours] / calc_series[valid_hours]

    if len(ratios) == 0:
        return 1.0

    # 2. Find the "typical" ratio (median)
    median_ratio = ratios.median()

    # 3. Create a mask for ratios that are within 20% of the median
    # This keeps the "stable" days and removes the outliers (days 4-7)
    factor_mask = (ratios > median_ratio * 0.8) & (ratios < median_ratio * 1.2)
    
    if factor_mask.sum() > 10:
        # Apply mask to the subset of valid_hours
        clean_calc = calc_series[valid_hours][factor_mask]
        clean_actual = act_series[valid_hours][factor_mask]
        
        # Calculate factor from the clean data
        factor = clean_actual.sum() / clean_calc.sum()
    else:
        # Fallback
        factor = act_series.sum() / calc_series.sum()
        
    return factor
