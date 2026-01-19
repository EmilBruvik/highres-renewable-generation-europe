# High-Resolution European Renewable Energy Generation Dataset

This repository contains scripts and tools to model, calibrate, and analyze high-resolution hourly power generation for Solar PV, Onshore Wind, and Offshore Wind across Europe. The project combines physical asset databases with reanalysis weather data to produce both country-aggregated and spatially gridded estimates.

### Overview

The main script "weather_energy_monthly.py" estimates power output by mapping individual power plant locations to a high-resolution weather grid. It calculates physical power output based on asset specifications (capacity, installation type, commissioning year) and local weather conditions. The model is calibrated against historical actual generation data from ENTSO-E.

### Data Sources
The modeling framework relies on the following input datasets:
Copernicus European Regional Reanalysis (CERRA) for meteorological variables (Surface solar radiation downwards (ssrd), 10m/100m wind speed, temperature, and albedo) (https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-height-levels?tab=download & https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels?tab=download).
Global Energy Monitor (GEM): Global Solar Power Tracker (https://globalenergymonitor.org/projects/global-solar-power-tracker/download-data/) & Global Wind Power Tracker (https://globalenergymonitor.org/projects/global-wind-power-tracker/download-data/).
ENTSOE-E for actual aggregated generation per production type (https://transparency.entsoe.eu/)

