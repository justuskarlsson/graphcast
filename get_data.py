import os
import sys
import cdsapi
import datetime
from google.cloud import storage
import haiku as hk
import jax
import math
import numpy as np
import xarray
import pytz
import scipy
from typing import Dict
from graphcast.solar_radiation import (
    get_toa_incident_solar_radiation,
)

client = cdsapi.Client()

singlelevelfields = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "geopotential",
    "land_sea_mask",
    "mean_sea_level_pressure",
    "toa_incident_solar_radiation",
    "total_precipitation",
]
pressurelevelfields = [
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "specific_humidity",
    "temperature",
    "vertical_velocity",
]
predictionFields = [
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "specific_humidity",
    "temperature",
    "vertical_velocity",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]
pressure_levels = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]
pi = math.pi
gap = 6
predictions_steps = 4
watts_to_joules = 3600
first_prediction = datetime.datetime(2022, 1, 1, 18, 0)


def trim_time(data):
    # Select every 6th sample along the time axis
    return data.isel(time=slice(5, None, 6))


year = [2019]
month = [10]

day = list(range(30))


# Getting the single and pressure level values.
def get_surface(path):
    if not os.path.exists(path):
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": singlelevelfields,
                "grid": "0.25/0.25",
                "year": year,
                "month": month,
                "day": day,
                "time": [f"{i:02d}:00" for i in range(24)],
                "format": "netcdf",
            },
            path,
        )
    surface = xarray.open_dataset(path)
    surface = surface.rename(
        {var: singlelevelfields[i] for i, var in enumerate(surface.data_vars)}
    )
    surface = surface.rename({"geopotential": "geopotential_at_surface"})

    # Rename axes
    surface = surface.rename(
        {"valid_time": "time", "latitude": "lat", "longitude": "lon"}
    )

    # Calculating the sum of the last 6 hours of rainfall
    surface = surface.sortby("time")
    surface["total_precipitation_6hr"] = (
        surface["total_precipitation"].rolling(time=6).sum()
    )
    surface = surface.drop_vars("total_precipitation")
    surface = trim_time(surface)
    return surface


def get_atmo(path):
    if not os.path.exists(path):
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": pressurelevelfields,
                "grid": "0.25/0.25",
                "year": year,
                "month": month,
                "day": day,
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "pressure_level": pressure_levels,
                "format": "netcdf",
            },
            path,
        )
    atmo = xarray.open_dataset(path)
    atmo = atmo.rename(
        {var: pressurelevelfields[i] for i, var in enumerate(atmo.data_vars)}
    )

    # Rename axes for pressurelevel as well
    atmo = atmo.rename(
        {
            "valid_time": "time",
            "latitude": "lat",
            "longitude": "lon",
            "pressure_level": "level",
        }
    )

    return atmo


if __name__ == "__main__":
    import time

    start_time = time.time()
    path = sys.argv[1]
    get_surface(path + f"surface.nc")
    surface_end_time = time.time()
    print(
        f"Time to get surface data: {surface_end_time - start_time:.2f} seconds"
    )

    atmo_start_time = time.time()
    get_atmo(path + f"atmo.nc")
    atmo_end_time = time.time()
    print(
        f"Time to get atmospheric data: {atmo_end_time - atmo_start_time:.2f} seconds"
    )
