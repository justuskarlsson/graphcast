import os
import time
import cdsapi
from datetime import datetime, timedelta
from google.cloud import storage

import math
import numpy as np
import xarray

xr = xarray

client = cdsapi.Client()

gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")

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


class AssignCoordinates:

    coordinates = {
        "2m_temperature": ["batch", "lon", "lat", "time"],
        "mean_sea_level_pressure": ["batch", "lon", "lat", "time"],
        "10m_v_component_of_wind": ["batch", "lon", "lat", "time"],
        "10m_u_component_of_wind": ["batch", "lon", "lat", "time"],
        "total_precipitation_6hr": ["batch", "lon", "lat", "time"],
        "temperature": ["batch", "lon", "lat", "level", "time"],
        "geopotential": ["batch", "lon", "lat", "level", "time"],
        "u_component_of_wind": ["batch", "lon", "lat", "level", "time"],
        "v_component_of_wind": ["batch", "lon", "lat", "level", "time"],
        "vertical_velocity": ["batch", "lon", "lat", "level", "time"],
        "specific_humidity": ["batch", "lon", "lat", "level", "time"],
        "toa_incident_solar_radiation": ["batch", "lon", "lat", "time"],
        "year_progress_cos": ["batch", "time"],
        "year_progress_sin": ["batch", "time"],
        "day_progress_cos": ["batch", "lon", "time"],
        "day_progress_sin": ["batch", "lon", "time"],
        "geopotential_at_surface": ["lon", "lat"],
        "land_sea_mask": ["lon", "lat"],
    }


year = [2022]
month = [1]
day = list(range(1, 4))


# Getting the single and pressure level values.
def download_surface(path):
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


surface_fmt = "{}/{}_{:02d}_surface.nc"


def get_surface(root_path, start: datetime, end: datetime):
    def load_dt(dt):
        surface = xr.open_dataset(
            surface_fmt.format(root_path, dt.year, dt.month)
        )
        surface = surface.sel(valid_time=slice(start, end)).compute()
        print("surface sel:", dt, surface)
        return surface

    start = start - timedelta(hours=6)
    if start.month == end.month:
        surface = load_dt(start)
    else:
        surface = xr.merge(
            [load_dt(start), load_dt(end)],
        )
    print(f"{surface=}")
    surface = surface.compute()
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


def download_atmo(path):
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


atmo_fmt = "{}/{}_{:02d}_atmo.nc"


def get_atmo(root_path, start: datetime, end: datetime):
    def load_dt(dt):
        atmo = xr.open_dataset(atmo_fmt.format(root_path, dt.year, dt.month))
        atmo = atmo.sel(valid_time=slice(start, end)).compute()
        return atmo

    if start.month == end.month:
        atmo = load_dt(start)
    else:
        atmo = xr.merge(
            [load_dt(start), load_dt(end)],
        )
    atmo = atmo.compute()
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
    atmo = atmo.isel(time=slice(1, None))
    return atmo


def drop_extra_coords(data):
    return data.drop_vars("number").drop_vars("expver")


def fix_time(data):
    # Add "datetime" as an extra coordinate with the same values as "time"
    data = data.expand_dims(dim="batch")

    dt_value = np.expand_dims(data.time.values, axis=0)

    # Calculate timedelta since the first datetime
    first_datetime = data.time.values[0]
    data["time"] = (data.time - first_datetime).astype("timedelta64[ns]")
    data = data.assign_coords(
        datetime=xarray.Variable(["batch", "time"], dt_value)
    )

    return data


def trim_time(data):
    # Select every 6th sample along the time axis
    return data.isel(time=slice(6, None, 6))


def modify_coordinates(data: xarray.Dataset):

    for var in list(data.data_vars):
        varArray: xarray.DataArray = data[var]
        nonIndices = list(
            set(list(varArray.coords)).difference(
                set(AssignCoordinates.coordinates[var])
            )
        )
        if "datetime" in nonIndices:
            nonIndices.remove("datetime")
        data[var] = varArray.isel(**{coord: 0 for coord in nonIndices})
    # Add batch dimension of length 1
    return data


def fix_axis(data):
    return data.reindex(lat=data["lat"][::-1], level=data["level"][::-1])


def pipeline(data):
    data = drop_extra_coords(data)
    data = fix_time(data)
    data = modify_coordinates(data)
    data = fix_axis(data)
    return data


def load_data(root_path: str, start: datetime, end: datetime):
    start_time = time.time()
    surface = get_surface(root_path, start, end)
    surface_end_time = time.time()
    print(
        f"Time to get surface data: {surface_end_time - start_time:.2f} seconds"
    )

    atmo_start_time = time.time()
    atmo = get_atmo(root_path, start, end)
    atmo_end_time = time.time()
    print(
        f"Time to get atmospheric data: {atmo_end_time - atmo_start_time:.2f} seconds"
    )

    org_data = xarray.merge([surface, atmo])
    # with open("data/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc", "rb") as f:
    #     ref_data = xarray.load_dataset(f)
    # ref_data = ref_data.isel(time=slice(1, None))
    return pipeline(org_data)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--year", type=int, default=2019, help="Year(s) to process"
    )
    parser.add_argument(
        "--month",
        type=int,
        default=10,
        help="Month(s) to process",
    )
    parser.add_argument(
        "--days",
        type=int,
        nargs="+",
        default=list(range(1, 32)),
        help="Day(s) to process",
    )
    args = parser.parse_args()
    year = [args.year]
    month = [args.month]
    day = args.days
    start_time = time.time()
    download_surface(surface_fmt.format("data", args.year, args.month))
    surface_end_time = time.time()
    print(
        f"Time to get surface data: {surface_end_time - start_time:.2f} seconds"
    )

    atmo_start_time = time.time()
    download_atmo(atmo_fmt.format("data", args.year, args.month))
    atmo_end_time = time.time()
    print(
        f"Time to get atmospheric data: {atmo_end_time - atmo_start_time:.2f} seconds"
    )
