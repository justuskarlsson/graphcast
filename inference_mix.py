import dataclasses
import os
import cdsapi
import datetime
import functools
from google.cloud import storage
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
)
import haiku as hk
import isodate
import jax
import math
import numpy as np
import xarray as xr
import pytz
import scipy
from typing import Dict
from graphcast.solar_radiation import (
    get_toa_incident_solar_radiation,
)

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
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]
pi = math.pi
gap = 6
predictions_steps = 4
watts_to_joules = 3600
first_prediction = datetime.datetime(2024, 1, 1, 18, 0)


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


with gcs_bucket.blob(
    f"params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
).open("rb") as model:
    ckpt = checkpoint.load(model, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

with gcs_bucket.blob("stats/diffs_stddev_by_level.nc").open("rb") as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()

with gcs_bucket.blob("stats/mean_by_level.nc").open("rb") as f:
    mean_by_level = xr.load_dataset(f).compute()

with gcs_bucket.blob("stats/stddev_by_level.nc").open("rb") as f:
    stddev_by_level = xr.load_dataset(f).compute()


def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig
):

    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)

    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):

    predictor = construct_wrapped_graphcast(model_config, task_config)

    return predictor(
        inputs, targets_template=targets_template, forcings=forcings
    )


def with_configs(fn):

    return functools.partial(
        fn, model_config=model_config, task_config=task_config
    )


def with_params(fn):

    return functools.partial(fn, params=params, state=state)


def drop_state(fn):

    return lambda **kw: fn(**kw)[0]


run_forward_jitted = drop_state(
    with_params(jax.jit(with_configs(run_forward.apply)))
)


class Predictor:

    @classmethod
    def predict(cls, inputs, targets, forcings) -> xr.Dataset:

        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets,
            forcings=forcings,
        )

        return predictions


def nans(*args) -> list:

    return np.full((args), np.nan)


def deltaTime(dt, **delta) -> datetime.datetime:

    return dt + datetime.timedelta(**delta)


# Getting the single and pressure level values.
def getSingleAndPressureValues():
    if not os.path.exists("single-level.nc"):
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": singlelevelfields,
                "grid": "0.25/0.25",
                "year": [2024],
                "month": [1],
                "day": [1],
                "time": [
                    "00:00",
                    "01:00",
                    "02:00",
                    "03:00",
                    "04:00",
                    "05:00",
                    "06:00",
                    "07:00",
                    "08:00",
                    "09:00",
                    "10:00",
                    "11:00",
                    "12:00",
                ],
                "format": "netcdf",
            },
            "single-level.nc",
        )
    singlelevel = xr.open_dataset("single-level.nc")
    singlelevel = singlelevel.rename(
        {
            var: singlelevelfields[i]
            for i, var in enumerate(singlelevel.data_vars)
        }
    )
    singlelevel = singlelevel.rename(
        {"geopotential": "geopotential_at_surface"}
    )

    # Rename axes
    singlelevel = singlelevel.rename(
        {"valid_time": "time", "latitude": "lat", "longitude": "lon"}
    )

    # Calculating the sum of the last 6 hours of rainfall
    singlelevel = singlelevel.sortby("time")
    singlelevel["total_precipitation_6hr"] = (
        singlelevel["total_precipitation"].rolling(time=6).sum()
    )
    singlelevel = singlelevel.drop_vars("total_precipitation")

    if not os.path.exists("pressure-level.nc"):
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": pressurelevelfields,
                "grid": "1.0/1.0",
                "year": [2024],
                "month": [1],
                "day": [1],
                "time": ["06:00", "12:00"],
                "pressure_level": pressure_levels,
                "format": "netcdf",
            },
            "pressure-level.nc",
        )
    pressurelevel = xr.open_dataset("pressure-level.nc")
    pressurelevel = pressurelevel.rename(
        {
            var: pressurelevelfields[i]
            for i, var in enumerate(pressurelevel.data_vars)
        }
    )

    # Rename axes for pressurelevel as well
    pressurelevel = pressurelevel.rename(
        {
            "valid_time": "time",
            "latitude": "lat",
            "longitude": "lon",
            "pressure_level": "level",
        }
    )

    return xr.merge([pressurelevel, singlelevel])


if __name__ == "__main__":

    values: Dict[str, xr.Dataset] = {}

    data = getSingleAndPressureValues()

    eval_inputs, eval_targets, eval_forcings = (
        data_utils.extract_inputs_targets_forcings(
            data,
            target_lead_times=slice("6h", f"{predictions_steps*6}h"),
            **dataclasses.asdict(task_config),
        )
    )

    predictions = Predictor.predict(eval_inputs, eval_targets, eval_forcings)
    predictions.to_netcdf("predictions.nc")
