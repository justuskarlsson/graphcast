import cdsapi
import functools
from google.cloud import storage
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    graphcast,
    normalization,
    rollout,
)
import haiku as hk
import jax
import math
import xarray as xr
import numpy as np

client = cdsapi.Client()

gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")


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

pi = math.pi


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
            targets_template=targets * np.nan,
            forcings=forcings,
        )

        return predictions
