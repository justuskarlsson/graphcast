import xarray
from graphcast import data_utils
import dataclasses
from google.cloud import storage
import cdsapi
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
)

client = cdsapi.Client()

gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")

with open(
    "data/source-era5_date-2022-01-01_res-0.25_levels-13_steps-01.nc", "rb"
) as f:
    ref_data = xarray.load_dataset(f)

with gcs_bucket.blob(
    f"params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
).open("rb") as model:
    ckpt = checkpoint.load(model, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

with gcs_bucket.blob("stats/diffs_stddev_by_level.nc").open("rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()

with gcs_bucket.blob("stats/mean_by_level.nc").open("rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()

with gcs_bucket.blob("stats/stddev_by_level.nc").open("rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()

levels = ref_data.level
print(levels)


def extract(data):
    eval_inputs, eval_targets, eval_forcings = (
        data_utils.extract_inputs_targets_forcings(
            data,
            target_lead_times=slice("6h", f"{predictions_steps*6}h"),
            **dataclasses.asdict(task_config),
        )
    )
