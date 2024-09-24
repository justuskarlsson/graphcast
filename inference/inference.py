import dataclasses
import sys
import os
import time
from graphcast import (
    data_utils,
)
from data import load_data
from datetime import datetime, timedelta
import xarray as xr

days_ahead = 5
predictions_steps = days_ahead * 4
month = int(sys.argv[1])
day = int(sys.argv[2])
BATCH_SIZE = int(sys.argv[3])

root_path = "data"
dst_path = os.path.join("data", "inference")
os.makedirs(dst_path, exist_ok=True)

selected_pressure_levels = [1000, 925, 850, 700, 500, 300, 200]


def inference(data, dt: datetime):
    begin_time = time.time()
    from model import task_config, run_forward_jitted, jax, np
    from graphcast import rollout

    dst_file_fmt = os.path.join(dst_path, datetime.strftime(dt, "%Y_%m_%d"))
    dst_file_fmt = dst_file_fmt + "_{}.nc"
    start_time = time.time()
    eval_inputs, eval_targets, eval_forcings = (
        data_utils.extract_inputs_targets_forcings(
            data,
            target_lead_times=slice("6h", f"{predictions_steps*6}h"),
            **dataclasses.asdict(task_config),
        )
    )
    print("Time to extract inputs:", time.time() - start_time)
    print("Inputs:  ", eval_inputs.dims.mapping)
    print("Targets: ", eval_targets.dims.mapping)
    print("Forcings:", eval_forcings.dims.mapping)
    """
    Inputs:   {'batch': 1, 'time': 2, 'lat': 721, 'lon': 1440, 'level': 37}
    Targets:  {'batch': 1, 'time': 1, 'lat': 721, 'lon': 1440, 'level': 37}
    Forcings: {'batch': 1, 'time': 1, 'lat': 721, 'lon': 1440}
    """
    start_time = time.time()
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )
    print("Time to predict:", time.time() - start_time)
    print(predictions)
    start_time = time.time()
    predictions = predictions.sel(level=selected_pressure_levels)
    eval_targets = eval_targets.sel(level=selected_pressure_levels)
    predictions.to_netcdf(dst_file_fmt.format("pred"))
    eval_targets.to_netcdf(dst_file_fmt.format("gt"))
    print("Time to save:", time.time() - start_time)
    print("Total time:", time.time() - begin_time)


final_start = datetime(2019, month, day)
start = final_start
end = start + timedelta(days=BATCH_SIZE)
end += timedelta(days=days_ahead)
print("Start:", start, "End:", end)
# Pad with lead time
start -= timedelta(days=1)
start = start.replace(hour=18)
# Pad with forecast length
data = load_data(root_path, start, end)
batch = []
for i in range(BATCH_SIZE):
    val = data.isel(time=slice(i * 4, i * 4 + days_ahead * 4 + 2))
    val["time"] = val.time - val.time.values[0]
    print("len vs expected:", len(val.time), days_ahead * 4 + 2)
    if len(val.time) < days_ahead * 4 + 2:
        print("Too short")
        break

    inference(val, start + timedelta(days=i + 1))
