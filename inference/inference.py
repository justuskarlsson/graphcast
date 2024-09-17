import dataclasses
import sys
from graphcast import (
    data_utils,
)
from data import load_data

predictions_steps = 9
nc_path = sys.argv[1] if len(sys.argv) > 1 else "data/22_3d_"

data = load_data(nc_path)
data = data.isel(time=slice(0, 3))
print(data.time.values)
# data.time.values -= np.timedelta64(12, "h")
from model import task_config, run_forward_jitted, jax, np
from graphcast import rollout

eval_inputs, eval_targets, eval_forcings = (
    data_utils.extract_inputs_targets_forcings(
        data,
        target_lead_times=slice("6h", f"{predictions_steps*6}h"),
        **dataclasses.asdict(task_config),
    )
)
print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)
"""
Inputs:   {'batch': 1, 'time': 2, 'lat': 721, 'lon': 1440, 'level': 37}
Targets:  {'batch': 1, 'time': 1, 'lat': 721, 'lon': 1440, 'level': 37}
Forcings: {'batch': 1, 'time': 1, 'lat': 721, 'lon': 1440}
"""

predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings,
)
print(predictions)
predictions.to_netcdf("predictions.nc")
