import dataclasses
import sys
from graphcast import (
    data_utils,
)
from data import load_data

predictions_steps = 4
nc_path = sys.argv[1] if len(sys.argv) > 1 else "data/22_3d_"

data = load_data(nc_path)
print(data.time.values)
# data.time.values -= np.timedelta64(12, "h")
from model import task_config, Predictor

eval_inputs, eval_targets, eval_forcings = (
    data_utils.extract_inputs_targets_forcings(
        data,
        target_lead_times=slice("6h", f"{predictions_steps*6}h"),
        **dataclasses.asdict(task_config),
    )
)
print(f"{eval_inputs.time=}")
print(f"{eval_targets.time=}")
predictions = Predictor.predict(eval_inputs, eval_targets, eval_forcings)
print(predictions)
predictions.to_netcdf("predictions.nc")
