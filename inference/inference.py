import dataclasses
from graphcast import (
    data_utils,
)
from data import load_data
from model import task_config, Predictor

predictions_steps = 4

data = load_data()
eval_inputs, eval_targets, eval_forcings = (
    data_utils.extract_inputs_targets_forcings(
        data,
        target_lead_times=slice("6h", f"{predictions_steps*6}h"),
        **dataclasses.asdict(task_config),
    )
)
predictions = Predictor.predict(eval_inputs, eval_targets, eval_forcings)
predictions.to_netcdf("predictions.nc")
