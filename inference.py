# @title Imports

import dataclasses
import datetime
import functools
import math
import re
from typing import Optional

import cartopy.crs as ccrs
from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray


def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))


gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")


# @title Load the model

params_file_options = [
    name
    for blob in gcs_bucket.list_blobs(prefix="params/")
    if (name := blob.name.removeprefix("params/"))
]  # Drop empty string.
params_file = params_file_options[0]
with gcs_bucket.blob(f"params/{params_file.value}").open("rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config
print("Model description:\n", ckpt.description, "\n")
print("Model license:\n", ckpt.license, "\n")

print(model_config)
