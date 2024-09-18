import xarray as xr
from datetime import datetime

abs_path = "/proj/cvl/users/x_juska/repos/graphcast/data/"
atmo = xr.open_dataset(abs_path + "22_3d_atmo.nc")
surface = xr.open_dataset(abs_path + "22_3d_atmo.nc")

atmo1 = atmo.sel(
    valid_time=slice(datetime(2022, 1, 1), datetime(2022, 1, 1, 6))
).compute()
atmo2 = atmo.sel(
    valid_time=slice(datetime(2022, 1, 2), datetime(2022, 1, 2, 18))
).compute()
print(atmo1.valid_time.values)
print(atmo2.valid_time.values)
atmo = xr.concat([atmo1, atmo2], dim="valid_time")
print(atmo.valid_time.values)
