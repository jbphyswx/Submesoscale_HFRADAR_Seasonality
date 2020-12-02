# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import xarray as xr
from xarray import open_mfdataset

import netCDF4 as netcdf
import numpy as np
import datetime

import sys
import os


# see https://www.unidata.ucar.edu/software/tds/current/reference/Services.html
#data_source = "https://hfrnet-tds.ucsd.edu/thredds/ncss/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd" # doesn't work, NetCDF Subset Service, see see https://hfrnet-tds.ucsd.edu/thredds/ncss/grid/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd/dataset.html
data_source =  "https://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd" # works,  OPeNDAP DAP2 (ncss->dodsC)
#data_source = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USEGC/6km/hourly/RTV/HFRADAR_US_East_and_Gulf_Coast_6km_Resolution_Hourly_RTV_best.ncd" # 6km allegedly

xr_data = xr.open_dataset(data_source, decode_times=True,chunks={'time':1000}) # decoding times seems to work, dask chunks for speed


#%%

filename_prefix = 'HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best'
script_path = os.path.abspath(os.path.dirname(__file__))
save_path = "../Data/HF_Radar/2km/temp_holding/"
save_path = os.path.normpath(os.path.join(script_path, save_path)) + '/'
dates = np.datetime_as_string(xr_data.time.values,unit='D')
for date in np.unique(dates):
    date = str(date) # cast numpy_str to str
    print(date)
    xrd = xr_data.sel(time=slice(date,date))
    filename = save_path + filename_prefix + '_' + date.replace('-','_') + '.nc'
    print(filename)
    if os.path.exists(filename)
        if overwrite:
            verbose_print('Writing data for date ' + date ' + to file ' + filename)
            xrd.to_netcdf(path=filename)
        else:
            verbose_print('Data for date ' + date ' already exists in file ' + filename)
    else:
        verbose_print('Writing data for date ' + date ' + to file ' + filename)
        xrd.to_netcdf(path=filename)
        

