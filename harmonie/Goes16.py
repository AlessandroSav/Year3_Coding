#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:11:49 2024

@author: acmsavazzi
"""
#%% GOES-16.py

#%%                             Libraries
###############################################################################
import numpy as np
import pandas as pd
import xarray as xr
import os
import netCDF4
from datetime import datetime, timedelta
from netCDF4 import Dataset
import dask
import dask.array as da
dask.config.set({"array.slicing.split_large_chunks": True})
from intake import open_catalog
#%%
max_BrT = 290 # K
min_BrT = 277 #K
my_data_dir = os.path.abspath('../../HARMONIE_paper/data')+'/'
#%%
## Get HARMONIE control 3d exp 
exp = 'HA43h22tg3_clim_noHGTQS'
harm3d_snap = xr.open_mfdataset(my_data_dir+exp+'/'+exp[16:]+'_harm3d_snapshots.nc',\
                                  combine='by_coords')
## Get EUREC4A catalogue
cat = open_catalog("https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml")
#%%
lat_min = harm3d_snap.lat.min()-0.01
lat_max = harm3d_snap.lat.max()+0.01
lon_min = harm3d_snap.lon.min()-0.01
lon_max = harm3d_snap.lon.max()+0.01
data = xr.concat([cat.satellites.GOES16['latlongrid'](date=d).to_dask().chunk({"time": 4,'lat':1000,'lon':1000}) 
                  for d in pd.date_range('2020-01-12','2020-02-14') 
                  if d.strftime('%Y-%m-%d') not in ['2020-01-21', '2020-01-24']], dim='time')\
                .sel(lat=slice(lat_max,lat_min)).sel(lon=slice(lon_min,lon_max))

## Select an image every 30 minutes
data_30min = data.isel(time=(data['time.minute'] == 0)| 
                       (data['time.minute'] == 30)).chunk({"time":4,'lat':-1,'lon':-1})
## SLOW !!
## Remove images where the 25th percentile of brightness temperatures is lower than 285K
data_30min_clean = data_30min.where(data_30min.quantile(0.25,dim=('lat','lon'))['C13']>285,drop=True)

### Interpolate to HARMONIE grid will give 2.5km resolution 
data_interp = data_30min_clean.interp(lat=harm3d_snap.lat.isel(x=40).values)\
                              .interp(lon=harm3d_snap.lon.isel(y=40).values)

## find cloud mask 
data_interp['cl_mask'] = data_interp['C13'].where(
                (data_interp['C13']<max_BrT)&
                (data_interp['C13']>min_BrT),1,0)

#%% SAVE
print('Saving')
data_interp.to_netcdf(my_harm_dir+'Goes16_interp.nc', compute=True)