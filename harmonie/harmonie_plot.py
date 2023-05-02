#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:16:49 2021

@author: acmsavazzi
"""
#%% HARMONIE_plot.py

#%%                             Libraries
###############################################################################
import numpy as np
import xarray as xr
import netCDF4
import os
from glob import glob
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
from netCDF4 import Dataset

my_source_dir = os.path.abspath('{}/../../../../My_source_codes')
sys.path.append(my_source_dir)
from My_thermo_fun import *


#%% initial 
dt = 75                 # model  timestep [seconds]
step = 3600             # output timestep [seconds]
domain_name = 'BES'
lat_select = 13.2806    # HALO center 
lon_select = -57.7559   # HALO center 
buffer = 60             # buffer of 150 km around (75 km on each side) the gridpoint 30 * 2 * 2.5 km

srt_time   = np.datetime64('2020-01-03')
end_time   = np.datetime64('2020-02-29T23')

month = '0*'

exps = ['noHGTQS_','noHGTQS_noSHAL_']
exps = ['noHGTQS_noSHAL_',]

levels = 'z'      ## decide wether to open files model level (lev) or 
                    ## already interpolate to height (z)

my_harm_dir     = '/Users/acmsavazzi/Documents/WORK/PhD_Year3/DATA/HARMONIE/'
ifs_dir         = '/Users/acmsavazzi/Documents/WORK/Research/MyData/'

#%%
print("Reading ERA5.") 
era5=xr.open_dataset(ifs_dir+'My_ds_ifs_ERA5.nc')
era5['Date'] = era5.Date - np.timedelta64(4, 'h')
era5.Date.attrs["units"] = "Local Time"

#%% Import Harmonie
### Import Harmonie data
print("Reading HARMONIE.") 
## new files on height levels are empty ### !!!
## is it a problem of the interpolation? if yes: open the file _lev_all.nc 
## and do the intrpolation here. 
harm2d={}
harm3d={}
for exp in exps:
    file2d = my_harm_dir+exp+month+'_avg*'+levels+'*.nc'
    harm2d[exp] = xr.open_mfdataset(file2d, combine='by_coords')
    harm2d[exp] = harm2d[exp].interpolate_na(dim='z')
    # drop duplicate hour between the 2 months 
    harm2d[exp].drop_duplicates('time')
    #remove first 2 days 
    harm2d[exp] = harm2d[exp].sel(time=slice(srt_time,end_time))
    
    file3d = my_harm_dir+exp+month+'_3d*'+levels+'*.nc'
    harm3d[exp] = xr.open_mfdataset(file3d, combine='by_coords')


#%%
var = 'wa'
harm3d[exp][var].sel(z=slice(500,30)).mean('z')\
    .plot.hist(yscale='log',xlim=(-0.3,0.4),bins=500)
plt.axvline(harm3d[exp][var].sel(z=slice(500,30)).mean('z')\
            .quantile(0.5),c='r',lw=0.5)
plt.axvline(harm3d[exp][var].sel(z=slice(500,30)).mean('z')\
            .quantile(0.05),c='r',lw=0.5,ls='--')
plt.axvline(harm3d[exp][var].sel(z=slice(500,30)).mean('z')\
            .quantile(0.95),c='r',lw=0.5,ls='--')
plt.axvline(0,c='k',lw=0.5)


#############################################################################
#%%                     ####### PLOT #######
#############################################################################

var = 'clw'
for exp in exps:
    plt.figure(figsize=(15,7))
    harm2d[exp][var].plot(x='time',vmin=-0.0002)
    plt.ylim([0,5000])
    plt.axvline('2020-02-02',c='k',lw=1)
    plt.axvline('2020-02-10T12',c='k',lw=1)
    plt.title(exp)
    
    
    plt.figure()
    # harm[exp]
#%%

acc_time=3600*1
layer=[200,1000]
sty=['-',':']

var='u'
for exp in ['noHGTQS_noSHAL_',]:
    plt.figure(figsize=(15,7))
    (acc_time*harm2d[exp]['dt'+var+'_dyn'])\
            .groupby(harm2d[exp].time.dt.hour).mean().plot(x='hour')
    plt.title('Dynamical tendency '+exp)
    plt.ylim([0,7000])
    
    plt.figure(figsize=(15,7))
    (acc_time*harm2d[exp]['dt'+var+'_dyn']).plot(x='time')
    plt.title('Dynamical tendency '+exp)
    plt.ylim([0,5000])
    
    
    
    plt.figure()
    acc_time*(harm2d[exp]['dt'+var+'_dyn']+\
              harm2d[exp]['dt'+var+'_phy']).mean('time').plot(y='z')
    (acc_time*harm2d[exp]['dt'+var+'_dyn']).mean('time').plot(y='z')
    (acc_time*harm2d[exp]['dt'+var+'_phy']).mean('time').plot(y='z')
    plt.xlim([-0.01,0.01])





fig, axs = plt.subplots(2,1,figsize=(19,15))
for ide, exp in enumerate(exps):
    h_clim_to_plot = harm2d[exp].sel(z=slice(layer[0],layer[1])).mean('z')
    for idx,var in enumerate(['u','v']):
        ## HARMONIE cy43 clim
        acc_time*(h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy'])\
            .groupby(h_clim_to_plot.time.dt.hour).mean().\
                plot(c='r',ls=sty[ide],label='H.clim cy43: Tot',ax=axs[idx])
        acc_time*h_clim_to_plot.groupby(h_clim_to_plot.time.dt.hour).mean()\
            ['dt'+var+'_dyn'].\
                plot(c='k',ls=sty[ide],label='H.clim cy43: Dyn',ax=axs[idx])
        acc_time*h_clim_to_plot.groupby(h_clim_to_plot.time.dt.hour).mean()\
            ['dt'+var+'_phy'].\
                plot(c='c',ls=sty[ide],label='H.clim cy43: Phy',ax=axs[idx]) 

