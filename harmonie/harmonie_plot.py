#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:16:49 2021

@author: acmsavazzi
"""

#%% HARMONIE_read_save.py



#%%                             Libraries
###############################################################################
import numpy as np
import xarray as xr
import netCDF4
import os
from glob import glob
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

srt_time   = np.datetime64('2020-02-02')
end_time   = np.datetime64('2020-02-12')
harmonie_time_to_keep = '202002010000-'


my_harm_dir     = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/cy43_clim/average_300km/'
ifs_dir         = '/Users/acmsavazzi/Documents/WORK/Research/MyData/'

#%%
print("Reading ERA5.") 
era5=xr.open_dataset(ifs_dir+'My_ds_ifs_ERA5.nc')
era5['Date'] = era5.Date - np.timedelta64(4, 'h')
era5.Date.attrs["units"] = "Local Time"

#%% Import Harmonie
### Import Harmonie data
# This is too slow... need to find a better way. Maybe in a separate file open
# and save only the points and time neede for comparison.
### HARMONIE clim spatially averaged
print("Reading HARMONIE clim spatial average.") 
file = my_harm_dir+'LES_forcing_2020020200.nc'
harm_clim_avg = xr.open_mfdataset(file)
harm_clim_avg['time'] = harm_clim_avg.time - np.timedelta64(4, 'h')
harm_clim_avg.time.attrs["units"] = "Local Time"
# harm_clim_avg = harm_clim_avg.mean(dim=['x', 'y'])

harm_clim_avg = harm_clim_avg.sel(time=~harm_clim_avg.get_index("time").duplicated())
harm_clim_avg = harm_clim_avg.interpolate_na('time')

#
z_ref = harm_clim_avg.z.mean('time')
zz    = harm_clim_avg.z
for var in list(harm_clim_avg.keys()):
    if 'level' in harm_clim_avg[var].dims:
        print("interpolating variable "+var)
        x = np.empty((len(harm_clim_avg['time']),len(harm_clim_avg['level'])))
        x[:] = np.NaN
        for a in range(len(harm_clim_avg.time)):
            x[a,:] = np.interp(z_ref,zz[a,:],harm_clim_avg[var].isel(time = a))            
        harm_clim_avg[var] = (("time","level"), x)    
# convert model levels to height levels
harm_clim_avg = harm_clim_avg.rename({'z':'geo_height'})
harm_clim_avg = harm_clim_avg.rename({'level':'z','clw':'ql','cli':'qi'})
harm_clim_avg["z"] = (z_ref-z_ref.min()).values
harm_clim_avg['z'] = harm_clim_avg.z.assign_attrs(units='m',long_name='Height')


harm_clim_avg = harm_clim_avg.rename({'dtq_dyn':'dtqt_dyn','dtq_phy':'dtqt_phy'})
harm_clim_avg['rho'] = calc_rho(harm_clim_avg['p'],harm_clim_avg['T'],harm_clim_avg['qt'])
harm_clim_avg['wspd']= np.sqrt(harm_clim_avg['u']**2 + harm_clim_avg['v']**2)
harm_clim_avg['th']  = calc_th(harm_clim_avg['T'],harm_clim_avg['p'])
harm_clim_avg['thl'] = calc_thl(harm_clim_avg['th'],harm_clim_avg['ql'],harm_clim_avg['p'])
for ii in ['phy','dyn']:
    harm_clim_avg['dtthl_'+ii]=calc_th(harm_clim_avg['dtT_'+ii],harm_clim_avg.p) - Lv / \
        (cp *calc_exner(harm_clim_avg.p)) * harm_clim_avg['dtqc_'+ii]

#%%


#############################################################################
#%%                     ####### PLOT #######
#############################################################################






