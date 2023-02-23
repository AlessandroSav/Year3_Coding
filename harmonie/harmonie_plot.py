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
harm={}
for exp in exps:
    file = my_harm_dir+exp+month+'*'+levels+'*.nc'
    harm[exp] = xr.open_mfdataset(file, combine='by_coords')
    harm[exp] = harm[exp].interpolate_na(dim='z')
    # drop duplicate hour between the 2 months 
    harm[exp].drop_duplicates('time')
    #remove first 2 days 
    harm[exp] = harm[exp].sel(time=slice(srt_time,end_time))

    # def calc_geo_height(ds_,fliplevels=False):
    #     # pressure in Pa
    #     if fliplevels==True:
    #         ds_['level']=np.flip(ds_.level)
        
    #     rho = calc_rho(ds_.p,ds_.T,ds_.qt)
    #     k = np.arange(ds_.level[0]+(ds_.level[1]-ds_.level[0])/2,\
    #                   ds_.level[-1]+(ds_.level[1]-ds_.level[0])/2,\
    #                   ds_.level[1]-ds_.level[0])
    #     rho_interp = rho.interp(level=k)
    #     zz = np.zeros((len(ds_['time']),len(ds_['level'])))
    #     zz = ((ds_.p.diff(dim='level').values)/(-1*rho_interp*g)).cumsum(dim='level')
    #     z = zz.interp(level=ds_.level,kwargs={"fill_value": "extrapolate"})
    #     ds_['z']=z -z.min('level')
    #     return (ds_)

    # # rename variables
    # harm_clim_avg        = harm[exp].rename({'ta':'T','hus':'qt','lev':'level','va':'v','ua':'u'})
    # #calculate height in meters
    # print("Calculating height levels")
    # harm_clim_avg        = calc_geo_height(harm_clim_avg,fliplevels=True) 
    # harm_clim_avg        = harm_clim_avg.sortby('level')
    # ##interpolate variables to heigth levels 
    # z_ref = harm_clim_avg.z.mean('time')
    # zz    = harm_clim_avg.z
    # for var in list(harm_clim_avg.keys()):
    #     if 'level' in harm_clim_avg[var].dims:
    #         print("interpolating variable "+var)
    #         x = np.empty((len(harm_clim_avg['time']),len(harm_clim_avg['level'])))
    #         x[:] = np.NaN
    #         for a in range(len(harm_clim_avg.time)):
    #             x[a,:] = np.interp(z_ref,zz[a,:],harm_clim_avg[var].isel(time = a))            
    #         harm_clim_avg[var] = (("time","level"), x)    
    # # convert model levels to height levels
    # harm_clim_avg = harm_clim_avg.rename({'z':'geo_height'})
    # harm_clim_avg = harm_clim_avg.rename({'level':'z'})
    # harm_clim_avg["z"] = (z_ref-z_ref.min()).values
    # harm_clim_avg['z'] = harm_clim_avg.z.assign_attrs(units='m',long_name='Height')
    # print("saving averaged heigth profiles")
    # harm_clim_avg.to_netcdf(my_harm_dir+exp+month+'_avg_z_all.nc')
    # del harm_clim_avg

#%%
######## old files in PhD_Year2 directory ###
# harm_clim_avg = harm_clim_avg.rename({'dtq_dyn':'dtqt_dyn','dtq_phy':'dtqt_phy'})
# harm_clim_avg['rho'] = calc_rho(harm_clim_avg['p'],harm_clim_avg['T'],harm_clim_avg['qt'])
# harm_clim_avg['wspd']= np.sqrt(harm_clim_avg['u']**2 + harm_clim_avg['v']**2)
# harm_clim_avg['th']  = calc_th(harm_clim_avg['T'],harm_clim_avg['p'])
# harm_clim_avg['thl'] = calc_thl(harm_clim_avg['th'],harm_clim_avg['ql'],harm_clim_avg['p'])
# for ii in ['phy','dyn']:
#     harm_clim_avg['dtthl_'+ii]=calc_th(harm_clim_avg['dtT_'+ii],harm_clim_avg.p) - Lv / \
#         (cp *calc_exner(harm_clim_avg.p)) * harm_clim_avg['dtqc_'+ii]

#%%


#############################################################################
#%%                     ####### PLOT #######
#############################################################################

var = 'clw'
for exp in exps:
    plt.figure(figsize=(15,7))
    harm[exp][var].plot(x='time',vmin=-0.0002)
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
for exp in ['noHGTQS_',]:
    plt.figure(figsize=(15,7))
    (acc_time*harm[exp]['dt'+var+'_dyn'])\
            .groupby(harm[exp].time.dt.hour).mean().plot(x='hour')
    plt.title('Dynamical tendency '+exp)
    plt.ylim([0,7000])
    
    plt.figure(figsize=(15,7))
    (acc_time*harm[exp]['dt'+var+'_dyn']).plot(x='time')
    plt.title('Dynamical tendency '+exp)
    plt.ylim([0,5000])
    
    
    
    plt.figure()
    acc_time*(harm[exp]['dt'+var+'_dyn']+\
              harm[exp]['dt'+var+'_phy']).mean('time').plot(y='z')
    (acc_time*harm[exp]['dt'+var+'_dyn']).mean('time').plot(y='z')
    (acc_time*harm[exp]['dt'+var+'_phy']).mean('time').plot(y='z')
    plt.xlim([-0.01,0.01])





fig, axs = plt.subplots(2,1,figsize=(19,15))
for ide, exp in enumerate(exps):
    h_clim_to_plot = harm[exp].sel(z=slice(layer[0],layer[1])).mean('z')
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

