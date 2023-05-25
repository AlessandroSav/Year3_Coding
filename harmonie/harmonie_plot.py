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
import pandas as pd
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

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 24,
         'axes.titlesize':'large',
         'xtick.labelsize':20,
         'ytick.labelsize':20,
         'figure.figsize':[10,7],
         'figure.titlesize':24}
pylab.rcParams.update(params)


#%% initial 
dt          = 75                 # model  timestep [seconds]
step        = 3600             # output timestep [seconds]
domain_name = 'BES'
lat_select  = 13.2806    # HALO center 
lon_select  = -57.7559   # HALO center 
domain      = 200            # km

srt_time   = np.datetime64('2020-01-03T00:30')
end_time   = np.datetime64('2020-02-29T23')

months = ['01',]

month='0*'

exps = ['noHGTQS_','noHGTQS_noSHAL_']
col=['r','g']
sty=['-',':']


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
    ### slow option to avoid open_mfdataset
    
    for idx,month in enumerate(months):
        file2d = my_harm_dir+exp[:-1]+'/'+exp+month+'_avg'+str(domain)+'_'+levels+'_all.nc'
        if idx==0 :
            harm2d[exp] = xr.open_dataset(file2d)
        else:
            harm2d[exp] = xr.concat([harm2d[exp],xr.open_dataset(file2d)],dim='time')#.to_dask().chunk()
            
        file3d = my_harm_dir+exp[:-1]+'/'+exp+month+'_3d_'+str(domain)+'_'+levels+'_all.nc'
        if idx==0:
            harm3d[exp] = xr.open_dataset(file3d)
        else:
            try:
                harm3d[exp] = xr.concat([harm3d[exp],xr.open_dataset(file3d)],dim='time')#.to_dask().chunk()
            except:
                harm3d[exp] = temp3d
        
    ####################################
    ####################################
    # ### open_mfdataset is not working! 
    # ### TypeError: from_array() got an unexpected keyword argument 'inline_array'
    # file2d = my_harm_dir+exp[:-1]+'/'+exp+month+'_avg'+str(domain)+'*'+levels+'*.nc'
    # harm2d[exp] = xr.open_mfdataset(file2d, combine='by_coords',decode_times=False)
    
    harm2d[exp]['z'] = np.sort(harm2d[exp]['z'].values)
    # harm2d[exp] = harm2d[exp].sortby('z')
    
    ## needed if there are nan values in z coordinate 
    harm2d[exp] = harm2d[exp].interpolate_na(dim='z')
    # drop duplicate hour between the 2 months 
    harm2d[exp] = harm2d[exp].drop_duplicates(dim='time',keep='first')
    #remove first 2 days 
    harm2d[exp] = harm2d[exp].sel(time=slice(srt_time,end_time))
    
    # file3d = my_harm_dir+exp+month+'_3d_'+str(domain)+'*'+levels+'*.nc'
    # harm3d[exp] = xr.open_mfdataset(file3d, combine='by_coords',parallel=True)
    harm3d[exp] = harm3d[exp].drop_duplicates(dim="time", keep="first")
    # harm3d[exp] = harm3d[exp].sortby('z')
    ####################################
    ####################################
    
    
    # convert model levels to height levels
    harm3d[exp] = harm3d[exp].rename({'lev':'z'})
    harm3d[exp]['z'] = harm3d[exp].z.assign_attrs(units='m',long_name='Height')
    harm3d[exp] = harm3d[exp].sortby('z')
    #remove first 2 days 
    harm3d[exp] = harm3d[exp].sel(time=slice(srt_time,end_time))
    
#%% Import organisation metrics
ds_org = {}
for exp in exps:    
    fileorg = my_harm_dir+'df_metrics_'+exp[:-1]+'.h5'    
    ds_org[exp] = pd.read_hdf(fileorg)
    ds_org[exp].index.names = ['time']
    ds_org[exp] = ds_org[exp].to_xarray()
#%%  Define Groups of organisation 
time_g1={}
time_g2={}
time_g3={}
### grouping by rain rate   
group = 'rain'
         
time_g1[group] = harm2d[exp].where(harm2d[exp]['rain'].sel(z=slice(0,100)).sum('z')\
                   <= np.quantile(harm2d[exp]['rain'].sel(z=slice(0,100)).sum('z').\
                                values,0.25),drop=True).time
time_g3[group] = harm2d[exp].where(harm2d[exp]['rain'].sel(z=slice(0,100)).sum('z')\
                   >= np.quantile(harm2d[exp]['rain'].sel(z=slice(0,100)).sum('z').\
                                values,0.75),drop=True).time
time_g2[group] = harm2d[exp].where(np.logical_not(harm2d[exp].time.\
                        isin(xr.concat((time_g1[group],time_g3[group]),'time'))),drop=True).time

#%% calculated resolved fluxes
for exp in exps:
    for var in ['ua','va','wa']:
        harm3d[exp][var+'_p'] = harm3d[exp][var] - harm3d[exp][var].mean(['x','y'])
    
    harm3d[exp]['uw']= harm3d[exp]['ua_p']*harm3d[exp]['wa_p']
    harm3d[exp]['vw']= harm3d[exp]['va_p']*harm3d[exp]['wa_p']

    for var in ['u','v']:
    ## save a variable for total parameterised momentum flux
        harm2d[exp][var+'_flx_param_tot']=harm2d[exp][var+'flx_turb']+\
          harm2d[exp][var+'flx_conv_moist']+\
          harm2d[exp][var+'flx_conv_dry']
#%%

#%% #########   QUESTIONS   #########

###DOES THE DISTRIBUTION OF CLOUDS (CLOUD SIZE,UPDRAFTS) DETERMINES THE FLUXES? 

### 1) How tilted are the eddies in different groups and runs?
### 2) How is the pdf of w changing with organisation and in the differnet runs?
### 3)

#############################################################################
#%%                     ####### PLOT #######
#############################################################################

#%% Cloud cover selecting different heights

plt.figure(figsize=(11,6))
harm3d[exp]['cl'].sel(z=slice(0,3000)).max('z').\
    mean(['x','y']).groupby('time.hour').mean('time').plot(x='hour',label='0 - 3km')
harm3d[exp]['cl'].sel(z=slice(1200,3000)).max('z').\
    mean(['x','y']).groupby('time.hour').mean('time').plot(x='hour',label='1.2 - 3km')
harm3d[exp]['cl'].sel(z=slice(1500,3000)).max('z').\
    mean(['x','y']).groupby('time.hour').mean('time').plot(x='hour',label='1.5 - 3km')
harm3d[exp]['cl'].sel(z=slice(1700,3000)).max('z').\
    mean(['x','y']).groupby('time.hour').mean('time').plot(x='hour',label='1.7 - 3km')
harm3d[exp]['cl'].sel(z=slice(0,1200)).max('z').\
    mean(['x','y']).groupby('time.hour').mean('time').plot(x='hour',label='0 - 1.2km')
plt.legend()
#%% PDF of vertical velocity 
plt.figure()
var = 'wa'
# harm3d[exp][var].sel(time=time_g1[group].values).sel(z=slice(500,30)).mean('z')\
#     .plot.hist(yscale='log',xlim=(-0.3,0.4),bins=500)
plt.axvline(harm3d[exp][var].sel(time=time_g1[group]).sel(z=slice(30,500)).mean('z')\
            .quantile(0.5),c='r',lw=0.5)
plt.axvline(harm3d[exp][var].sel(time=time_g1[group]).sel(z=slice(30,500)).mean('z')\
            .quantile(0.05),c='r',lw=0.5,ls='--')
plt.axvline(harm3d[exp][var].sel(time=time_g1[group]).sel(z=slice(30,500)).mean('z')\
            .quantile(0.95),c='r',lw=0.5,ls='--')
plt.axvline(0,c='k',lw=0.5)

# harm3d[exp][var].sel(time=time_g3[group]).sel(z=slice(500,30)).mean('z')\
#     .plot.hist(yscale='log',xlim=(-0.3,0.4),bins=500)
plt.axvline(harm3d[exp][var].sel(time=time_g3[group]).sel(z=slice(30,500)).mean('z')\
            .quantile(0.5),c='g',lw=0.5)
plt.axvline(harm3d[exp][var].sel(time=time_g3[group]).sel(z=slice(30,500)).mean('z')\
            .quantile(0.05),c='g',lw=0.5,ls='--')
plt.axvline(harm3d[exp][var].sel(time=time_g3[group]).sel(z=slice(30,500)).mean('z')\
            .quantile(0.95),c='g',lw=0.5,ls='--')
plt.axvline(0,c='k',lw=0.5)

#%% time series 
var = 'clw'
for exp in exps:
    plt.figure(figsize=(15,7))
    harm2d[exp][var].plot(x='time',vmin=-0.0002)
    plt.ylim([0,5000])
    plt.axvline('2020-02-02',c='k',lw=1)
    plt.axvline('2020-02-10T12',c='k',lw=1)
    plt.title(exp)
    
#%% Tendency evolution over the day 
acc_time=3600*1
layer=[200,1000]

var='u'
for exp in exps:
    ## diurnal cycle
    plt.figure(figsize=(15,7))
    (acc_time*harm2d[exp]['dt'+var+'_dyn'])\
            .groupby(harm2d[exp].time.dt.hour).mean().plot(x='hour')
    plt.title('Dynamical tendency '+exp)
    plt.ylim([0,5000])
    
    ## time series
    # plt.figure(figsize=(15,7))
    # (acc_time*harm2d[exp]['dt'+var+'_dyn']).plot(x='time')
    # plt.title('Dynamical tendency '+exp)
    # plt.ylim([0,5000])


fig, axs = plt.subplots(2,1,figsize=(19,15))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    h_clim_to_plot = harm2d[exp].sel(z=slice(layer[0],layer[1])).mean('z')
    for idx,var in enumerate(['u','v']):
        ## HARMONIE cy43 clim
        acc_time*(h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy'])\
            .groupby(h_clim_to_plot.time.dt.hour).mean().\
                plot(c='r',ls=sty[ide],label=lab+': Tot',ax=axs[idx])
        acc_time*h_clim_to_plot.groupby(h_clim_to_plot.time.dt.hour).mean()\
            ['dt'+var+'_dyn'].\
                plot(c='k',ls=sty[ide],label=lab+': Dyn',ax=axs[idx])
        acc_time*h_clim_to_plot.groupby(h_clim_to_plot.time.dt.hour).mean()\
            ['dt'+var+'_phy'].\
                plot(c='c',ls=sty[ide],label=lab+': Phy',ax=axs[idx]) 
        axs[idx].set_title(var+' direction',fontsize=25)
plt.legend(fontsize=20)
plt.tight_layout()

#%% Wind fluxes
step = 3600
fig, axs = plt.subplots(1,2,figsize=(13,11))
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
            ((harm2d[exp][var+'flx_conv_dry'].diff('time')) * step**-1)\
                .mean('time').isel(z=slice(1,-1)).plot(y='z',\
                        ls='--',ax=axs[idx],label=lab+' conv_dry',lw=2,c=col[ide])
                    
            ((harm2d[exp][var+'flx_conv_moist'].diff('time')) * step**-1)\
                .mean('time').isel(z=slice(1,-1)).plot(y='z',\
                        ls=':',ax=axs[idx],label=lab+' conv_moist',lw=2,c=col[ide])
            ((harm2d[exp][var+'flx_turb'].diff('time')) * step**-1)\
                .mean('time').isel(z=slice(1,-1)).plot(y='z',\
                        ls='-',ax=axs[idx],label=lab+' turb',lw=2,c=col[ide])
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        ## mean of all 
        # ((harm2d[exp][var+'_flx_param_tot']).diff('time') * step**-1+\
        #     harm3d[exp][var+'w'].mean(['x','y']))\
        #         .mean('time').isel(z=slice(1,-1)).plot(y='z',\
        #                 ls='-',ax=axs[idx],label=lab+' total',lw=4,c=col[ide])  
        ((harm2d[exp][var+'_flx_param_tot']).diff('time') * step**-1)\
                .mean('time').isel(z=slice(1,-1)).plot(y='z',\
                        ls='-.',ax=axs[idx],label=lab+' param',lw=4,c=col[ide])  
        # harm3d[exp][var+'w'].mean(['x','y','time']).plot(y='z',\
        #                 ls=':',ax=axs[idx],label=lab+' resolved',lw=2,c=col[ide])
        
            
        axs[idx].axhline(layer[0],c='grey',lw=0.3)
        axs[idx].axhline(layer[1],c='grey',lw=0.3)
        axs[idx].axvline(0,c='k',lw=0.5)
        axs[idx].set_ylim([0,3500])
axs[1].get_yaxis().set_visible(False) 
axs[0].set_xlabel(r'Zonal wind flux ($m^{2} s^{-2}$)')
axs[1].set_xlabel(r'Meridional wind flux ($m^{2} s^{-2}$)')

axs[0].legend(fontsize=21)   
plt.tight_layout()
           
#%% Wind profiles over the day
layer=[500,1500]
fig, axs = plt.subplots(2,2,figsize=(19,15),gridspec_kw={'width_ratios': [1,3]})
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        ## mean of all 
        
        # harm3d[exp][var].isel(z=slice(1,-1)).mean(['x','y','time'])\
        #     .plot(y='z',ls=sty[ide],ax=axs[idx,0],label=lab,lw=3,c=col[ide])
        
        harm2d[exp][var]\
                .mean('time').isel(z=slice(1,-1)).plot(y='z',\
                        ls=sty[ide],ax=axs[idx,0],label=lab,lw=3,c=col[ide])
        axs[idx,0].axhline(layer[0],c='grey',lw=0.3)
        axs[idx,0].axhline(layer[1],c='grey',lw=0.3)
        # axs[idx,0].axvline(0,c='k',lw=0.5)
        axs[idx,0].set_ylim([0,4000])
        
        
        ## diurnal cycle
        # harm3d[exp][var].isel(z=slice(1,-1)).mean(['x','y'])\
        #     .groupby('time.hour').mean().sel(z=slice(layer[0],layer[1])).mean('z')\
        #         .plot(x='hour',ls=sty[ide],ax=axs[idx,1],label=lab,lw=3,c=col[ide])
        
        harm2d[exp][var].sel(z=slice(layer[0],layer[1]))\
                .groupby('time.hour').mean().sel(z=slice(500,1500)).mean('z').plot(x='hour',\
                                ls=sty[ide],ax=axs[idx,1],lw=3,c=col[ide])
        axs[idx,1].set_xlim(0,23)
        
        axs[idx,0].set_title('Mean '+var,fontsize=30)
        axs[idx,1].set_title('Height between: '+str(layer[0])+'-'+str(layer[1])+' m',fontsize=30)

axs[0,0].set_xlim([-10,-0.5])
axs[1,0].set_xlim([-2,-0.7])
axs[0,0].legend(fontsize=25)
plt.tight_layout()
#%% Humidity profiles (lower and upper quartiles of humidity in the field)
var = 'hus'
plt.figure(figsize=(10,13))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
        
    (harm3d[exp][var]*1000).isel(z=slice(1,-1)).quantile(0.25,dim=['x','y']).\
        mean('time').plot(y='z',ls=sty[ide],c='orange',lw=3,label=lab+' dry')
    (harm3d[exp][var]*1000).isel(z=slice(1,-1)).quantile(0.75,dim=['x','y']).\
        mean('time').plot(y='z',ls=sty[ide],c='c',lw=3,label=lab+' moist')

plt.xlabel(r'Specific humidity ($g kg^{-1}$)')

plt.ylim([0,4000])
plt.legend(fontsize=25)
plt.title('Humidity quantiles',fontsize=30)
plt.tight_layout()

#%% Liquid water in air
plt.figure(figsize=(10,13))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    (harm2d[exp]['clw']*1000).mean('time').plot(y='z',lw=3,ls=sty[ide],\
                                                c=col[ide],label=lab)
plt.axvline(0,c='k',lw=0.5)
plt.ylim([0,4000])
plt.xlabel(r'Liquid water content ($g kg^{-1}$)')
plt.title('Liquid water',fontsize=30)
plt.legend(fontsize=25)
plt.tight_layout()
#%% rain distribution 
var = 'rain'
plt.figure()    
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    (harm3d[exp][var]*1000).where(harm3d[exp][var]>0.0001).isel(z=1)\
    .plot.hist(bins=500,color=col[ide],histtype=u'step')
    plt.axvline(harm3d[exp][var].where(harm3d[exp][var]>0.0001).isel(z=1)\
                .quantile(0.5)*1000,c=col[ide],lw=3.5,ls=sty[ide],label=lab)
    plt.axvline(harm3d[exp][var].where(harm3d[exp][var]>0.0001).isel(z=1)\
                .quantile(0.05)*1000,c=col[ide],lw=2.5,ls=sty[ide])
    plt.axvline(harm3d[exp][var].where(harm3d[exp][var]>0.0001).isel(z=1)\
                .quantile(0.95)*1000,c=col[ide],lw=2.5,ls=sty[ide])
plt.xlim([0.08,0.8])
plt.xlabel(r'Surface rain ($g kg^{-1}$)')
plt.ylabel(r'Count')
# ax = plt.gca()
# ax.get_yaxis().set_visible(False) 
plt.legend(fontsize=25)
plt.tight_layout()


#%%
print('End.')