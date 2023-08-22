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
import matplotlib.colors as mcolors
from matplotlib import cm
import sys
from datetime import datetime, timedelta
from netCDF4 import Dataset
from scipy import ndimage as ndi
my_source_dir = os.path.abspath('{}/../../../../My_source_codes')
sys.path.append(my_source_dir)
from My_thermo_fun import *

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
proj=ccrs.PlateCarree()
coast = cartopy.feature.NaturalEarthFeature(\
        category='physical', scale='50m', name='coastline',
        facecolor='none', edgecolor='r')

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 24,
         'axes.titlesize':'large',
         'xtick.labelsize':20,
         'ytick.labelsize':20,
         'figure.figsize':[10,7],
         'figure.titlesize':24}
pylab.rcParams.update(params)
import dask
dask.config.set({"array.slicing.split_large_chunks": True})

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
#%% initial 
dt          = 75                 # model  timestep [seconds]
step        = 3600             # output timestep [seconds]
domain_name = 'BES'
lat_select  = 13.2806    # HALO center 
lon_select  = -57.7559   # HALO center 
domain      = 200            # km
srt_time   = np.datetime64('2020-01-03T00:30')
end_time   = np.datetime64('2020-01-29T23')

months = ['01',]
month='0*'

exps = ['noHGTQS_','noHGTQS_noSHAL_','noHGTQS_noUVmix_']
exps = ['noHGTQS_','noHGTQS_noUVmix_']
col=['r','g','k']
col=['k','r']
sty=['-',':','--']
sty=['--','-']

levels = 'z'      ## decide wether to open files model level (lev) or 
                    ## already interpolate to height (z)
my_harm_dir     = '/Users/acmsavazzi/Documents/WORK/PhD_Year3/DATA/HARMONIE/'
ifs_dir         = '/Users/acmsavazzi/Documents/WORK/Research/MyData/'
#%%
print("Reading ERA5.") 
era5=xr.open_mfdataset(ifs_dir+'My_ds_ifs_ERA5.nc',chunks={'Date':-1})
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
harm_srf={}
for exp in exps:
    file2d = my_harm_dir+exp[:-1]+'/'+exp+month+'_avg'+str(domain)+'*'+levels+'*.nc'
    harm2d[exp] = xr.open_mfdataset(file2d, combine='by_coords',chunks={'time':-1})
    harm2d[exp]['z'] = np.sort(harm2d[exp]['z'].values)
    harm2d[exp] = harm2d[exp].sortby('z')
    
    ## needed if there are nan values in z coordinate 
    harm2d[exp] = harm2d[exp].interpolate_na(dim='z')
    # drop duplicate hour between the 2 months 
    harm2d[exp] = harm2d[exp].drop_duplicates(dim='time',keep='first')
    #remove first 2 days 
    harm2d[exp] = harm2d[exp].sel(time=slice(srt_time,end_time))
    # convert to local time
    harm2d[exp]['time'] = harm2d[exp]['time'] - np.timedelta64(4, 'h')
    harm2d[exp].time.attrs["units"] = "Local Time"
    
    ####################################
    file3d = my_harm_dir+exp[:-1]+'/'+exp+month+'_3d_'+str(domain)+'*'+levels+'*.nc'
    harm3d[exp] = xr.open_mfdataset(file3d, combine='by_coords',chunks={'time':-1})
    harm3d[exp] = harm3d[exp].drop_duplicates(dim="time", keep="first")
    ####################################    
    # convert model levels to height levels
    harm3d[exp] = harm3d[exp].rename({'lev':'z'})
    harm3d[exp]['z'] = harm3d[exp].z.assign_attrs(units='m',long_name='Height')
    harm3d[exp] = harm3d[exp].sortby('z')
    #remove first 2 days 
    harm3d[exp] = harm3d[exp].sel(time=slice(srt_time,end_time))
    # convert to local time
    harm3d[exp]['time'] = harm3d[exp]['time'] - np.timedelta64(4, 'h')
    harm3d[exp].time.attrs["units"] = "Local Time"

    # read surface 2d fields
    file_srf = my_harm_dir+exp[:-1]+'/'+exp+month+'_2d_'+str(domain)+'.nc'
    harm_srf[exp] = xr.open_mfdataset(file_srf, combine='by_coords')
    harm_srf[exp]['time'] = np.sort(harm_srf[exp]['time'].values)
    # drop duplicate hour between the 2 months 
    harm_srf[exp] = harm_srf[exp].drop_duplicates(dim='time',keep='first')
    #remove first 2 days 
    harm_srf[exp] = harm_srf[exp].sel(time=slice(srt_time,end_time))
    # convert to local time
    harm_srf[exp]['time'] = harm_srf[exp]['time'] - np.timedelta64(4, 'h')
    harm_srf[exp].time.attrs["units"] = "Local Time"
    
#%% Import organisation metrics
ds_org = {}
for exp in exps:    
    fileorg = my_harm_dir+'df_metrics_'+exp[:-1]+'.h5'    
    ds_org[exp] = pd.read_hdf(fileorg)
    ds_org[exp].index.names = ['time']
    ds_org[exp] = ds_org[exp].to_xarray()
    ds_org[exp] = ds_org[exp].sel(time=slice(srt_time,end_time))
    # convert to local time
    ds_org[exp]['time'] = ds_org[exp]['time'] - np.timedelta64(4, 'h')
    ds_org[exp].time.attrs["units"] = "Local Time"
#%%  Define Groups of organisation 
time_g1={}
time_g2={}
time_g3={}
### grouping by rain rate   
group = 'rain'
time_g1[group]={}
time_g2[group]={}
time_g3[group]={}
for exp in exps:   
    time_g1[group][exp] = harm2d[exp].where((harm2d[exp]['rain'].sel(z=slice(0,100)).sum('z')\
                       <= np.quantile(harm2d[exp]['rain'].sel(z=slice(0,100)).sum('z').\
                                    values,0.25)).compute(),drop=True).time
    time_g3[group][exp] = harm2d[exp].where((harm2d[exp]['rain'].sel(z=slice(0,100)).sum('z')\
                       >= np.quantile(harm2d[exp]['rain'].sel(z=slice(0,100)).sum('z').\
                                    values,0.75)).compute(),drop=True).time
    time_g2[group][exp] = harm2d[exp].where((np.logical_not(harm2d[exp].time.\
                            isin(xr.concat((time_g1[group][exp],time_g3[group][exp])\
                                           ,'time')))).compute(),drop=True).time
### grouping by Iorg   
group = 'iorg'
time_g1[group]={}
time_g2[group]={}
time_g3[group]={}
## use quartiles as treshold
# for exp in exps:        
#     time_g1[group][exp] = harm2d[exp].where((ds_org[exp].drop_duplicates('time')[group]\
#                        <= np.quantile(ds_org[exp][group].\
#                                     values,0.25)).compute(),drop=True).time
#     time_g3[group][exp] = harm2d[exp].where((ds_org[exp].drop_duplicates('time')[group]\
#                        >= np.quantile(ds_org[exp][group].\
#                                     values,0.75)).compute(),drop=True).time
#     time_g2[group][exp] = harm2d[exp].where((np.logical_not(harm2d[exp].time.\
#                             isin(xr.concat((time_g1[group][exp],time_g3[group][exp])\
#                                            ,'time')))).compute(),drop=True).time
## use a fixed treshold
for exp in exps:
    time_g1[group][exp] = harm2d[exp].where((ds_org[exp].drop_duplicates('time')[group]\
                       <= 0.54).compute(),drop=True).time
    time_g3[group][exp] = harm2d[exp].where((ds_org[exp].drop_duplicates('time')[group]\
                       >= 0.67).compute(),drop=True).time
    time_g2[group][exp] = harm2d[exp].where((np.logical_not(harm2d[exp].time.\
                            isin(xr.concat((time_g1[group][exp],time_g3[group][exp])\
                                           ,'time')))).compute(),drop=True).time

#%% calculated resolved fluxes
for exp in exps:
    for var in ['ua','va','wa']:
        harm3d[exp][var+'_p'] = harm3d[exp][var] - harm3d[exp][var].mean(['x','y'])
    
    harm3d[exp]['uw']= harm3d[exp]['ua_p']*harm3d[exp]['wa_p']
    harm3d[exp]['vw']= harm3d[exp]['va_p']*harm3d[exp]['wa_p']
    
    for id_ds, ds in enumerate([harm2d[exp],harm3d[exp]]): 
        for var in ['u','v']:
        ## save a variable for total parameterised momentum flux
            ds[var+'flx_turb'] = ds[var+'flx_turb'].diff('time') * step**-1
            ds[var+'flx_conv_dry'] = ds[var+'flx_conv_dry'].diff('time') * step**-1
            if exp == 'noHGTQS_noUVmix_':  # flx_conv_moist is missing in this experiment... why??        
                ds[var+'_flx_param_tot']=ds[var+'flx_turb']+\
                  ds[var+'flx_conv_dry']     
            else:
                ds[var+'flx_conv_moist'] = ds[var+'flx_conv_moist'].diff('time') * step**-1
                ds[var+'_flx_param_tot']=ds[var+'flx_turb']+\
                  ds[var+'flx_conv_moist']+\
                  ds[var+'flx_conv_dry']
        if id_ds ==0:
            harm2d[exp]=ds
        elif id_ds==1:
            harm3d[exp]=ds
#%% Cloud top height 
# define cloud top in each pixel using Cloud Area Fraction (cl)
var = 'cl'
thres = 0.1
for exp in exps:
    #height of zero cloud fraction after maximum
    zmax = harm3d[exp][var].sel(z=slice(0,5000)).idxmax('z')  # height of max cloud cover
    temp = harm3d[exp][var].sel(z=slice(0,5000)).where(harm3d[exp]['z']>=zmax)
    harm3d[exp][var+'_top'] = temp.where(lambda x: x<thres).idxmax(dim='z') 
    # exclude areas with no clouds (cloud top below 500 m)
    harm3d[exp][var+'_top'] = harm3d[exp][var+'_top'].where(harm3d[exp][var+'_top']>500)
    
    harm3d[exp][var+'_top_std'] = harm3d[exp][var+'_top'].std(['x','y'])
    
    ### Calculate variances 
    harm3d[exp]['u_var'] = harm3d[exp]['ua_p']**2
    harm3d[exp]['v_var'] = harm3d[exp]['va_p']**2
    harm3d[exp]['w_var'] = harm3d[exp]['wa_p']**2
    
    ### Calculate TKE 
    harm3d[exp]['tke']=\
        harm3d[exp]['ua_p']**2+\
        harm3d[exp]['ua_p']**2+\
        harm3d[exp]['ua_p']**2
        
    ### calculate cloud cover below 4km
    harm2d[exp]['cc_4km'] = \
    ((harm3d[exp]['cl'].sel(z=slice(0,4000))>0.5).any(dim='z').\
    sum(dim=('x','y'))/(len(harm3d[exp].x)*len(harm3d[exp].y)))
        
    ## calculate th
    harm3d[exp]['thv']= calc_th(calc_Tv(harm3d[exp]['ta'],harm3d[exp]['p'],\
                       calc_rh(harm3d[exp]['p'],harm3d[exp]['hus'],harm3d[exp]['ta'])),\
                               harm3d[exp]['p'])
        
    ## buoyancy
    harm3d[exp]['buoy'] = calc_buoy(harm3d[exp]['thv'],harm3d[exp]['thv'].mean(dim=('x','y')))
    
    
    

#%% #########   QUESTIONS   #########
###DOES THE DISTRIBUTION OF CLOUDS (CLOUD SIZE,UPDRAFTS) DETERMINES THE FLUXES? 

### 1) How tilted are the eddies in different groups and runs?
### 2) How is the pdf of w changing with organisation and in the differnet runs?
### 3)

#############################################################################
#%%                     ####### PLOT #######
#############################################################################
print("Plotting.") 
#%% Cloud cover selecting different heights

plt.figure(figsize=(11,6))
harm3d[exp]['cl'].sel(z=slice(0,13000)).max('z').\
    mean(['x','y']).groupby('time.hour').mean('time').plot(x='hour',label='0 - 13km')
harm3d[exp]['cl'].sel(z=slice(0,6000)).max('z').\
    mean(['x','y']).groupby('time.hour').mean('time').plot(x='hour',label='0 - 6km')
# harm3d[exp]['cl'].sel(z=slice(1500,3000)).max('z').\
#     mean(['x','y']).groupby('time.hour').mean('time').plot(x='hour',label='1.5 - 3km')
# harm3d[exp]['cl'].sel(z=slice(1700,3000)).max('z').\
#     mean(['x','y']).groupby('time.hour').mean('time').plot(x='hour',label='1.7 - 3km')
# harm3d[exp]['cl'].sel(z=slice(0,1200)).max('z').\
#     mean(['x','y']).groupby('time.hour').mean('time').plot(x='hour',label='0 - 1.2km')
plt.legend()


#%% Cloud statistics
fig, axs = plt.subplots(3,1,figsize=(19,15))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    for idx,var in enumerate(['mean_length_scale','num_objects','open_sky']):
        if var =='cloud_fraction':
            factor = 1
            title  = 'Cloud fraction'
            unit   = r'fraction '
        elif var =='num_objects':
            factor = 1
            title  = 'Number of clouds'
            unit   = r'number #'
        elif var =='iorg':
            factor = 1
            title  = r'$I_{org}$'
            unit   = r'$I_{org}$'
        elif var == 'mean_length_scale':
            factor = 1
            title  ='Mean length scale'
            unit   = r'km'
        elif var == 'open_sky':
            factor = 1
            title  ='Open sky'
            unit   = r''
        else:
            factor = 1
            title = var
        
        ds_org[exp][var].groupby('time.hour').mean('time').plot(\
                    x='hour',ls=sty[ide],ax=axs[idx],lw=3,c=col[ide],label=lab)
        
        # if var =='cloud_fraction':
        #     # harm_srf[exp].cll.mean(dim=('x','y')).groupby('time.hour').mean('time')\
        #     #     .plot(x='hour',ls=sty[ide],ax=axs[idx],lw=1,c=col[ide],label='low CC')
        #     harm3d[exp]['cl'].sel(z=slice(0,6000)).max('z').\
        #         mean(['x','y']).groupby('time.hour').mean('time').\
        #             plot(x='hour',c=col[ide],ls='--',lw=3,ax=axs[idx],label='0 - 6km')
            
        # Fill the area between the vertical lines
        axs[idx].axvspan(20, 23, alpha=0.1, color='grey')
        axs[idx].axvspan(0, 6, alpha=0.1, color='grey')
        axs[idx].set_xlim(0,23)
        axs[idx].set_title(title,fontsize =28)
        axs[idx].set_ylabel(unit)
axs[0].set_xlabel('')
axs[1].set_xlabel('')
axs[2].set_xlabel('hour LT')
axs[0].legend(fontsize=21)   
plt.tight_layout()

## cloud cover 
fig, axs = plt.subplots(1,figsize=(19,7))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    harm2d[exp]['cc_4km'].\
        groupby('time.hour').mean()\
            .plot(x='hour',ls=sty[ide],lw=3,c=col[ide])
    plt.axhline(harm2d[exp]['cc_4km'].\
                mean('time'),ls=sty[ide],lw=1,c=col[ide])

# Fill the area between the vertical lines
axs.axvspan(20, 23, alpha=0.1, color='grey')
axs.axvspan(0, 6, alpha=0.1, color='grey')
axs.set_xlim([0,23])
axs.set_ylabel(r'fraction')
axs.set_title(r'Cloud cover in the lower 4 km',fontsize=25)
axs.set_xlabel(r'hour LT')
plt.tight_layout()

#%% Surface fluxes and precipitation 
fig, axs = plt.subplots(3,1,figsize=(19,15))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    for idx,var in enumerate(['hfls','hfss','pr']):
        
        if var =='pr':
            factor = 3600
            titel  = 'Precipitation'
            unit   = r'$mm \, hour^{-1}$'
        elif var =='hfls':
            factor = 1
            titel  = 'Latent heat flux'
            unit   = r'$J \, m^{-2}$'
        elif var =='hfss':
            factor = 1
            titel  = 'Sensible heat flux'
            unit   = r'$J \, m^{-2}$'
        else: factor =1
        (factor*harm_srf[exp][var]).mean(dim=('x','y')).groupby('time.hour').mean('time')\
                .plot(x='hour',ls=sty[ide],ax=axs[idx],lw=3,c=col[ide],label=lab)
        axs[idx].axhline((factor*harm_srf[exp][var]).mean(dim=('x','y')).mean('time')\
                ,ls=sty[ide],lw=1,c=col[ide])
            
        # Fill the area between the vertical lines
        axs[idx].axvspan(20, 23, alpha=0.1, color='grey')
        axs[idx].axvspan(0, 6, alpha=0.1, color='grey')
        axs[idx].set_xlim(0,23)
        axs[idx].set_title(titel,fontsize =23)
        axs[idx].set_ylabel(unit)
axs[0].set_xlabel('')
axs[1].set_xlabel('')
axs[2].set_xlabel('hour LT')
axs[0].legend(fontsize=21)   
plt.tight_layout()
#%% PDF of vertical velocity 
level = [0,100]
group='rain'
var='hus'
plt.figure(figsize=(15,7))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    # Flatten the multidimensional array into a 1D array
    temp = harm3d[exp][var].sel(time=time_g1[group][exp]).sel(z=slice(level[0],level[1])).mean('z').values.ravel()
    plt.boxplot(temp,positions=[ide+1],labels=[lab+'_D'], showfliers=False,showmeans=True,widths=[0.5])
    
    temp = harm3d[exp][var].sel(time=time_g3[group][exp]).sel(z=slice(level[0],level[1])).mean('z').values.ravel()
    plt.boxplot(temp,positions=[ide+1+3],labels=[lab+'_R'], showfliers=False,showmeans=True,widths=[0.5])
    
plt.grid(axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
plt.axvline(3.5,lw=4,c='k')
plt.title('Distribution between '+str(level[0])+' - '+str(level[-1])+' m',fontsize=22)
plt.xlabel('')
plt.ylabel('Vertical velocity '+var)

#%% time series contours 
# var = 'clw'
# for exp in exps:
#     plt.figure(figsize=(15,7))
#     harm2d[exp][var].plot(x='time',vmin=-0.0002)
#     plt.ylim([0,5000])
#     plt.axvline(pd.to_datetime('2020-02-02', format = '%Y-%m-%d'),c='k',lw=1)
#     plt.axvline(pd.to_datetime('2020-02-10T12', format = '%Y-%m-%dT%H'),c='k',lw=1)
#     plt.title(exp)
#%% Tendency evolution over the day 
layer=[0,500]
var='u'
# for exp in exps:
#     ## diurnal cycle
#     plt.figure(figsize=(15,7))
#     (harm2d[exp]['dt'+var+'_dyn']*step)\
#             .groupby(harm2d[exp].time.dt.hour).mean().plot(x='hour')
#     plt.title('Dynamical tendency '+exp)
#     plt.ylim([0,5000])
    
    ## time series
    # plt.figure(figsize=(15,7))
    # (harm2d[exp]['dt'+var+'_dyn']*step).plot(x='time')
    # plt.title('Dynamical tendency '+exp)
    # plt.ylim([0,5000])

fig, axs = plt.subplots(2,1,figsize=(19,15))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    h_clim_to_plot = harm2d[exp].sel(z=slice(layer[0],layer[1])).mean('z')
    for idx,var in enumerate(['u','v']):
        ## HARMONIE cy43 clim
        ((h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy'])*step)\
            .groupby(h_clim_to_plot.time.dt.hour).mean().\
                plot(c=col[ide],ls='-',lw=3,label=lab+': Tot',ax=axs[idx])
        (h_clim_to_plot*step).groupby(h_clim_to_plot.time.dt.hour).mean()\
            ['dt'+var+'_dyn'].\
                plot(c=col[ide],ls=':',lw=3,label=lab+': Dyn',ax=axs[idx])
        (h_clim_to_plot*step).groupby(h_clim_to_plot.time.dt.hour).mean()\
            ['dt'+var+'_phy'].\
                plot(c=col[ide],ls='-.',lw=3,label=lab+': Phy',ax=axs[idx]) 
        axs[idx].set_title(var+' direction',fontsize=25)
        axs[idx].axhline(0,c='k',lw=0.5)
plt.legend(['Total','Resolved','Parameterised'],fontsize=20)
plt.tight_layout()

#%% Wind fluxes
layer=[0,250]
fig, axs = plt.subplots(2,2,figsize=(19,15),gridspec_kw={'width_ratios': [1,3]})
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
        ## mean of all 
        (harm2d[exp][var+'_flx_param_tot']+\
            harm3d[exp][var+'w'].mean(['x','y']))\
                .mean('time').isel(z=slice(1,-1)).plot(y='z',\
                        ls='-',ax=axs[idx,0],label=lab+' total',lw=4,c=col[ide])      
        ## diurnal cycle
        (harm2d[exp][var+'_flx_param_tot']+\
            harm3d[exp][var+'w'].mean(['x','y']))\
                .groupby('time.hour').mean().sel(z=slice(layer[0],layer[1])).mean('z')\
                    .plot(x='hour',ls=sty[ide],ax=axs[idx,1],lw=3,c=col[ide])
                
    axs[idx,0].set_title('Mean '+var+' flux',fontsize=30)
    axs[idx,1].set_title('Height between: '+str(layer[0])+'-'+str(layer[1])+' m',fontsize=30)
    axs[idx,1].set_xlim(0,23)
    axs[idx,0].axhline(layer[0],c='grey',lw=0.3)
    axs[idx,0].axhline(layer[1],c='grey',lw=0.3)
    axs[idx,0].axvline(0,c='k',lw=0.5)
    axs[idx,0].set_ylim([0,3500])
    axs[idx,0].set_xlabel(r'($m^{2} s^{-2}$)')
axs[0,1].set_xlabel(r'')

axs[0,0].legend(fontsize=21)   
plt.tight_layout()
#%% Wind fluxes profiles
fig, axs = plt.subplots(1,2,figsize=(13,11))
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
        resol = harm3d[exp][var+'w'].mean(['x','y'])
        param = harm2d[exp][var+'_flx_param_tot']
        
        ## mean of all 
        (resol+param)\
                .mean('time').plot(y='z',\
                        ls='-',ax=axs[idx],label=lab+' total',lw=3,c=col[ide])  
        ## parameterised 
        # param\
        #         .mean('time').plot(y='z',\
        #                 ls='--',ax=axs[idx],label=lab+' param',lw=2,c=col[ide])  
        ## resolved
        resol\
                .mean('time').plot(y='z',\
                        ls=':',ax=axs[idx],label=lab+' resol',lw=3,c=col[ide])  
        
        
        # axs[idx].axhline(layer[0],c='grey',lw=0.3)
        # axs[idx].axhline(layer[1],c='grey',lw=0.3)
        axs[idx].axvline(0,c='grey',lw=0.5)
        axs[idx].set_ylim([0,4000])
        axs[idx].set_xlabel(r'$m^{2} \, s^{-2}$')
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
# axs[0].set_xlim([-10,-0.5])
# axs[1].set_xlim([-2,-0.7])   
axs[1].set_yticks([]) 
axs[1].set_ylabel('')
axs[0].set_title('Zonal momentum flux',fontsize=25)
axs[1].set_title('Meridional momentum flux',fontsize=25)
axs[0].legend(fontsize=25)
plt.tight_layout()

#%% wind fluxes by group 
group = 'iorg'
fig, axs = plt.subplots(2,2,figsize=(15,20))
# for idx,var in enumerate(['u','v']):
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
            
        resol = harm3d[exp][var+'w'].mean(['x','y'])
        param = harm2d[exp][var+'_flx_param_tot']
        temp_g1 = (resol+param).sel(time=time_g1[group][exp]).mean('time')
        temp_g3 = (resol+param).sel(time=time_g3[group][exp]).mean('time')  
        ## mean of all         
        temp_g1.plot(y='z',\
                        ls=sty[ide],ax=axs[0,idx],label=lab,lw=3,c=col[ide])
        temp_g3.plot(y='z',\
                        ls=sty[ide],ax=axs[1,idx],label=lab,lw=3,c=col[ide])
        axs[idx,0].axvline(0,c='k',lw=0.5)
        axs[idx,1].axvline(0,c='k',lw=0.5)
        axs[0,idx].title.set_text(group+' Group 1')
        axs[1,idx].title.set_text(group+' Group 3')
        axs[0,idx].title.set_fontsize(20)
        axs[1,idx].title.set_fontsize(20)
        
        
    axs[idx,0].set_ylim([0,4000])
    axs[idx,1].set_ylim([0,4000])
# limits for fluxes
axs[0,0].set_xlim([-0.02,0.1])
axs[1,0].set_xlim([-0.02,0.1])
axs[0,1].set_xlim([-0.02,0.03])
axs[1,1].set_xlim([-0.02,0.03])
#%%
### Organisation influences resolved fluxes?
group = 'iorg'
fig, axs = plt.subplots(1,2,figsize=(15,13))
# for idx,var in enumerate(['u','v']):
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
            
        resol = harm3d[exp][var+'w'].mean(['x','y'])
        param = harm2d[exp][var+'_flx_param_tot']
        
        resol_perc = np.fabs(resol) / (np.fabs(resol)+np.fabs(param))
        param_perc = np.fabs(param) / (np.fabs(resol)+np.fabs(param))
       
        resol_perc_g1 = resol_perc.sel(time=time_g1[group][exp]).mean('time')
        resol_perc_g3 = resol_perc.sel(time=time_g3[group][exp]).mean('time')

        
        resol_g1 = resol.sel(time=time_g1[group][exp]).mean('time')
        resol_g3 = resol.sel(time=time_g3[group][exp]).mean('time') 
        param_g1 = param.sel(time=time_g1[group][exp]).mean('time')
        param_g3 = param.sel(time=time_g3[group][exp]).mean('time') 
        
        resol_g1_perc = np.fabs(resol_g1) / (np.fabs(resol_g1)+np.fabs(param_g1))
        resol_g3_perc = np.fabs(resol_g3) / (np.fabs(resol_g3)+np.fabs(param_g3))
       
        ## group 3
        resol_perc_g3.\
            plot(y='z',\
            ls='-',ax=axs[idx],label=lab+' Organised',lw=3,c=col[ide])
        
       ## group 1
        resol_perc_g1.\
            plot(y='z',\
            ls=':',ax=axs[idx],lw=3,c=col[ide],label=lab+' Un-organised')    
             
    axs[idx].axvline(0.5,c='grey',lw=0.5)
    axs[idx].title.set_fontsize(23)
    axs[idx].spines['top'].set_visible(False)
    axs[idx].spines['right'].set_visible(False)
        
    axs[idx].set_ylim([0,4000])
    axs[idx].set_xlabel(r'fraction of total flux')
axs[0].title.set_text(' Resolved zonal flux')
axs[1].title.set_text(' Resolved meridional flux')
axs[1].set_yticks([]) 
axs[1].set_ylabel('')
axs[0].legend(fontsize=21)   
plt.tight_layout()
#%% similar to plot above 
fig, axs = plt.subplots(2,2,figsize=(15,20))
# for idx,var in enumerate(['u','v']):
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
            
        resol = harm3d[exp][var+'w'].mean(['x','y'])
        param = harm2d[exp][var+'_flx_param_tot']
        resol_perc = np.fabs(resol) / (np.fabs(resol)+np.fabs(param))
        param_perc = np.fabs(param) / (np.fabs(resol)+np.fabs(param))
            
        temp_g1 = resol_perc.sel(time=time_g1[group][exp]).mean('time')
        temp_g3 = resol_perc.sel(time=time_g3[group][exp]).mean('time')        
        ## group 1
        temp_g1.\
            plot(y='z',\
            ls=':',ax=axs[0,idx],lw=3,c=col[ide])
        temp_g1.where((resol.sel(time=time_g1[group][exp]).mean('time') * \
         (resol+param).sel(time=time_g1[group][exp]).mean('time')) > 0).\
            plot(y='z',\
            ls='-',ax=axs[0,idx],label=lab+' resolved',lw=3,c=col[ide])         
            
        ## group 3
        temp_g3.\
            plot(y='z',\
            ls='-',ax=axs[1,idx],label=lab+' resolved',lw=3,c=col[ide])
        temp_g3.where((resol.sel(time=time_g3[group][exp]).mean('time') * \
         (resol+param).sel(time=time_g3[group][exp]).mean('time')) > 0).\
            plot(y='z',\
            ls='-',ax=axs[1,idx],lw=3,c=col[ide])
                
        ## group 1       
        # param_perc.sel(time=time_g1[group][exp]).mean('time').\
        #     plot(y='z',\
        #     ls='--',ax=axs[0,idx],label=lab+' param',lw=2,c=col[ide])
        
        # (harm2d[exp][var+'_flx_param_tot']+\
        #     harm3d[exp][var+'w'].mean(['x','y']))\
        #         .mean('time').isel(z=slice(1,-1)).plot(y='z',\
        #                 ls='-',ax=axs[0,idx],label=lab+' total',lw=4,c=col[ide])  
        
        # (harm3d[exp][var+'w'].mean(['x','y']).mean('time'))\
        #             .plot(y='z',\
        #                 ls='-',ax=axs[0,idx],label=lab+' resolved',lw=2,c=col[ide])
        
        # (harm2d[exp].sel(time=time_g1[group][exp])[var+'_flx_param_tot'])\
        #         .mean('time')\
        #             .plot(y='z',\
        #                 ls='--',ax=axs[0,idx],label=lab+' param',lw=2,c=col[ide])
        
        ## group 3
        # param_perc.sel(time=time_g3[group][exp]).mean('time').\
        #     plot(y='z',\
        #     ls='--',ax=axs[1,idx],label=lab+' param',lw=2,c=col[ide])
                
        
        # harm2d[exp].sel(time=time_g3[group][exp])[var+'flx_turb']\
        #         .mean('time').plot(y='z',\
        #                 ls='-',ax=axs[1,idx],label=lab,lw=3,c=col[ide])
                    
                    
        axs[idx,0].axvline(0.5,c='grey',lw=0.5)
        axs[idx,1].axvline(0.5,c='grey',lw=0.5)
        axs[0,idx].title.set_text(group+' Group 1')
        axs[1,idx].title.set_text(group+' Group 3')
        axs[0,idx].title.set_fontsize(22)
        axs[1,idx].title.set_fontsize(22)
        
    axs[idx,0].set_ylim([0,4000])
    axs[idx,1].set_ylim([0,4000])
axs[0,0].legend(fontsize=21)   
plt.tight_layout()


#%%Level of max subgrid transport
step = 3600
# fig, axs = plt.subplots(2,1,figsize=(13,11))
# for idx,var in enumerate(['u','v']):
#     for ide, exp in enumerate(exps):
#         if exp == 'noHGTQS_':
#             lab='Control'
#         elif exp == 'noHGTQS_noSHAL_':
#             lab='NoShal'
#         elif exp == 'noHGTQS_noUVmix_':
#                 lab='NoMom'
#         (harm2d[exp][var+'_flx_param_tot'])\
#                 .groupby('time.hour').mean('time').idxmin('z').plot(\
#                         ls='-',ax=axs[idx],label=lab+' param',lw=4,c=col[ide])
#         # (harm2d[exp][var+'_flx_param_tot'])\
#         #         .idxmin('z').groupby('time.hour').mean('time').plot(\
#         #                 ls='-',ax=axs[idx],label=lab+' param',lw=4,c=col[ide])
        
#         axs[idx].set_xlim([0,23])
# axs[0].set_ylim([1000,2500])
# axs[1].set_ylim([350,1400])
# axs[0].set_xlabel('')
# axs[0].set_title(r'Zonal wind flux ($m^{2} s^{-2}$)',fontsize=22)
# axs[1].set_title(r'Meridional wind flux ($m^{2} s^{-2}$)',fontsize=22)

# axs[0].legend(fontsize=21)   
# plt.tight_layout()
var='v'
fig, axs = plt.subplots(3,1,figsize=(13,11))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
    (harm2d[exp][var+'_flx_param_tot'])\
            .groupby('time.hour').mean('time').plot(x='hour',\
                                    ax=axs[ide],vmin=-0.015,vmax=0.02,\
                                cmap=cm.PiYG_r,norm=mcolors.TwoSlopeNorm(0))
    if exp != 'noHGTQS_':  
        (harm2d[exp][var+'_flx_param_tot'])\
            .groupby('time.hour').mean('time').idxmin('z').plot(\
                        ls=':',ax=axs[0],lw=3,c=col[ide])
    (harm2d[exp][var+'_flx_param_tot'])\
        .groupby('time.hour').mean('time').idxmin('z').plot(\
                    ls=':',ax=axs[ide],lw=4,c=col[ide])
                    
    axs[ide].set_xlim([0,23])
    axs[ide].set_ylim([0,3500])
    axs[ide].set_title(lab,fontsize=22)
axs[0].set_xlabel('')
axs[1].set_xlabel('')
plt.tight_layout()
#%% Wind tendencies
fig, axs = plt.subplots(1,2,figsize=(12,10))
for idx,var in enumerate(['dtu','dtv']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
        
        ## mean of all 
        # ((harm2d[exp][var+'_phy'] + harm2d[exp][var+'_dyn'])*step)\
        #     .mean('time').isel(z=slice(1,-1))\
        #     .plot(y='z',\
        #           ls='-',ax=axs[idx],label=lab+' total',lw=4,c=col[ide])   
        ## parameterised 
        (harm2d[exp][var+'_phy']*step).mean('time').isel(z=slice(1,-1))\
            .plot(y='z',\
                  ls='-.',ax=axs[idx],label=lab+' param',lw=3,c=col[ide])  
        ## resolved
        (harm2d[exp][var+'_dyn']*step).mean('time').isel(z=slice(1,-1))\
            .plot(y='z',\
                  ls=':',ax=axs[idx],label=lab+' resolved',lw=3,c=col[ide])
        
        
        # axs[idx].axhline(layer[0],c='grey',lw=0.3)
        # axs[idx].axhline(layer[1],c='grey',lw=0.3)
        axs[idx].axvline(0,c='grey',lw=0.5)
        axs[idx].set_ylim([0,4000])
        axs[idx].set_xlabel(r'$m \, s^{-1} \, /hour$')
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
# axs[0].set_xlim([-10,-0.5])
# axs[1].set_xlim([-2,-0.7])   
axs[1].set_yticks([]) 
axs[1].set_ylabel('')
axs[0].set_title('Zonal tendency',fontsize=25)
axs[1].set_title('Meridional tendency',fontsize=25)
# axs[0].legend(fontsize=25)
fig.legend(['Param','Resolved'],fontsize=23,loc=[0.4,0.8])
plt.tight_layout()

#%% wind tendencies by group
group = 'iorg'
fig, axs = plt.subplots(2,2,figsize=(15,20))
# for idx,var in enumerate(['u','v']):
for idx,var in enumerate(['dtu','dtv']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
        ## mean of all         
        # ((harm2d[exp][var+'_phy'] + harm2d[exp][var+'_dyn'])*step)\
        #     .sel(time=time_g1[group][exp]).mean('time').isel(z=slice(1,-1))\
        #     .plot(y='z',\
        #           ls='-',ax=axs[0,idx],label=lab+' total',lw=4,c=col[ide])
        # ((harm2d[exp][var+'_phy'] + harm2d[exp][var+'_dyn'])*step)\
        #     .sel(time=time_g3[group][exp]).mean('time').isel(z=slice(1,-1))\
        #     .plot(y='z',\
        #           ls='-',ax=axs[1,idx],label=lab+' total',lw=4,c=col[ide])
        ## parameterised 
        (harm2d[exp][var+'_phy']*step).sel(time=time_g1[group][exp]).mean('time').isel(z=slice(1,-1))\
            .plot(y='z',\
                  ls='-.',ax=axs[0,idx],label=lab+' param',lw=2,c=col[ide]) 
        (harm2d[exp][var+'_phy']*step).sel(time=time_g3[group][exp]).mean('time').isel(z=slice(1,-1))\
            .plot(y='z',\
                  ls='-.',ax=axs[1,idx],label=lab+' param',lw=2,c=col[ide]) 
        ## resolved 
        (harm2d[exp][var+'_dyn']*step).sel(time=time_g1[group][exp]).mean('time').isel(z=slice(1,-1))\
            .plot(y='z',\
                  ls=':',ax=axs[0,idx],label=lab+' resolved',lw=2,c=col[ide])
        (harm2d[exp][var+'_dyn']*step).sel(time=time_g3[group][exp]).mean('time').isel(z=slice(1,-1))\
            .plot(y='z',\
                  ls=':',ax=axs[1,idx],label=lab+' resolved',lw=2,c=col[ide])
                
        axs[idx,0].axvline(0,c='k',lw=0.5)
        axs[idx,1].axvline(0,c='k',lw=0.5)
        axs[0,idx].title.set_text('Un-organised')
        axs[1,idx].title.set_text('Organised')
        axs[0,idx].title.set_fontsize(28)
        axs[1,idx].title.set_fontsize(28)
        axs[1,idx].set_xlabel(r'$m \, s^{-1} \, /hour$')
        
    axs[idx,0].set_ylim([0,4000])
    axs[idx,1].set_ylim([0,4000])
    axs[idx,1].set_ylabel('')
    axs[idx,1].set_yticks([]) 
# limits for tendencies
# axs[0,0].set_xlim([-0.1,0.1])
# axs[1,0].set_xlim([-0.1,0.1])
# axs[0,1].set_xlim([-0.165,0.15])
# axs[1,1].set_xlim([-0.165,0.15])

#%% Tendencies inside/outside a cloudy pixel

## you need to have tendencies in 'harm3d'. Go back to raw files and save it.

#%% Wind profiles over the day
layer=[0,250]
fig, axs = plt.subplots(2,2,figsize=(19,15),gridspec_kw={'width_ratios': [1,3]})
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
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
        
        harm2d[exp][var]\
                .groupby('time.hour').mean().sel(z=slice(layer[0],layer[1])).mean('z').plot(x='hour',\
                                ls=sty[ide],ax=axs[idx,1],lw=3,c=col[ide])
        axs[idx,1].set_xlim(0,23)
        
        axs[idx,0].set_title('Mean '+var,fontsize=30)
        axs[idx,1].set_title('Height between: '+str(layer[0])+'-'+str(layer[1])+' m',fontsize=30)

axs[0,0].set_xlim([-10,-0.5])
axs[1,0].set_xlim([-2,-0.7])
axs[0,0].legend(fontsize=25)
plt.tight_layout()

#%% Wind profiles 
layer=[0,200]
fig, axs = plt.subplots(1,2,figsize=(13,10))
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
        ## mean of all 
        
        # harm3d[exp][var].isel(z=slice(1,-1)).mean(['x','y','time'])\
        #     .plot(y='z',ls=sty[ide],ax=axs[idx,0],label=lab,lw=3,c=col[ide])
        
        harm2d[exp][var]\
                .mean('time').isel(z=slice(1,-1)).plot(y='z',\
                        ls=sty[ide],ax=axs[idx],label=lab,lw=3,c=col[ide])
        axs[idx].axhline(layer[0],c='grey',lw=0.3)
        axs[idx].axhline(layer[1],c='grey',lw=0.3)
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        # axs[idx,0].axvline(0,c='k',lw=0.5)
        axs[idx].set_ylim([0,4000])
        axs[idx].set_xlabel(r'$m \, s^{-1}$')
axs[0].set_xlim([-10,-0.5])
axs[1].set_xlim([-2,-0.7])   
axs[1].set_yticks([]) 
axs[1].set_ylabel('')
axs[0].set_title('Zonal wind',fontsize=25)
axs[1].set_title('Meridional wind',fontsize=25)
axs[0].legend(fontsize=25)
plt.tight_layout()
        
## wind speed         
fig, axs = plt.subplots(1,figsize=(19,7))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    np.sqrt(harm2d[exp]['u']**2 + harm2d[exp]['v']**2).\
        groupby('time.hour').mean().sel(z=slice(layer[0],layer[1])).mean('z')\
            .plot(x='hour',ls=sty[ide],lw=3,c=col[ide])
    plt.axhline(np.sqrt(harm2d[exp]['u']**2 + harm2d[exp]['v']**2).\
                mean('time').sel(z=slice(layer[0],layer[1])).mean('z'),ls=sty[ide],lw=1,c=col[ide])

# Fill the area between the vertical lines
axs.axvspan(20, 23, alpha=0.1, color='grey')
axs.axvspan(0, 6, alpha=0.1, color='grey')
axs.set_xlim([0,23])
axs.set_ylabel(r'$m \, s^{-1}$')
axs.set_title(r'Wind speed in the lower '+str(layer[1])+' m',fontsize=25)
axs.set_xlabel(r'hour LT')
plt.tight_layout()

#%% Wind profiles by groups
group = 'iorg'
fig, axs = plt.subplots(2,2,figsize=(15,20))
# for idx,var in enumerate(['u','v']):
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
        elif exp == 'noHGTQS_noUVmix_':
            lab='NoMom'
        ## mean of all         
        harm2d[exp].sel(time=time_g1[group][exp])[var]\
                .mean('time').plot(y='z',\
                        ls=sty[ide],ax=axs[0,idx],label=lab,lw=3,c=col[ide])
        harm2d[exp].sel(time=time_g3[group][exp])[var]\
                .mean('time').plot(y='z',\
                        ls=sty[ide],ax=axs[1,idx],label=lab,lw=3,c=col[ide])
        axs[idx,0].axvline(0,c='k',lw=0.5)
        axs[idx,1].axvline(0,c='k',lw=0.5)
        axs[0,idx].title.set_text(group+' Group 1')
        axs[1,idx].title.set_text(group+' Group 3')
        axs[0,idx].title.set_fontsize(20)
        axs[1,idx].title.set_fontsize(20)
        
    axs[idx,0].set_ylim([0,4000])
    axs[idx,1].set_ylim([0,4000])
# limits for wind
axs[0,0].set_xlim([-14,4])
axs[1,0].set_xlim([-14,4])
axs[0,1].set_xlim([-4.5,1.2])
axs[1,1].set_xlim([-4.5,1.2])
# limits for fluxes
# axs[0,0].set_xlim([-0.018,0.1])
# axs[1,0].set_xlim([-0.018,0.1])
# axs[0,1].set_xlim([-0.015,0.03])
# axs[1,1].set_xlim([-0.015,0.03])

#%% Humidity profiles (lower and upper quartiles of humidity in the field)
var = 'hus'
plt.figure(figsize=(10,13))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    deviations = harm3d[exp][var] - harm3d['noHGTQS_'][var].mean(dim=['x','y','time']).interp(z=harm3d[exp].z)
    
    (deviations*1000).isel(z=slice(1,-1)).quantile(0.25,dim=['x','y']).\
        mean('time').plot(y='z',ls='-',c=col[ide],lw=3,label=lab+' dry')
    (deviations*1000).isel(z=slice(1,-1)).quantile(0.75,dim=['x','y']).\
        mean('time').plot(y='z',ls='--',c=col[ide],lw=3,label=lab+' moist')

plt.xlabel(r'Specific humidity ($g kg^{-1}$)')

plt.ylim([0,4000])
plt.axvline(0,c='k',lw=0.5)
plt.legend(fontsize=25)
plt.title('Humidity quantiles',fontsize=30)
plt.tight_layout()


harm3d[exp]['rain'].mean(dim=['x','y','time'])

#%% Plot fluxes for moister and dryer pixels

#%% Liquid water in air
fig, axs = plt.subplots(1,3,figsize=(15,10))

for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    
    harm2d[exp]['cl'].mean('time').plot(y='z',lw=3,ls=sty[ide],\
                                                c=col[ide],label=lab,ax=axs[0])
        
    (1000*harm2d[exp]['clw']/(harm2d[exp]['cl'])).where(harm2d[exp]['cl']>0.02).mean('time').plot(y='z',lw=3,ls=sty[ide],\
                                                c=col[ide],label=lab,ax=axs[1])
    
    # (1000*harm2d[exp]['clw']*(1-harm2d[exp]['cl'])).mean('time').plot(y='z',lw=3,ls=sty[ide],\
    #                                             c=col[ide],ax=axs[1])

    # ((1000*harm3d[exp]['rain']).mean(dim=['x','y'])\
    #     /(harm2d[exp]['cl'])).where(harm2d[exp]['cl']>0.02).mean('time').\
    #     plot(y='z',lw=3,ls=sty[ide],c=col[ide],label=lab,ax=axs[2])
    # (1000*harm3d[exp]['rain']).mean(dim=['x','y']).mean('time').\
    #     plot(y='z',lw=3,ls=sty[ide],c=col[ide],label=lab,ax=axs[2])
        
    ((harm3d[exp]['hus'].mean(dim=['x','y']).interp(z=harm3d['noHGTQS_'].z) - harm3d['noHGTQS_']['hus'].mean(dim=['x','y']))*100).\
        mean('time').plot(y='z',ls=sty[ide],c=col[ide],lw=3,ax=axs[2])

axs[0].set_ylim([0,4000])
axs[1].set_ylim([0,4000])
axs[2].set_ylim([0,4000])
# plt.axvline(0,c='k',lw=0.5)
axs[1].set_ylabel('')
axs[1].set_yticks([])
axs[2].set_ylabel('')
axs[2].set_yticks([])
axs[0].set_xlabel(r'fraction')
axs[1].set_xlabel(r'Liquid water ($g \, kg^{-1}$)')
axs[2].set_xlabel(r'$q_t$ ($g \, kg^{-1}$)')
axs[0].set_title('Cloud fraction',fontsize=30)
axs[1].set_title('In-cloud LW content',fontsize=30)
axs[2].set_title(r'$\Delta q_t$ (NoMom - ctrl)',fontsize=30)
axs[0].legend(fontsize=25)
plt.tight_layout()
#%% temporaneo
var = 'hus'
plt.figure(figsize=(10,13))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    deviations = harm3d[exp][var] #- harm3d['noHGTQS_'][var].mean(dim=['x','y','time']).interp(z=harm3d[exp].z)
    
    (deviations*1000).isel(z=slice(1,-1)).quantile(0.25,dim=['x','y']).\
        mean('time').plot(y='z',ls='-',c=col[ide],lw=3,label=lab+' dry')
    (deviations*1000).isel(z=slice(1,-1)).quantile(0.75,dim=['x','y']).\
        mean('time').plot(y='z',ls='--',c=col[ide],lw=3,label=lab+' moist')

plt.ylim([0,4000])
plt.axvline(0,c='k',lw=0.5)
plt.legend(fontsize=25)
plt.title('Humidity quantiles',fontsize=30)
plt.tight_layout()



#%% rain distribution 
var = 'rain'
thres = +0.0001
plt.figure()    
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    (harm3d[exp][var]*1000).where(harm3d[exp][var]>thres).isel(z=1)\
    .plot.hist(bins=500,color=col[ide],histtype=u'step', density=False)
    plt.axvline(harm3d[exp][var].where(harm3d[exp][var]>thres).isel(z=1)\
                .quantile(0.5)*1000,c=col[ide],lw=3.5,ls=sty[ide],label=lab)
    plt.axvline(harm3d[exp][var].where(harm3d[exp][var]>thres).isel(z=1)\
                .quantile(0.05)*1000,c=col[ide],lw=2.5,ls=sty[ide])
    plt.axvline(harm3d[exp][var].where(harm3d[exp][var]>thres).isel(z=1)\
                .quantile(0.95)*1000,c=col[ide],lw=2.5,ls=sty[ide])
        
        # harm3d[exp][var].isel(z=1))>tresh).sum()
plt.xlim([0.05,0.8])
# plt.ylim([0,13])
plt.xlabel(r'Surface rain ($g kg^{-1}$)')
plt.ylabel(r'Density')
# ax = plt.gca()
# ax.get_yaxis().set_visible(False) 
plt.legend(fontsize=25)
plt.tight_layout()

#%% Define cloud base 

# variable exists! import from 2d fields 

#%% Temporal variability in cloud top 
##  count objects with cloud top below 1.5km and with CT between 2.5 and 3km
#   plot the diurnal cycle of these counts
var = 'cl_top'
plt.figure(figsize=(13,9) )
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    (harm3d[exp][var]<1500).sum(dim=('x','y')).groupby('time.hour').mean()\
        .plot(color=col[ide],label=lab+'_Sh',lw=3)
    ((harm3d[exp][var]<4000)&(harm3d[exp][var]>2500)).sum(dim=('x','y')).groupby('time.hour').mean()\
            .plot(color=col[ide],label=lab+'_Dp',ls='--',lw=3)
plt.ylabel(r'count of pixels')
plt.legend(fontsize=25)
plt.title('Cloud top counts for Deeper and Shallower pixels',fontsize=22)
plt.xlim([0,23])
plt.tight_layout()  
#%%
def convert_rain_intensity(intensity_kg_kg):
    density_water_vapor = 0.9  # kg/m³
    conversion_factor = 1000 / 3600  # Conversion from kg/m³ to g/m³ and seconds to hours

    intensity_mm_hour = intensity_kg_kg * density_water_vapor * conversion_factor
    return intensity_mm_hour
var = 'rain'
plt.figure(figsize=(13,8) )
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    rain_rate = convert_rain_intensity(harm3d[exp][var]).mean(dim=('x','y'))  
    rain_rate.sel(z=0).groupby('time.hour').mean()\
        .plot(color=col[ide],label=lab,lw=3)
plt.ylabel(r'mm/hour')
plt.legend(fontsize=25)
plt.title('Rain rate',fontsize=22)
plt.xlim([0,23])
plt.tight_layout()  

#%%
var = 'cl_top'
plt.figure(figsize=(13,9) )
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    harm3d[exp][var].plot.hist(bins=4,lw=3, color=col[ide],\
                               histtype=u'step',label=lab, density=True)
plt.ylabel(r'Distribution of CL top')
plt.legend(fontsize=25)
plt.tight_layout()  

# plt.figure(figsize=(13,9) )   
# for ide, exp in enumerate(exps):
#     if exp == 'noHGTQS_':
#         lab='Control'
#     elif exp == 'noHGTQS_noSHAL_':
#         lab='NoShal'
#     elif exp == 'noHGTQS_noUVmix_':
#         lab='NoMom'
#     harm3d[exp][var].plot.hist(bins=40,lw=3, color=col[ide],\
#                                               histtype=u'step',label=lab, density=True)
#     plt.axvline(harm3d[exp][var]\
#                 .quantile(0.25),c=col[ide],lw=1.5,ls=sty[ide])
#     plt.axvline(harm3d[exp][var]\
#                     .quantile(0.75),c=col[ide],lw=1.5,ls=sty[ide])
#     plt.axvline(harm3d[exp][var]\
#                 .mean(),c=col[ide],lw=3.5,ls=sty[ide],label='mean')
# plt.ylabel(r'CL top variance (m)')
# plt.legend(fontsize=25)
# plt.tight_layout()       

#%% Pixels from drier to wettest
## you should use integrated LWP available as 'prw' in the 2d fields 
var = 'va'
plt.figure(figsize=(13,9) )
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    lwp = harm3d[exp]['hus'].sel(z=slice(0,5000)).sum('z')
    lwp_deviation = lwp -lwp.mean(dim=('x','y'))
    shear = (harm3d[exp][var].sel(z=2000, method='nearest') - \
             harm3d[exp][var].sel(z=500, method='nearest'))
    shear.groupby_bins(lwp,100).mean().plot(\
                color=col[ide],label=lab,lw=3)
    # harm3d[exp][var].sel(z=1500,method='nearest').\
    #     groupby_bins(lwp,100).mean().plot(\
    #                 color=col[ide],label=lab,lw=2)
    
    
# plt.xlabel(r'Deviation from mean LWP')
# plt.axvline(0,c='k',lw=0.5)
plt.axhline(0,c='k',lw=0.5)
plt.xlabel(r'LWP')
plt.ylabel(r'Shear $m\,s^{-1}$')
plt.title('Difference between winds at 2 km and winds at 500m',fontsize=25)
plt.legend(fontsize=25)
plt.tight_layout()  


#%% Convergence and divergence 



harm3d[exp][var]


#%% aereal snapshot
import geopy
import geopy.distance
idtime= np.datetime64('2020-01-10T19')
plt.figure()
harm3d[exp].sel(time=idtime).cl.sel(z=slice(0,3500)).sum('z').plot()
###
plt.figure()
harm3d[exp].sel(time=idtime).vw.sel(z=slice(0,1500)).mean('z').plot()


ds_pr = xr.open_mfdataset('/Users/acmsavazzi/Documents/WORK/PhD_Year3/DATA/HARMONIE/noHGTQS/pr_his_BES_HA43h22tg3_clim_noHGTQS_1hr_202001110000-202001210000.nc')
var = 'pr'
ii=10

les_centre = [13.3,-54.01]
##medium
Dx = geopy.distance.distance(kilometers = 200)
Dy = geopy.distance.distance(kilometers = 200)
lat_max = Dy.destination(point=les_centre, bearing=0)
lat_min = Dy.destination(point=les_centre, bearing=180)
lon_max = Dx.destination(point=les_centre, bearing=270)
lon_min = Dx.destination(point=les_centre, bearing=90)
medium_ocean =[lat_min[0], lon_min[1], lat_max[0], lon_max[1]]
##large
Dx = geopy.distance.distance(kilometers = 400)
Dy = geopy.distance.distance(kilometers = 400)
lat_max = Dy.destination(point=les_centre, bearing=0)
lat_min = Dy.destination(point=les_centre, bearing=180)
lon_max = Dx.destination(point=les_centre, bearing=270)
lon_min = Dx.destination(point=les_centre, bearing=90)
large_ocean =[lat_min[0], lon_min[1], lat_max[0], lon_max[1]]



small     = [12.39, -58.6, 14.16, -56.86]
medium    = [11.47, -59.61, 15.067, -55.91]
large     = [9.65, -61.5, 16.86, -54.01]


plt.figure()
# ax =ds_pr.isel(time=ii)[var].plot(vmin=0,vmax=1,\
#                     cmap=plt.cm.Blues_r,x='lon',y='lat',\
#                     subplot_kws=dict(projection=proj))
ax = plt.axes(projection=proj)
ax.add_feature(coast, lw=2, zorder=7)
plt.xlim([ds_pr.lon[0,0].values,ds_pr.lon[0,-1].values])
plt.ylim([ds_pr.lat[0,-1].values,ds_pr.lat[-1,-1].values])


## small domain
ax.plot([small[1],small[3]],[small[0],small[0]],c='g',ls='-')
ax.plot([small[1],small[3]],[small[2],small[2]],c='g',ls='-')
ax.plot([small[1],small[1]],[small[0],small[2]],c='g',ls='-')
ax.plot([small[3],small[3]],[small[0],small[2]],c='g',ls='-')
## medium domain
ax.plot([medium[1],medium[3]],[medium[0],medium[0]],c='b',ls='-')
ax.plot([medium[1],medium[3]],[medium[2],medium[2]],c='b',ls='-')
ax.plot([medium[1],medium[1]],[medium[0],medium[2]],c='b',ls='-')
ax.plot([medium[3],medium[3]],[medium[0],medium[2]],c='b',ls='-')
## large domain ocean 
ax.plot([medium_ocean[1],medium_ocean[3]],[medium_ocean[0],medium_ocean[0]],c='b',ls='--')
ax.plot([medium_ocean[1],medium_ocean[3]],[medium_ocean[2],medium_ocean[2]],c='b',ls='--')
ax.plot([medium_ocean[1],medium_ocean[1]],[medium_ocean[0],medium_ocean[2]],c='b',ls='--')
ax.plot([medium_ocean[3],medium_ocean[3]],[medium_ocean[0],medium_ocean[2]],c='b',ls='--')
## large domain
ax.plot([large[1],large[3]],[large[0],large[0]],c='k',ls='-')
ax.plot([large[1],large[3]],[large[2],large[2]],c='k',ls='-')
ax.plot([large[1],large[1]],[large[0],large[2]],c='k',ls='-')
ax.plot([large[3],large[3]],[large[0],large[2]],c='k',ls='-')
## large domain ocean 
ax.plot([large_ocean[1],large_ocean[3]],[large_ocean[0],large_ocean[0]],c='k',ls='--')
ax.plot([large_ocean[1],large_ocean[3]],[large_ocean[2],large_ocean[2]],c='k',ls='--')
ax.plot([large_ocean[1],large_ocean[1]],[large_ocean[0],large_ocean[2]],c='k',ls='--')
ax.plot([large_ocean[3],large_ocean[3]],[large_ocean[0],large_ocean[2]],c='k',ls='--')

gl = ax.gridlines(crs=proj, draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.suptitle(exp)

#%%cross section
crossyz = harm3d[exp].sel(time=idtime).sel(x=152500,y=slice(65000,145000))

###############
### caclculate tendency by differenciating the flux ###
crossyz['v_tend']=crossyz['vw'].differentiate(coord='z')
###### check this function!! 
###############

for section in ['xz','yz']:
    if section =='xz':
        mask = np.nan_to_num(crossyz['cl'].where(crossyz['cl']>0.15).values)

mask[mask > 0] = 3
kernel = np.ones((4,4))
C      = ndi.convolve(mask, kernel, mode='constant', cval=0)
outer  = np.where( (C>=3) & (C<=12 ), 1, 0)
# add variable cloud contour
# works only for 1 time stamp 
if section =='yz':
    crossyz['cloud'] = (('z', 'y'), outer)

plt.figure(figsize=(15,6))
temp = crossyz.coarsen(y=1, boundary='trim').mean()
temp = temp.coarsen(z=1, boundary="trim").mean()
temp = temp.interp(z=np.linspace(temp.z.min(),temp.z.max(), num=30))
im_1a = crossyz['v_tend'].plot(x='y')
im_1b = temp.plot.\
    streamplot('y','z','va_p','wa_p',hue='vw',vmin=-0.001,\
                     density=[0.6, 0.6],\
                    linewidth=3.5,arrowsize=4,\
                arrowstyle='fancy',cmap='PiYG_r')

crossyz['cloud'].where(crossyz['cloud'] > 0).plot(cmap='binary',\
                                add_colorbar=False,vmin=0,vmax=0.5)
cbar = im_1a.colorbar
cbar.remove()
cbar = im_1b.colorbar
cbar.remove()
plt.ylim([0,3500])
plt.tight_layout()


#%% Winds only inside a cloud
fig, axs = plt.subplots(1,3,figsize=(15,11))
for ide, exp in enumerate(exps):
# for ide, exp in enumerate(['noHGTQS_','noHGTQS_noSHAL_']):
    if exp == 'noHGTQS_':
        lab ='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab ='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab ='NoMom'
    Sh_pixels = harm3d[exp].where((harm3d[exp]['cl_top']<1500)&(harm3d[exp]['cl_top']>900))
    Dp_pixels = harm3d[exp].where((harm3d[exp]['cl_top']<3500)&(harm3d[exp]['cl_top']>2500))
    Nc_pixels = harm3d[exp].where((harm3d[exp]['cl_top'].isnull())|(harm3d[exp]['cl_top']>6000))
    Nc_pixels['count'] = ((harm3d[exp]['cl_top'].isnull())|(harm3d[exp]['cl_top']>6000)).sum(('x','y'))
    
    for idx,var in enumerate(['wa','tke','buoy']):
        ### for each scene calculate mean flux in Sh (and Dp) pixels
        #   multiply by the fraction of Sh (and Dp) pixels
        #   average over time 
        
        ## deep pixels 
        Dp_pixels[var].mean(('x','y'))\
            .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                    label=lab+'_Dp',lw=3,c=col[ide],ls='-')
    
        ## shallow pixels 
        Sh_pixels[var].mean(('x','y'))\
            .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                    label=lab+'_Sh',lw=3,c=col[ide],ls='--')
        ## non-cloudy pixels 
        # Nc_pixels[var].mean(('x','y'))\
        #     .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
        #                             label=lab+'_Nc',lw=3,c=col[ide],ls=':')
        axs[idx].axvline(0,c='k',lw=0.5)
        axs[idx].set_ylim([0,4000])
        axs[idx].set_xlabel(r'$m\,s^{-1}$')
        if var == 'ua':
            axs[idx].set_title('Zonal wind',fontsize=24)
        elif var == 'va':
            axs[idx].set_title('Meridional wind',fontsize=24)
        elif var == 'wa':
            axs[idx].set_title('Vertical velocity',fontsize=24)
        elif var == 'tke':
            axs[idx].set_title('TKE',fontsize=24)
            axs[idx].set_xlabel(r'$m^{2}\,s^{-2}$')
        elif var == 'buoy':
            axs[idx].set_title('Buoyancy',fontsize=24)
            axs[idx].set_xlabel(r'? $N$ ?')
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        
# axs[1].set_xlim([-11,0])
# axs[2].set_xlim([-3,0])
# axs[0].set_xlim([-0.05,0.1])
axs[1].get_yaxis().set_visible(False) 
# axs[2].get_yaxis().set_visible(False) 
axs[0].legend(fontsize=18)
plt.tight_layout()

#%% Flux only inside a cloud
fig, axs = plt.subplots(1,2,figsize=(13,11))
for ide, exp in enumerate(exps):
# for ide, exp in enumerate(['noHGTQS_','noHGTQS_noSHAL_']):
    if exp == 'noHGTQS_':
        lab ='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab ='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab ='NoMom'
    Sh_pixels = harm3d[exp].where((harm3d[exp]['cl_top']<1500)&(harm3d[exp]['cl_top']>900))
    Dp_pixels = harm3d[exp].where((harm3d[exp]['cl_top']<3500)&(harm3d[exp]['cl_top']>2500))
    Nc_pixels = harm3d[exp].where((harm3d[exp]['cl_top'].isnull())|(harm3d[exp]['cl_top']>6000))
    Nc_pixels['count'] = ((harm3d[exp]['cl_top'].isnull())|(harm3d[exp]['cl_top']>6000)).sum(('x','y'))
    
    for idx,var in enumerate(['u','v']):
        ### for each scene calculate mean flux in Sh (and Dp) pixels
        #   multiply by the fraction of Sh (and Dp) pixels
        #   average over time 
        
        ## shallow pixels 
        ((Sh_pixels[var+'w']+Sh_pixels[var+'_flx_param_tot']).mean(('x','y')) * Sh_pixels.count(('x','y'))['cl_top']\
                                    /(len(harm3d[exp].x) * len(harm3d[exp].y)))\
            .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                    label=lab+'_Sh',lw=3,c=col[ide],ls='-')
        ## deep pixels 
        ((Dp_pixels[var+'w']+Dp_pixels[var+'_flx_param_tot']).mean(('x','y')) * Dp_pixels.count(('x','y'))['cl_top']\
                                    /(len(harm3d[exp].x) * len(harm3d[exp].y)))\
            .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                    label=lab+'_Dp',lw=3,c=col[ide],ls=':')
        ## non-cloudy pixels 
        # ((Nc_pixels[var+'w']+Nc_pixels[var+'_flx_param_tot']).mean(('x','y')) * Nc_pixels['count']\
        #                             /(len(harm3d[exp].x) * len(harm3d[exp].y)))\
        #     .mean('time').plot(y='z',ax=axs[idx],\
        #                             label=lab+'_Nc',lw=3,c=col[ide],ls=':')
        axs[idx].axvline(0,c='k',lw=0.5)
        axs[idx].set_ylim([0,4000])
        if var == 'u':
            axs[idx].set_title('Zonal momentum flux',fontsize=22)
        elif var == 'v':
            axs[idx].set_title('Meridional momentum flux',fontsize=22)
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        axs[idx].set_xlabel(r'$m^{2}\,s^{-2}$')
# axs[0].set_xlim([-0.009,0.003])
# axs[1].set_xlim([-0.0012,0.0022])
axs[1].get_yaxis().set_visible(False) 
axs[0].legend(fontsize=18)
plt.tight_layout()
   
# var = 'uw'
# plt.figure(figsize=(6,10))
# Sh_pixels[var].mean(('x','y','time')).plot(y='z',label='Sh',lw=3)
# Dp_pixels[var].mean(('x','y','time')).plot(y='z',label='Dp',lw=3)
# plt.axvline(0,c='k',lw=0.5)
# plt.legend()
# plt.ylim([0,4000])

## how many Sh pixels and how many Dp pixels?
# plt.figure(figsize=(10,6))
# (Sh_pixels.count(('x','y'))['cl_top']\
#                             /(len(Sh_pixels.x) * len(Sh_pixels.y))).plot(label='Sh',lw=3)
# (Dp_pixels.count(('x','y'))['cl_top']\
#                             /(len(Sh_pixels.x) * len(Sh_pixels.y))).plot(label='Dp',lw=3)

# (Nc_pixels['count']/(len(Sh_pixels.x) * len(Sh_pixels.y))).plot(label='Nc',lw=3)
# plt.title('Percentage of Sh and Dp pixels')
# plt.ylabel('%')
# plt.legend()

#%% Shear
## Do all experiments have the same amount of forward and backward shear?
    # or they have different distributions of shear?




#%%
print('End.')