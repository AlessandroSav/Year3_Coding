#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:31:52 2024

@author: acmsavazzi
"""

#%% EUREC4A Observations 


#%%                             Libraries
###############################################################################
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from datetime import datetime, timedelta
from intake import open_catalog
import dask
import dask.array as da
dask.config.set({"array.slicing.split_large_chunks": True})


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
############################################################
############################################################
#%% Iport Observations
cat = open_catalog("https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml")

#%%


joanne = cat.dropsondes.JOANNE.level3.to_dask()

goes = cat.satellites.GOES16['latlongrid'].to_dask()
goes_all = xr.concat([cat.satellites.GOES16['latlongrid'](date=d).to_dask().chunk() for d in pd.date_range('2020-01-12','2020-01-14')], dim='time')


#%%

domain_name = 'BES'
lat_select  = 13.2806    # HALO center 
lon_select  = -57.7559   # HALO center 

srt_time    = np.datetime64('2020-01-03T00:30')
end_time    = np.datetime64('2020-01-29T23')

month='0*'
plot=False
apply_filter = False

# exps = ['noHGTQS_','noHGTQS_noSHAL_']
exps = ['noHGTQS_','noHGTQS_noUVmix_','noHGTQS_noSHAL_']
col=['k','r','g']
# col=['k','r']
sty=['--','-',':']
# sty=['--','-']

levels = 'z'      ## decide wether to open files model level (lev) or 
                    ## already interpolate to height (z)
my_harm_dir     = '/Users/acmsavazzi/Documents/WORK/PhD_Year3/DATA/HARMONIE/'
ifs_dir         = '/Users/acmsavazzi/Documents/WORK/Research/MyData/'
save_dir        = '/Users/acmsavazzi/Documents/WORK/PhD_Year3/Figures/HARMONIE/'
#%%

####################################
file3d = my_harm_dir+'noHGTQS/noHGTQS_01_3d_200_z_all.nc'
harm3d = xr.open_mfdataset(file3d, combine='by_coords',chunks={'time':24})
harm3d = harm3d.drop_duplicates(dim="time", keep="first")
####################################    
# convert model levels to height levels
harm3d = harm3d.rename({'lev':'z'})
harm3d['z'] = harm3d.z.assign_attrs(units='m',long_name='Height')
harm3d = harm3d.sortby('z')
#remove first 2 days 
harm3d = harm3d.sel(time=slice(srt_time,end_time))
# convert to local time
harm3d['time'] = harm3d['time'] - np.timedelta64(4, 'h')
harm3d.time.attrs["units"] = "Local Time"

harm2d = xr.open_mfdataset(my_harm_dir+'noHGTQS/noHGTQS_harm2d_200.nc', combine='by_coords',chunks={'time':10*24})


#%% Import organisation metrics
ds_org = {}
for exp in exps:    
    fileorg = my_harm_dir+'df_metrics_'+exp[:-1]+'.h5'    
    ds_org[exp] = pd.read_hdf(fileorg)
    ds_org[exp] = ds_org[exp].astype(np.float64)
    ds_org[exp] = ds_org[exp].to_xarray()
    ds_org[exp] = ds_org[exp].rename({'index':'time'})
    ds_org[exp] = ds_org[exp].sel(time=slice(srt_time,end_time))
    # convert to local time
    ds_org[exp]['time'] = ds_org[exp]['time'] - np.timedelta64(4, 'h')
    ds_org[exp].time.attrs["units"] = "Local Time"
    
    

exp = exps[0]

fileorg = my_harm_dir+'df_metrics_cc_4km_'+exp[:-1]+'.nc'    
ds_org_4km = xr.open_mfdataset(fileorg, combine='by_coords')
ds_org_4km['time'] = ds_org[exp]['time']
ds_org_4km = ds_org_4km.drop('index')
ds_org_4km = ds_org_4km.interpolate_na('time')    

fileorg_smoc_conv = my_harm_dir+exp[:-1]+'/'+exp[:-1]+'_smoc_conv_metrics_5klp.nc'
fileorg_smoc_dive = my_harm_dir+exp[:-1]+'/'+exp[:-1]+'_smoc_dive_metrics_5klp.nc'

ds_org_smoc_conv = xr.open_mfdataset(fileorg_smoc_conv, combine='by_coords')
ds_org_smoc_conv = ds_org_smoc_conv.set_index(time='index')

ds_org_smoc_dive = xr.open_mfdataset(fileorg_smoc_dive, combine='by_coords')
ds_org_smoc_dive = ds_org_smoc_dive.set_index(time='index')

#%% time of large and smole SMOCS

tm_cf_large = ds_org_smoc_conv.where(\
              ds_org_smoc_conv.compute()['mean_length_scale']>\
              ds_org_smoc_conv['mean_length_scale'].compute().median(),drop=True)['time']
    
tm_cf_small = ds_org_smoc_conv.where(\
              ds_org_smoc_conv.compute()['mean_length_scale']<=\
              ds_org_smoc_conv['mean_length_scale'].compute().median(),drop=True)['time']
    
# tm_cf_large = ds_org_smoc_dive.where(\
#               ds_org_smoc_dive.compute()['mean_length_scale']>\
#               ds_org_smoc_dive['mean_length_scale'].compute().median(),drop=True)['time']
# tm_cf_small = ds_org_smoc_dive.where(\
#               ds_org_smoc_dive.compute()['mean_length_scale']<=\
#               ds_org_smoc_dive['mean_length_scale'].compute().median(),drop=True)['time']
#%% Wind tendency from shallow convective parameterisation 
n_xplots = 3
n_yplots = 2
cmap_ = cm.coolwarm
sc_layer_base = 0
sc_layer_top = 600
c_layer_base = 900
c_layer_top = 1500

para_res = 'conv'

sel_time=harm2d.time

# sel_time = harm2d['time'].where\
#     (harm2d['u_flx_param_tot'].compute().sel(z=700,method='nearest')<\
#      harm2d['u_flx_param_tot'].compute().sel(z=700,method='nearest').median(),\
#          drop=True).values

# sel_time = tm_cf_large

fig, axs = plt.subplots(n_yplots,n_xplots,figsize=(11,11))
for idx, var in enumerate(['u','v']):
    if var == 'u':
        write_title = 'Zonal tendency'
    elif var == 'v':
        write_title = 'Meridional tendency'
    ## Winds
    # # harm2d[wind_var].sel(z=slice(0,4000)).mean(('x','y','time'))\
    # #         .plot(y='z',lw=1.5,ax=axs[idx,0],c='k',label='Mean')
    # harm2d[var+'_on_conv'].sel(time=sel_time).sel(z=slice(10,4000)).mean('time')\
    #         .plot(y='z',lw=2,ax=axs[idx,0],c='orangered',label=r'$D_{sc}>0$')
    # harm2d[var+'_on_dive'].sel(time=sel_time).sel(z=slice(10,4000)).mean('time')\
    #         .plot(y='z',lw=2,ax=axs[idx,0],c='royalblue',label=r'$D_{sc}<0$')
    # harm2d[var+'_off_smoc'].sel(time=sel_time).sel(z=slice(10,4000)).mean('time')\
    #         .plot(y='z',lw=3,ax=axs[idx,0],c='olive',label=r'NoSMOC')    
    
    ## Fluxes
    # harm2d[flx_var].sel(z=slice(0,4000)).mean(('x','y','time'))\
    #         .plot(y='z',lw=1.5,ax=axs[idx,0],c='k',label='Mean')
    harm2d[var+'flx_'+para_res+'_on_conv'].sel(time=sel_time).sel(z=slice(0,4000)).mean('time')\
            .plot(y='z',lw=2,ax=axs[idx,0],c='orangered',label=r'$D_{sc}>0$')
    harm2d[var+'flx_'+para_res+'_on_dive'].sel(time=sel_time).sel(z=slice(0,4000)).mean('time')\
            .plot(y='z',lw=2,ax=axs[idx,0],c='royalblue',label=r'$D_{sc}<0$')
    harm2d[var+'flx_'+para_res+'_off_smoc'].sel(time=sel_time).sel(z=slice(0,4000)).mean('time')\
            .plot(y='z',lw=3,ax=axs[idx,0],c='olive',label=r'NoSMOC')

    ## Tendency 
    ## Tendency normalised by the wind
    # (harm2d[var].sel(z=slice(0,4000)).mean(('x','y','time'))\
    #     /harm2d[wind_var].sel(z=slice(0,4000)).mean(('x','y','time')))\
    #         .plot(y='z',lw=1.5,ax=axs[idx,1],c='k',label='Mean')
    (harm2d[var+'tend_'+para_res+'_on_conv'].sel(time=sel_time).sel(z=slice(0,4000)).mean('time')\
        /(harm2d[var+'_on_conv']/np.abs(harm2d[var+'_on_conv'])).sel(time=sel_time).sel(z=slice(0,4000)).mean('time'))\
            .plot(y='z',lw=2,ax=axs[idx,1],c='orangered',label=r'$D_{sc}>0$')
    (harm2d[var+'tend_'+para_res+'_on_dive'].sel(time=sel_time).sel(z=slice(0,4000)).mean('time')\
        /(harm2d[var+'_on_dive']/np.abs(harm2d[var+'_on_dive'])).sel(time=sel_time).sel(z=slice(0,4000)).mean('time'))\
            .plot(y='z',lw=2,ax=axs[idx,1],c='royalblue',label=r'$D_{sc}<0$')
    (harm2d[var+'tend_'+para_res+'_off_smoc'].sel(time=sel_time).sel(z=slice(0,4000)).mean('time')\
        /(harm2d[var+'_off_smoc']/np.abs(harm2d[var+'_off_smoc'])).sel(time=sel_time).sel(z=slice(0,4000)).mean('time'))\
            .plot(y='z',lw=3,ax=axs[idx,1],c='olive',label=r'NoSMOC')
            
    ## K = u'w' du/dz 
    ## Tendency normalised by the wind
    # (harm2d[var].sel(z=slice(0,4000)).mean(('x','y','time'))\
    #     /harm2d[wind_var].sel(z=slice(0,4000)).mean(('x','y','time')))\
    #         .plot(y='z',lw=1.5,ax=axs[idx,2],c='k',label='Mean')
    (harm2d[var+'flx_'+para_res+'_on_conv'].sel(time=sel_time).mean('time')\
     *harm2d[var+'_on_conv'].differentiate('z').sel(time=sel_time).mean('time'))\
             .sel(z=slice(0,4000))\
            .plot(y='z',lw=2,ax=axs[idx,2],c='orangered',label=r'$D_{sc}>0$')
    (harm2d[var+'flx_'+para_res+'_on_dive'].sel(time=sel_time).mean('time')\
     *harm2d[var+'_on_dive'].differentiate('z').sel(time=sel_time).mean('time'))\
             .sel(z=slice(0,4000))\
            .plot(y='z',lw=2,ax=axs[idx,2],c='royalblue',label=r'$D_{sc}<0$')
    (harm2d[var+'flx_'+para_res+'_off_smoc'].sel(time=sel_time).mean('time')\
     *harm2d[var+'_off_smoc'].differentiate('z').sel(time=sel_time).mean('time'))\
             .sel(z=slice(0,4000))\
            .plot(y='z',lw=3,ax=axs[idx,2],c='olive',label=r'NoSMOC')
    ##
    axs[idx,1].set_yticks([])
    axs[idx,1].set_ylabel('')
    axs[idx,2].set_yticks([])
    axs[idx,2].set_ylabel('')
    for idy in [0,1,2]:
        axs[idx,idy].axvline(0,c='grey',lw=0.5)
        axs[idx,idy].set_ylim([0,4000])
        axs[idx,idy].axhspan(sc_layer_base, sc_layer_top, alpha=0.1, color='grey')
        axs[idx,idy].axhspan(c_layer_base, c_layer_top, alpha=0.1, color='grey')
    # axs[idx,2].set_xlim([-0.05,0.03])
    # axs[idx,2].set_xticks(np.linspace(-0.03, 0.03, 3))
    ##
axs[0,0].set_xlim([-0.007,0.025])
axs[1,0].set_xlim([-0.004,0.003])
axs[0,1].set_xlim([-0.14,0.036])
axs[1,1].set_xlim([-0.09,0.04])
axs[0,2].set_xlim([-2.7e-5,0.9e-5])
axs[1,2].set_xlim([-0.3e-5,0.15e-5])
axs[1,0].set_xticks(np.linspace(-0.003, 0.003, 3))
axs[0,0].set_xlabel('')
axs[0,1].set_xlabel('')
axs[0,2].set_xlabel('')
# axs[0,0].set_title('Zonal wind',fontsize=20)
# axs[1,0].set_title('Meridional wind',fontsize=20)
axs[0,0].set_title(r"$u'w'$ "+para_res+'.',fontsize=20)
axs[1,0].set_title(r"$v'w'$ "+para_res+'.',fontsize=20)
axs[0,1].set_title(r"$\frac{\bar{u}}{|\bar{u}|}\ \frac{du'w'}{dz}$ "+para_res+'.',fontsize=20)
axs[1,1].set_title(r"$\frac{\bar{v}}{|\bar{v}|}\ \frac{dv'w'}{dz}$ "+para_res+'.',fontsize=20)
axs[0,2].set_title(r"$u'w'\ \frac{d\bar{u}}{dz}$ "+para_res+'.',fontsize=20)
axs[1,2].set_title(r"$v'w'\ \frac{d\bar{v}}{dz}$ "+para_res+'.',fontsize=20)
# axs[idx,0].set_xlabel(r'$m s^{-1}$')
axs[idx,0].set_xlabel(r'$m^2 s^{-2}$')
axs[idx,1].set_xlabel(r'$m s^{-2}$')
axs[idx,2].set_xlabel(r'$m^2 s^{-3}$')
# axs[idx,1].set_xlabel(r'$hour^{-1}$')
axs[0,0].legend(fontsize=14)
plt.tight_layout()
# plt.savefig(save_dir+para_res+'_flx_byregion.pdf')
#%%




