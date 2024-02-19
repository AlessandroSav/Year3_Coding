#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon October 2 15:50:49 2021

@author: acmsavazzi
"""
#%% compute_cl_metrics.py

#%%                             Libraries
###############################################################################
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import os
from glob import glob
import sys
from datetime import datetime, timedelta
from netCDF4 import Dataset
import cloudmetrics
import inspect
my_source_dir = os.path.abspath('{}/../../../../My_source_codes')
sys.path.append(my_source_dir)
import dask
dask.config.set({"array.slicing.split_large_chunks": True})

#%% initial 
exps = ['noHGTQS_','noHGTQS_noUVmix_','noHGTQS_noSHAL_']
exps = ['Goes16']
srt_time    = np.datetime64('2020-01-03T00:30')
end_time    = np.datetime64('2020-01-29T23')
month='0*'
domain_sml  = 200            # km

mask_on_clouds = False
mask_on_smoc_conv = False
mask_on_smoc_dive = False
mask_on_goes16 =True
## If the subcloud layer is weakly converging or weakly diverging disregard the smoc:
threshold_div = 0.1e-5 ## threshold for subcloud layer convergence and divergence.
threshold_cl = 0.5
####################
## select which level to use to calculate the metrcis ##
cc_level = 'cc_4km'
## select which resolution to use for mesoscale divergence  ##
klp = 5
iz  = 200
####################

levels = 'z'      ## decide wether to open files model level (lev) or 
                    ## already interpolate to height (z)
my_harm_dir     = '/Users/acmsavazzi/Documents/WORK/PhD_Year3/DATA/HARMONIE/'
#%% Import Harmonie
if mask_on_goes16:
    print('Reading Goes16')
    goes16 = xr.open_mfdataset(my_harm_dir+'Goes16_interp.nc',
                               combine='by_coords',chunks={'time':48})
    
else:
    print("Reading HARMONIE.") 
    ## new files on height levels are empty ### !!!
    ## is it a problem of the interpolation? if yes: open the file _lev_all.nc 
    ## and do the intrpolation here. 
    harm3d   = {}
    filtered = {}
    for exp in exps:
        ####################################
        file3d = my_harm_dir+exp[:-1]+'/'+exp+month+'_3d_'+str(domain_sml)+'*'+levels+'*.nc'
        harm3d[exp] = xr.open_mfdataset(file3d, combine='by_coords',chunks={'time':24})
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
    
    
        ### import filtered fields 
        file_filtered = my_harm_dir+exp[:-1]+'/filtered_'+exp[:-1]+'.nc'
        filtered[exp] = xr.open_mfdataset(file_filtered, combine='by_coords',chunks={'time':24})
        
        ###############
        ### Cloud top height 
        # define cloud top in each pixel using Cloud Area Fraction (cl) 
        ### calculate cloud cover below 4km
        harm3d[exp]['cc_4km']  = (harm3d[exp]['cl'].sel(z=slice(0,4000))>threshold_cl).any(dim='z') *1
        ### calculate cloud cover below 2.5km (this is to check with standard output of low cloud cover CLL)
        harm3d[exp]['cc_2_5km']  = (harm3d[exp]['cl'].sel(z=slice(0,2500))>threshold_cl).any(dim='z') *1
        ### calculate cloud cover below 1.5km ()
        harm3d[exp]['cc_1_5km']  = (harm3d[exp]['cl'].sel(z=slice(0,1500))>threshold_cl).any(dim='z') *1
        ### calculate cloud cover below 1km ()
        harm3d[exp]['cc_1km']  = (harm3d[exp]['cl'].sel(z=slice(0,1000))>threshold_cl).any(dim='z') *1
        ### calculate cloud cover between 1 and  4km
        harm3d[exp]['cc_1to4km'] = harm3d[exp]['cc_4km'] - harm3d[exp]['cc_1km']
        ### calculate cloud cover between 1.5 and  4km
        harm3d[exp]['cc_1_5to4km'] = harm3d[exp]['cc_4km'] - harm3d[exp]['cc_1_5km']

#%% Calculate organisation metrics on layers of clouds
metrics = [
           'cloud_fraction',
           'fractal_dimension',
           'open_sky',
           'cop',
           'iorg',
           'scai',
           'max_length_scale',
           'mean_eccentricity',
           'mean_length_scale',
           'mean_perimeter_length',
           'num_objects',
           'orientation',
           'spectral_length_moment',
           'spectral_anisotropy',
           'spectral_slope',
           'woi1',
           'woi2',
           'woi3',
           'mean',
           'var',
           'cth',
           'cthVar'
          ]

# These are the metrics you can choose from
available_mask_metrics = dict(inspect.getmembers(cloudmetrics.mask, inspect.isfunction))
available_object_metrics = dict(inspect.getmembers(cloudmetrics.objects, inspect.isfunction))
available_scalar_metrics = dict(inspect.getmembers(cloudmetrics.scalar, inspect.isfunction))

for exp in exps:
    print("Processing exp "+exp) 
    if   mask_on_clouds == True:
        df_metrics = pd.DataFrame(index=harm3d[exp].time.values, columns=metrics)
    elif (mask_on_smoc_conv    == True) | (mask_on_smoc_dive    == True):
        df_metrics = pd.DataFrame(index=filtered[exp].time.values, columns=metrics)
    elif mask_on_goes16 == True:
        df_metrics = pd.DataFrame(index=goes16.time.values, columns=metrics)
        
    for ix, ii in enumerate(df_metrics.index):
        if ii.hour in [0,12]:
            print("Done to "+str(ii)[:-3]) 
        
        if  mask_on_goes16 == True:
            cloud_mask = xr.where(goes16['cl_mask']>10,1,0).sel(time=ii).values
        elif   mask_on_clouds == True:
            cloud_mask = harm3d[exp][cc_level].sel(time=ii).values
        elif (mask_on_smoc_conv    == True) | (mask_on_smoc_dive    == True):
            ## Define divergence anomaly in the subcloud layer 
            filtered[exp]['Dsc'] = filtered[exp]['div_f'].sel(z=slice(0,600)).mean('z') - \
                        filtered[exp]['div_f'].sel(z=slice(0,600)).mean('z').mean(('x','y'))
            ## Define divergence anomaly in the cloud layer 
            filtered[exp]['Dc']  = filtered[exp]['div_f'].sel(z=slice(900,1500)).mean('z') - \
                        filtered[exp]['div_f'].sel(z=slice(900,1500)).mean('z').mean(('x','y'))
            ## identify dipols / smocs
            filtered[exp]['smoc'] = (filtered[exp]['Dsc']/filtered[exp]['Dc'])<0
            ## distinguis converging and diverging smocs
            filtered[exp]['smoc_conv'] = filtered[exp]['smoc'].where((filtered[exp]['Dsc']<-threshold_div) & (filtered[exp]['smoc']>0))
            filtered[exp]['smoc_dive'] = filtered[exp]['smoc'].where((filtered[exp]['Dsc']>+threshold_div) & (filtered[exp]['smoc']>0))
            # cloud_mask = (filtered[exp].where(filtered[exp]['div_f']>\
            #                                   threshold_div).notnull()\
            #               .sel(time=ii,klp=klp).sel(z=iz,method='nearest')['div_f']).values
            if mask_on_smoc_conv      == True:
                cloud_mask = filtered[exp]['smoc_conv'].sel(time=ii,klp=klp).notnull().values
            elif mask_on_smoc_dive    == True:
                cloud_mask = filtered[exp]['smoc_dive'].sel(time=ii,klp=klp).notnull().values
            else: print('Mask type not defiend. Unable to compute metrics')
            
        # Continue if there aren't any clouds
        if np.sum(cloud_mask) < 1:
            continue

        # Compute selected metrics
        computed_object_labels = False
        computed_spectra = False
        
        for j in range(len(metrics)):
            ### Cloud object metrics ###
            if metrics[j] in available_object_metrics.keys():
                # Compute object labels if not done yet
                if not computed_object_labels:
                    object_labels = cloudmetrics.objects.label(cloud_mask)
                    computed_object_labels = True
                # Compute metric
                fn_metric = available_object_metrics[metrics[j]]
                df_metrics.iloc[ix, df_metrics.columns.get_loc(metrics[j])] = fn_metric(object_labels)
                
            ### Cloud mask metrics ###
            elif metrics[j] in available_mask_metrics.keys():
                fn_metric = available_mask_metrics[metrics[j]]
                # Open sky exception - just take the mean open sky area (second function output)
                if 'open_sky' in metrics[j]:
                    _, df_metrics.iloc[ix, df_metrics.columns.get_loc(metrics[j])] = fn_metric(cloud_mask)
                else:
                    df_metrics.iloc[ix, df_metrics.columns.get_loc(metrics[j])] = fn_metric(cloud_mask)

            ### Cloud scalar metrics ###
            elif metrics[j] in available_scalar_metrics.keys():
                fn_metric = available_scalar_metrics[metrics[j]]
                # Spectral metrics exception
                if 'spectral' in metrics[j]:

                    # Compute spectra if not done yet
                    if not computed_spectra:
                        wavenumbers, psd_1d_radial, psd_1d_azimuthal = cloudmetrics.scalar.compute_spectra(cloud_mask)
                        computed_spectra = True

                    # Compute metrics
                    if 'anisotropy' in metrics[j]:
                        df_metrics.iloc[ix, df_metrics.columns.get_loc(metrics[j])] = fn_metric(psd_1d_azimuthal)
                    else:
                        df_metrics.iloc[ix, df_metrics.columns.get_loc(metrics[j])] = fn_metric(wavenumbers, psd_1d_radial)

                # All other scalar metrics computed normally
                else:
                    df_metrics.iloc[ix, df_metrics.columns.get_loc(metrics[j])] = fn_metric(cloud_mask)
    #Store
    print("Saving exp "+exp) 
    df_metrics = df_metrics.astype(np.float64)
    df_metrics = df_metrics.to_xarray()
    df_metrics = df_metrics.rename_dims({'index':'time'})
    df_metrics.time.attrs["units"] = "Local Time"

    if mask_on_clouds == True:
        df_metrics.to_netcdf(my_harm_dir+'df_metrics_'+cc_level+'_'+exp[:-1]+'.nc')
    elif mask_on_smoc_conv == True:
        df_metrics.to_netcdf(my_harm_dir+'smoc_conv_metrics_'+str(klp)+'klp_'+exp[:-1]+'.nc')
    elif mask_on_smoc_dive == True:
        df_metrics.to_netcdf(my_harm_dir+'smoc_dive_metrics_'+str(klp)+'klp_'+exp[:-1]+'.nc')
    elif mask_on_goes16 == True:
        df_metrics.to_netcdf(my_harm_dir+'Goes16_metrics.nc')
#%%
print('End.')