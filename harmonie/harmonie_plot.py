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
import geopy
import geopy.distance
from datetime import datetime, timedelta
from netCDF4 import Dataset
from scipy import ndimage as ndi

my_source_dir = os.path.abspath('{}/../../../../My_source_codes')
sys.path.append(my_source_dir)
from My_thermo_fun import *
sys.path.insert(1, '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Coding/Scale_separation/')
from functions import *

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

def convert_rain_intensity(intensity_kg_kg):
    density_water_vapor = 0.9  # kg/m³
    conversion_factor = 1000 / 3600  # Conversion from kg/m³ to g/m³ and seconds to hours

    intensity_mm_hour = intensity_kg_kg * density_water_vapor * conversion_factor
    return intensity_mm_hour
#%% initial 
dt          = 75               # model  timestep [seconds]
step        = 3600           # output timestep [seconds]
domain_name = 'BES'
lat_select  = 13.2806    # HALO center 
lon_select  = -57.7559   # HALO center 

domain_sml  = 200           # km
domain_med  = 400
grid        = 2.5 # km
srt_time    = np.datetime64('2020-01-03T00:30')
end_time    = np.datetime64('2020-01-29T23')

months = ['01',]
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

##medium domain
buffer_sml = int(domain_sml/(2*grid)) 
buffer_med = int(domain_med/(2*grid))  

# 100km to the east 
Dx = geopy.distance.distance(kilometers = 100)
new_centre = Dx.destination(point=[lat_select,lon_select], bearing=90)
Dx = geopy.distance.distance(kilometers = 200)
Dy = geopy.distance.distance(kilometers = 200)
lat_max = Dy.destination(point=new_centre, bearing=0)
lat_min = Dy.destination(point=new_centre, bearing=180)
lon_max = Dx.destination(point=new_centre, bearing=270)
lon_min = Dx.destination(point=new_centre, bearing=90)
medium_ocean =[lat_min[0], lon_min[1], lat_max[0], lon_max[1]]
#%%
print("Reading ERA5.") 
era5=xr.open_mfdataset(ifs_dir+'My_ds_ifs_ERA5.nc',chunks={'Date':24})
era5['Date'] = era5.Date - np.timedelta64(4, 'h')
era5.Date.attrs["units"] = "Local_Time"

#%% Import Harmonie
### Import Harmonie data
print("Reading HARMONIE.") 
## new files on height levels are empty ### !!!
## is it a problem of the interpolation? if yes: open the file _lev_all.nc 
## and do the intrpolation here. 
harm2d={}
harm3d={}
harm_srf={}
harm_srf_sml = {}
harm_srf_med  = {}
filtered = {}
spectral = {}
for exp in exps:
    file2d = my_harm_dir+exp[:-1]+'/'+exp+month+'_avg'+str(domain_sml)+'*'+levels+'*.nc'
    harm2d[exp] = xr.open_mfdataset(file2d, combine='by_coords',chunks={'time':24})
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

    # read surface 2d fields
    # file_srf = my_harm_dir+exp[:-1]+'/'+exp+month+'_2d_'+str(domain_sml)+'.nc'
    file_srf = my_harm_dir+exp[:-1]+'/'+exp+month+'_2d_1100.nc'
    harm_srf[exp] = xr.open_mfdataset(file_srf, combine='by_coords',chunks={'time':24})
    harm_srf[exp]['time'] = np.sort(harm_srf[exp]['time'].values)
    # drop duplicate hour between the 2 months 
    harm_srf[exp] = harm_srf[exp].drop_duplicates(dim='time',keep='first')
    #remove first 2 days 
    harm_srf[exp] = harm_srf[exp].sel(time=slice(srt_time,end_time))
    # convert to local time
    harm_srf[exp]['time'] = harm_srf[exp]['time'] - np.timedelta64(4, 'h')
    harm_srf[exp].time.attrs["units"] = "Local Time"
    
    
    # select a smaller area 
    j,i = np.unravel_index(np.sqrt((harm_srf[exp].lon-lon_select)**2 + (harm_srf[exp].lat-lat_select)**2).argmin(), harm_srf[exp].lon.shape)
    harm_srf_sml[exp] = harm_srf[exp].isel(x=slice(i-buffer_sml,i+buffer_sml),y=slice(j-buffer_sml,j+buffer_sml))
    # select a medium area 
    j,i = np.unravel_index(np.sqrt((harm_srf[exp].lon-new_centre[1])**2 + (harm_srf[exp].lat-lat_select)**2).argmin(), harm_srf[exp].lon.shape)
    harm_srf_med[exp] = harm_srf[exp].isel(x=slice(i-buffer_med,i+buffer_med),y=slice(j-buffer_med,j+buffer_med))
    
    #get rid of useles variables
    if 'Lambert_Conformal' in harm_srf_sml[exp]:
        harm_srf_sml[exp] = harm_srf_sml[exp].drop(['Lambert_Conformal'])
        harm_srf_med[exp] = harm_srf_med[exp].drop(['Lambert_Conformal'])
        harm_srf[exp] = harm_srf[exp].drop(['Lambert_Conformal'])
    if 'time_bnds' in harm_srf_sml[exp]:
        harm_srf_sml[exp] = harm_srf_sml[exp].drop(['time_bnds'])
        harm_srf_med[exp] = harm_srf_med[exp].drop(['time_bnds'])
        harm_srf[exp] = harm_srf[exp].drop(['time_bnds'])
    
    ### import filtered fields 
    filtered[exp] = xr.open_mfdataset(my_harm_dir+exp[:-1]+'/filtered_'+exp[:-1]+'.nc',chunks={'time':24})
    spectral[exp] = xr.open_mfdataset(my_harm_dir+exp[:-1]+'/spectral_'+exp[:-1]+'.nc',chunks={'time':24})
    
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
    
    
ds_org_4km = {}
ds_org_div = {}
ds_org_smoc_conv = {}
ds_org_smoc_dive = {}
for exp in exps:
    fileorg = my_harm_dir+'df_metrics_cc_4km_'+exp[:-1]+'.nc'    
    ds_org_4km[exp] = xr.open_mfdataset(fileorg, combine='by_coords')
    ds_org_4km[exp]['time'] = ds_org[exp]['time']
    ds_org_4km[exp] = ds_org_4km[exp].drop('index')
    ds_org_4km[exp] = ds_org_4km[exp].interpolate_na('time')    
    
    fileorg_smoc_conv = my_harm_dir+exp[:-1]+'/'+exp[:-1]+'_smoc_conv_metrics_5klp.nc'
    # fileorg_smoc_dive = my_harm_dir+exp[:-1]+'/'+exp[:-1]+'_smoc_dive_metrics_5klp.nc'
    
    ds_org_smoc_conv[exp] = xr.open_mfdataset(fileorg_smoc_conv, combine='by_coords')
    ds_org_smoc_conv[exp] = ds_org_smoc_conv[exp].set_index(time='index')
    
    # ds_org_smoc_dive[exp] = xr.open_mfdataset(fileorg_smoc_dive, combine='by_coords')
    # ds_org_smoc_dive[exp] = ds_org_smoc_dive[exp].set_index(time='index')
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
        
### grouping by parameterised mom flux in control exp
# group = 'tau_par'
# mom_flux_q2 = harm3d[exps[0]]['tau_par'].sel(z=slice(0,1500)).chunk({'time': -1}).mean(('x','y','z')).quantile(0.5,dim=('time')).values
# time_g1[group] = harm3d[exps[0]].where((harm3d[exps[0]]['tau_par'].\
#                                         sel(z=slice(0,1500)).chunk({'time': -1}).\
#                                             mean(('x','y','z'))\
#                                         <= mom_flux_q2).compute(),drop=True).time
# time_g2[group] = harm3d[exps[0]].where((harm3d[exps[0]]['tau_par'].\
#                                         sel(z=slice(0,1500)).chunk({'time': -1}).\
#                                             mean(('x','y','z'))\
#                                         > mom_flux_q2).compute(),drop=True).time


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
    for var in ['ua','va','wa','hus','ta']:
        harm3d[exp][var+'_p'] = harm3d[exp][var] - harm3d[exp][var].mean(['x','y'])
    
    harm3d[exp]['uw']= harm3d[exp]['ua_p']*harm3d[exp]['wa_p']
    harm3d[exp]['vw']= harm3d[exp]['va_p']*harm3d[exp]['wa_p']
    harm3d[exp]['tw']= harm3d[exp]['ta_p']*harm3d[exp]['wa_p']
    harm3d[exp]['qw']= harm3d[exp]['hus_p']*harm3d[exp]['wa_p']
    
    for id_ds, ds in enumerate([harm2d[exp],harm3d[exp]]): 
        if id_ds ==0:
            vars = ['u','v','Thl','rt']
        else: 
            vars = ['u','v']
        for var in vars:
            ## save a variable for total parameterised momentum flux
            if (exp == 'noHGTQS_noUVmix_') & (var in ['Thl','rt']):
                print('Param. '+var+' fluxes missing in '+exp)
            elif (exp == 'noHGTQS_noUVmix_') & (var in ['u','v']):
                #deaccumulate 
                ds[var+'flx_turb'] = ds[var+'flx_turb'].diff('time') * step**-1
                ds[var+'flx_conv_dry'] = ds[var+'flx_conv_dry'].diff('time') * step**-1
                # sum parameterised components 
                ds[var+'_flx_param_tot']=ds[var+'flx_turb']+\
                                         ds[var+'flx_conv_dry']      
            else:
                #deaccumulate 
                ds[var+'flx_turb'] = ds[var+'flx_turb'].diff('time') * step**-1
                ds[var+'flx_conv_dry'] = ds[var+'flx_conv_dry'].diff('time') * step**-1
                if var == 'Thl':
                    ds[var+'flx_conv_moist'] = ds[var+'flx_conv_mois'].diff('time') * step**-1
                else:
                    ds[var+'flx_conv_moist'] = ds[var+'flx_conv_moist'].diff('time') * step**-1
                # sum parameterised components 
                ds[var+'_flx_param_tot']=ds[var+'flx_turb']+\
                  ds[var+'flx_conv_moist']+\
                  ds[var+'flx_conv_dry']
        if id_ds ==0:
            harm2d[exp]=ds
        elif id_ds==1:
            harm3d[exp]=ds
            
    ### Convergence and divergence ['dudx']+['dvdy']
    harm3d[exp]['div']   = harm3d[exp]['ua_p'].differentiate('x')   + harm3d[exp]['va_p'].differentiate('y')
#%% Cloud top height 
# define cloud top in each pixel using Cloud Area Fraction (cl)
var = 'cl'
thres = 0.5     # /kg
for exp in exps:  
    #height of zero cloud fraction after maximum
    zmax = harm3d[exp][var].sel(z=slice(0,5000)).idxmax('z')  # height of max cloud cover
    temp = harm3d[exp][var].sel(z=slice(0,5000)).where(harm3d[exp]['z']>=zmax)
    harm3d[exp][var+'_top'] = temp.where(lambda x: x>thres).idxmax(dim='z') 
    # exclude areas with no clouds (cloud top below 500 m)
    harm3d[exp][var+'_top'] = harm3d[exp][var+'_top'].where(harm3d[exp][var+'_top']>500)
    
    harm3d[exp][var+'_top_std'] = harm3d[exp][var+'_top'].std(['x','y'])
    
    ### Calculate variances 
    harm3d[exp]['u_var'] = harm3d[exp]['ua_p'] **2
    harm3d[exp]['v_var'] = harm3d[exp]['va_p'] **2
    harm3d[exp]['w_var'] = harm3d[exp]['wa_p'] **2
    harm3d[exp]['q_var'] = harm3d[exp]['hus_p']**2
    harm3d[exp]['t_var'] = harm3d[exp]['ta_p'] **2
    
    ### Calculate TKE 
    harm3d[exp]['tke']=\
        harm3d[exp]['ua_p']**2+\
        harm3d[exp]['ua_p']**2+\
        harm3d[exp]['ua_p']**2
        
    ### calculate cloud cover below 4km
    harm3d[exp]['cc_4km']  = (harm3d[exp]['cl'].sel(z=slice(0,4000))>thres).any(dim='z') *1
    harm2d[exp]['cc_4km'] = \
    ((harm3d[exp]['cl'].sel(z=slice(0,4000))>thres).any(dim='z').\
    sum(dim=('x','y'))/(len(harm3d[exp].x)*len(harm3d[exp].y)))
        
    ### calculate cloud cover below 2.5km (this is to check with standard output of low cloud cover CLL)
    harm3d[exp]['cc_2_5km']  = (harm3d[exp]['cl'].sel(z=slice(0,2500))>thres).any(dim='z') *1
    harm2d[exp]['cc_2_5km'] = \
    ((harm3d[exp]['cl'].sel(z=slice(0,2500))>0.5).any(dim='z').\
    sum(dim=('x','y'))/(len(harm3d[exp].x)*len(harm3d[exp].y)))

    ### calculate cloud cover below 1.5km ()
    harm3d[exp]['cc_1_5km']  = (harm3d[exp]['cl'].sel(z=slice(0,1500))>thres).any(dim='z') *1
    harm2d[exp]['cc_1_5km'] = \
    ((harm3d[exp]['cl'].sel(z=slice(0,1500))>thres).any(dim='z').\
    sum(dim=('x','y'))/(len(harm3d[exp].x)*len(harm3d[exp].y)))
    ### calculate cloud cover below 1km ()
    harm3d[exp]['cc_1km']  = (harm3d[exp]['cl'].sel(z=slice(0,1000))>thres).any(dim='z') *1
    harm2d[exp]['cc_1km'] = \
    ((harm3d[exp]['cl'].sel(z=slice(0,1000))>thres).any(dim='z').\
    sum(dim=('x','y'))/(len(harm3d[exp].x)*len(harm3d[exp].y)))
        
    ### calculate cloud cover between 1 and  4km
    harm3d[exp]['cc_1to4km'] = harm3d[exp]['cc_4km'] - harm3d[exp]['cc_1km']
    harm2d[exp]['cc_1to4km'] = harm3d[exp]['cc_1to4km'].sum(dim=('x','y'))/\
                                (len(harm3d[exp].x)*len(harm3d[exp].y))
    ### calculate cloud cover between 1.5 and  4km
    harm3d[exp]['cc_1_5to4km'] = harm3d[exp]['cc_4km'] - harm3d[exp]['cc_1_5km']
    harm2d[exp]['cc_1_5to4km'] = harm3d[exp]['cc_1_5to4km'].sum(dim=('x','y'))/\
                                (len(harm3d[exp].x)*len(harm3d[exp].y))
        
        
    ## calculate th
    harm3d[exp]['thv']= calc_th(calc_Tv(harm3d[exp]['ta'],harm3d[exp]['p'],\
                       calc_rh(harm3d[exp]['p'],harm3d[exp]['hus'],harm3d[exp]['ta'])),\
                               harm3d[exp]['p'])
    harm2d[exp]['thv'] = harm3d[exp]['thv'].mean(('x','y'))
    ################################  
    ## buoyancy
    harm3d[exp]['buoy'] = calc_buoy(harm3d[exp]['thv'],harm3d[exp]['thv'].mean(dim=('x','y')))
    ################################
    
    ## total momentum flux tau
    harm3d[exp]['tau_res'] = np.sqrt(harm3d[exp]['uw']**2 + harm3d[exp]['vw']**2)
    harm3d[exp]['tau_par'] = np.sqrt(harm3d[exp]['u_flx_param_tot']**2 + \
                                     harm3d[exp]['v_flx_param_tot']**2 )
    harm3d[exp]['tau_turb'] = np.sqrt(harm3d[exp]['uflx_turb']**2 + \
                                     harm3d[exp]['vflx_turb']**2 )
    if exp == 'noHGTQS_':
        harm3d[exp]['tau_conv'] = np.sqrt((harm3d[exp]['uflx_conv_dry']+\
                                          harm3d[exp]['uflx_conv_moist'])**2 + \
                                         (harm3d[exp]['vflx_conv_dry']+\
                                          harm3d[exp]['vflx_conv_moist'])**2 )

#%% Identify 2D convergence objects 
## use these objects as a cloud mask for cloudmetrics 
threshold_div = 0.00007 
div_mask = {}
for exp in exps:
    filtered[exp].transpose('y','x','time','z','klp')
    
    div_mask[exp] = (filtered[exp].where(filtered[exp]['div_f']>threshold_div,other=1)['div_f'])
    
    ### Calculate TKE 
    filtered[exp]['tke']=\
        filtered[exp]['u_pf']**2+\
        filtered[exp]['u_pf']**2+\
        filtered[exp]['u_pf']**2
        
        
    ## Define divergence anomaly in the subcloud layer 
    filtered[exp]['Dsc'] = filtered[exp]['div_f'].sel(z=slice(0,600)).mean('z') - \
                filtered[exp]['div_f'].sel(z=slice(0,600)).mean('z').mean(('x','y'))
    ## Define divergence anomaly in the cloud layer 
    filtered[exp]['Dc']  = filtered[exp]['div_f'].sel(z=slice(900,1500)).mean('z') - \
                filtered[exp]['div_f'].sel(z=slice(900,1500)).mean('z').mean(('x','y'))
    ## identify dipols / smocs
    filtered[exp]['smoc'] = (filtered[exp]['Dsc']/filtered[exp]['Dc'])<0


#%% Save intermediate 
    harm_srf_med_synopt[exp] = harm_srf_med[exp].mean(dim=('x','y')).chunk(dict(time=-1)).\
                        interpolate_na(dim='time').rolling(time=32,center=True).mean()
                        
     
                        
    harm_srf_med_synopt[exp][['cape','pr']].to_netcdf(my_harm_dir+exp+'/'+exp[16:]+'_harm_srf_med_synopt.nc', compute=True)
    #
    harm_srf_sml[exp].to_netcdf(my_harm_dir+exp+'/'+exp[16:]+'harm_srf_sml.nc', compute=True)
#%% filtered fields in separate script

#%% #########   QUESTIONS   #########
###DOES THE DISTRIBUTION OF CLOUDS (CLOUD SIZE,UPDRAFTS) DETERMINES THE FLUXES? 

### 1) How tilted are the eddies in different groups and runs?
### 2) How is the pdf of w changing with organisation and in the differnet runs?
### 3)

#############################################################################
#%%                     ####### PLOT #######
#############################################################################
if plot == False:
    sys.exit("Stopped before plotting.")

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
### ful time series  
fig, axs = plt.subplots(3,1,figsize=(19,15))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    for idx,var in enumerate(['cloud_fraction','num_objects','open_sky']):
        if var =='cloud_fraction':
            factor = 1
            title  = 'Cloud fraction'
            unit   = r'fraction '
            lim =   [-0.5,0.5]
        elif var =='num_objects':
            factor = 1
            title  = 'Number of clouds'
            unit   = r'number #'
            lim =   [-250,250]
        elif var =='iorg':
            factor = 1
            title  = r'$I_{org}$'
            unit   = r'$I_{org}$'
        elif var == 'mean_length_scale':
            factor = 1
            title  ='Mean length scale'
            unit   = r'km'
            lim =   [-15,15]
        elif var == 'open_sky':
            factor = 1
            title  ='Open sky'
            unit   = r''
            lim =   [-0.5,0.5]
        else:
            factor = 1
            title = var
        
        # (ds_org_4km[exp][var]-ds_org_4km['noHGTQS_'][var]).plot(\
        #             x='time',ls=sty[ide],ax=axs[idx],lw=1,c=col[ide],label=lab)
            
        # (ds_org_4km[exp][var]-ds_org_4km['noHGTQS_'][var]).rolling(time=24).mean().plot(\
        #             x='time',ls=sty[ide],ax=axs[idx],lw=3,c=col[ide],label=lab)
        
            
        (ds_org[exp][var]-ds_org['noHGTQS_'][var]).rolling(time=24).mean().plot(\
                    x='time',ls=sty[ide],ax=axs[idx],lw=3,c=col[ide],label=lab)
        
            
        # Fill the area between the vertical lines
        axs[idx].axvspan(np.datetime64('2020-01-25T00:30'),\
                         np.datetime64('2020-01-26T00:30'), alpha=0.1, color='grey')
        axs[idx].axvspan(np.datetime64('2020-01-05T00:30'),\
                         np.datetime64('2020-01-06T00:30'), alpha=0.1, color='grey')
        # axs[idx].set_xlim(0,23)
        axs[idx].set_title(title,fontsize =28)
        axs[idx].set_ylabel(unit)
        axs[idx].set_ylim(lim)
plt.suptitle('cloud mask from 0 to 4 km')
plt.suptitle('cloud mask from 0 to 2.5 km')
axs[0].set_xlabel('')
axs[0].tick_params(labelbottom=False) 
axs[1].set_xlabel('')
axs[1].tick_params(labelbottom=False) 
axs[2].set_xlabel('time')
axs[0].legend(fontsize=21)   
plt.tight_layout()

#%% cloud statistics 
## 3 panel plot 
fig, axs = plt.subplots(3,1,figsize=(19,15))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    # for idx,var in enumerate(['mean_length_scale','num_objects','open_sky']):
    for idx,var in enumerate(['mean_length_scale','num_objects','cc']):
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
        elif var == 'cc':
            title  ='Cloud cover'
            unit   = r''
        else:
            factor = 1
            title = var
        
        if var in list(ds_org[exp]):
            ds_org[exp][var].groupby('time.hour').mean('time').plot(\
                        x='hour',ls='-',ax=axs[idx],lw=3,c=col[ide],label=lab)
        
        if var in list(harm_srf[exp]):
            harm_srf[exp][var].mean(dim=('x','y')).groupby('time.hour').mean('time')\
                .plot(x='hour',ls=sty[ide],ax=axs[idx],lw=3,c=col[ide],label=lab)
        #     harm3d[exp]['cl'].sel(z=slice(0,6000)).max('z').\
        #         mean(['x','y']).groupby('time.hour').mean('time').\
        #             plot(x='hour',c=col[ide],ls='--',lw=3,ax=axs[idx],label='0 - 6km')
        
        
        if var == 'cc':
            harm2d[exp]['cc_1_5km'].\
                groupby('time.hour').mean()\
                    .plot(x='hour',ax=axs[idx],ls='-',lw=3,c=col[ide])
            harm2d[exp]['cc_1_5to4km'].\
                groupby('time.hour').mean()\
                    .plot(x='hour',ax=axs[idx],ls='--',lw=3,c=col[ide])
            
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

var = 'cc_4km'
## cloud cover 
fig, axs = plt.subplots(1,figsize=(19,7))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    # harm2d[exp][var].\
    #     groupby('time.hour').mean()\
    #         .plot(x='hour',ls='-',lw=3,c=col[ide])
    # plt.axhline(harm2d[exp][var].\
    #             mean('time'),ls=sty[ide],lw=1,c=col[ide])
    
    harm2d[exp]['cc_2_5km'].\
        groupby('time.hour').mean()\
            .plot(x='hour',ls='--',lw=3,c=col[ide])

# Fill the area between the vertical lines
axs.axvspan(20, 23, alpha=0.1, color='grey')
axs.axvspan(0, 6, alpha=0.1, color='grey')
axs.set_xlim([0,23])
axs.set_ylabel(r'fraction')
axs.set_title(r'Cloud cover',fontsize=25)
axs.set_xlabel(r'hour LT')
plt.tight_layout()

#%% cloud metrics on divergence 
fig, axs = plt.subplots(4,1,figsize=(15,10))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    for idx,var in enumerate(['cloud_fraction','open_sky','mean_length_scale','num_objects']):
            
        # ds_org_smoc_conv[exp][var].plot(\
        #             x='time',ls='--',ax=axs[idx],lw=2,c=col[ide],label=lab)
        axs[idx].axhline(ds_org_smoc_conv[exp][var].mean('time'),ls='--',lw=1,c=col[ide])
        
        
        ds_org_smoc_conv[exp].interpolate_na('time').groupby('time.hour').mean()[var]\
            .plot(x='hour',ls='-',ax=axs[idx],lw=3,c=col[ide],label=lab)
        
        # ds_org_smoc_conv[exp].interpolate_na('time').rolling(time=31).mean()[var].plot(\
        #             x='time',ls='-',ax=axs[idx],lw=3,c=col[ide],label=lab)

        # Fill the area between the vertical lines
        axs[idx].axvspan(20, 23, alpha=0.1, color='grey')
        axs[idx].axvspan(0, 6, alpha=0.1, color='grey')
        axs[idx].set_xlim([0,23])
        axs[idx].set_ylabel(var)
        axs[idx].set_title(var,fontsize=25) 
        axs[idx].set_xlabel('')
        axs[idx].set_xticks(())
axs[idx].set_xlabel(r'hour LT')
axs[0].legend(fontsize=21)  
plt.tight_layout()

#%% boxplot for organisation metrics 

fig, axs = plt.subplots(2,1,figsize=(13,7))
for idx, smoc in enumerate([ds_org_smoc_conv,]):
    if 'lab' in locals(): del lab
    if 'x_ax' in locals(): del x_ax
    iteration=-0.4                            
    for var in ['cloud_fraction','open_sky',]:
        iteration +=0.4
        for exp in exps:
            iteration +=0.3
            axs[idx].boxplot(smoc[exp][var].values,\
                        positions=[round(iteration,1)],\
                whis=1.8,showfliers=False,showmeans=True,meanline=False,widths=0.25,\
                    medianprops=dict(color="r", lw=2))  
                
            if 'lab' in locals():
                lab=np.append(lab,max(exp[8:-1],'Control'))
                x_ax=np.append(x_ax,iteration)
            else:
                lab = max(exp[8:-1],'Control')
                x_ax=iteration
                
    axs[idx].set_xticklabels(lab, rotation=45 )
    axs[idx].tick_params(axis='x', which='major', labelsize=16)
axs[0].set_ylabel('Converging\n SMOCS')
axs[1].set_ylabel('Diverging\n SMOCS')
axs[1].set_xticklabels(lab, rotation=45 )
axs[0].set_xticklabels('')
axs[0].set_title('cloud_fraction,                      open_sky',fontsize=23)
            
            
fig, axs = plt.subplots(2,1,figsize=(13,7))
for idx, smoc in enumerate([ds_org_smoc_conv,]):
    if 'lab' in locals(): del lab
    if 'x_ax' in locals(): del x_ax
    iteration=-0.4                            
    for var in ['mean_length_scale','num_objects',]:
        iteration +=0.4
        for exp in exps:
            iteration +=0.3
            axs[idx].boxplot(smoc[exp][var].values,\
                        positions=[round(iteration,1)],\
                whis=1.8,showfliers=False,showmeans=True,meanline=False,widths=0.25,\
                    medianprops=dict(color="r", lw=2))   
            if 'lab' in locals():
                lab=np.append(lab,max(exp[8:-1],'Control'))
                x_ax=np.append(x_ax,iteration)
            else:
                lab = max(exp[8:-1],'Control')
                x_ax=iteration
                
    axs[idx].set_xticklabels(lab, rotation=45 )
    axs[idx].tick_params(axis='x', which='major', labelsize=16)
axs[0].set_ylabel('Converging\n SMOCS')
axs[1].set_ylabel('Diverging\n SMOCS')
axs[1].set_xticklabels(lab, rotation=45 )
axs[0].set_xticklabels('')
axs[0].set_title('mean_length_scale,                      num_objects',fontsize=23)

#%% Surface fluxes and precipitation 
fig, axs = plt.subplots(2,1,figsize=(18,15))
for ide, exp in enumerate(exps[:-1]):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    for idx,var in enumerate(['cape','pr']):
    # for idx,var in enumerate(['ct','cb','cll']):
        
        if var =='pr':
            factor = 3600 *(24/0.0346)
            title  = 'Precipitation'
            # unit   = r'$mm \, hour^{-1}$'
            unit   = r'$W \, m^{-2}$'
        elif var =='hfls':
            factor = 1
            title  = 'Latent heat flux'
            unit   = r'$W \, m^{-2}$'
        elif var =='hfss':
            factor = 1
            title  = 'Sensible heat flux'
            unit   = r'$W \, m^{-2}$'
        elif var =='cape':
            factor = 1
            title  = 'CAPE'
            unit   = r'$J \, kg^{-1}$'
        else: 
            factor =1
            title =  var
            unit = '?'
        
        ## cropped 200km
        # (factor*harm_srf_200[exp][var]).mean(dim=('x','y')).groupby('time.hour').mean('time')\
        #         .plot(x='hour',ls='-',ax=axs[idx],lw=3,c=col[ide],label=lab)
        # axs[idx].axhline((factor*harm_srf_200[exp][var]).mean(dim=('x','y')).mean('time')\
        #         ,ls='-',lw=0.5,c=col[ide])
        # ## new cropped 200km
        # (factor*harm_srf_sml[exp][var]).mean(dim=('x','y')).groupby('time.hour').mean('time')\
        #         .plot(x='hour',ls='-',ax=axs[idx],lw=2,c=col[ide],label=lab)
        # axs[idx].axhline((factor*harm_srf_sml[exp][var]).mean(dim=('x','y')).mean('time')\
        #         ,ls='-.',lw=1,c=col[ide])
        ## new cropped 400
        (factor*harm_srf_med[exp][var]).mean(dim=('x','y')).groupby('time.hour').mean('time')\
                .plot(x='hour',ls='-',ax=axs[idx],lw=3,c=col[ide],label=lab)
        axs[idx].axhline((factor*harm_srf_med[exp][var]).mean(dim=('x','y')).mean('time')\
                ,ls=sty[ide],lw=1,c=col[ide])
        ## full domain 
        # (factor*harm_srf[exp][var]).mean(dim=('x','y')).groupby('time.hour').mean('time')\
        #         .plot(x='hour',ls=sty[ide],ax=axs[idx],lw=3,c=col[ide],label=lab)
        # axs[idx].axhline((factor*harm_srf[exp][var]).mean(dim=('x','y')).mean('time')\
        #         ,ls=sty[ide],lw=1,c=col[ide])
            
        # Fill the area between the vertical lines
        axs[idx].axvspan(20, 23, alpha=0.1, color='grey')
        axs[idx].axvspan(0, 6, alpha=0.1, color='grey')
        axs[idx].set_xlim(0,23)
        axs[idx].set_title(title,fontsize =28)
        axs[idx].set_ylabel(unit)
axs[0].set_xlabel('')
axs[1].set_xlabel('')
axs[2].set_xlabel('hour LT')
axs[1].set_ylim(185,237)
axs[2].set_ylim(9,62)
axs[0].legend(fontsize=23)   
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
        lab='UVmixOFF'
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
#%%  calculate tendencies from wind components
##for Louise 
exp= 'noHGTQS_'
for var in ['u','v']:
    # harm3d[exp][var+'_tend']=-harm3d[exp][var+'w'].differentiate(coord='z')
    # harm2d[exp][var+'_tend'] = harm3d[exp][var+'_tend'].mean(('x','y'))


    harm2d[exp][var+'w'] = harm3d[exp][var+'w'].mean(('x','y'))
    harm2d[exp][var+'_tend_diftime'] = harm2d[exp][var].diff('time')
    
    harm2d[exp][var+'_tend_diftime_'] = -(harm3d[exp][var+'a']*harm3d[exp]['wa']).differentiate(coord='z').mean(('x','y'))


    harm3d_filter[exp][var+'_tend'] = harm3d_filter[exp][var+'a_f'].diff('time')
#%% for Louise 
layer=[0,200]

fig, axs = plt.subplots(2,1,figsize=(19,15))
exp = 'noHGTQS_'
lab='Control'
h_clim_to_plot = harm2d[exp].sel(z=slice(layer[0],layer[1])).mean('z')
for idx,var in enumerate(['u','v']):
    ## HARMONIE cy43 clim
    ((h_clim_to_plot['dt'+var+'_dyn']+h_clim_to_plot['dt'+var+'_phy'])*step)\
        .groupby(h_clim_to_plot.time.dt.hour).mean().\
            plot(c=col[ide],ls='-',lw=3,label=lab+': Tot',ax=axs[idx])
    (h_clim_to_plot*step).groupby(h_clim_to_plot.time.dt.hour).mean()\
        ['dt'+var+'_dyn'].\
            plot(c=col[ide],ls=':',lw=3,label=lab+': Dyn',ax=axs[idx])


    ## calculated tendencies
    # (harm2d[exp][var+'_tend']*step).groupby('time.hour').mean()\
    #     .sel(z=slice(layer[0],layer[1])).mean('z').plot(ax=axs[idx])
        
        
    (harm2d[exp][var+'_tend_diftime']).groupby('time.hour').mean()\
        .sel(z=slice(layer[0],layer[1])).mean('z').plot(ax=axs[idx])
        
    (harm2d[exp][var+'_tend_diftime_']*step).groupby('time.hour').mean()\
        .sel(z=slice(layer[0],layer[1])).mean('z').plot(ax=axs[idx])
    
    
    
    axs[idx].set_title(var+' direction',fontsize=25)
    axs[idx].axhline(0,c='k',lw=0.5)
plt.legend(['Total','Resolved','Parameterised'],fontsize=20)
plt.tight_layout()


#%% Tendency evolution over the day 
layer=[0,200]
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
        lab='UVmixOFF'
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
            lab='UVmixOFF'
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
#%% Fluxes profiles
fig, axs = plt.subplots(1,2,figsize=(13,11))
for idx,var in enumerate(['u','v']):
    for ide, exp in enumerate(exps):
        if exp == 'noHGTQS_':
            lab='Control'
            conv  = harm2d[exp][var+'flx_conv_dry'] + harm2d[exp][var+'flx_conv_moist']
            param = harm2d[exp][var+'_flx_param_tot']
        elif exp == 'noHGTQS_noSHAL_':
            lab='NoShal'
            param = harm2d[exp][var+'_flx_param_tot']
            conv  = harm2d[exp][var+'flx_conv_dry'] + harm2d[exp][var+'flx_conv_moist']
        elif exp == 'noHGTQS_noUVmix_':
            lab='UVmixOFF'
            if var in ['u','v']:
                param = harm2d[exp][var+'_flx_param_tot']
        
        if var == 'Thl':
            resol = harm3d[exp]['tw'].mean(['x','y'])
        elif var == 'rt':
            resol = harm3d[exp]['qw'].mean(['x','y'])
        else: 
            resol = harm3d[exp][var+'w'].mean(['x','y'])
            # resol = (harm3d[exp][var+'a']*harm3d[exp]['wa']).mean(['x','y'])
        
        
        # ## mean of all 
        # (resol+param).isel(z=slice(1,-1))\
        #         .mean('time').plot(y='z',\
        #                 ls='-',ax=axs[idx],label=lab+' total',lw=3,c=col[ide])  
        # # # parameterised 
        # param.isel(z=slice(1,-1))\
        #         .mean('time').plot(y='z',\
        #                 ls='--',ax=axs[idx],label=lab+' param',lw=2,c=col[ide]) 
        ## parameterised convection
        if exp == 'noHGTQS_':
            ## mean of all 
            (resol+param)\
                    .mean('time').plot(y='z',\
                            ls='-',ax=axs[idx],label=lab+' total',lw=3,c=col[ide])  
            param\
                    .mean('time').plot(y='z',\
                            ls='--',ax=axs[idx],label=lab+' param',lw=2,c=col[ide]) 
        # conv\
        #         .mean('time').plot(y='z',\
        #                 ls='--',ax=axs[idx],label=lab+' conv.',lw=2,c=col[ide]) 
        # harm2d[exp][var+'flx_turb']\
        #         .mean('time').plot(y='z',\
        #                 ls=':',ax=axs[idx],label=lab+' turb.',lw=2,c=col[ide]) 
                        
            # ## total - convection
            # (resol+param - conv)\
            #     .mean('time').plot(y='z',\
            #                 ls='--',ax=axs[idx],label=lab+' conv.',lw=1,c='r') 
        
            ## resolved
            resol\
                    .mean('time').plot(y='z',\
                            ls=':',ax=axs[idx],label=lab+' resol.',lw=2,c=col[ide])  
        
        
        # axs[idx].axhline(layer[0],c='grey',lw=0.3)
        # axs[idx].axhline(layer[1],c='grey',lw=0.3)
        axs[idx].axvline(0,c='grey',lw=0.5)
        axs[idx].set_ylim([0,4000])
        axs[idx].set_xlabel(r'$m^{2} \, s^{-2}$')
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
axs[0].set_xlim([-0.02,0.082])
axs[1].set_xlim([-0.015,0.03])   
axs[1].set_yticks([]) 
axs[1].set_ylabel('')
# axs[0].set_title('Temperature flux',fontsize=26)
# axs[1].set_title('Humidity flux',fontsize=26)
axs[0].set_title('Zonal momentum flux',fontsize=25)
axs[1].set_title('Meridional momentum flux',fontsize=25)
axs[0].legend(fontsize=21)
plt.tight_layout()
#%% Resolved filtered fluxes profiles 
################ FOR LOUISE ################
################################################################################
pltnight = False
pltday = True
alltime = False

fig, axs = plt.subplots(1,2,figsize=(13,11))
for idx,var in enumerate(['u','v']):
    exp = 'noHGTQS_'
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
        
    if pltnight:
        resol  = harm3d[exp][var+'w'].mean(['x','y']).where((harm3d[exp]['time.hour']>= 20) | (harm3d[exp]['time.hour']< 6))
        up_fi  = (harm3d_filter[exp][var+'a_pf'] * harm3d_filter[exp]['wa_pf']).mean(['x','y']).where((harm3d[exp]['time.hour']>= 20) | (harm3d[exp]['time.hour']< 6))
    elif pltday:    
        resol  = harm3d[exp][var+'w'].mean(['x','y']).where((harm2d[exp]['time.hour']>= 6) & (harm2d[exp]['time.hour']< 20))
        up_fi  = (harm3d_filter[exp][var+'a_pf'] * harm3d_filter[exp]['wa_pf']).mean(['x','y']).where((harm2d[exp]['time.hour']>= 6) & (harm2d[exp]['time.hour']< 20))
    elif alltime:
        resol  = harm3d[exp][var+'w'].mean(['x','y'])
        up_fi  = (harm3d_filter[exp][var+'a_pf'] * harm3d_filter[exp]['wa_pf']).mean(['x','y'])
    
    sub_fi = resol - up_fi
    
    ## resolved
    resol\
            .mean('time').plot(y='z',\
                    ls='-',ax=axs[idx],label='Resol.',lw=3,c='k')  
    up_fi\
            .mean('time').plot(y='z',\
                    ls='--',ax=axs[idx],label='>25km',lw=2,c='r')  
    sub_fi\
            .mean('time').plot(y='z',\
                    ls=':',ax=axs[idx],label='<25km',lw=2,c='r')  
    
    
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
axs[1].legend(fontsize=24)
#%% 
#### diurnal cycle of variances 
layer = [0,200]
fig, axs = plt.subplots(2,1,figsize=(15,15))
exp = 'noHGTQS_'
if exp == 'noHGTQS_':
    lab='Control'
    unit = r'$m^2\,s^{-2}$'
elif exp == 'noHGTQS_noSHAL_':
    lab='NoShal'
elif exp == 'noHGTQS_noUVmix_':
    lab='UVmixOFF'
for idx,var in enumerate(['u','v']):
    resol = harm3d[exp][var+'_var'].mean(['x','y'])
    up_fi = (harm3d_filter[exp][var+'a_pf']**2).mean(['x','y'])
    sub_fi = resol - up_fi
        
    ## resolved
    resol.sel(z=slice(layer[0],layer[1])).groupby('time.hour').mean(['z','time'])\
            .plot(x='hour',ls='-',ax=axs[idx],lw=3,c='k',label='Resol.')
    axs[idx].axhline(resol.sel(z=slice(layer[0],layer[1])).mean(['z','time'])\
            ,ls=sty[ide],lw=3,c='k')
    ## upfilter
    up_fi.sel(z=slice(layer[0],layer[1])).groupby('time.hour').mean(['z','time'])\
            .plot(x='hour',ls='--',ax=axs[idx],lw=3,c='r',label='>25km')
    axs[idx].axhline(up_fi.sel(z=slice(layer[0],layer[1])).mean(['z','time'])\
            ,ls=sty[ide],lw=2,c='r')
    ## subfilter
    sub_fi.sel(z=slice(layer[0],layer[1])).groupby('time.hour').mean(['z','time'])\
            .plot(x='hour',ls=':',ax=axs[idx],lw=3,c='r',label='<25km')
    axs[idx].axhline(sub_fi.sel(z=slice(layer[0],layer[1])).mean(['z','time'])\
            ,ls=sty[ide],lw=2,c='r')
            
    # Fill the area between the vertical lines
    axs[idx].axvspan(20, 23, alpha=0.1, color='grey')
    axs[idx].axvspan(0, 6, alpha=0.1, color='grey')
    axs[idx].set_xlim(0,23)
    axs[idx].set_title(var+' variance',fontsize =28)
    axs[idx].set_ylabel(unit)
axs[0].set_xlabel('')
axs[1].set_xlabel('hour LT')
axs[0].legend(fontsize=23)   
plt.tight_layout()

################ end plots for Louise ################
################################################################################
################################################################################
#%% total momentum flux tau
import itertools
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

fig, axs = plt.subplots(1,1,figsize=(7,10))
var = 'tau'
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
        conv  = harm3d[exp][var+'_conv'].mean(['x','y']).isel(z=slice(1,-1))
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    resol = harm3d[exp][var+'_res'].mean(['x','y']).isel(z=slice(1,-1))
    param = harm3d[exp][var+'_par'].mean(['x','y']).isel(z=slice(1,-1))
    turb  = harm3d[exp][var+'_turb'].mean(['x','y']).isel(z=slice(1,-1))
    
    
    
    ## mean of all 
    (resol+param)\
            .mean('time').plot(y='z',\
                    ls='-',label=lab+' total',lw=3,c=col[ide])  
    ## resolved
    resol\
            .mean('time').plot(y='z',\
                    ls=':',label=lab+' resol.',lw=3,c=col[ide])  
    
    ## parameterised 
    param\
            .mean('time').plot(y='z',\
                    ls='--',label=lab+' param.',lw=2,c=col[ide]) 
    # turb\
    #         .mean('time').plot(y='z',\
    #                 ls='--',label=lab+' turb.',lw=2,c=col[ide]) 
    # ## parameterised convection
    # if exp == 'noHGTQS_':
    #     conv\
    #             .mean('time').plot(y='z',\
    #                     ls=':',label=lab+' conv.',lw=1.5,c=col[ide]) 
        # param\
        #         .mean('time').plot(y='z',\
        #                 ls='-',label=lab+' param.',lw=2,c=col[ide]) 
        # turb\
        #         .mean('time').plot(y='z',\
        #                 ls='--',label=lab+' turb.',lw=2,c=col[ide]) 
                    
        # ## total - convection
        # (resol+param - conv)\
        #     .mean('time').plot(y='z',\
        #                 ls='--',label='tot - conv',lw=3,c='r') 
    

    
    # axs[idx].axhline(layer[0],c='grey',lw=0.3)
    # axs[idx].axhline(layer[1],c='grey',lw=0.3)
    # axs.axvline(0,c='grey',lw=0.5)
    axs.set_ylim([0,4000])
    axs.set_xlabel(r'$m^{2} \, s^{-2}$')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
 
axs.set_title('Total momentum flux',fontsize=25)
# axs.set_title('Resolved momentum flux',fontsize=25)
# axs.set_title('Parameterised momentum flux',fontsize=25)

axs.set_xlim([0,0.091])
handles, labels = axs.get_legend_handles_labels()

# axs.legend(flip(handles, 2), flip(labels, 2),fontsize=21,loc='upper center',\
#             frameon=False,bbox_to_anchor=(0.5, -0.1), ncol=2)
axs.legend(fontsize=19,loc='upper center',\
            frameon=False,bbox_to_anchor=(0.5, -0.1), ncol=2)
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
            lab='UVmixOFF'
            
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
            lab='UVmixOFF'
            
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
            lab='UVmixOFF'
            
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
#                 lab='UVmixOFF'
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
            lab='UVmixOFF'
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
            lab='UVmixOFF'
        
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
            lab='UVmixOFF'
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
            lab='UVmixOFF'
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
            lab='UVmixOFF'
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
        lab='UVmixOFF'
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
            lab='UVmixOFF'
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
        lab='UVmixOFF'
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
pltnight = False
pltday = False
alltime = True
fig, axs = plt.subplots(1,3,figsize=(15,10))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    
    night_2d = harm2d[exp].where((harm2d[exp]['time.hour']>= 20) | (harm2d[exp]['time.hour']< 6))
    day_2d = harm2d[exp].where((harm2d[exp]['time.hour']>= 6) & (harm2d[exp]['time.hour']< 20))
    night_3d = harm3d[exp].where((harm3d[exp]['time.hour']>= 20) | (harm3d[exp]['time.hour']< 6))
    day_3d = harm3d[exp].where((harm3d[exp]['time.hour']>= 6) & (harm3d[exp]['time.hour']< 20))
    
    if pltnight:
        ## night time
        # cloud fraction  
        night_2d['cl'].mean('time').plot(y='z',lw=3,ls=sty[ide],\
                                                    c=col[ide],label=lab,ax=axs[0])
        # in cloud LWC  
        (1000*night_2d['clw']/(night_2d['cl'])).where(night_2d['cl']>0.02).mean('time').plot(y='z',lw=3,ls=sty[ide],\
                                                    c=col[ide],label=lab,ax=axs[1])
        # delta q  
        ((night_3d['hus'].mean(dim=['x','y']).interp(z=harm3d['noHGTQS_'].z) - harm3d['noHGTQS_']['hus'].mean(dim=['x','y']))*100).\
            mean('time').plot(y='z',ls=sty[ide],c=col[ide],lw=3,ax=axs[2])
    
    if pltday:
        ## day time 
        # cloud fraction  
        day_2d['cl'].mean('time').plot(y='z',lw=3,ls=sty[ide],\
                                                    c=col[ide],label=lab,ax=axs[0])
            # in cloud LWC
        (1000*day_2d['clw']/(day_2d['cl'])).where(day_2d['cl']>0.02).mean('time').plot(y='z',lw=3,ls=sty[ide],\
                                                    c=col[ide],label=lab,ax=axs[1])
        
        # delta q
        ((day_3d['hus'].mean(dim=['x','y']).interp(z=harm3d['noHGTQS_'].z) - harm3d['noHGTQS_']['hus'].mean(dim=['x','y']))*100).\
            mean('time').plot(y='z',ls=sty[ide],c=col[ide],lw=3,ax=axs[2])
    if alltime:
        # cloud fraction  
        harm2d[exp]['cl'].mean('time').plot(y='z',lw=3,ls=sty[ide],\
                                                    c=col[ide],label=lab,ax=axs[0])
            # in cloud LWC
        (1000*harm2d[exp]['clw']/(harm2d[exp]['cl'])).where(harm2d[exp]['cl']>0.02).mean('time').plot(y='z',lw=3,ls=sty[ide],\
                                                    c=col[ide],label=lab,ax=axs[1])
        
        # delta q
        ((harm3d[exp]['hus'].mean(dim=['x','y']).interp(z=harm3d['noHGTQS_'].z) - harm3d['noHGTQS_']['hus'].mean(dim=['x','y']))*100).\
            mean('time').plot(y='z',ls=sty[ide],c=col[ide],lw=3,ax=axs[2])
            
        # delta t
        # ((harm3d[exp]['ta'].mean(dim=['x','y']).interp(z=harm3d['noHGTQS_'].z) - harm3d['noHGTQS_']['ta'].mean(dim=['x','y']))).\
        #     mean('time').plot(y='z',ls=sty[ide],c=col[ide],lw=3,ax=axs[2])
        
    
    ## all hours
    # harm2d[exp]['cl'].mean('time').plot(y='z',lw=3,ls=sty[ide],\
    #                                             c=col[ide],label=lab,ax=axs[0])
        
    # (1000*harm2d[exp]['clw']/(harm2d[exp]['cl'])).where(harm2d[exp]['cl']>0.02).mean('time').plot(y='z',lw=3,ls=sty[ide],\
    #                                             c=col[ide],label=lab,ax=axs[1])
        
    # ((harm3d[exp]['hus'].mean(dim=['x','y']).interp(z=harm3d['noHGTQS_'].z) - harm3d['noHGTQS_']['hus'].mean(dim=['x','y']))*100).\
    #     mean('time').plot(y='z',ls=sty[ide],c=col[ide],lw=3,ax=axs[2])
    
    
    #########
    ### old in-cloud LWC attempts 
    # (1000*harm2d[exp]['clw']*(1-harm2d[exp]['cl'])).mean('time').plot(y='z',lw=3,ls=sty[ide],\
    #                                             c=col[ide],ax=axs[1])
    # ((1000*harm3d[exp]['rain']).mean(dim=['x','y'])\
    #     /(harm2d[exp]['cl'])).where(harm2d[exp]['cl']>0.02).mean('time').\
    #     plot(y='z',lw=3,ls=sty[ide],c=col[ide],label=lab,ax=axs[2])
    # (1000*harm3d[exp]['rain']).mean(dim=['x','y']).mean('time').\
    #     plot(y='z',lw=3,ls=sty[ide],c=col[ide],label=lab,ax=axs[2])
     #########   


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
axs[2].set_title(r'$\Delta q_t$ (UVmixOFF - ctrl)',fontsize=30)
# axs[2].set_title(r'$\Delta q_t$ (NoSHAL - ctrl)',fontsize=30)
axs[0].legend(fontsize=25)
plt.tight_layout()

#%% rain distribution 
var = 'rain'
thres = +0.0001
factor = 1000
plt.figure()    
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    (harm3d[exp][var]*factor).where(harm3d[exp][var]>thres).isel(z=1)\
    .plot.hist(bins=500,color=col[ide],histtype=u'step', density=False)
    plt.axvline(harm3d[exp][var].where(harm3d[exp][var]>thres).isel(z=1)\
                .quantile(0.5)*factor,c=col[ide],lw=3.5,ls=sty[ide],label=lab)
    plt.axvline(harm3d[exp][var].where(harm3d[exp][var]>thres).isel(z=1)\
                .quantile(0.05)*factor,c=col[ide],lw=2.5,ls=sty[ide])
    plt.axvline(harm3d[exp][var].where(harm3d[exp][var]>thres).isel(z=1)\
                .quantile(0.95)*factor,c=col[ide],lw=2.5,ls=sty[ide])
        
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
        lab='UVmixOFF'
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
################################################################
## plot cloud time series as in DALES paper.
## plot it as a difference to the control .
################################################################
var = 'cl_top'
plt.figure(figsize=(13,9) )
for ide, exp in enumerate(exps):
    harm3d[exp][var].mean(dim=('x','y')).groupby('time.hour').mean()\
        .plot(color=col[ide],label=lab,lw=3)
    # ((harm3d[exp][var]<4000)&(harm3d[exp][var]>2500)).sum(dim=('x','y')).groupby('time.hour').mean()\
    #         .plot(color=col[ide],label=lab+'_Dp',ls='--',lw=3)
plt.ylabel(r'Cloud top')
plt.legend(fontsize=25)
plt.title('Diurnality of mean cloud top',fontsize=22)
plt.xlim([0,23])
plt.tight_layout()  


#%%
var = 'rain'
plt.figure(figsize=(13,8) )
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    rain_rate = convert_rain_intensity(harm3d[exp][var]).mean(dim=('x','y'))  
    rain_rate.sel(z=0).groupby('time.hour').mean()\
        .plot(color=col[ide],label=lab,lw=3)
plt.ylabel(r'mm/hour')
plt.legend(fontsize=25)
plt.title('Rain rate',fontsize=22)
plt.xlim([0,23])
plt.tight_layout()  

#%% Distribution of cloud top
var = 'cl_top'
plt.figure(figsize=(13,9) )
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    harm3d[exp][var].plot.hist(bins=15,lw=3, color=col[ide],\
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
#         lab='UVmixOFF'
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
        lab='UVmixOFF'
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

#%% Convergence and divergence ['dudx']+['dvdy']

klp = filtered[exp].klp[1]
# it = filtered[exp].time[15]
it= np.datetime64('2020-01-08T12')
iz = 200
var = 'div'
# plt.figure(figsize=(13,5))
# for ide, exp in enumerate(exps):
#     filtered[exp][var+'_f'].var(('x','y')).sel(klp=klp).sel(z=iz,method='nearest').plot(c=col[ide],lw=3)
#     plt.suptitle('variance of divergence')

## fields of divergence
fig, axs = plt.subplots(1,2,figsize=(19,7))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    filtered[exp].sel(klp=klp,time=it).sel(z=iz,method='nearest')[var+'_f']\
        .plot(ax=axs[ide],vmax=0.0003)
    axs[ide].set_title(lab,fontsize=23)
axs[1].set_ylabel('')
axs[1].set_yticks([])

fig, axs = plt.subplots(1,2,figsize=(19,7))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    harm3d[exp][var].sel(time=it).sel(z=iz,method='nearest').\
        plot(ax=axs[ide],vmax=0.0007)
    axs[ide].set_title(lab,fontsize=23)
axs[1].set_ylabel('')
axs[1].set_yticks([])

    
# ## profiles of mean divergence
# fig, axs = plt.subplots(2,1,figsize=(19,7))
# for ide, exp in enumerate(exps):
#     if exp == 'noHGTQS_':
#         lab='Control'
#     elif exp == 'noHGTQS_noSHAL_':
#         lab='NoShal'
#     elif exp == 'noHGTQS_noUVmix_':
#         lab='UVmixOFF'
#     filtered[exp][var+'_f'].mean(('x','y')).sel(klp=klp).\
#         plot(ax=axs[ide],vmax=0.00005)
#     axs[ide].set_title(lab,fontsize=23)
#     axs[ide].set_ylim([0,6000])
# axs[0].set_xlabel('')
# axs[0].set_xticks([])

    
# ## profiles of mean divergence from non coarsened fields
# fig, axs = plt.subplots(2,1,figsize=(19,7))
# for ide, exp in enumerate(exps):
#     if exp == 'noHGTQS_':
#         lab='Control'
#     elif exp == 'noHGTQS_noSHAL_':
#         lab='NoShal'
#     elif exp == 'noHGTQS_noUVmix_':
#         lab='UVmixOFF'
#     harm3d[exp][var].mean(('x','y')).\
#         plot(x='time',ax=axs[ide])
#     axs[ide].set_title(lab,fontsize=23)
#     axs[ide].set_ylim([0,6000])
# axs[0].set_xlabel('')
# axs[0].set_xticks([])

#%% aereal snapshot
idtime= np.datetime64('2020-01-08T12')
# idtime= filtered[exp].time[15]
klp= 5
var= 'cc_4km'
# var= 'cl'
var_2 = 'Dsc'

ticklabels= np.array([50,100,150]) # in km 


fig, axs = plt.subplots(2,3,figsize=(15,7))
for ide,exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
        
    if ide == 2:
        ##########
        ## filtered has x and y axis inverted because in "main_scale_sep... .py" 
        ## the dataframe is initialised wrongly.
        ############
        harm3d[exp].sel(time=idtime)[var].plot(x='x',ax=axs[0,ide],cmap='Blues_r',add_colorbar=True)
        filtered[exp].sel(klp=klp,time=idtime)[var_2].plot(x='y',ax=axs[1,ide],vmin=-0.0002,add_colorbar=True)
    else:
        harm3d[exp].sel(time=idtime)[var].plot(x='x',ax=axs[0,ide],cmap='Blues_r',add_colorbar=False)
        filtered[exp].sel(klp=klp,time=idtime)[var_2].plot(x='y',ax=axs[1,ide],vmin=-0.0002,add_colorbar=False)
    
    # (harm_srf_sml[exp].sel(time=idtime)[var]).plot(ax=axs[ide],cmap='Blues_r',vmax=0.9)
    # (harm_srf_sml[exp].sel(time=idtime)[var]>0.5).plot(ax=axs[ide],cmap='Blues_r')
    axs[0,ide].set_title(lab,fontsize =25)
    ## x
    axs[0,ide].set_xlabel('')
    axs[0,ide].set_xticks([])
    axs[1,ide].set_xlabel('km')
    axs[1,ide].set_xticks(ticklabels*1000 +min(filtered[exp].y).values)
    axs[1,ide].set_xticklabels(ticklabels)
    ## y
    axs[0,ide].set_ylabel('')
    axs[1,ide].set_ylabel('')
    axs[0,ide].set_yticks([])
    axs[1,ide].set_yticks([])
    axs[1,0].set_yticks(ticklabels*1000 +min(filtered[exp].x).values)
    axs[1,0].set_yticklabels(ticklabels)
    axs[0,0].set_yticks(ticklabels*1000 +min(harm3d[exp].y).values)
    axs[0,0].set_yticklabels(ticklabels)

axs[0,0].set_ylabel('km')
axs[1,0].set_ylabel('km')
plt.suptitle(idtime,fontsize =28)
###


#%% plot domains 
ds_pr = xr.open_mfdataset('/Users/acmsavazzi/Documents/WORK/PhD_Year3/DATA/HARMONIE/noHGTQS/pr_his_BES_HA43h22tg3_clim_noHGTQS_1hr_202001110000-202001210000.nc')
var = 'pr'
ii=10

# ##large
# Dx = geopy.distance.distance(kilometers = 400)
# Dy = geopy.distance.distance(kilometers = 400)
# lat_max = Dy.destination(point=les_centre, bearing=0)
# lat_min = Dy.destination(point=les_centre, bearing=180)
# lon_max = Dx.destination(point=les_centre, bearing=270)
# lon_min = Dx.destination(point=les_centre, bearing=90)
# large_ocean =[lat_min[0], lon_min[1], lat_max[0], lon_max[1]]



small = [harm_srf_sml[exp].lat.min().values, harm_srf_sml[exp].lon.min().values,\
          harm_srf_sml[exp].lat.max().values, harm_srf_sml[exp].lon.max().values]
    
medium = [harm_srf_med[exp].lat.min().values, harm_srf_med[exp].lon.min().values,\
          harm_srf_med[exp].lat.max().values, harm_srf_med[exp].lon.max().values]

# medium    = [11.47, -59.61, 15.067, -55.91]
# large     = [10.5, -61, 20.3, -51]
large = [harm_srf[exp].lat.min().values, harm_srf[exp].lon.min().values,\
          harm_srf[exp].lat.max().values, harm_srf[exp].lon.max().values]


plt.figure()
# ax =ds_pr.isel(time=ii)[var].plot(vmin=0,vmax=1,\
#                     cmap=plt.cm.Blues_r,x='lon',y='lat',\
#                     subplot_kws=dict(projection=proj))
    
# ax =harm_srf_sml[exp].isel(time=ii)['pr'].plot(\
#                     cmap=plt.cm.Blues_r,x='lon',y='lat',\
#                     subplot_kws=dict(projection=proj))

    
ax = plt.axes(projection=proj)
ax.add_feature(coast, lw=2, zorder=7,color='darkgreen',alpha=0.3)
plt.xlim([ds_pr.lon[0,0].values,ds_pr.lon[0,-1].values])
plt.ylim([ds_pr.lat[0,-1].values,ds_pr.lat[-1,-1].values])


## small domain
ax.plot([small[1],small[3]],[small[0],small[0]],c='r',ls='-')
ax.plot([small[1],small[3]],[small[2],small[2]],c='r',ls='-')
ax.plot([small[1],small[1]],[small[0],small[2]],c='r',ls='-')
ax.plot([small[3],small[3]],[small[0],small[2]],c='r',ls='-')
# ## medium domain
# ax.plot([medium[1],medium[3]],[medium[0],medium[0]],c='r',ls='--')
# ax.plot([medium[1],medium[3]],[medium[2],medium[2]],c='r',ls='--')
# ax.plot([medium[1],medium[1]],[medium[0],medium[2]],c='r',ls='--')
# ax.plot([medium[3],medium[3]],[medium[0],medium[2]],c='r',ls='--')
# ## medium domain ocean 
# ax.plot([medium_ocean[1],medium_ocean[3]],[medium_ocean[0],medium_ocean[0]],c='r',ls='--')
# ax.plot([medium_ocean[1],medium_ocean[3]],[medium_ocean[2],medium_ocean[2]],c='r',ls='--')
# ax.plot([medium_ocean[1],medium_ocean[1]],[medium_ocean[0],medium_ocean[2]],c='r',ls='--')
# ax.plot([medium_ocean[3],medium_ocean[3]],[medium_ocean[0],medium_ocean[2]],c='r',ls='--')
## large domain
ax.plot([large[1],large[3]],[large[0],large[0]],c='k',ls='-',lw=1)
ax.plot([large[1],large[3]],[large[2],large[2]],c='k',ls='-',lw=1)
ax.plot([large[1],large[1]],[large[0],large[2]],c='k',ls='-',lw=1)
ax.plot([large[3],large[3]],[large[0],large[2]],c='k',ls='-',lw=1)

gl = ax.gridlines(crs=proj, draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.suptitle(exp)
         

#%% snapshots of wind anomalies
iz=1500
it=79
var='w'
fig, axs = plt.subplots(1,2,figsize=(15,7))
if var =='w':
    vmin=-0.06
elif var =='u':
    vmin=-3.5
for ide,exp in enumerate(exps):
    harm3d_filter[exp][var+'_pf'].sel(z=iz,method='nearest')\
        .isel(time=it).plot(cmap='jet',ax=axs[ide],vmin=vmin)

fig, axs = plt.subplots(1,2,figsize=(15,7))
if var =='w':
    vmin=-0.15
    vmax=0.3
elif var =='u':
    vmin=-3.5
for ide,exp in enumerate(exps):
    harm3d[exp][var+'a_p'].sel(z=iz,method='nearest')\
        .isel(time=it).plot(cmap='jet',ax=axs[ide],vmin=vmin,vmax=vmax)
# plt.figure()
# harm3d[exp][var+'a'].sel(z=iz,method='nearest')\
#     .isel(time=it).plot(cmap='jet')

#%%cross section
exp = 'noHGTQS_'
# exp = 'noHGTQS_noSHAL_'
idtime= np.datetime64('2020-01-25T19')


crossyz = harm3d[exp].sel(time=idtime).sel(x=170000,method='nearest').sel(y=slice(50000,225000))

crossxz = harm3d[exp].sel(time=idtime).sel(y=84000,method='nearest').sel(x=slice(150000,300000))

###############
### caclculate tendency by differenciating the flux ###
crossyz['v_tend']=crossyz['vw'].differentiate(coord='z')
crossxz['u_tend']=crossxz['uw'].differentiate(coord='z')
###### check this function!! 
###############

for section in ['xz','yz']:
    if section =='xz':
        mask = np.nan_to_num(crossxz['cl'].where(crossxz['cl']>0.5).values)
        mask[mask > 0] = 3
        kernel = np.ones((4,4))
        C      = ndi.convolve(mask, kernel, mode='constant', cval=0)
        outer  = np.where( (C>=3) & (C<=12 ), 1, 0)

        crossxz['cloud'] = (('z', 'x'), outer)
    if section =='yz':        
        mask = np.nan_to_num(crossyz['cl'].where(crossyz['cl']>0.5).values)
        mask[mask > 0] = 3
        kernel = np.ones((4,4))
        C      = ndi.convolve(mask, kernel, mode='constant', cval=0)
        outer  = np.where( (C>=3) & (C<=12 ), 1, 0)
        
        crossyz['cloud'] = (('z', 'y'), outer)
        


# plot YZ 
plt.figure(figsize=(15,6))
temp = crossyz.coarsen(y=1, boundary='trim').mean()
temp = temp.coarsen(z=1, boundary="trim").mean()
temp = temp.interp(z=np.linspace(temp.z.min(),temp.z.max(), num=30))
im_1a = crossyz['v_tend'].plot(x='y')
im_1b = temp.plot.\
    streamplot('y','z','va_p','wa_p',hue='vw',vmin=-0.001,\
                     density=[0.4, 0.4],\
                    linewidth=3.5,arrowsize=4,\
                arrowstyle='fancy',cmap='PiYG_r')

crossyz['cloud'].where(crossyz['cloud'] > 0).plot(cmap='binary',\
                                add_colorbar=False,vmin=0,vmax=0.5)
cbar = im_1a.colorbar
cbar.remove()
cbar = im_1b.colorbar
cbar.remove()
plt.ylim([0,4500])
plt.tight_layout()

# plot XZ 
plt.figure(figsize=(15,6))
temp = crossxz.coarsen(x=1, boundary='trim').mean()
temp = temp.coarsen(z=1, boundary="trim").mean()
temp = temp.interp(z=np.linspace(temp.z.min(),temp.z.max(), num=30))
im_1a = crossxz['u_tend'].plot(x='x')
im_1b = temp.plot.\
    streamplot('x','z','ua_p','wa_p',hue='uw',vmin=-0.001,\
                     density=[0.4, 0.4],\
                    linewidth=3.5,arrowsize=4,\
                arrowstyle='fancy',cmap='PiYG_r')

crossxz['cloud'].where(crossxz['cloud'] > 0).plot(cmap='binary',\
                                add_colorbar=False,vmin=0,vmax=0.5)
cbar = im_1a.colorbar
cbar.remove()
cbar = im_1b.colorbar
cbar.remove()
plt.ylim([0,4500])
plt.tight_layout()

#%% Variances profiles 

#### Maybe plot the delta normalised by the control. 

day = '2020'
day = filtered[exp].time
klp = 5
f_scale = 100*domain_sml/(klp*2)
fig, axs = plt.subplots(1,4,figsize=(18,11))
for ide, exp in enumerate(exps):
# for ide, exp in enumerate(['noHGTQS_','noHGTQS_noSHAL_']):
    if exp == 'noHGTQS_':
        lab ='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab ='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab ='UVmixOFF'
    
    for idx,var in enumerate(['u','v','w','tke']):
        ### for each scene calculate mean flux in Sh (and Dp) pixels
        #   multiply by the fraction of Sh (and Dp) pixels
        #   average over time 
        
        if var == 'tke':
            ## all pixels 
            filtered[exp][var].sel(time=day).mean(('x','y'))\
                .mean('time').sel(klp=klp).isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                        label=lab,lw=3,c=col[ide],ls=':')
            harm3d[exp][var].sel(time=day).mean(('x','y'))\
                .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                        label=lab,lw=5,c=col[ide],ls='-',alpha=0.9)

            
        
        else: 
            ## delta variance 
            if exp != 'noHGTQS_':
                ## normalised
                ## filtered
                (((filtered[exp][var+'_pf']**2).sel(time=day).mean(('x','y','time')).interp(z=harm3d['noHGTQS_'].z)-\
                    (filtered['noHGTQS_'][var+'_pf']**2).sel(time=day).mean(('x','y','time')))\
                    /(filtered['noHGTQS_'][var+'_pf']**2).sel(time=day).mean(('x','y','time')))\
                    .sel(klp=klp).sel(z=slice(1,4000)).plot(y='z',ax=axs[idx],\
                                    label=lab+' '+str(np.around(f_scale/100,1))+'km',\
                                        lw=3,c=col[ide],ls=':')   
                ## 2.5 km resolutio 
                ((harm3d[exp][var+'_var'].sel(time=day).mean(('x','y','time')).interp(z=harm3d['noHGTQS_'].z)-\
                    harm3d['noHGTQS_'][var+'_var'].sel(time=day).mean(('x','y','time')))\
                    /harm3d['noHGTQS_'][var+'_var'].sel(time=day).mean(('x','y','time')))\
                    .sel(z=slice(1,4000)).plot(y='z',ax=axs[idx],\
                                    label=lab+' 2.5km',lw=3,c=col[ide],ls='-',alpha=0.9)
               ## delta at each time 
                # ## 2.5 km resolutio 
                # ((harm3d[exp][var+'_var'].sel(time=day).interp(z=harm3d['noHGTQS_'].z)-\
                #     harm3d['noHGTQS_'][var+'_var'].sel(time=day)).mean(('x','y','time')))\
                #     .sel(z=slice(1,4000)).plot(y='z',ax=axs[idx],\
                #                     label=lab+' 2.5km',lw=2,c=col[ide],ls='-',alpha=0.7)
                # ## filtered variance - filtered variance of control 
                # (((filtered[exp][var+'_pf']**2).sel(time=day).interp(z=harm3d['noHGTQS_'].z)-\
                #     (filtered['noHGTQS_'][var+'_pf']**2).sel(time=day)).mean(('x','y','time')))\
                #     .sel(klp=klp).sel(z=slice(1,4000)).plot(y='z',ax=axs[idx],\
                #                     label=lab+' '+str(np.around(f_scale/100,1))+'km',\
                #                         lw=3,c=col[ide],ls=':')  
                ## filtered variance - 2.5km variance of control 
                # (((filtered[exp][var+'_pf']**2).sel(time=day).interp(z=harm3d['noHGTQS_'].z)-\
                #     (harm3d['noHGTQS_'][var+'_var'].sel(time=day))).mean(('x','y','time')))\
                #     .sel(klp=klp).sel(z=slice(1,4000)).plot(y='z',ax=axs[idx],\
                #                     label=lab+' '+str(np.around(f_scale/100,1))+'km',\
                #                         lw=3,c=col[ide],ls=':')  
        
            ## actual variance 
            # ## 2.5 km resolutio 
            # harm3d[exp][var+'_var'].sel(time=day).mean(('x','y','time'))\
            #     .isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
            #                     label=lab+' 2.5km',lw=2,c=col[ide],ls='-')
            # ## filtered
            # (filtered[exp][var+'_pf']**2).sel(klp=klp,time=day).mean(('x','y','time'))\
            #     .isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
            #                     label=lab+' '+str(np.around(f_scale/1000,1))+'km',\
            #                         lw=2,c=col[ide],ls=':')


        axs[idx].axvline(0,c='k',lw=0.5)
        axs[idx].set_ylim([0,4000])
        axs[idx].set_xlabel(r'$m^2\,s^{-2}$')
        axs[idx].set_xlabel(r'fraction')

        if var == 'u':
            # axs[idx].set_title('Zonal wind variance',fontsize=24)
            axs[idx].set_title(r" $\Delta$ u' $^2$",fontsize=24)
        elif var == 'v':
            # axs[idx].set_title('Meridional wind variance',fontsize=24)
            axs[idx].set_title(r"$\Delta$ v' $^2$",fontsize=24)
        elif var == 'w':
            # axs[idx].set_title('Vertical velocity variance',fontsize=24)
            axs[idx].set_title(r"$\Delta$ w' $^2$",fontsize=24)
            axs[idx].set_xlim(left=0,right=4.5)
        elif var == 'hus':
            axs[idx].set_title('Specific humidity variance',fontsize=24)
        elif var == 'ta':
            axs[idx].set_title('Temperature variance',fontsize=24)
        elif var == 'tke':
            axs[idx].set_title('Resolved TKE',fontsize=24)
            axs[idx].set_xlabel(r'$m^{2}\,s^{-2}$')
            axs[idx].set_xlim(left=0)
        elif var == 'buoy':
            axs[idx].set_title('Buoyancy',fontsize=24)
            axs[idx].set_xlabel(r'? $N$ ?')
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        
# axs[1].set_xlim([-11,0])
# axs[2].ticklabel_format(style='sci', scilimits=(0,0))
# axs[0].set_xlim([-0.05,0.1])
axs[1].get_yaxis().set_visible(False) 
axs[2].get_yaxis().set_visible(False) 
axs[3].get_yaxis().set_visible(False) 
axs[0].legend(fontsize=18)
plt.tight_layout()


#%% Winds only inside a cloud
pltday =False
pltnight = False
alltime = True
fig, axs = plt.subplots(1,2,figsize=(12,11))
for ide, exp in enumerate(exps):
# for ide, exp in enumerate(['noHGTQS_','noHGTQS_noSHAL_']):
    if exp == 'noHGTQS_':
        lab ='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab ='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab ='UVmixOFF'
        
        
    updr_piels = harm3d[exp].where(harm3d[exp]['wa']>=1)
    dndr_piels = harm3d[exp].where(harm3d[exp]['wa']<=-1)
    
    
    
    Sh_pixels = harm3d[exp].where((harm3d[exp]['cl_top']<1500)&(harm3d[exp]['cl_top']>900))
    Dp_pixels = harm3d[exp].where((harm3d[exp]['cl_top']<3500)&(harm3d[exp]['cl_top']>2500))
    cl_pixels = harm3d[exp].where((harm3d[exp]['cl_top']<4000)&(harm3d[exp]['cl_top']>200))
    Nc_pixels = harm3d[exp].where((harm3d[exp]['cl_top'].isnull())|(harm3d[exp]['cl_top']>4000))
    Nc_pixels['count'] = ((harm3d[exp]['cl_top'].isnull())|(harm3d[exp]['cl_top']>6000)).sum(('x','y'))
    
    clcore_piels = cl_pixels.where(cl_pixels['wa']>0)
    rest_pixels = harm3d[exp].where((harm3d[exp]['cl_top'].isnull())|\
                                    (harm3d[exp]['cl_top']>4000)|\
                                        (harm3d[exp]['wa']<0))
    
    for idx,var in enumerate(['wa','buoy']):
        ### for each scene calculate mean flux in Sh (and Dp) pixels
        #   multiply by the fraction of Sh (and Dp) pixels
        #   average over time 
        
        if pltnight:
            plt.suptitle('Nighttime',fontsize=24)
            if var =='tke':
                ## all pixels 
                harm3d[exp][var].where((harm2d[exp]['time.hour']>= 20) | (harm2d[exp]['time.hour']< 6)).mean(('x','y'))\
                    .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                            label=lab,lw=3,c=col[ide],ls='-')
                        
            if (var == 'buoy') or (var =='wa'):
                clcore_piels[var].where((harm2d[exp]['time.hour']>= 20) | (harm2d[exp]['time.hour']< 6)).mean(('x','y'))\
                    .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                            label=lab,lw=3,c=col[ide],ls='-')
            if var =='wa':
                ## up and down pixels
                Nc_pixels[var].where((harm2d[exp]['time.hour']>= 20) | (harm2d[exp]['time.hour']< 6)).mean(('x','y'))\
                    .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                            lw=3,c=col[ide],ls='--')
        if pltday:
            plt.suptitle('Day time',fontsize=24)
            if var =='tke':
                ## all pixels 
                harm3d[exp][var].where((harm2d[exp]['time.hour']>= 6) & (harm2d[exp]['time.hour']< 20)).mean(('x','y'))\
                    .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                            label=lab,lw=3,c=col[ide],ls='-')
                        
            if (var == 'buoy') or (var =='wa'):
            # if (var == 'buoy'):
                clcore_piels[var].where((harm2d[exp]['time.hour']>= 6) & (harm2d[exp]['time.hour']< 20)).mean(('x','y'))\
                    .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                            label=lab,lw=3,c=col[ide],ls='-')
            if var =='wa':
                Nc_pixels[var].where((harm2d[exp]['time.hour']>= 6) & (harm2d[exp]['time.hour']< 20)).mean(('x','y'))\
                    .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                            lw=3,c=col[ide],ls='--')
        if alltime:
            if var =='tke':
                ## all pixels 
                harm3d[exp][var].mean(('x','y'))\
                    .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                            label=lab,lw=3,c=col[ide],ls='-')
                        
            if (var == 'buoy') or (var =='wa'):
            # if (var == 'buoy'):
                clcore_piels[var].mean(('x','y'))\
                    .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                            label=lab,lw=3,c=col[ide],ls='-')
            if var =='wa':
                rest_pixels[var].mean(('x','y'))\
                    .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                            lw=3,c=col[ide],ls='--')
            
                
        # ## deep pixels 
        # Dp_pixels[var].mean(('x','y'))\
        #     .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
        #                             label=lab+'_Dp',lw=3,c=col[ide],ls='-')
    
        ## shallow pixels 
        # Sh_pixels[var].mean(('x','y'))\
        #     .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
        #                             label=lab+'_Sh',lw=3,c=col[ide],ls='--')
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
            axs[idx].set_xlabel(r'$N$ ')
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        
# axs[1].set_xlim([-11,0])
# axs[2].set_xlim([-3,0])
# axs[0].set_xlim([-0.05,0.1])
axs[1].get_yaxis().set_visible(False) 
axs[1].ticklabel_format(style='sci', scilimits=(0,0))
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
        lab ='UVmixOFF'
    Sh_pixels = harm3d[exp].where((harm3d[exp]['cl_top']<1500)&(harm3d[exp]['cl_top']>900))
    Dp_pixels = harm3d[exp].where((harm3d[exp]['cl_top']<3500)&(harm3d[exp]['cl_top']>2500))
    Nc_pixels = harm3d[exp].where((harm3d[exp]['cl_top'].isnull())|(harm3d[exp]['cl_top']>6000))
    Nc_pixels['count'] = ((harm3d[exp]['cl_top'].isnull())|(harm3d[exp]['cl_top']>6000)).sum(('x','y'))
    
    for idx,var in enumerate(['u','v']):
        ### for each scene calculate mean flux in Sh (and Dp) pixels
        #   multiply by the fraction of Sh (and Dp) pixels
        #   average over time 
        
        
        ## deep pixels 
        ((Dp_pixels[var+'w']+Dp_pixels[var+'_flx_param_tot']).mean(('x','y')) * Dp_pixels.count(('x','y'))['cl_top']\
                                    /(len(harm3d[exp].x) * len(harm3d[exp].y)))\
            .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                    label=lab+'_Dp',lw=3,c=col[ide],ls='-')
        ## shallow pixels 
        ((Sh_pixels[var+'w']+Sh_pixels[var+'_flx_param_tot']).mean(('x','y')) * Sh_pixels.count(('x','y'))['cl_top']\
                                    /(len(harm3d[exp].x) * len(harm3d[exp].y)))\
            .mean('time').isel(z=slice(1,-1)).plot(y='z',ax=axs[idx],\
                                    label=lab+'_Sh',lw=3,c=col[ide],ls='--')
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