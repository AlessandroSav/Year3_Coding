#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:13:10 2022

@author: acmsavazzi
"""
#%%                             Libraries
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import gc
import os
from glob import glob
import sys
import argparse
import xarray as xr
sys.path.insert(1, '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Coding/Scale_separation/')
from functions import *

sys.path.insert(1, '/Users/acmsavazzi/Documents/WORK/My_source_codes')
from My_thermo_fun import *




#%% initial 
dt          = 75                 # model  timestep [seconds]
step        = 3600             # output timestep [seconds]
domain_name = 'BES'
lat_select  = 13.2806    # HALO center 
lon_select  = -57.7559   # HALO center 
domain      = 200            # km
srt_time   = np.datetime64('2020-01-03T00:30')
end_time   = np.datetime64('2020-01-03T23')

months = ['01',]
month='0*'

exps = ['noHGTQS_','noHGTQS_noUVmix_','noHGTQS_noSHAL_']
exps = ['noHGTQS_','noHGTQS_noUVmix_']
col=['k','r','g']
sty=['--','-',':']

levels = 'z'      ## decide wether to open files model level (lev) or 
                    ## already interpolate to height (z)
my_harm_dir     = '/Users/acmsavazzi/Documents/WORK/PhD_Year3/DATA/HARMONIE/'
write_dir = os.path.abspath('{}/../../../DATA/HARMONIE/')+'/'

#%%
itmin = 1
itmax = 24
di    = 8       # delta time for plots 
zmin = 0
zmax = 5000
store = False
# klps = [187.5,75,30,10]

klps = [10,]         ## halfh the number of grids after coarsening 
#domain size from namotions
xsize      =  200000 # m
ysize      =  200000 # m

dx = 2500               # model resoluton in m
dt = 75                 # model  timestep [seconds]
step = 3600             # output timestep [seconds]
domain_name = 'BES'
lat_select = 13.2806    # HALO center 
lon_select = -57.7559   # HALO center 
buffer = 30             # buffer of 150 km around (75 km on each side) the gridpoint 30 * 2 * 2.5 km

##############################
##### NOTATIONS #####
# _av = domain average 
# _p  = domain perturbation (prime)
# sf  = sub filter scale 
# f   = filter scale

# t   = total grid laevel 
# m   = middle of the grid 
##############################
#%%
### Import Harmonie data
print("Reading HARMONIE.") 
## new files on height levels are empty ### !!!
## is it a problem of the interpolation? if yes: open the file _lev_all.nc 
## and do the intrpolation here. 
harm3d={}
#
spec_u  = {}
spec_v  = {}
spec_w  = {}
spec_uw = {}
spec_vw = {}
uw_p_av = {}
vw_p_av = {}
for exp in exps: 
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

#%% Time and hight for plotting
    # 
    plttime = harm3d[exp].time.values
    z = harm3d[exp].z.values
    pltheights    = [1500,]
    pltz = []
    for ii in pltheights:
        pltz = np.append(pltz,np.argmin(abs(z-ii))).astype(int)
    #%% filtering 
    xt = harm3d[exp].x
    dx = xsize/xt.size  # in metres
    ### Loop in time
    for i in range(len(plttime)):
        print('Processing time step', i+1, '/', len(plttime))
        u = harm3d[exp]['ua'].sel(time=plttime[i]).values
        v = harm3d[exp]['va'].sel(time=plttime[i]).values
        w = harm3d[exp]['wa'].sel(time=plttime[i]).values
    
        u_av  = np.mean(u,axis=(1,2))
        v_av  = np.mean(v,axis=(1,2))
        w_av  = np.mean(w,axis=(1,2))
        u_p   = u - u_av[:,np.newaxis,np.newaxis]
        v_p   = v - v_av[:,np.newaxis,np.newaxis]
        w_p   = w - w_av[:,np.newaxis,np.newaxis]
        
        for k in range(len(klps)):
            print('Processing scale', k+1, '/', len(klps))
            klp=klps[k]
            #
            if klp > 0:
                f_scale = xsize/(klp*2)  # m
            elif klp == 0:
                f_scale = xsize
            else: print('Warning: Cutoff wavenumber for lw-pass filter smaller than 0.')
            
            # Mask for low-pass filtering
            circ_mask = np.zeros((xt.size,xt.size))
            rad = getRad(circ_mask)
            circ_mask[rad<=klp] = 1
        
            #filtered U
            u_pf  = lowPass(u_p, circ_mask)
            u_psf = u_p - u_pf
    #%% Initialize np arrays
    # Wave lenght related variables
    N = harm3d[exp].x.size; N2 = N//2
    ## initialise variables for spectral analysis 
    spec_u[exp]  = np.zeros((len(plttime),len(pltz),N2))
    spec_v[exp]  = np.zeros((len(plttime),len(pltz),N2))
    spec_w[exp]  = np.zeros((len(plttime),len(pltz),N2))
    spec_uw[exp] = np.zeros((len(plttime),len(pltz),N2))
    spec_vw[exp] = np.zeros((len(plttime),len(pltz),N2))
    
    uw_p_av[exp] = np.zeros((len(plttime),len(z)))
    vw_p_av[exp] = np.zeros((len(plttime),len(z)))
#%% Loop in time
    for i in range(len(plttime)):
        print('Processing time step', i+1, '/', len(plttime))
        
        # 3D fields
        # qt = dl.load_qt(plttime[i], izmin, izmax)
        # thlp = dl.load_thl(plttime[i], izmin, izmax)
        # qlp = dl.load_ql(plttime[i], izmin, izmax)
        u = harm3d[exp]['ua'].sel(time=plttime[i]).values
        v = harm3d[exp]['va'].sel(time=plttime[i]).values
        w = harm3d[exp]['wa'].sel(time=plttime[i]).values
    
        u_av  = np.mean(u,axis=(1,2))
        v_av  = np.mean(v,axis=(1,2))
        w_av  = np.mean(w,axis=(1,2))
        u_p   = u - u_av[:,np.newaxis,np.newaxis]
        v_p   = v - v_av[:,np.newaxis,np.newaxis]
        w_p   = w - w_av[:,np.newaxis,np.newaxis]
        
        uw_p  = u_p * w_p
        vw_p  = v_p * w_p
        
        uw_p_av[exp][i,:] = np.mean(uw_p,axis=(1,2))
        vw_p_av[exp][i,:] = np.mean(vw_p,axis=(1,2))
    
        ### spectral analysis at specific levels
        for iz in range(len(pltz)):
            print('Computing spectra at time step', i+1, '/', len(plttime),
                  ', height', iz+1,'/',len(pltz))
            k1d,spec_u[exp][i,iz,:]  = compute_spectrum(u[pltz[iz],:,:], dx)
            k1d,spec_v[exp][i,iz,:]  = compute_spectrum(v[pltz[iz],:,:], dx)
            k1d,spec_w[exp][i,iz,:]  = compute_spectrum(w[pltz[iz],:,:], dx)
            k1d,spec_uw[exp][i,iz,:] = compute_spectrum(u[pltz[iz],:,:], dx,\
                                                   cloud_scalar_2=w[pltz[iz],:,:])
            k1d,spec_vw[exp][i,iz,:] = compute_spectrum(v[pltz[iz],:,:], dx,\
                                                   cloud_scalar_2=w[pltz[iz],:,:])
    
    if store:        
        np.save(write_dir+exp[0:-1]+'/spec_time_HAR.npy',plttime)
        np.save(write_dir+'spec_plttime_HAR.npy',plttime)
        np.save(write_dir+exp[0:-1]+'/spec_pltz_HAR.npy',pltz)
        # np.save(lp+'/spec_zt_HAR.npy',ztlim)
        
        np.save(write_dir+exp[0:-1]+'/spec_u_HAR.npy',spec_u[exp])
        np.save(write_dir+exp[0:-1]+'/spec_v_HAR.npy',spec_v[exp])
        np.save(write_dir+exp[0:-1]+'/spec_w_HAR.npy',spec_w[exp])
        np.save(write_dir+'spec_uw_HAR.npy',spec_uw)
        np.save(write_dir+'spec_vw_HAR.npy',spec_vw)
        np.save(write_dir+'spec_k1d_HAR.npy',k1d)
        
        np.save(write_dir+exp[0:-1]+'/scale_up_wp_HAR.npy',uw_p_av[exp])
        np.save(write_dir+exp[0:-1]+'/scale_vp_wp_HAR.npy',vw_p_av[exp])
        np.save(write_dir+'scale_zt_HAR.npy',z)


#%% PLOTTING
xsize = N*dx
lam = (xsize*(k1d/np.pi))
nx  = np.pi/k1d
# k1d = frquency
# lam = wavelenght 
# lam = xsize when k1d=pi

#%%
wavelenght = np.pi/(k1d*1000)
## Spectral analysis 
# for it in range(len(s_plttime)):
iz=min(1,len(pltheights)-1)
it = min(5,len(plttime)-1)

# plt.figure()
# for iz in range(len(pltz)):
#     plt.plot(wavelenght,k1d*spec_uw[it,iz,:],\
#              ls='--',label='z: '+str(int(z[pltz[iz]]))+' m')
# plt.axvline(2.5,c='k',lw=0.5)
# plt.xscale('log')
# plt.xlabel('Wavelength  [km]',fontsize=17)
# plt.ylabel('Spectral density',fontsize=17)
# plt.legend(fontsize=15)
# # plt.ylim([None,0.11])
# plt.title('Spectra uw at '+np.datetime_as_string(time[it], unit='m'),fontsize=18)
# plt.title('Spectra uw.  Time: Feb-3 21:00UTC',fontsize=18)
# plt.savefig(save_dir+'uw_spectra.pdf')

# cumulative
plt.figure()
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='NoMom'
    plt.plot(wavelenght,1- np.cumsum(k1d*spec_uw[exp][it,iz,:])\
                /max(np.cumsum(k1d*spec_uw[exp][it,iz,:]))\
                  ,label=lab)
    ## time average     
    plt.plot(wavelenght,1- np.cumsum(k1d*spec_uw[exp][:,iz,:].mean(0))\
                /max(np.cumsum(k1d*spec_uw[exp][:,iz,:].mean(0)))\
                  ,label=lab,c=col[ide])
    # plt.plot(wavelenght,np.cumsum(k1d*spec_uw[it,iz,:])\
    #             /max(np.cumsum(k1d*spec_uw[it,iz,:]))\
    #               ,label=None,ls='--')
plt.axvline(2.5,c='k',lw=0.5)
plt.xscale('log')
plt.xlabel('Wavelength  [km]')
plt.ylabel('%')
plt.legend()
plt.title('Cumulative spectra uw at '+str(int(z[pltz[iz]]))+' m')
    
#%% 
for var in ['uflx_turb']:
    fig, axs = plt.subplots(figsize=(19,5))
    harm_clim_avg[var].sel(z=slice(0,4500)).plot(x='time',vmax=0.05)
    for ii in np.arange(srt_time, end_time):
        plt.axvline(x=ii,c='k')
    plt.suptitle('Param momentum flux form HARMONIE')




