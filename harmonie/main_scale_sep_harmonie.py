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
end_time   = np.datetime64('2020-01-29T23')

months = ['01',]
month='0*'

# exps = ['noHGTQS_','noHGTQS_noUVmix_','noHGTQS_noSHAL_']
# exps = ['noHGTQS_','noHGTQS_noUVmix_']
exps = ['noHGTQS_noSHAL_']
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
read_existing_filtered = False
read_existing_spectral = False
store = True 
# klps = [187.5,75,30,10]

klps = [10,5]         ## halfh the number of grids after coarsening 
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
filtered = {}
spectral ={}
for exp in exps: 
    print('Processing exp '+exp)
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
        
    if read_existing_filtered == True:
        filtered[exp] = xr.open_mfdataset(my_harm_dir+exp[0:-1]+'/filtered_'+exp[0:-1]+'.nc')
    if read_existing_spectral == True:
        spectral[exp] = xr.open_mfdataset(my_harm_dir+exp[0:-1]+'/spectral_'+exp[0:-1]+'.nc')

#%% calculated resolved fluxes
    for var in ['ua','va','wa','hus','ta']:
        harm3d[exp][var+'_p'] = harm3d[exp][var] - harm3d[exp][var].mean(['x','y'])
    
    harm3d[exp]['div'] = harm3d[exp]['ua_p'].differentiate('x') + harm3d[exp]['va_p'].differentiate('y')
    
    harm3d[exp]['uw']= harm3d[exp]['ua_p']*harm3d[exp]['wa_p']
    harm3d[exp]['vw']= harm3d[exp]['va_p']*harm3d[exp]['wa_p']
    harm3d[exp]['tw']= harm3d[exp]['ta_p']*harm3d[exp]['wa_p']
    harm3d[exp]['qw']= harm3d[exp]['hus_p']*harm3d[exp]['wa_p']

    vars_ = ['u','v']
    for var in vars_:
        ## save a variable for total parameterised momentum flux

        if exp == 'noHGTQS_noUVmix_':
            #deaccumulate 
            harm3d[exp][var+'flx_turb']     = harm3d[exp][var+'flx_turb'].diff('time') * step**-1
            harm3d[exp][var+'flx_conv_dry'] = harm3d[exp][var+'flx_conv_dry'].diff('time') * step**-1
            # sum parameterised components 
            harm3d[exp][var+'_flx_param_tot']=  harm3d[exp][var+'flx_turb']+\
                                                harm3d[exp][var+'flx_conv_dry']      
        else:
            #deaccumulate 
            harm3d[exp][var+'flx_turb']         = harm3d[exp][var+'flx_turb'].diff('time') * step**-1
            harm3d[exp][var+'flx_conv_dry']     = harm3d[exp][var+'flx_conv_dry'].diff('time') * step**-1
            harm3d[exp][var+'flx_conv_moist']   = harm3d[exp][var+'flx_conv_moist'].diff('time') * step**-1
            # sum parameterised components 
            harm3d[exp][var+'_flx_param_tot']=  harm3d[exp][var+'flx_turb']+\
                                                harm3d[exp][var+'flx_conv_moist']+\
                                                harm3d[exp][var+'flx_conv_dry']
#%% Time and hight for plotting
    # 
    plttime = harm3d[exp].time.values
    plttime_filter = harm3d[exp].where(harm3d[exp]['time.hour']==12,drop=True).time.values
    z = harm3d[exp].z.values
    pltheights    = [1500,]
    pltz = []
    for ii in pltheights:
        pltz = np.append(pltz,np.argmin(abs(z-ii))).astype(int)

    # Wave lenght related variables
    N = harm3d[exp].x.size; N2 = N//2
    #%% filtering 
    if read_existing_filtered ==False:
    
        ## initialise variables for spectral analysis 
        u_pf  = np.zeros((len(z),len(harm3d[exp].x),len(harm3d[exp].y),\
                               len(plttime_filter),len(klps)))
        v_pf  = np.zeros((len(z),len(harm3d[exp].x),len(harm3d[exp].y),\
                               len(plttime_filter),len(klps)))
        w_pf  = np.zeros((len(z),len(harm3d[exp].x),len(harm3d[exp].y),\
                               len(plttime_filter),len(klps)))
        div_f = np.zeros((len(z),len(harm3d[exp].x),len(harm3d[exp].y),\
                               len(plttime_filter),len(klps)))
        
        
        xt = harm3d[exp].x
        dx = xsize/xt.size  # in metres
        ### Loop in time
        for it in range(len(plttime_filter)):
            print('Processing time step', it+1, '/', len(plttime_filter))
            u = harm3d[exp]['ua'].sel(time=plttime_filter[it]).values
            v = harm3d[exp]['va'].sel(time=plttime_filter[it]).values
            w = harm3d[exp]['wa'].sel(time=plttime_filter[it]).values
            
            div = harm3d[exp]['div'].sel(time=plttime_filter[it]).values
        
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
                u_pf_temp  = lowPass(u_p, circ_mask)
                v_pf_temp  = lowPass(v_p, circ_mask)
                w_pf_temp  = lowPass(w_p, circ_mask)
                div_f_temp  = lowPass(div, circ_mask)
                
                u_pf[:,:,:,it,k] = u_pf_temp
                v_pf[:,:,:,it,k] = v_pf_temp
                w_pf[:,:,:,it,k] = w_pf_temp
                div_f[:,:,:,it,k] = div_f_temp
                
        ### filtered
        filtered_exp = xr.DataArray(
                    u_pf,
                    coords={'z': z,'x':harm3d[exp].x,'y':harm3d[exp].y,
                            'time':plttime_filter,'klp':klps}, 
                    dims=["z", "x", "y","time","klp"]
                    )
        filtered_exp = filtered_exp.to_dataset(name='u_pf')
        filtered_exp.time.attrs["units"] = "Local Time"
        ## 
        filtered_exp['v_pf'] = xr.DataArray(
                    v_pf,
                    coords={'z': z,'x':harm3d[exp].x,'y':harm3d[exp].y,
                            'time':plttime_filter,'klp':klps}, 
                    dims=["z", "x", "y","time","klp"]
                    )
        ## 
        filtered_exp['w_pf'] = xr.DataArray(
                    w_pf,
                    coords={'z': z,'x':harm3d[exp].x,'y':harm3d[exp].y,
                            'time':plttime_filter,'klp':klps}, 
                    dims=["z", "x", "y","time","klp"]
                    )
        ##div
        filtered_exp['div_f'] = xr.DataArray(
                    div_f,
                    coords={'z': z,'x':harm3d[exp].x,'y':harm3d[exp].y,
                            'time':plttime_filter,'klp':klps}, 
                    dims=["z", "x", "y","time","klp"]
                    )
        
        if store:        
            print('Saving filtered data ')
            filtered_exp.to_netcdf(write_dir+exp[0:-1]+'/filtered_'+exp[0:-1]+'.nc')
        
        #%% Initialize np arrays
    if read_existing_spectral == False:
        ## initialise variables for spectral analysis 
        spec_u[exp]  = np.zeros((len(plttime),len(pltz),N2))
        spec_v[exp]  = np.zeros((len(plttime),len(pltz),N2))
        spec_w[exp]  = np.zeros((len(plttime),len(pltz),N2))
        spec_uw[exp] = np.zeros((len(plttime),len(pltz),N2))
        spec_vw[exp] = np.zeros((len(plttime),len(pltz),N2))
    #%% Loop in time
        for i in range(len(plttime)):            
            # 3D fields
            # qt = dl.load_qt(plttime[i], izmin, izmax)
            # thlp = dl.load_thl(plttime[i], izmin, izmax)
            # qlp = dl.load_ql(plttime[i], izmin, izmax)
            u = harm3d[exp]['ua'].sel(time=plttime[i]).values
            v = harm3d[exp]['va'].sel(time=plttime[i]).values
            w = harm3d[exp]['wa'].sel(time=plttime[i]).values
        
            # u_av  = np.mean(u,axis=(1,2))
            # v_av  = np.mean(v,axis=(1,2))
            # w_av  = np.mean(w,axis=(1,2))
            # u_p   = u - u_av[:,np.newaxis,np.newaxis]
            # v_p   = v - v_av[:,np.newaxis,np.newaxis]
            # w_p   = w - w_av[:,np.newaxis,np.newaxis]
            
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
#%% Data to xarray     
    ### spectral 
        ## u
        spec_exp = xr.DataArray(
                    spec_u[exp],
                    coords={'time': plttime,'z':z[pltz],'klp':k1d}, 
                    dims=["time", "z", "klp"]
                    )
        spec_exp = spec_exp.to_dataset(name='spec_u')
        spec_exp.time.attrs["units"] = "Local Time"
        ## v 
        spec_exp['spec_v'] = xr.DataArray(
                    spec_v[exp],
                    coords={'time': plttime,'z':z[pltz],'klp':k1d}, 
                    dims=["time", "z", "klp"]
                    )
        ## w    
        spec_exp['spec_w'] = xr.DataArray(
                    spec_w[exp],
                    coords={'time': plttime,'z':z[pltz],'klp':k1d}, 
                    dims=["time", "z", "klp"]
                    )
        ## uw    
        spec_exp['spec_uw'] = xr.DataArray(
                    spec_uw[exp],
                    coords={'time': plttime,'z':z[pltz],'klp':k1d}, 
                    dims=["time", "z", "klp"]
                    )
        ## vw    
        spec_exp['spec_vw'] = xr.DataArray(
                    spec_vw[exp],
                    coords={'time': plttime,'z':z[pltz],'klp':k1d}, 
                    dims=["time", "z", "klp"]
                    )
        ## vw    
        spec_exp['spec_vw'] = xr.DataArray(
                    spec_vw[exp],
                    coords={'time': plttime,'z':z[pltz],'klp':k1d}, 
                    dims=["time", "z", "klp"]
                    )
        if store:        
            print('Saving data')
            spec_exp.to_netcdf(write_dir+exp[0:-1]+'/spectral_'+exp[0:-1]+'.nc')



#%% 
sum_spectral = {}
for exp in exps:    
    sum_spectral[exp]= spectral[exp].cumsum(dim='klp')
    sum_spectral[exp]['klp']= spectral[exp]['klp']

#%% PLOTTING
xsize = N*dx
lam = (xsize*(spectral[exp].klp/np.pi))
nx  = np.pi/spectral[exp].klp
# k1d = frquency
# lam = wavelenght 
# lam = xsize when k1d=pi

#%%
my_res = 2.5 # resoltion in km
klp = np.pi/(my_res*1000)

var = 'spec_uw'
plt.figure(figsize=(12,5))
for ide, exp in enumerate(exps):

    plt.plot(spectral[exp].time,
             np.cumsum(spectral[exp].klp*spectral[exp][var])\
        .sel(klp = klp ,method='nearest').isel(z=0))
        
    # spectral[exp].sel(klp = klp ,method='nearest').isel(z=0)[var].\
    #     plot(lw=2,c=col[ide],ls=sty[ide])

#%%
wavelenght = np.pi/(spectral[exp].klp*1000)
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

var = 'u'
param = abs(harm3d[exp].isel(z=pltz).mean(('x','y'))[var+'_flx_param_tot'].values)


temp_flx = globals()['spec_'+var+'w']

# cumulative
plt.figure()
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    ## time average     
    ## normalised 0-1
    # plt.plot(wavelenght,1- np.cumsum(spectral[exp].klp*spectral[exp]['spec_'+var+'w'].mean('time'))\
    #             /max(np.cumsum(spectral[exp].klp*spectral[exp]['spec_'+var+'w'].mean('time'))+np.nanmean(param))\
    #               ,label=lab,c=col[ide])
        
        
    flux     =  (spectral[exp].klp*spectral[exp]['spec_'+var+'w']).cumsum(dim='klp')
    max_flux = ((spectral[exp].klp*spectral[exp]['spec_'+var+'w']).cumsum(dim='klp')+param).max('klp')


    plt.plot(wavelenght,(1- flux\
                /max_flux).mean('time')\
                  ,label=lab,c=col[ide],ls=':')
        
    plt.plot(wavelenght,(max_flux - flux).mean('time').isel(z=iz)\
                  ,label=lab,c=col[ide],ls='-')
    
        
        
    # ##         
    # plt.plot(wavelenght,np.cumsum(\
    #          spectral[exp]['spec_'+var+'w'].mean('time').isel(z=iz)).values.max() - \
    #          np.cumsum(\
    #          spectral[exp]['spec_'+var+'w'].mean('time').isel(z=iz)),
    #          label=None,ls='--',c=col[ide])

plt.axvline(2.5,c='k',lw=0.5)
plt.xscale('log')
# plt.ylim([0,1])
plt.xlabel('Wavelength  [km]')
plt.ylabel('%')
plt.legend()
plt.title('Cumulative spectra uw at '+str(int(z[pltz[iz]]))+' m')


    
#%% filtered fields 
klp = klps[1]
it = plttime_filter[15]
iz = 3000
var = 'w_pf'
fig, axs = plt.subplots(1,2,figsize=(19,7))
for ide, exp in enumerate(exps):
    if exp == 'noHGTQS_':
        lab='Control'
    elif exp == 'noHGTQS_noSHAL_':
        lab='NoShal'
    elif exp == 'noHGTQS_noUVmix_':
        lab='UVmixOFF'
    filtered[exp].sel(klp=klp,time=it).sel(z=iz,method='nearest')[var]\
        .plot(ax=axs[ide],vmax=0.3)
    axs[ide].set_title(lab,fontsize=23)


#%% DIVERGENCE ['dudx']+['dvdy']





iz = 7
plt.figure(figsize=(13,5))
for ide, exp in enumerate(exps):
    filtered[exp]['div'].var(('x','y')).isel(klp=0,z=iz).plot(c=col[ide],lw=3)
    plt.suptitle('variance of divergence')
    







