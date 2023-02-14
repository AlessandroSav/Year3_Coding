#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:13:10 2022

@author: acmsavazzi
"""
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import gc
import os
import sys
sys.path.insert(1, os.path.abspath('.'))
from functions import *
from dataloader import DataLoaderDALES
import argparse
import xarray as xr
#%%
##### NOTATIONS
# _av = domain average 
# _p  = domain perturbation (prime)
# sf  = sub filter scale 
# f   = filter scale
# t   = total grid laevel 
# m   = middle of the grid 

# for casenr in ['001','002','003','004','005','006','007','008','009','010',\
#                 '011','012','013','014','015','016','017']:
# for casenr in ['001','011','012','013','014','015','016','017']:
for casenr in ['004',]:
   
    print('################## \n ### Exp_'+casenr+'###')
    # pltheights = 200  # in m  # height at which to compute the scale separation 
    pltheights = 'subCL'
    # pltheights = 'midCL'
    
    ## running on staffumbrella
    # lp = os.path.abspath('../../../Raw_Data/Les/Eurec4a/20200202_12/Exp_'+casenr)
    # save_dir   = lp
    ## running on Local
    lp =  '/Users/acmsavazzi/Documents/Mount1/Raw_Data/Les/Eurec4a/20200202_12_clim/Exp_'+casenr
    # lp =  '/Users/acmsavazzi/Documents/WORK/PhD_Year1/DATA/DALES/DALES_atECMWF/outputs/20200202_12_clim/Exp_'+casenr
    save_dir   = '/Users/acmsavazzi/Documents/WORK/PhD_Year3/DATA/DALES/scalesep/Exp_'+casenr
    
    itmin = 1       # first timestep to consider
    itmax = 24      # last timestep to consider
    di    = 2       # delta time to consider (1-> 30 min) 
    store = False
    #domain size from namotions
    xsize      =  150000 # m
    ysize      =  150000 # m
    cu         = -6 # m/s
    vu         = 0 # m/s
    
    nchunks_x = 1
    nchunks_y = 1
    xsize = xsize/nchunks_x
    ysize = ysize/nchunks_y
    
    x= np.floor(xsize/1000/2)
    
    klps = [30,]         ## halfh the number of grids after coarsening (klp=30 -> coarsen to 2.5km)
    nr_klps = 10 # number of equally spaced filters
    klps = xsize/1000/(2*np.logspace(-1,2,nr_klps,base=10))
    # klps = xsize/1000/(2*np.logspace(-1,np.log10(x)/np.log10(10),nr_klps,base=10))
    klps = np.sort(np.unique(np.append(klps,[0.5,30])))[::-1]
    
    #%% import profiles
    srt_time   = np.datetime64('2020-02-01T20')
    profiles = xr.open_mfdataset(lp+'/profiles.'+casenr+'.nc', combine='by_coords')
    profiles['time'] = srt_time + profiles.time.astype("timedelta64[s]")
    profiles.time.attrs["units"] = "Local Time"
    #%%
    dl = DataLoaderDALES(lp,casenr=casenr)
    time = dl.time
    zt = dl.zt
    zm = dl.zm
    xt = dl.xt
    xm = dl.xm
    yt = dl.yt
    ym = dl.ym
    
    # FIXME temporary hardcoding of dx/dy for data that does not have xf/yf as variables
    dx = xsize/xt.size*nchunks_x  # in metres
    dy = ysize/yt.size*nchunks_y  # in metres
    ##########  All these dz to be checked !!!
    # Vertical differences
    dzt = np.zeros(zm.shape)
    dzt[:-1] = np.diff(zm) # First value is difference top 1st cell and surface
    dzt[-1] = dzt[-2]
    dzm = np.zeros(zt.shape)
    dzm[1:] = np.diff(zt) # First value is difference mid 1st cell and mid 1st cell below ground
    dzm[0] = 2*zt[1]
    
    plttime = np.arange(itmin, itmax, di)
    # plttime = np.unique(np.sort(np.append(plttime,3)))
    
    ##################
    ############
    if (int(casenr) % 2) == 0:
        start_d = int(casenr)//2 +1
    else:
        start_d = int(casenr)//2 +2
    start_h = 0
    ###### Exp_001 and Exp_002 have wrong times
    if casenr == '001' or casenr=='002':
        time = time + 34385400  +1800
    time = np.array(time,dtype='timedelta64[s]') + (np.datetime64('2020-02-'+str(start_d).zfill(2)+'T'+str(start_h).zfill(2)+':00'))
    ############
    ##################
    
    #### initialise variables for scale separation
    u_p_avtime          = np.zeros((plttime.size,nchunks_x,nchunks_y))
    v_p_avtime          = np.zeros((plttime.size,nchunks_x,nchunks_y))
    w_p_avtime          = np.zeros((plttime.size,nchunks_x,nchunks_y))
    #
    u_pf_avtime         = np.zeros((len(klps),plttime.size,nchunks_x,nchunks_y))
    v_pf_avtime         = np.zeros((len(klps),plttime.size,nchunks_x,nchunks_y))
    w_pf_avtime         = np.zeros((len(klps),plttime.size,nchunks_x,nchunks_y))
    #
    u_pfw_pf_avtime     = np.zeros((len(klps),plttime.size,nchunks_x,nchunks_y))
    u_psfw_psf_avtime   = np.zeros((len(klps),plttime.size,nchunks_x,nchunks_y))
    v_pfw_pf_avtime     = np.zeros((len(klps),plttime.size,nchunks_x,nchunks_y))
    v_psfw_psf_avtime   = np.zeros((len(klps),plttime.size,nchunks_x,nchunks_y))
    #
    u_pw_p_avtime        = np.zeros((plttime.size,nchunks_x,nchunks_y))
    v_pw_p_avtime        = np.zeros((plttime.size,nchunks_x,nchunks_y))
    
    #%% 
    # Before looping in time reduce the size of profiles and interpolate to 
    # the available time
    profiles = profiles.interp(time=time)
    
    # Now calculate the Boundary layer height  as the height above 
    # cloudbase where ql is back at 0
    # maxql = profiles['ql'].max('z')    # max of humiidity 
    # imax = profiles['ql'].argmax('z')  # index of max humidity 
    zmax = profiles['ql'].idxmax('zt')  # height of max humidity
    temp = profiles['ql'].where(profiles['zt']>=zmax)
    hc_ql = temp.where(lambda x: x<0.0000001).idxmax(dim='zt')  #height of zero humidity after maximum
    
    # Now calculate cloud base as the height where ql becomes >0 
    temp = profiles['ql'].where(profiles['zt']<=zmax)
    cl_base = temp.where(lambda x: x<0.0000001).idxmax(dim='zt')  
    #%% Loop in time
    ## make pltz time dependent so that ath each time in plttime you can select cloud top and cloud base
    
    for i in range(len(plttime)):
        print('Processing time step', i+1, '/', len(plttime))
        ########## first define the height 
        if type(pltheights) == int:
            pltz    = np.argmin(abs(zt.values-pltheights)).astype(int)
        elif pltheights == 'subCL':
            pltz_ideal = (cl_base/2).sel(time=time[i]).values
            pltz    = np.argmin(abs(zt.values-pltz_ideal)).astype(int)
        elif pltheights ==' midCL':
            pltz_ideal = ((hc_ql-cl_base)/2 + cl_base).sel(time=time[i]).values
            pltz    = np.argmin(abs(zt.values-pltz_ideal)).astype(int)
        else: 
            sys.exit("errors: Variable pltheights not recognised !")
        ######## 
        
        # 3D fields
        # qt = dl.load_qt(plttime[i], izmin, izmax)
        wm1 = dl.load_wm(plttime[i], pltz)
        wm2 = dl.load_wm(plttime[i],pltz+1)
        # thlp = dl.load_thl(plttime[i], izmin, izmax)
        # qlp = dl.load_ql(plttime[i], izmin, izmax)
        u = dl.load_u(plttime[i], pltz) + cu
        v = dl.load_v(plttime[i], pltz) + vu
        w = (wm1 + wm2)*0.5 ### grid is stretched !!! # from w at midlevels caclculate w at full levels
        print('Fields loaded')
        
        for ii in range(0,nchunks_x):
            for jj in range(0,nchunks_y):
                
                print('Processing chunck '+str(ii)+','+str(jj))
                # averages and perturbations 
                u_av  = np.mean(u[ii*int(xt.size/nchunks_x):(ii+1)*int(xt.size/nchunks_x),\
                                  jj*int(xt.size/nchunks_x):(jj+1)*int(xt.size/nchunks_x)],axis=(0,1))
                v_av  = np.mean(v[ii*int(xt.size/nchunks_x):(ii+1)*int(xt.size/nchunks_x),\
                                  jj*int(xt.size/nchunks_x):(jj+1)*int(xt.size/nchunks_x)],axis=(0,1))
                w_av  = 0
                u_p   = u[ii*int(xt.size/nchunks_x):(ii+1)*int(xt.size/nchunks_x),\
                                  jj*int(xt.size/nchunks_x):(jj+1)*int(xt.size/nchunks_x)] - u_av[np.newaxis,np.newaxis]
                v_p   = v[ii*int(xt.size/nchunks_x):(ii+1)*int(xt.size/nchunks_x),\
                                  jj*int(xt.size/nchunks_x):(jj+1)*int(xt.size/nchunks_x)] - v_av[np.newaxis,np.newaxis]
                w_p   = w[ii*int(xt.size/nchunks_x):(ii+1)*int(xt.size/nchunks_x),\
                                  jj*int(xt.size/nchunks_x):(jj+1)*int(xt.size/nchunks_x)] - w_av
        

        
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
                    circ_mask = np.zeros((int(xt.size/nchunks_x),int(yt.size/nchunks_y)))
                    rad = getRad(circ_mask)
                    circ_mask[rad<=klp] = 1
                
                    #filtered U
                    u_pf  = lowPass(u_p, circ_mask)
                    u_psf = u_p - u_pf
                    #filtered V
                    v_pf = lowPass(v_p, circ_mask)
                    v_psf = v_p - v_pf
                    #filtered W total level
                    w_pf  = lowPass(w_p, circ_mask)
                    w_psf = w_p - w_pf   
                     
                    ## Fluxes
                    # filtered and sub-filtered fluxes without the cross-terms
                    u_pfw_pf   = u_pf  * w_pf 
                    u_psfw_psf = u_psf * w_psf
                    v_pfw_pf   = v_pf  * w_pf 
                    v_psfw_psf = v_psf * w_psf
                    # Fluxes with the cross-terms
                    # uw_p = (u_pf + u_psf) * (w_pf + w_psf)  
                    uw_p  = u_p * w_p
                    vw_p  = v_p * w_p
                    
                    # # # filtered fluxes)
                    # uw_pf = lowPass(uw_p, circ_mask)
                    # vw_pf   = lowPass(vw_p, circ_mask)
                    # # # subgrid fluxes
                    # uw_psf = uw_p - uw_pf
                    # vw_psf    = vw_p - vw_pf    
                    
                    ## Put results into variables 
                    print('Averaging fields...')
    
                    #
                    u_pf_avtime[k,i,ii,jj] = np.mean(u_pf,axis=(0,1))
                    v_pf_avtime[k,i,ii,jj] = np.mean(v_pf,axis=(0,1))
                    w_pf_avtime[k,i,ii,jj] = np.mean(w_pf,axis=(0,1))
                    #
                    u_pfw_pf_avtime[k,i,ii,jj]   = np.mean(u_pfw_pf,axis=(0,1))
                    u_psfw_psf_avtime[k,i,ii,jj] = np.mean(u_psfw_psf,axis=(0,1))
                    v_pfw_pf_avtime[k,i,ii,jj]   = np.mean(v_pfw_pf,axis=(0,1))
                    v_psfw_psf_avtime[k,i,ii,jj] = np.mean(v_psfw_psf,axis=(0,1))
                    #
                    gc.collect()
                    #### Momentum fluxes divergence 
                    # to be added...
                #
                u_p_avtime[i,ii,jj] = np.mean(u_p,axis=(0,1))
                v_p_avtime[i,ii,jj] = np.mean(v_p,axis=(0,1))
                w_p_avtime[i,ii,jj] = np.mean(w_p,axis=(0,1))
                #
                u_pw_p_avtime[i,ii,jj]   = np.mean(uw_p,axis=(0,1))
                v_pw_p_avtime[i,ii,jj]   = np.mean(vw_p,axis=(0,1))
    
    if store:  
        print('Saving data...')     
        # df = xr.DataArray(u_pf_avtime, coords=[('klp',klps),('time', time), ('z', ztlim)])
        np.save(save_dir+'/scale_chunks_time_'+str(pltheights)+'_'+casenr+'.npy',time[plttime])
        np.save(save_dir+'/scale_chunks_plttime_'+str(pltheights)+'_'+casenr+'.npy',plttime)
        np.save(save_dir+'/scale_chunks_zt_'+str(pltheights)+'_'+casenr+'.npy',zt[pltz].values)
        np.save(save_dir+'/scale_chunks_klps_'+str(pltheights)+'_'+casenr+'.npy',klps)
        print('Sved general variables') 
        np.save(save_dir+'/scale_chunks_u_'+str(pltheights)+'_'+casenr+'.npy',u_p_avtime)
        np.save(save_dir+'/scale_chunks_v_'+str(pltheights)+'_'+casenr+'.npy',v_p_avtime)
        np.save(save_dir+'/scale_chunks_w_'+str(pltheights)+'_'+casenr+'.npy',w_p_avtime)
        np.save(save_dir+'/scale_chunks_u_pf_'+str(pltheights)+'_'+casenr+'.npy',u_pf_avtime)
        np.save(save_dir+'/scale_chunks_v_pf_'+str(pltheights)+'_'+casenr+'.npy',v_pf_avtime)
        np.save(save_dir+'/scale_chunks_w_pf_'+str(pltheights)+'_'+casenr+'.npy',w_pf_avtime)
        print('Sved u, v, w')
        np.save(save_dir+'/scale_chunks_chunks_u_pfw_pf_'+str(pltheights)+'_'+casenr+'.npy',u_pfw_pf_avtime)
        np.save(save_dir+'/scale_chunks_u_psfw_psf_'+str(pltheights)+'_'+casenr+'.npy',u_psfw_psf_avtime)
        np.save(save_dir+'/scale_chunks_v_pfw_pf_'+str(pltheights)+'_'+casenr+'.npy',v_pfw_pf_avtime)
        np.save(save_dir+'/scale_chunks_v_psfw_psf_'+str(pltheights)+'_'+casenr+'.npy',v_psfw_psf_avtime)
        np.save(save_dir+'/scale_chunks_uw_p_'+str(pltheights)+'_'+casenr+'.npy',u_pw_p_avtime)
        np.save(save_dir+'/scale_chunks_vw_p_'+str(pltheights)+'_'+casenr+'.npy',v_pw_p_avtime)
        print('Sved fluxes')
print('END')    

#%% Quick plots

f_scale = xsize/(klps*2) *0.001

u_psfw_psf_avtime_norm = u_psfw_psf_avtime#/u_psfw_psf_avtime[-1,:,:,:]

# plt.figure()
# for j in range(0,5):
#     plt.plot(f_scale,u_psfw_psf_avtime_norm[:,1,:,j])
#     # plt.scatter(25,u_pw_p_avtime[1,1,j])
#     # plt.scatter(25,u_psfw_psf_avtime[-1,1,:,j])
# plt.xscale('log')
# plt.ylim([-0.5,+1.5])
# plt.axvline(2.5,c='k',lw=0.5)
# plt.axhline(0,c='k',lw=0.5)
# # plt.axhline(1,c='k',lw=0.5)

##
median_inspace = np.median(u_psfw_psf_avtime_norm,[2,3])
std_inspace = np.std(u_psfw_psf_avtime_norm,(2,3))
##
median_intime = np.median(u_psfw_psf_avtime_norm,1)
std_intime = np.std(u_psfw_psf_avtime_norm,1)

### plot single boxes
plt.figure()
for i in np.sort(range(0,nchunks_x)):
    for j in np.sort(range(0,nchunks_y)):
        plt.plot(f_scale,median_intime[:,i,j])
        # plt.fill_between(f_scale,median_intime[:,i,j]-std_intime[:,i,j],\
        #                  median_intime[:,i,j]+std_intime[:,i,j],alpha=0.2)
plt.xscale('log')
# plt.ylim([-0.2,+1.3])
plt.ylim([None,+0.044])
plt.axvline(2.5,c='k',lw=0.5)
plt.axhline(0,c='k',lw=0.5)
# plt.axhline(1,c='k',lw=0.5)

plt.figure()
for i in np.sort(range(0,nchunks_x)):
    for j in np.sort(range(0,nchunks_y)):
        plt.plot(f_scale,std_intime[:,i,j])
# plt.ylim([0,10])
plt.xscale('log')
plt.axvline(2.5,c='k',lw=0.5)

### plot single hours 
plt.figure()
plt.plot(f_scale,median_inspace[:,:])
# plt.plot(f_scale,median_inspace[:,idtime],lw=4,c='k')
# for idtime in range(len(plttime)):
#     plt.fill_between(f_scale,median_inspace[:,idtime]-std_inspace[:,idtime],\
#                      median_inspace[:,idtime]+std_inspace[:,idtime],alpha=0.2)
plt.legend(time[plttime][0:3])
plt.xscale('log')
# plt.ylim([-0.2,+1.3])
plt.ylim([None,+0.044])
plt.axvline(2.5,c='k',lw=0.5)
plt.axhline(0,c='k',lw=0.5)
# plt.axhline(1,c='k',lw=0.5)

plt.figure()
plt.plot(f_scale,std_inspace[:,:])
plt.legend(time[plttime][0:3])
# plt.ylim([0,10])
plt.xscale('log')
plt.axvline(2.5,c='k',lw=0.5)

#%%
idtime = 3
plt.figure()
for i in np.sort(range(0,nchunks_x)):
    for j in np.sort(range(0,nchunks_y)):
        
        plt.plot(f_scale,u_psfw_psf_avtime_norm[:,idtime,i,j])
plt.plot(f_scale,median_inspace[:,idtime],lw=4,c='k')
plt.fill_between(f_scale,median_inspace[:,idtime]-std_inspace[:,idtime],\
                 median_inspace[:,idtime]+std_inspace[:,idtime],alpha=0.2)
plt.xscale('log')
# plt.ylim([-0.2,+1.6])
plt.ylim([None,+0.054])
plt.axvline(2.5,c='k',lw=0.5)
plt.axhline(0,c='k',lw=0.5)
plt.title(time[idtime])



