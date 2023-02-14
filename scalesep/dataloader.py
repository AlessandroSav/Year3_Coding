#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:24:33 2020

@author: martinjanssens
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr

class DataLoaderDALES:

    def __init__(self, load_path,casenr='001'):
        print('Initialising dataloader...')
        self.lp = load_path

        try:
            self.ds    = xr.open_mfdataset(self.lp+'/merged_fielddump_wind.'+casenr+'.nc',decode_times=False)
            self.time  = self.ds['time']
            self.xt    = self.ds['xt']
            self.xm    = self.ds['xm']
            self.yt    = self.ds['yt']
            self.ym    = self.ds['ym']
        except:
            print('Warning: No 3D fields loaded, load_ routines that rely on \
                   these fields will fail.')

        self.ds1    = xr.open_mfdataset(self.lp+'/profiles.'+casenr+'.nc')
        self.zt     = self.ds1['zt']
        self.zm     = self.ds1['zm']     
        self.time1d = self.ds1['time']
        self.rhobt  = self.ds1['rhobf']
        self.rhobm  = self.ds1['rhobh']
        try: 
            self.ilp = np.loadtxt(self.lp+'/lscale.inp.'+casenr, skiprows=2)
            self.zf_inp = self.ilp[:,0]
            self.wfls = self.ilp[:,3]
            self.dqdt_ls = self.ilp[:,6]
            self.dthldt_ls = self.ilp[:,7]
        except:
            print('Warning: No lscale.inp, need to read ls_flux.inp')
        print('Set paths to all datasets and extracted dimensions')

    def load_qt(self, it, izmin, izmax):
        return np.ma.getdata(self.ds.variables['qt'][it,izmin:izmax,:,:])
    
    def load_u(self, it, izmin, izmax=None):
        # return np.ma.getdata(self.ds.variables['u'][it,izmin:izmax,:,:])
        if izmax!=None:
            return self.ds.isel(time=it,zt=slice(izmin,izmax))['u'].values
        else:
            return self.ds.isel(time=it).isel(zt=izmin)['u'].values

    def load_v(self, it, izmin, izmax=None):
        # return np.ma.getdata(self.ds.variables['v'][it,izmin:izmax,:,:])
        if izmax!=None:
            return self.ds.isel(time=it,zt=slice(izmin,izmax))['v'].values
        else:
            return self.ds.isel(time=it).isel(zt=izmin)['v'].values

    def load_wm(self, it, izmin, izmax=None):
        # return np.ma.getdata(self.ds.variables['w'][it,izmin:izmax+1,:,:])
        if izmax!=None:
            return self.ds.isel(time=it,zm=slice(izmin,izmax))['w'].values
        else:
            return self.ds.isel(time=it).isel(zm=izmin)['w'].values

    def load_thl(self, it, izmin, izmax):
        return np.ma.getdata(self.ds.variables['thl'][it,izmin:izmax,:,:])

    def load_ql(self, it, izmin, izmax):
        return np.ma.getdata(self.ds.variables['ql'][it,izmin:izmax,:,:])

    def load_e12(self, it, izmin, izmax):
        return np.ma.getdata(self.ds.variables['e12'][it,izmin:izmax,:,:])

    def load_p(self, it, izmin, izmax):
        return np.ma.getdata(self.ds.variables['p'][it,izmin:izmax,:,:])

    def load_mcr(self, it, izmin, izmax):
        # mcr_masked = self.ds.variables['qtpmcr'][it,izmin:izmax,:,:]
        # np.ma.set_fill_value(mcr_masked, 0.) # Assuming undefined points are just zero in non-cloudy regions
        return np.ma.getdata(self.ds.variables['qtpmcr'][it,izmin:izmax,:,:])

    def load_rad(self, it, izmin, izmax):
        return np.ma.getdata(self.ds.variables['thlprad'][it,izmin:izmax,:,:])

    def load_presh(self, it, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['presh'][it,izmin:izmax])

    def load_qtav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['qt'][:,izmin:izmax])

    def load_thlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['thl'][:,izmin:izmax])

    def load_qlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['ql'][:,izmin:izmax])

    def load_wqtav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wqtt'][:,izmin:izmax])

    def load_wthlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wthlt'][:,izmin:izmax])

    def load_wthvav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wthvt'][:,izmin:izmax])

    def load_wqlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wqlt'][:,izmin:izmax])

    def load_wqtrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wqtr'][:,izmin:izmax])

    def load_wthlrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wthlr'][:,izmin:izmax])

    def load_wqlrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wqlr'][:,izmin:izmax])

    def load_wthvrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wthvr'][:,izmin:izmax])
    
    def load_w2tav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['w2r'][:,izmin:izmax] +
                             self.ds1.variables['w2s'][:,izmin:izmax])
    
    def load_dissav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['diss'][:,izmin:izmax])
    
    def load_qt2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['qt2r'][:,izmin:izmax])
    
    def load_ql2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['ql2r'][:,izmin:izmax])
    
    def load_thl2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['thl2r'][:,izmin:izmax])
    
    def load_thv2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['thv2r'][:,izmin:izmax])

class DataLoaderDALESSeparate:

    def __init__(self, load_path):
        print('Initialising dataloader...')
        self.lp = load_path

        try:
            self.dsqt = nc.Dataset(self.lp+'/fielddump-qt.nc')
            self.dsthl = nc.Dataset(self.lp+'/fielddump-thl.nc')
            self.dsql = nc.Dataset(self.lp+'/fielddump-ql.nc')
            self.dsqr = nc.Dataset(self.lp+'/fielddump-qr.nc')
            self.dsw = nc.Dataset(self.lp+'/fielddump-w.nc')
            self.dsu = nc.Dataset(self.lp+'/fielddump-u.nc')
            self.dsv = nc.Dataset(self.lp+'/fielddump-v.nc')
            
            self.time  = np.ma.getdata(self.dsqt.variables['time'][:]) / 3600
            self.xf    = np.ma.getdata(self.dsqt.variables['xt'][:]) 
            self.xh    = np.ma.getdata(self.dsu.variables['xm'][:])
            self.yf    = np.ma.getdata(self.dsqt.variables['yt'][:])
            self.yh    = np.ma.getdata(self.dsv.variables['ym'][:])
        except:
            print('Warning: No 3D fields loaded, load_ routines that rely on \
                   these fields will fail.')

        self.ds1= nc.Dataset(self.lp+'/profiles.001.nc')
        self.zf     = np.ma.getdata(self.ds1.variables['zt'][:])
        self.zh     = np.ma.getdata(self.ds1.variables['zm'][:])        
        self.time1d = np.ma.getdata(self.ds1.variables['time'][:])
        self.rhobf  = np.ma.getdata(self.ds1.variables['rhobf'][:])
        self.rhobh  = np.ma.getdata(self.ds1.variables['rhobh'][:])

        self.ilp = np.loadtxt(self.lp+'/lscale.inp.001')
        self.zf_inp = self.ilp[:,0]
        self.wfls = self.ilp[:,3]
        self.dqdt_ls = self.ilp[:,6]
        self.dthldt_ls = self.ilp[:,7]
        print('Set paths to all datasets and extracted dimensions')

    def load_qt(self, it, izmin, izmax):
        return np.ma.getdata(self.dsqt.variables['qt'][it,izmin:izmax,:,:])

    def load_wh(self, it, izmin, izmax):
        return np.ma.getdata(self.dsw.variables['w'][it,izmin:izmax+1,:,:])

    def load_thl(self, it, izmin, izmax):
        return np.ma.getdata(self.dsthl.variables['thl'][it,izmin:izmax,:,:])

    def load_ql(self, it, izmin, izmax):
        return np.ma.getdata(self.dsql.variables['ql'][it,izmin:izmax,:,:])

    def load_u(self, it, izmin, izmax):
        return np.ma.getdata(self.dsu.variables['u'][it,izmin:izmax,:,:])

    def load_v(self, it, izmin, izmax):
        return np.ma.getdata(self.dsv.variables['v'][it,izmin:izmax,:,:])

    def load_e12(self, it, izmin, izmax):
        raise NotImplementedError('These runs does not these runs do not return e12')

    def load_p(self, it, izmin, izmax):
        raise NotImplementedError('These runs does not these runs do not return p')
    
    def load_presh(self, it, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['presh'][it,izmin:izmax])

    def load_qtav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['qt'][:,izmin:izmax])

    def load_thlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['thl'][:,izmin:izmax])

    def load_qlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['ql'][:,izmin:izmax])

    def load_wqtav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wqtt'][:,izmin:izmax])

    def load_wthlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wthlt'][:,izmin:izmax])

    def load_wthvav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wthvt'][:,izmin:izmax])

    def load_wqlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wqlt'][:,izmin:izmax])

    def load_wqtrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wqtr'][:,izmin:izmax])

    def load_wthlrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wthlr'][:,izmin:izmax])

    def load_wqlrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wqlr'][:,izmin:izmax])

    def load_wthvrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['wthvr'][:,izmin:izmax])
    
    def load_w2tav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['w2r'][:,izmin:izmax] +
                             self.ds1.variables['w2s'][:,izmin:izmax])
    
    def load_dissav(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['diss'][:,izmin:izmax])
    
    def load_qt2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['qt2r'][:,izmin:izmax])
    
    def load_ql2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['ql2r'][:,izmin:izmax])
    
    def load_thl2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['thl2r'][:,izmin:izmax])
    
    def load_thv2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1.variables['thv2r'][:,izmin:izmax])
    

class DataLoaderMicroHH:

    def __init__(self, load_path, case='bomex'):
        print('Initialising dataloader...')
        self.lp = load_path
        
        try:
            self.dsqt = nc.Dataset(self.lp+'/qt.nc')
            self.dsthl = nc.Dataset(self.lp+'/thl.nc')
            self.dsql = nc.Dataset(self.lp+'/ql.nc')
            self.dsw = nc.Dataset(self.lp+'/w.nc')
            self.dsu = nc.Dataset(self.lp+'/u.nc')
            self.dsv = nc.Dataset(self.lp+'/v.nc')
            self.dsp = nc.Dataset(self.lp+'/p.nc')
    
            self.time  = np.ma.getdata(self.dsqt.variables['time'][:]) / 3600
            self.xf    = np.ma.getdata(self.dsqt.variables['x'][:])
            self.xh    = np.ma.getdata(self.dsu.variables['xh'][:])
            self.yf    = np.ma.getdata(self.dsqt.variables['y'][:])
            self.yh    = np.ma.getdata(self.dsv.variables['yh'][:])
        except:
            print('Warning: No 3D fields loaded, load_ routines that rely on \
                  these fields will fail.')

        self.ds1    = nc.Dataset(self.lp+'/'+case+'.default.0000000.nc')
        self.zf     = np.ma.getdata(self.ds1.variables['z'][:])
        self.zh     = np.ma.getdata(self.ds1.variables['zh'][:])
        self.time1d = np.ma.getdata(self.ds1.variables['time'][:])
        self.rhobf  = np.ma.getdata(self.ds1['thermo']['rho'][:])
        self.rhobh  = np.ma.getdata(self.ds1['thermo']['rhoh'][:])

        self.ilp = nc.Dataset(self.lp+'/'+case+'_input.nc')
        self.zf_inp = np.ma.getdata(self.ilp['z'][:])
        self.wfls = np.ma.getdata(self.ilp['init']['w_ls'][:])
        self.dqdt_ls = np.ma.getdata(self.ilp['init']['qt_ls'][:])
        self.dthldt_ls = np.ma.getdata(self.ilp['init']['thl_ls'][:])
        print('Set paths to all datasets and extracted dimensions')

    def load_qt(self, it, izmin, izmax):
        return np.ma.getdata(self.dsqt.variables['qt'][it,izmin:izmax,:,:])

    def load_wh(self, it, izmin, izmax):
        return np.ma.getdata(self.dsw.variables['w'][it,izmin:izmax+1,:,:])

    def load_thl(self, it, izmin, izmax):
        return np.ma.getdata(self.dsthl.variables['thl'][it,izmin:izmax,:,:])

    def load_ql(self, it, izmin, izmax):
        return np.ma.getdata(self.dsql.variables['ql'][it,izmin:izmax,:,:])

    def load_u(self, it, izmin, izmax):
        return np.ma.getdata(self.dsu.variables['u'][it,izmin:izmax,:,:])

    def load_v(self, it, izmin, izmax):
        return np.ma.getdata(self.dsv.variables['v'][it,izmin:izmax,:,:])

    def load_e12(self, it, izmin, izmax):
        raise NotImplementedError('MicroHH does not solve for e12')

    def load_p(self, it, izmin, izmax):
        return np.ma.getdata(self.dsp.variables['p'][it,izmin:izmax,:,:])

    def load_presh(self, it, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['phydroh'][it,izmin:izmax])

    def load_qtav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['qt'][:,izmin:izmax])

    def load_thlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['thl'][:,izmin:izmax])

    def load_qlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['ql'][:,izmin:izmax])

    def load_wqtav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['qt_flux'][:,izmin:izmax])

    def load_wthlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['thl_flux'][:,izmin:izmax])

    def load_wthvav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['thv_flux'][:,izmin:izmax])

    def load_wqlav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['ql_flux'][:,izmin:izmax])

    def load_wqtrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['qt_w'][:,izmin:izmax])

    def load_wthlrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['thl_w'][:,izmin:izmax])

    def load_wqlrav(self, izmin, izmax):
        print('Warning: MicroHH does not output wql. Returning zeros...')
        return np.zeros((len(self.time1d),izmax-izmin))

    def load_wthvrav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['thv_w'][:,izmin:izmax])

    def load_w2tav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['default']['w_2'][:,izmin:izmax])

    def load_dissav(self, izmin, izmax):
        return np.ma.getdata(self.ds1['budget']['tke_diff'][:,izmin:izmax])
    
    def load_qt2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['qt_2'][:,izmin:izmax])

    def load_ql2av(self, izmin, izmax):
        raise NotImplementedError('MicroHH does not solve for e12')
    
    def load_thl2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['thl_2'][:,izmin:izmax])
    
    def load_thv2av(self, izmin, izmax):
        return np.ma.getdata(self.ds1['thermo']['thv_2'][:,izmin:izmax])
    
