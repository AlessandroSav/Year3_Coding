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
import sys
from datetime import datetime, timedelta
from netCDF4 import Dataset
sys.path.insert(1, os.path.abspath('.'))
my_source_dir = os.path.abspath('{}/../../../../My_source_codes')
sys.path.append(my_source_dir)
from My_thermo_fun import *

#%%
domain  = 150               # size of the domain where to average (in km)
exp     = 'cy43_clim'       # name of the experiment 

plot    = False
#%% initial 

dt = 75                 # model  timestep [seconds]
step = 3600             # output timestep [seconds]
grid = 2.5              # horizontal grid size in km
domain_name = 'BES'
lat_select = 13.2806    # HALO center 
lon_select = -57.7559   # HALO center 

buffer = int(domain/(2*grid))          # buffer of 150 km around (75 km on each side) the gridpoint 30 * 2 * 2.5 km

srt_time   = np.datetime64('2020-02-02')
end_time   = np.datetime64('2020-02-12')
harmonie_time_to_keep = '202002010000-'

## running on Local
read_dir  = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/'+exp+'/'
write_dir = os.path.abspath('{}/../../../DATA/HARMONIE')+'/'
## runnin with mounted staff-umbrella
# read_dir  = os.path.abspath('/Users/acmsavazzi/Documents/Mount1/harmonie_data/Eurec4a_climrun/')+'/'
# write_dir = os.path.abspath('{}/../../DATA/HARMONIE')+'/'
## running on VrLab
# read_dir  = os.path.expanduser('~/net/shared-staff/projects/cmtrace/Data/harmonie_data/'+exp)
# write_dir = os.path.expanduser('~/net/labdata/alessandro/harmonie_data/'+exp)
## running on DelftBlue
# read_dir = os.path.abspath('../../data/HARMONIE/'+exp)
# write_dir = read_dir


#%%
def calc_geo_height(ds_,fliplevels=False):
    # pressure in Pa
    if fliplevels==True:
        ds_['level']=np.flip(ds_.level)
    
    rho = calc_rho(ds_.p,ds_.T,ds_.qt)
    k = np.arange(ds_.level[0]+(ds_.level[1]-ds_.level[0])/2,\
                  ds_.level[-1]+(ds_.level[1]-ds_.level[0])/2,\
                  ds_.level[1]-ds_.level[0])
    rho_interp = rho.interp(level=k)
    zz = np.zeros((len(ds_['time']),len(ds_['level'])))
    zz = ((ds_.p.diff(dim='level').values)/(-1*rho_interp*g)).cumsum(dim='level')
    z = zz.interp(level=ds_.level,kwargs={"fill_value": "extrapolate"})
    ds_['z']=z -z.min('level')
    return (ds_)

#%% Open files

# # file for converting model levels
# sigma = (pd.read_csv(read_dir+'H43lev65.txt',header=None,index_col=[0],delim_whitespace=True))[2].values[:-1]

# nlev  = len(sigma)      # Number of full vertical levels
# nlevh = nlev + 1        # Number of half vertical levels

#%%         # Read in model level outputs

### Import raw Harmonie data
# This is too slow... need to find a better way. 
print("Reading HARMONIE 3D output.") 
### 3D fields

## first open and trim one by one the files saving a small portion 
## This becasue it is too heavy to load all files together and then cut the area
nc_files    = []
all_files = []
EXT = "*_BES_BES_*.nc"
for path,subdir,files in os.walk(read_dir):
    for file in glob(os.path.join(path, EXT)):
        all_files.append(file)

## open files
try:
    nc_data_3d  = xr.open_mfdataset(all_files, combine='by_coords')
except TypeError:
    nc_data_3d  = xr.open_mfdataset(all_files)
#get rid of useles variables
nc_data_3d = nc_data_3d.drop(['Lambert_Conformal','time_bnds'])
# select a smaller area 
j,i = np.unravel_index(np.sqrt((nc_data_3d.lon-lon_select)**2 + (nc_data_3d.lat-lat_select)**2).argmin(), nc_data_3d.lon.shape)
nc_data_3d = nc_data_3d.isel(x=slice(i-buffer,i+buffer),y=slice(j-buffer,j+buffer))
# Deaccumulate tendencies 
for var in list(nc_data_3d.keys()):
    if 'dt' in var:
        print("deaccumulating "+var)
        nc_data_3d[var] = (nc_data_3d[var].diff('time')) * step**-1  # gives values per second    
## select only lower levels
nc_data_3d = nc_data_3d.sel(lev=slice(15,65)) # MAKE THIS SELECTION ONLY WHEN SAVING
## average over the domain
harm_clim_avg = nc_data_3d.mean(dim=['x', 'y'])
##### save #####
# print("saving level "+var)
# harm_clim_avg.to_netcdf(write_dir+exp[16::]+'_avg_lev_'+var+'.nc')
print("saving averaged level profiles")
harm_clim_avg.to_netcdf(write_dir+exp[16::]+'_avg_lev_all.nc')

del harm_clim_avg
# free some memory
del nc_data_3d



#%%

## now open all trimmed files together and save as one netcdf
EXT = exp[16::]+'_avg_lev_*.nc'   
for file in glob(os.path.join(write_dir, EXT)):
    nc_files.append(file)
try:
    harm_clim_avg  = xr.open_mfdataset(nc_files, combine='by_coords')
except TypeError:
    harm_clim_avg  = xr.open_mfdataset(nc_files)

# save at it should be for creating LES forcings
print("saving averaged level profiles")
harm_clim_avg.to_netcdf(write_dir+exp[16::]+'_avg_lev_all.nc')

# rename variables
harm_clim_avg        = harm_clim_avg.rename({'ta':'T','hus':'qt','lev':'level','va':'v','ua':'u'})
#calculate height in meters
harm_clim_avg        = calc_geo_height(harm_clim_avg,fliplevels=True) 
harm_clim_avg        = harm_clim_avg.sortby('level')
##interpolate variables to heigth levels 
z_ref = harm_clim_avg.z.mean('time')
zz    = harm_clim_avg.z
for var in list(harm_clim_avg.keys()):
    if 'level' in harm_clim_avg[var].dims:
        print("interpolating variable "+var)
        x = np.empty((len(harm_clim_avg['time']),len(harm_clim_avg['level'])))
        x[:] = np.NaN
        for a in range(len(harm_clim_avg.time)):
            x[a,:] = np.interp(z_ref,zz[a,:],harm_clim_avg[var].isel(time = a))            
        harm_clim_avg[var] = (("time","level"), x)    
# convert model levels to height levels
harm_clim_avg = harm_clim_avg.rename({'z':'geo_height'})
harm_clim_avg = harm_clim_avg.rename({'level':'z'})
harm_clim_avg["z"] = (z_ref-z_ref.min()).values
harm_clim_avg['z'] = harm_clim_avg.z.assign_attrs(units='m',long_name='Height')
print("saving averaged heigth profiles")
harm_clim_avg.to_netcdf(write_dir+exp[16::]+'_avg_z_all.nc')
del harm_clim_avg

#%%         # Read cloud fraction
print("Reading 2D HARMONIE data.") 
nc_files = []
for EXT in ["clt_his*.nc","cll_his*.nc","clm_his*.nc","clh_his*.nc","clwvi_his*.nc","clivi_his*.nc",'prw*']:
    for file in glob(os.path.join(harmonie_dir, EXT)):
        if harmonie_time_to_keep in file:
            nc_files.append(file) 
try:
    nc_data_cl  = xr.open_mfdataset(nc_files, combine='by_coords')
except TypeError:
    nc_data_cl  = xr.open_mfdataset(nc_files)
nc_data_cl.to_netcdf(write_dir+exp[16::]+'_2D.nc')
#%%         # Read in surface (or first level) outputs 
print("Reading surface HARMONIE data.") 
nc_files = []
for EXT in ['hfss*','hfls*','cape*','ps*','ts*','tos*']:
    for file in glob(os.path.join(harmonie_dir, EXT)):
        if harmonie_time_to_keep in file:
            nc_files.append(file) 
try:
    nc_data_surf  = xr.open_mfdataset(nc_files, combine='by_coords')
except TypeError:
    nc_data_surf  = xr.open_mfdataset(nc_files)
## select 10 days in February 
nc_data_surf = nc_data_surf.sel(time=slice(srt_time,end_time))
# select a smaller area for comparison with DALES
j,i = np.unravel_index(np.sqrt((nc_data_surf.lon-lon_select)**2 + (nc_data_surf.lat-lat_select)**2).argmin(), nc_data_surf.lon.shape)
nc_data_surf = nc_data_surf.isel(x=slice(i-buffer,i+buffer),y=slice(j-buffer,j+buffer))
# Deaccumulate tendencies 
for var in list(nc_data_surf.keys()):
    nc_data_surf[var] = (nc_data_surf[var].diff('time')) * step**-1  # gives values per second
# ## average over the domain
# nc_data_surf = nc_data_surf.mean(dim=['x', 'y'])
print("saving surface variables")
nc_data_surf.to_netcdf(write_dir+exp[16::]+'_surf.nc')


#%%
print('End.')

#%%
# # ##### calculate pressure #####

# # ahalf= (pd.read_csv('/nfs/home/users/theeuwes/work/DALES_runs/ecf/scr/data/H43_65lev.txt',
# #                      header=None,index_col=[0],delim_whitespace=True))[1].values[:]
# # bhalf= (pd.read_csv('/nfs/home/users/theeuwes/work/DALES_runs/ecf/scr/data/H43_65lev.txt',
# #                      header=None,index_col=[0],delim_whitespace=True))[2].values[:]

# # ph = np.array([ahalf + (p * bhalf) for p in df['ps'].values])
# # p = np.zeros((df.ta.values).shape)
# # for z in range(0,len(df.lev)):
# #     p[:,z] = 0.5 * (ph[:,z] + ph[:,z+1])

# # df['p'] =  xr.DataArray(data=p,dims = dict(time = df.time, lev = df.lev))

# #%% calculate some varibles
# ## density
# nc_data_3d['rho']=calc_rho(nc_data_3d.p,nc_data_3d.ta,nc_data_3d.hus)

# ##################################################################
# ########### !!! PROBABLY WRONG INTEGRATION !!! ###########
# z =(-(nc_data_3d.p.diff(dim='lev'))/(-1*nc_data_3d.rho.sel(lev=slice(0, None))*g)).cumsum(dim='lev')
# # LWP
# lwp = (nc_data_3d.rho * nc_data_3d.clw * (-nc_data_3d.z.diff(dim='lev'))).sum('lev')
# ##################################################################






