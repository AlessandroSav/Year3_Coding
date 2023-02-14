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

my_source_dir = os.path.abspath('{}/../../../My_source_codes')
sys.path.append(my_source_dir)
from My_thermo_fun import *


#%% initial 
dt = 75                 # model  timestep [seconds]
step = 3600             # output timestep [seconds]
domain_name = 'BES'
lat_select = 13.2806    # HALO center 
lon_select = -57.7559   # HALO center 
buffer = 60             # buffer of 150 km around (75 km on each side) the gridpoint 30 * 2 * 2.5 km

srt_time   = np.datetime64('2020-02-02')
end_time   = np.datetime64('2020-02-12')
harmonie_time_to_keep = '202002010000-'

## running on Local
read_dir  = os.path.abspath('/Users/acmsavazzi/Documents/Mount1/harmonie_data/Eurec4a_climrun/')+'/'
write_dir = os.path.abspath('{}/../../DATA/HARMONIE')+'/'
## running on VrLab
# read_dir  = os.path.abspath('{}/../')+'/'
# write_dir = os.path.abspath('{}/../average_150km/')+'/'
## running on Mounted staffumbrella
# read_dir   = '/Users/acmsavazzi/Documents/Mount/'
# harmonie_dir   = base_dir+'Raw_Data/HARMONIE/BES_harm43h22tg3_fERA5_exp0/2020/'

harmonie_dir = read_dir+'2020/'

plot = False
harm_3d = True

my_vars = ['rain']
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
#
#

## define general filename from HARMONIE output

# f_general = '*{1}_{2}_{3}_BES_harm43h22tg3_fERA5_exp0_1hr_{4:02d}{5:02d}{6:02d}{7:02d}{8:02d}-{9:02d}{10:02d}{11:02d}{12:02d}{13:02d}.nc'
#  *  : variable name
# {0} : variable name 
# {1} : levels (_Slev or '')
# {2} : history (his), fullpos (fp), or surfex (sfx)


# {3} : domain (BES or EUREC4Acircle)

# {2} : start year
# {3} : start month
# {4} : start day
# {5} : start hour
# {6} : start min
# {7} : end year
# {8} : end month
# {9} : end day
# {10}: end hour
# {11}: end min
#

# # file for converting model levels
# sigma = (pd.read_csv(read_dir+'H43lev65.txt',header=None,index_col=[0],delim_whitespace=True))[2].values[:-1]

# nlev  = len(sigma)      # Number of full vertical levels
# nlevh = nlev + 1        # Number of half vertical levels

#%%         # Read in model level outputs

### Import raw Harmonie data
# This is too slow... need to find a better way. 
if harm_3d:
    print("Reading HARMONIE raw outputs.") 
    ### 3D fields
    nc_files = []
    EXT = "*_Slev_*.nc"
    for file in glob(os.path.join(harmonie_dir, EXT)):
        if harmonie_time_to_keep in file:
            try:
                nc_data_3d  = xr.open_mfdataset(file, combine='by_coords')
            except TypeError:
                nc_data_3d  = xr.open_mfdataset(file)
            ## select 10 days in February 
            nc_data_3d = nc_data_3d.sel(time=slice(srt_time,end_time))
            # select a smaller area for comparison with DALES
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
            for var in list(nc_data_3d.keys()): # DOESN'T WORK, NEED A FOR LOOP
                if var in [my_vars]:
                    harm_clim_avg = nc_data_3d[var].mean(dim=['x', 'y'])
                    print("saving level "+var)
                    harm_clim_avg.to_netcdf(write_dir+'my_harm_clim_avg_lev_'+var+'.nc')

                    del harm_clim_avg
                else: pass
            # free some memory
            del nc_data_3d
            


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
nc_data_cl.to_netcdf(write_dir+'my_harm_clim_2D.nc')
#%%         # Read in surface (or first level) outputs 
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
print("saving my_harm_clim_surf")
nc_data_surf.to_netcdf(write_dir+'my_harm_clim_surf.nc')



# #%% PLOT
# if plot: 





