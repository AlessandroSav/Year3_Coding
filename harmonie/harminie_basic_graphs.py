#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:16:49 2021

@author: acmsavazzi
"""

#%% HARMONIE_BASIC_GRAPHS.py
# open HARMONIE outputs and plot some basic variables


#%%                             Libraries
###############################################################################
import numpy as np
import xarray as xr
import netCDF4
import os
from glob import glob
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from netCDF4 import Dataset
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 20,
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large',
         'figure.figsize':[10,7],
         'figure.titlesize':20}
pylab.rcParams.update(params)

my_source_dir = os.path.abspath('{}/../../../My_source_codes')
sys.path.append(my_source_dir)
from My_thermo_fun import *

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
proj=ccrs.PlateCarree()
coast = cartopy.feature.NaturalEarthFeature(\
        category='physical', scale='50m', name='coastline',
        facecolor='none', edgecolor='r')

#%% initial 
dt = 75                 # model  timestep [seconds]
step = 3600             # output timestep [seconds]
domain_name = 'BES'
lat_select = 13.2806    # HALO center 
lon_select = -57.7559   # HALO center 
buffer = 30             # buffer of 150 km around (75 km on each side) the gridpoint 30 * 2 * 2.5 km

srt_time   = np.datetime64('2020-02-02')
end_time   = np.datetime64('2020-02-12')
harmonie_time_to_keep = '202002010000-'

## running on Local
read_dir  = os.path.abspath('/Users/acmsavazzi/Documents/Mount/harmonie_data/Eurec4a_climrun/')+'/'
write_dir = os.path.abspath('{}/../../DATA/HARMONIE')+'/'
## running on VrLab
# read_dir  = os.path.abspath('{}/../')+'/'
# write_dir = os.path.abspath('{}/../average_150km/')+'/'
## running on Mounted staffumbrella
base_dir   = '/Users/acmsavazzi/Documents/Mount/'
# harmonie_dir   = base_dir+'Raw_Data/HARMONIE/BES_harm43h22tg3_fERA5_exp0/2020/'

harmonie_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/DATA/HARMONIE/'

ha_exps = ['cy43_clim','cy43_climnoUVmix','HA43h22tg3_clim_noHGTQS','HA43h22tg3_clim_noHGTQS_ECUME','HA43h22tg3_clim_noSHAL']

# path = read_dir+'2020/'

save_dir = '/Users/acmsavazzi/Documents/WORK/PhD_Year2/Figures/Harmonie/'

plot = True
harm_3d = False
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





#%% FOR MULTIPLES HARMNIE EXPERIMENTS        # Read in model level outputs

### Import raw Harmonie data
# This is too slow... need to find a better way. 
###

nc_data={}
nc_data_cl={}
for exp in ha_exps:
    # nc_files = []
    # for EXT in ['hfss*','hfls*']:
    #     for file in glob(os.path.join(harmonie_dir+exp+'/', EXT)):
    #         # if harmonie_time_to_keep in file:
    #         #     nc_files.append(file)
    #         nc_files.append(file)
    # nc_data[exp]  = xr.open_mfdataset(nc_files, combine='by_coords')
    
    # ## cut domain smaller based on buffer
    # j,i = np.unravel_index(np.sqrt((nc_data[exp].lon-lon_select)**2 + (nc_data[exp].lat-lat_select)**2).argmin(), nc_data[exp].lon.shape)
    # nc_data[exp] = nc_data[exp].isel(x=slice(i-buffer,i+buffer),y=slice(j-buffer,j+buffer))
    
    # for var in list(['hfss','hfls']):
    #      nc_data[exp][var] = (nc_data[exp][var].diff('time')) * 3600**-1  # gives values per second   



    print("Reading 2D HARMONIE data.") 
    nc_files= []
    for EXT in ['cll_his*.nc',]:
        for file in glob(os.path.join(harmonie_dir+exp+'/', EXT)):
            if os.path.exists(file):
                if harmonie_time_to_keep in file:
                    nc_files.append(file) 
                try:
                    nc_data_cl[exp]  = xr.open_mfdataset(nc_files, combine='by_coords')
                except TypeError:
                    nc_data_cl[exp]  = xr.open_mfdataset(nc_files)
    # nc_data_cl.to_netcdf(write_dir+'my_harm_clim_2D.nc')
#%%         # Read in model level outputs

### Import raw Harmonie data
# This is too slow... need to find a better way. 
###
# nc_files = []
# for EXT in ['hfss*','hfls*']:
#     for file in glob(os.path.join(harmonie_dir, EXT)):
#         nc_files.append(file)
# nc_data  = xr.open_mfdataset(nc_files, combine='by_coords')

# j,i = np.unravel_index(np.sqrt((nc_data.lon-lon_select)**2 + (nc_data.lat-lat_select)**2).argmin(), nc_data.lon.shape)
# nc_data = nc_data.isel(x=slice(i-buffer,i+buffer),y=slice(j-buffer,j+buffer))


# for var in list(['hfss','hfls']):
#     nc_data[var] = (nc_data[var].diff('time')) * 3600**-1  # gives values per second

######
############
####################################
############    
######
# Read observations from meteor

Aers_Dship_dir  = '/Users/acmsavazzi/Documents/WORK/Data/Aers-Dship/'
Meteor = xr.open_dataset(Aers_Dship_dir+'../EUREC4A_Meteor_surface_heat_fluxes_20200115_v1.0.nc')\
    #.sel(time=slice(srt_time,end_time))
####################################
############
######
####################################


#PLOT
col=['red','coral','maroon','blue','cornflowerblue','darkblue','green','lime','forestgreen','m']
temp_hrs   = [np.datetime64('2020-02-02'),np.datetime64('2020-02-12')]
# senisble flux
plt.figure(figsize=(19,5))
for exp in  ha_exps:
    nc_data[exp].hfss.mean(dim=['x','y']).plot(lw=2.5,label=exp)
xr.plot.scatter(Meteor,'time','SHF_bulk_mast',alpha = 0.6,s=12,c=col[2],label='Meteor')
xr.plot.scatter(Meteor,'time','SHF_EC_mast',alpha = 0.4,s=12,label='EC')
plt.ylim(-2,40)
plt.ylabel('LH (W/m2)')
plt.title('Surface sensible heat flux',size=20)
plt.legend(fontsize=15)
plt.xlim(temp_hrs)
plt.savefig(save_dir+'tmser_SHflux_HARMONIE_zoom.pdf', bbox_inches="tight")

#latent flux
plt.figure(figsize=(19,5))
for exp in  ha_exps:
    nc_data[exp].hfls.mean(dim=['x','y']).plot(lw=2.5,label=exp)
xr.plot.scatter(Meteor,'time','LHF_bulk_mast',alpha = 0.6,s=12,c=col[2],label='Meteor')
xr.plot.scatter(Meteor,'time','LHF_EC_mast',alpha = 0.4,s=12,label='EC')
plt.ylim(-10,400)
plt.ylabel('LH (W/m2)')
plt.title('Surface latent heat flux',size=20)
plt.legend(fontsize=15)
plt.xlim(temp_hrs)
plt.savefig(save_dir+'tmser_LHflux_HARMONIE_zoom.pdf', bbox_inches="tight")
###
#%%
if harm_3d:
    print("Reading HARMONIE raw outputs.") 
    ### 3D fields
    nc_files = []
    EXT = "hfls*.nc"
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
            if 'dt' in list(nc_data_3d.keys()):
                print("deaccumulating "+list(nc_data_3d.keys()))
                nc_data_3d = (nc_data_3d.diff('time')) * step**-1  # gives values per second
            ## average over the domain
            harm_clim_avg = nc_data_3d.mean(dim=['x', 'y'])
            # free some memory
            del nc_data_3d
            ## select only lower levels
            harm_clim_avg = harm_clim_avg.sel(lev=slice(15,65))
            var = list(harm_clim_avg.keys())[0]
            print("Read "+var)
            if var in ['rain']:
                print("saving level "+var)
                harm_clim_avg.to_netcdf(write_dir+'my_harm_clim_avg_lev_'+var+'.nc')
            del harm_clim_avg
    EXT = 'my_harm_clim_avg_lev_*.nc'   
    for file in glob(os.path.join(write_dir, EXT)):
        nc_files.append(file)
    try:
        harm_clim_avg  = xr.open_mfdataset(nc_files, combine='by_coords')
    except TypeError:
        harm_clim_avg  = xr.open_mfdataset(nc_files)
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
    # print("saving my_harm_clim_avg_profiles")
    # harm_clim_avg.to_netcdf(write_dir+'my_harm_clim_avg.nc')


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
for EXT in ['hfss*','hfls*','cape*','ps*','ts*','tos*','pr_his']:
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


#%% PLOT
if plot:
    ########  cloud variables ########
    for var in ['cll']:
        for exp in  nc_data_cl.keys():
            for ii in nc_data_cl[exp].time[58:70]:
                plt.figure()
                ax =nc_data_cl[exp].sel(time=ii)[var].plot(vmin=0,vmax=1,\
                                    cmap=plt.cm.Blues_r,x='lon',y='lat',\
                                    subplot_kws=dict(projection=proj))
                ax = plt.axes(projection=proj)
                ax.add_feature(coast, lw=2, zorder=7)
                plt.xlim([-60,-56.5])
                plt.ylim([12,14.5])
                gl = ax.gridlines(crs=proj, draw_labels=True)
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabels_top = False
                gl.ylabels_right = False
                plt.suptitle(exp)
    
                # plt.savefig(save_dir+var+'_D'+str(ii.dt.day.values)+'H'+str(ii.dt.hour.values)+'_'+exp+'.png')
    
#     ######### lwp ########
#     # for ii in nc_data_cl.time[0:13]:
#     #     plt.figure()
#     #     ax =lwp.sel(time=ii).plot(vmin=0,vmax=0.05,\
#     #                             cmap=plt.cm.Blues_r,x='lon',y='lat',\
#     #                             subplot_kws=dict(projection=proj),cbar_kwargs={'label': 'LWP [kg/m2]'})
#     #     ax = plt.axes(projection=proj)
#     #     ax.add_feature(coast, lw=2, zorder=7)
#     #     plt.xlim([-59.7,-56.5])
#     #     plt.ylim([12,14.5])
#     #     gl = ax.gridlines(crs=proj, draw_labels=True)
#     #     gl.xformatter = LONGITUDE_FORMATTER
#     #     gl.yformatter = LATITUDE_FORMATTER
#     #     gl.xlabels_top = False
#     #     gl.ylabels_right = False
#         # plt.savefig(fig_dir+'lwp_D'+str(ii.dt.day.values)+'H'+str(ii.dt.hour.values)+'_4video.png')
# #%%



# sel_lev = 62
# if plot:
#     ######## 3D variables ########
#     for var in ['dtu_conv','dtu_turb','dtu_phy','dtu_dyn']:
#         for ii in ds_deacc.time[12:13]:
#             plt.figure()
#             ax =ds_deacc[var].sel(time=ii,lev=sel_lev).plot(vmin=-1,vmax=+1,cmap='jet',\
#                                     x='lon',y='lat',\
#                                     subplot_kws=dict(projection=proj))
#             ax = plt.axes(projection=proj)
#             ax.add_feature(coast, lw=2, zorder=7)
#             plt.xlim([-60,-56.5])
#             plt.ylim([12,14.5])
#             gl = ax.gridlines(crs=proj, draw_labels=True)
#             gl.xformatter = LONGITUDE_FORMATTER
#             gl.yformatter = LATITUDE_FORMATTER
#             gl.xlabels_top = False
#             gl.ylabels_right = False
            

        
# #%% MAKE VIDEOS

# # import cv2
# # import glob
 
# # img_array = []
# # for filename in glob.glob(fig_dir+'/clm*_4video.png'):
# #     img = cv2.imread(filename)
# #     height, width, layers = img.shape
# #     size = (width,height)
# #     img_array.append(img)
 
 
# # out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
# # for i in range(len(img_array)):
# #     out.write(img_array[i])
# # out.release()








