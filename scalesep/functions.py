#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:10:43 2021

@author: janssens
"""
import numpy as np
from scipy import fftpack, ndimage
import dask
import matplotlib.pyplot as plt

# Constants
cp = 1004.
Rd = 287.04
Rv = 461.5
Lv = 2.53e6
grav = 9.81


def getRad(data):
    h  = data.shape[-2];  hc = h//2
    w  = data.shape[-1];  wc = w//2
   
    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)
    
    return r

def lowPass(field, mask): 
    # FFT
    F = fftpack.fft2(field)       # 2D FFT (no prefactor)
    F = fftpack.fftshift(F,axes=(-1,-2))       # Shift so k0 is centred
    if len(F.shape) == 2:
        F = F*mask
    elif len(F.shape) == 3:
        F = F*mask[np.newaxis,:,:]
    F = fftpack.fftshift(F,axes=(-1,-2))
    
    return np.real(fftpack.ifft2(F))

def scaleDecomposeFlux(xt,xp,yt,yp,mask):
    # *t -> Filtered quantity
    # *p -> Remainder
    
    Le = lowPass(xt*yt, mask)
    Cr = lowPass(xt*yp + xp*yt, mask)
    Re = lowPass(xp*yp, mask)
    
    return Le, Cr, Re

# Advection at the cell centre
def ddzwx_2nd(wh,x,dzf,dzh,rhobf=None):
    if type(rhobf) == type(None):
        rhobf = np.ones(x.shape[0])
    return ((wh[2:,:,:]*(rhobf[2:,np.newaxis,np.newaxis]*x[2:,:,:]*dzf[1:-1,np.newaxis,np.newaxis] + 
                         rhobf[1:-1,np.newaxis,np.newaxis]*x[1:-1,:,:]*dzf[2:,np.newaxis,np.newaxis])/
                         dzh[2:,np.newaxis,np.newaxis]
                         - 
             wh[1:-1,:,:]*(rhobf[:-2,np.newaxis,np.newaxis]*x[:-2,:,:]*dzf[1:-1,np.newaxis,np.newaxis] +
                           rhobf[1:-1,np.newaxis,np.newaxis]*x[1:-1,:,:]*dzf[:-2,np.newaxis,np.newaxis])/
                           dzh[1:-1,np.newaxis,np.newaxis])/
             (2*dzf[1:-1,np.newaxis,np.newaxis]*rhobf[1:-1,np.newaxis,np.newaxis]))

# Advection at half z-level (for w)
def ddzww_2nd(wh,dzh,rhobf=None,rhobh=None):
    if type(rhobf) == type(None): # then rhobh must also be None, else you're a moron
        rhobf = np.ones(wh.shape[0])
    return ((wh[2:,:,:] + wh[1:-1,:,:]) *
            (rhobh[2:,np.newaxis,np.newaxis]*wh[2:,:,:] + 
             rhobh[1:-1,np.newaxis,np.newaxis]*wh[1:-1,:,:]) -
            (wh[:-2,:,:] + wh[1:-1,:,:]) *
            (rhobh[:-2,np.newaxis,np.newaxis]*wh[:-2,:,:] + 
             rhobh[1:-1,np.newaxis,np.newaxis]*wh[1:-1,:,:])
            ) / (4*dzh[1:-1,np.newaxis,np.newaxis]*rhobf[1:-1,np.newaxis,np.newaxis])
            

# def ddzw2x_2nd(wh,x,dzh,rhobf=None):
#     # Assumes constant dz
#     if type(rhobf) == type(None):
#         rhobf = np.ones(x.shape[0])
#     return ((wh[2:,:,:]**2*(rhobf[2:,np.newaxis,np.newaxis]*x[2:,:,:] + 
#                             rhobf[1:-1,np.newaxis,np.newaxis]*x[1:-1,:,:]) - 
#              wh[1:-1,:,:]**2*(rhobf[:-2,np.newaxis,np.newaxis]*x[:-2,:,:]+
#                               rhobf[1:-1,np.newaxis,np.newaxis]*x[1:-1,:,:]))/
#              (2*dzh*rhobf[1:-1,np.newaxis,np.newaxis]))

# def kddza_2nd(k,x,zf,rhobf=None):
#     # Assumes constant dz
#     if type(rhobf) == type(None):
#         rhobf = np.ones(k.shape)
#     if k.shape != x.shape:
#         x = x[1:-1,:,:]
#         rhobf = rhobf[1:-1]
#     dzh = np.diff(zf)[0]
#     return (k[1:-1,:,:]*((rhobf[2:,np.newaxis,np.newaxis]*x[2:,:,:]+
#                            rhobf[1:-1,np.newaxis,np.newaxis]*x[1:-1,:,:]) - 
#                          (rhobf[:-2,np.newaxis,np.newaxis]*x[:-2,:,:] + 
#                           rhobf[1:-1,np.newaxis,np.newaxis]*x[1:-1,:,:]))/
#             (2*dzh*rhobf[1:-1,np.newaxis,np.newaxis]))

def ddxhuha_2nd(u,v,a,dx,dy):
    # Assumes constant dx, dy, shape (z,y,x)
    return ((np.roll(u,-1,axis=2)*(np.roll(a,-1,axis=2) + a) -
             u                   *(np.roll(a,1,axis=2) + a))/(2*dx) +
            (np.roll(v,-1,axis=1)*(np.roll(a,-1,axis=1) + a) -
             v                   *(np.roll(a,1,axis=1) + a))/(2*dy))

def ddxhuhwa_2nd(u,v,w,a,dx,dy):
    # Assumes constant dx, dy, shape (z,y,x)
    return (( np.roll(u[1:-1,:,:],-1,axis=2)*
             (np.roll(a[1:-1,:,:],-1,axis=2) + a[1:-1,:,:])*
             (w[2:,:,:] + w[1:-1,:,:] +
              np.roll(w[2:,:,:],-1,axis=2) + np.roll(w[1:-1,:,:],-1,axis=2))
             -
              u[1:-1,:,:]*
             (np.roll(a[1:-1,:,:],1,axis=2) + a[1:-1,:,:])*
             (np.roll(w[2:,:,:],1,axis=2) + np.roll(w[1:-1,:,:],1,axis=2) +
              w[2:,:,:] + w[1:-1,:,:]))
            / (8*dx) +
            
            ( np.roll(v[1:-1,:,:],-1,axis=1)*
             (np.roll(a[1:-1,:,:],-1,axis=1) + a[1:-1,:,:])*
             (w[2:,:,:] + w[1:-1,:,:] +
              np.roll(w[2:,:,:],-1,axis=1) + np.roll(w[1:-1,:,:],-1,axis=1))
             -
             v[1:-1,:,:]*
             (np.roll(a[1:-1,:,:],1,axis=1) + a[1:-1,:,:])*
             (np.roll(w[2:,:,:],1,axis=1) + np.roll(w[1:-1,:,:],1,axis=1)+
              w[2:,:,:] + w[1:-1,:,:]))
            / (8*dy))

def ddxhuhw_2nd(u,v,w,dx,dy):
    # Assumes constant dx, dy, shape (z,y,x), horizontal advection of w at half vertical level
    # Starts at half level above full level 0 (i.e. u_ijk=u[1:,:,:])
    # Return z-shape will thus be w.shape[0]-1
    return (((np.roll(u[1:,:,:],-1,axis=2) + np.roll(u[:-1,:,:],-1,axis=2))*
             (w[1:,:,:] + np.roll(w[1:,:,:],-1,axis=2)) -
             (u[1:,:,:] + u[:-1,:,:])*
             (np.roll(w[1:,:,:],1,axis=2) + w[1:,:,:]))
              / (4*dx) +
            ((np.roll(v[1:,:,:],-1,axis=1) + np.roll(v[:-1,:,:],-1,axis=1))*
             (w[1:,:,:] + np.roll(w[1:,:,:],-1,axis=1)) -
             (v[1:,:,:] + v[:-1,:,:])*
             (np.roll(w[1:,:,:],1,axis=1) + w[1:,:,:]))
             / (4*dy))


def wsubdxdz(wfls,x,dzh):
    # Assumes wfls < 0
    # returns values starting at the first half level above the surface, which is used at full level above and up
    # Shape in z is x.shape[0] - 1
    # wsub defined at cell centres -> Interpolate to edges 
    whls = (wfls[1:] +  wfls[:-1])*0.5
    # upwinding
    return whls[:,np.newaxis,np.newaxis]*(x[1:,:,:] - x[:-1,:,:])/dzh[1:,np.newaxis,np.newaxis]

def compute_dthvdz(thl,qt,ql,exnf,dzh):
    # Using 2nd order scheme, ignores first model level
    
    # Bodges to have an upper boundary (which will be screwed up)
    exnf = np.append(exnf,exnf[-1]) 
    
    thv = (thl + Lv*ql/(cp*exnf[:,np.newaxis,np.newaxis]))* (1+(Rv/Rd-1)*qt-Rv/Rd*ql)
    dthvdz = (thv[2:,:,:] - thv[:-2,:,:])/(dzh[1:-1,np.newaxis,np.newaxis]+dzh[2:,np.newaxis,np.newaxis])
    return dthvdz

def compute_ek(e12,dthvdz,thv_av,delta,cf=2.5,cn=0.76,Pri=3,ekmin=1e-6):
    # Assumes TKE-eq scheme, lmason=lanisotrop=ldelta=False
    alpha_kolm=1.5
    cm = cf / (2. * np.pi) * (1.5*alpha_kolm)**(-1.5)

    # Bodge to get shape right    
    e12 = e12[1:-1,:,:] # Remove first, last model level where you don't have dthvdz
    delta = delta[1:]
    thv_av = thv_av[1:]
        
    # Stable regime correction
    zlt = np.ones(e12.shape)*delta[:,np.newaxis,np.newaxis]
    mask = ( grav*np.abs(dthvdz) * zlt**2 > (cn*e12**2 * thv_av[:,np.newaxis,np.newaxis] ))
    zlt[mask] = cn*e12[mask]/(np.sqrt(grav/thv_av[:,np.newaxis,np.newaxis]*np.abs(dthvdz)))[mask]
    
    ekm = cm*zlt*e12
    ekh = (1 + (Pri-1)*zlt/delta[:,np.newaxis,np.newaxis])*ekm
    
    ekm[ekm<ekmin] = ekmin
    ekh[ekh<ekmin] = ekmin
    
    return ekm,ekh

def diffzeka(ekh,a,dzf,dzh,rhobf,rhobh):
    # Set shapes to ekh shape
    rhobf = rhobf[1:-1]
    rhobh = rhobh[1:-1]
    dzf = dzf[1:-1]
    dzh = dzh[1:-1]
    if ekh.shape != a.shape:
        a = a[1:-1,:,:]
    
    return 0.5*(rhobh[2:,np.newaxis,np.newaxis]/rhobf[1:-1,np.newaxis,np.newaxis]
               *(ekh[1:-1,:,:]*dzf[2:,np.newaxis,np.newaxis] + 
                 ekh[2:,:,:]*dzf[1:-1,np.newaxis,np.newaxis])*(a[2:,:,:] - a[1:-1,:,:])
               / dzh[2:,np.newaxis,np.newaxis]**2
              -
               rhobh[1:-1,np.newaxis,np.newaxis]/rhobf[1:-1,np.newaxis,np.newaxis]
               *(ekh[1:-1,:,:]*dzf[:-2,np.newaxis,np.newaxis] + 
                 ekh[:-2,:,:]*dzf[1:-1,np.newaxis,np.newaxis])*(a[1:-1,:,:] - a[:-2,:,:]) 
               / dzh[1:-1,np.newaxis,np.newaxis]**2) / dzf[1:-1,np.newaxis,np.newaxis]

def diffeka(ekh,a,dx,dy,dzf,dzh,rhobf=None,rhobh=None):
    # Assumes constant vertical spacing
    if type(rhobf) == type(None):
        rhobf = np.ones(zf.shape)
    if type(rhobh) == type(None):
        rhobh = np.ones(zf.shape)
    
    # set to ekh shape
    a = a[1:-1,:,:]
    
    diffxh = 0.5*(((np.roll(ekh,-1,axis=2) + ekh)*(np.roll(a,-1,axis=2) - a)
                -(ekh + np.roll(ekh,1,axis=2))*(a - np.roll(a,1,axis=2)))/(dx**2)
                +
                ((np.roll(ekh,-1,axis=1) + ekh)*(np.roll(a,-1,axis=1) - a)
                -(ekh + np.roll(ekh,1,axis=1))*(a - np.roll(a,1,axis=1)))/(dy**2))
    diffz = diffzeka(ekh, a, dzf, dzh, rhobf, rhobh)
    
    return diffxh[1:-1,:,:] + diffz

def diffekw(ekm,u,v,w,dx,dy,dzf,dzh,rhobf=None,rhobh=None):
    if type(rhobf) == type(None):
        rhobf = np.ones(dzh.shape)
    if type(rhobh) == type(None):
        rhobh = np.ones(dzf.shape)
    
    # Set to ekm shape
    u = u[1:-1,:,:]
    v = v[1:-1,:,:]
    w = w[1:-1,:,:]
    rhobf = rhobf[1:-1]
    rhobh = rhobh[1:-1]
    dzf = dzf[1:-1]
    dzh = dzh[1:-1]

    emom = (((ekm[1:-1,:,:] + np.roll(ekm[1:-1,:,:],1,axis=2))*dzf[:-2,np.newaxis,np.newaxis] +
             (ekm[:-2,:,:]  + np.roll(ekm[:-2, :,:],1,axis=2))*dzf[1:-1,np.newaxis,np.newaxis]) / 
           (4.*dzh[1:-1,np.newaxis,np.newaxis]))

    eomm = (((ekm[1:-1,:,:] + np.roll(ekm[1:-1,:,:],1,axis=1))*dzf[:-2,np.newaxis,np.newaxis] + 
             (ekm[:-2,:,:]  + np.roll(ekm[:-2, :,:],1,axis=1))*dzf[1:-1,np.newaxis,np.newaxis]) /
           (4.*dzh[1:-1,np.newaxis,np.newaxis]))

    eopm = (((ekm[1:-1,:,:] + np.roll(ekm[1:-1,:,:],-1,axis=1))*dzf[:-2,np.newaxis,np.newaxis] + 
             (ekm[:-2,:,:]  + np.roll(ekm[:-2,:,:], -1,axis=1))*dzf[1:-1,np.newaxis,np.newaxis]) /
           (4.*dzh[1:-1,np.newaxis,np.newaxis]))

    epom = (((ekm[1:-1,:,:] + np.roll(ekm[1:-1,:,:],-1,axis=2))*dzf[:-2,np.newaxis,np.newaxis] +
             (ekm[:-2,:,:]  + np.roll(ekm[:-2,:,:], -1,axis=2))*dzf[1:-1,np.newaxis,np.newaxis]) /
           (4.*dzh[1:-1,np.newaxis,np.newaxis]))

    return ((epom*((np.roll(w[1:-1,:,:],-1,axis=2) - w[1:-1,:,:]) / dx +
                   (np.roll(u[1:-1,:,:],-1,axis=2) - np.roll(u[:-2, :,:],-1,axis=2) / dzh[1:-1,np.newaxis,np.newaxis]))
            -emom*((w[1:-1,:,:] - np.roll(w[1:-1,:,:],1,axis=2)) / dx +
                   (u[1:-1,:,:] - u[:-2,:,:]) / dzh[1:-1,np.newaxis,np.newaxis])) / dx
           +
            (eopm*((np.roll(w[1:-1,:,:],-1,axis=1) - w[1:-1,:,:]) / dy +
                   (np.roll(v[1:-1,:,:],-1,axis=1) - np.roll(v[:-2, :,:],-1,axis=1) / dzh[1:-1,np.newaxis,np.newaxis]))
            -eomm*((w[1:-1,:,:] - np.roll(w[1:-1,:,:],1,axis=1)) / dy +
                   (v[1:-1,:,:] - v[:-2,:,:]) / dzh[1:-1,np.newaxis,np.newaxis])) / dy
           + 
            2./rhobh[1:-1,np.newaxis,np.newaxis]*
            (rhobf[1:-1,np.newaxis,np.newaxis]*ekm[1:-1,:,:]*(w[2:,:,:] - w[1:-1,:,:])/dzh[1:-1,np.newaxis,np.newaxis] -
             rhobf[:-2, np.newaxis,np.newaxis]*ekm[:-2, :,:]*(w[1:-1,:,:] - w[:-2,:,:])/dzh[:-2,np.newaxis,np.newaxis]) /
             dzh[1:-1,np.newaxis,np.newaxis])

def mean_mask(field,mask):
    # Assumes mask is 1-0
    if len(field.shape) == 2:
        return np.sum(field*mask) / np.sum(mask) 
    elif len(field.shape) == 3:
        return np.sum(field*mask[np.newaxis,:,:],axis=(1,2)) / np.sum(mask)
    else:
        print('Input field has wrong shape')
        return

def plot_2d(pltinp,xf,vmin=-1,vmax=1,fluct=True,cmap='RdYlBu'):
    if fluct:
        pltvar = pltinp - np.mean(pltinp)
    else:
        pltvar = pltinp
    plt.figure(figsize=(5.5,5))
    plt.imshow(pltvar,extent=np.array([xf.min(), xf.max(), xf.min(), xf.max()])/1000,
               vmin=vmin,vmax=vmax,cmap=cmap)
    plt.colorbar()
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.show()
    
def get_rad(data):
    # From https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
    h = data.shape[0]
    hc = h // 2
    w = data.shape[1]
    wc = w // 2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(X - wc, Y - hc)

    return r


def get_psd_1d_radial(psd_2d, dx):

    # TODO: Assumes even number of points in psd_2d and square domain

    N = np.min(psd_2d.shape)
    L = int(N * dx)

    # Index radii corresponding to horizontal wavenumber 2*pi/L*r
    r = get_rad(psd_2d)
    r_int = np.round(r).astype(int)
    rp = np.arange(1, N // 2 + 1)

    # SUM all psd_2d pixels with label 'kh' for 0<=kh<=N/2 * 2*pi*L
    # Will miss power contributions in 'corners' kh>N/2 * 2*pi*L
    # Ignores centre (mean, wavenumber 0)
    # This is still a variance quantity.
    psd_1d = ndimage.sum(psd_2d, r_int, index=rp)

    # Compute prefactor that converts to spectral density and corrects for
    # annulus discreteness
    Ns = ndimage.sum(np.ones(psd_2d.shape), r_int, index=rp)

    kp = 2 * np.pi / L * ndimage.sum(r, r_int, index=rp) / Ns

    psd_1d *= L ** 2 * kp / (2 * np.pi * N ** 2 * Ns)

    return psd_1d

def compute_spectrum(cloud_scalar,dx,cloud_scalar_2=None,sqrt=False):
    # FFT
    F = fftpack.fft2(cloud_scalar)  # 2D FFT (no prefactor)
    F = fftpack.fftshift(F)  # Shift so k0 is centred
    if type(cloud_scalar_2) == type(None):
        # psd
        psd_2d = np.abs(np.conjugate(F) * F)
    else:
        # coherence magnitude spectrum
        F2 = fftpack.fft2(cloud_scalar_2)
        F2 = fftpack.fftshift(F2)
        psd_2d = np.abs(np.conjugate(F) * F2)
    if sqrt:
        psd_2d = np.sqrt(psd_2d)
    psd_2d /= np.prod(cloud_scalar.shape)  # Energy-preserving 2D PSD
    psd_1d_rad = get_psd_1d_radial(psd_2d, dx)  # Azimuthal integral-> 1D radial PSD

    # Wavenumbers
    N = np.min(cloud_scalar.shape)
    L = dx * N
    k1d = 2 * np.pi / L * np.arange(1, N // 2 + 1)
    
    return k1d, psd_1d_rad

def tderive(var,time):
    return ((var[1:,1:-1] - var[:-1,1:-1])
           /(time[1:,np.newaxis] - time[:-1,np.newaxis])/3600)

def zderivef(var,dzh):
    ddz_var = ((var[:,1:] - var[:,:-1])/dzh)
    return (ddz_var[:,1:] + ddz_var[:,:-1])*0.5

def vint(field,rhob,z,plttime):
    if len(field.shape) == 3:
        var = np.trapz(rhob[:,np.newaxis,np.newaxis]*field[:,:,:],z,axis=0)
    elif len(field.shape) == 4:
        var = np.trapz(rhob[np.newaxis,:,np.newaxis,np.newaxis]*
                       field[plttime,:,:,:],z,axis=1)
    elif len(field.shape) == 2:
        var = np.trapz(rhob[np.newaxis,:]*field[plttime,:],z,axis=1)
    elif len(field.shape) == 1:
        var = np.trapz(rhob*field,z)
    return var
