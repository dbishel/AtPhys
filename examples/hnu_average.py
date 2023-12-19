#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:26:56 2023

@author: dbis

This script calculates the ionization and excitation-state-resolved average
transition energy.
"""

# Python modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp

import os                       # Used to e.g. change directory
import sys                      # Used to put breaks/exits for debugging
import re

sys.path.append('../')
from ScHyd import get_ionization, AvIon, dense_plasma
from saha_boltzmann_populations import saha, boltzmann
from ScHyd_AtomicData import AtDat

# %% Run AD
DIR = '../complexes/'

ZZ = 24 # Nuclear charge
A = 51.996 # Nucleon number

Zbar_min = ZZ - 15
nmax = 5 # Maximum allowed shell
exc_list = [0,1,2,] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
# exc_list = [0,1] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
pf = 1

# Run model
ad = AtDat(ZZ, A, Zbar_min, nmax, exc_list)
ad.get_atomicdata(vb=0,  DIR=DIR, pop_method='inline')
ad.get_hnu(np.array(ad.Zkeys).astype(int))
ad.tidy_arrays()

if 0: # Check against file method
    adf = AtDat(ZZ, A, Zbar_min, nmax, exc_list)
    adf.get_atomicdata(vb=0,  DIR=DIR, pop_method='file')
    adf.get_hnu(np.array(ad.Zkeys).astype(int))
    adf.tidy_arrays()
    
    zidx=0
    print(ad.Zkeys[zidx])
    [print(pi, pj) for pi,pj in zip(ad.Pnarrs['lo'][zidx], adf.Pnarrs['lo'][zidx])];
    sys.exit()
    
# print(ad.Pnarrs['lo'][1])
# sys.exit()

# Define T, Ne, rho grids for SB
Nn, NT = 10, 51 # Number of density, temperature gridpoints
KT = np.logspace(1,np.log10(2e3), num=NT) # eV, Temperature range, to find IPD

rho0 = np.logspace(-1,2, num=Nn) # g/cc
Zbar0 = 20 # Estimated Zbar
NE = rho0 / (A*ad.mp) * Zbar0 # 1/cm^3, Ne range

Nrho = 12
rho_grid = np.logspace(-1,3, num=Nrho)

# Estimate IPD for Saha and interpolate onto NE vector
if 1: # Skip while testing
    Zgrid_rho, __, __, __, __, __, CLgrid_rho = dense_plasma(Z=ZZ, Zbar=1, A=A, Ts=KT, rhos=rho_grid,
                                                              CL='IS Atzeni')
    
    Zgrid = []
    CLgrid = [] #np.zeros(shape=[NT, Nrho])
    for i,t in enumerate(KT):
        CLgrid.append(np.interp(NE, rho_grid / (A*ad.mp) * Zgrid_rho[i,:], CLgrid_rho[i,:]))
        Zgrid.append(np.interp(NE, rho_grid / (A*ad.mp) * Zgrid_rho[i,:], Zgrid_rho[i,:]))
    CLgrid = np.array(CLgrid) #NE-indexed
    Zgrid = np.array(Zgrid) # NE-indexed

if 0:
    # Check IPD interpolation
    fig, axs = plt.subplots(2, figsize=[4,6])
    im = axs[0].pcolormesh(NE,KT, CLgrid, shading='nearest');
    axs[0].set(xscale='log', yscale='log',
                  xlabel='Ne (1/cc)',
                  ylabel='kT (eV)',
                  title='Interpolated, used in S-B')
    plt.colorbar(im, ax=axs[0])
    
    im = axs[1].pcolormesh(rho_grid,KT, CLgrid_rho, shading='nearest');
    axs[1].set(xscale='log', yscale='log',
                  xlabel='rho (g/cc)',
                  ylabel='kT (eV)',
                  title='Calculated, not used')
    plt.colorbar(im, ax=axs[1])
    
    # Check ZBar
    fig, ax = plt.subplots(figsize=[4,3])
    im = ax.pcolormesh(NE, KT, Zgrid, shading='nearest');
    ax.set(xscale='log', yscale='log',
                  xlabel='Ne (1/cc)',
                  ylabel='kT (eV)',
                  title='ScHyd Zbar')
    plt.colorbar(im, ax=ax)


# %% Populations and gf
# Run Saha-Boltzmann
ad.saha_boltzmann(KT, NE, IPD=CLgrid) # NE-indexed
ad.saha_boltzmann_rho(rho_grid) # rho_grid-indexed

# View Zbar from Saha-Boltzmann
if 0:
    fig, axs = plt.subplots(2, figsize=[4,6])
    
    im = axs[0].pcolormesh(NE,KT, ad.Zbar, shading='nearest', vmin=0, vmax=ZZ)
    plt.colorbar(im, ax=axs[0])
    
    im = axs[1].pcolormesh(rho_grid,KT, ad.Zbar_rho, shading='nearest', vmin=0, vmax=ZZ)
    plt.colorbar(im, ax=axs[1])
    
    axs[0].set(xscale='log', yscale='log',
                  xlabel='NE (1/cc)',
                  ylabel='kT (eV)',
                  title='Saha-Boltzmann Zbar')
    axs[1].set(xscale='log', yscale='log',
                  xlabel='rho (g/cc)',
                  ylabel='kT (eV)',
                  title='Saha-Boltzmann Zbar')
    
    # Compare dense plasma Zbar to Saha-Boltzmann Zbar
    Zdiff = ad.Zbar-Zgrid
    plt.figure()
    plt.pcolormesh(NE,KT, Zdiff, shading='nearest',
                   vmin=-abs(Zdiff).max(), vmax=abs(Zdiff).max(),
                   cmap='bwr')
    plt.gca().set(xscale='log',yscale='log')
    plt.colorbar()

# Get oscillator strengths
gf = ad.get_gf(1,0,2,1, return_gs=False)

#### Average hnu
# Keep indexed to rho_grid throughout
# Satellite-resolved
sat_avg, pgf_sat = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='ionization',
                             return_weight=True) # Shape: [excitation, NT, Nrho, ionization]
# Line-complex resolved line centers
hnu_avg, pgf = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='line',
                                  return_weight=True) # Shape: [excitation, NT, Nrho, ionization]

# %% Opacity
if 0:
    # Generate spectra
    ad.append_lineshape(3*np.ones(ad.pstate_rho.shape), 'G') # Gaussian lineshape
    # ad.append_lineshape(np.ones(ad.pstate_rho.shape), 'L')
    ad.sum_linewidths()
    # linecenter = ad.get_linecenter() # To troubleshoot – unused
    
    ad.get_line_opacity(1, 0, 2, 1)
    
    hnu_minmax = [ad.hnuarrs.flatten()[ad.hnuarrs.flatten()>0].min(),
                  ad.hnuarrs.max()]
    hnu_axis = np.linspace(5400, 5800, num=2000)
    # ls = ad.generate_lineshapes(hnu_axis) # Unit-height line shapes. To troubleshoot – unused
    
    ad.generate_spectra(hnu_axis)
    
    # ad.print_table() # broken
    
    # Gifs
    # gifT = np.arange(0,len(KT))
    # gifrho = np.ones(len(KT), dtype=int)*-1
    
    # bins = np.arange(5400, 5800, 5)

# %% Check populations
print('Sum over Saha != 1:')
print('    ',np.where(abs(ad.psaha.sum(-1)-1)>1e-6))
print('Sum over Boltz != 1:')
print('    ', np.where(abs(ad.pboltz.sum(-1)-1)>1e-6))
print('Sum over state populations + population of bare ion != 1:')
print('    ', np.where(abs(ad.pstate.sum(-1).sum(-1)+ad.psaha[:,:,-1]-1)>1e-2))

# View populations within 1 ionization state
Tidx = -1
rhoidx = 5
zidx = 3
print()
print('pgf-weights within one ionization state:')
print('{0:0.2f} eV, {1:0.2f} g/cc'.format(KT[Tidx], rho_grid[rhoidx]))
print('{0:5s} | {1:3s} | {2:10s} | {3:10s}'.format('Zbar','exc', 'pop x gf', 'pop x gf (norm)'))
[print('{0:5s} | {1:3d} | {2:10.1e} | {3:10.1e}'.format(
    ad.Zkeys[zidx], e,
    pgf_sat[e,Tidx,rhoidx,zidx],
    pgf_sat[e,Tidx,rhoidx,zidx]/pgf_sat[0,Tidx,rhoidx,zidx])) for e in exc_list]

print('Saha balance')
[print('{0:s} : {1:8.1e}'.format(ad.Zkeys[zidx], ad.psaha[Tidx,rhoidx,zidx])) for zidx in range(ad.psaha.shape[-1]-1)];

# %% Check individual states
zidx = 4
print(ad.Zkeys[zidx])
for e in exc_list:
    locond = ad.excarrs['lo'][zidx]==e
    upcond = ad.excarrs['up'][zidx]==e
    print('Exc =', e)
    print('  ', np.array(locond).astype(int))
    print('  ', np.array(upcond).astype(int))
    
    [print('  ', p) for p in ad.Pnarrs['lo'][zidx][locond]]
    

# %% Plot: Excitation-resolved lines
rhoidx = 7
zidx = len(ad.Zkeys)-5 # 3 = N-like

# Single charge-state
plt.figure(figsize=[4,3])
[plt.semilogx(KT, sat_avg[eidx,:,rhoidx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d} new'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]
plt.gca().set(xlabel='kT (eV)',
              ylabel='hnu (eV)',)
plt.legend()

# All charge states
plt.figure(figsize=[4,3])
[plt.semilogx(KT, sat_avg[eidx,:,rhoidx,:],
          color='C{0:d}'.format(eidx),
          # label='Z*={0:s}, exc={1:d} new'.format(ad.Zkeys[zidx-eidx], eidx)
          )
      for eidx in exc_list if (zidx-eidx)>=0]
plt.gca().set(xlabel='kT (eV)',
              ylabel='hnu (eV)',
              title=r'Excitation-resolved $\langle h\nu \rangle$',
              ylim=[5400,5800])
# plt.legend()

# %% Plot: T-dependence of <hnu>

#### Single satellite complex
fig, axs = plt.subplots(2, figsize=[5,4], sharex=True)

axs[0].semilogx(KT, ad.Zbar_rho[:,rhoidx], color='k')
axs[0].set(ylabel='Zbar',
           title=r'$\rho$={0:0.1e} g/cm$^3$'.format(rho_grid[rhoidx]))

# Satellite resolved line centers
[axs[1].plot(KT, sat_avg[eidx,:,rhoidx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d}'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]

# Line-complex resolved line centers
axs[1].plot(KT, hnu_avg[:,rhoidx,zidx], label='Averaged', color='k')

axs[1].set(xlabel='kT (eV)',
          ylabel='hnu (eV)')

plt.legend(bbox_to_anchor=(1.,1))

#### All satellite complexes, average only
# plt.figure(figsize=[4,3])
# plt.semilogx(KT, hnu_avg[:,rhoidx,:], label=ad.Zkeys)
# plt.gca().set(ylim=[5350,5800],
#               xlabel='kT (eV)',
#               ylabel='hnu (eV)')
# plt.legend()


# %% Plot: Satellite resolved line centers
fig, axs = plt.subplots(2, figsize=[4,4], sharex=True)
axs[0].plot(KT, ad.Zbar_rho[:,rhoidx], color='k')
axs[0].set(ylabel='Zbar',
           title=r'Z*={1:s} satellites, $\rho=${0:0.2f} g/cm$^3$'.format(rho_grid[rhoidx],
                                                                        ad.Zkeys[zidx]))


[axs[1].plot(KT, sat_avg[eidx,:,rhoidx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d}'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]

# Line-complex resolved line centers
axs[1].scatter(KT, hnu_avg[:,rhoidx,zidx], label='Averaged', color='k',
            facecolor='None')
axs[1].scatter(KT, hnu_avg[:,rhoidx,zidx], color='k',
            alpha=pgf[:,rhoidx,zidx]/np.nanmax(pgf[:,rhoidx,zidx]))

axs[1].set(xlabel='kT (eV)',
          ylabel='hnu (eV)',
          # xscale='log',
            xlim=[0,1100],
          )


axs[1].legend(bbox_to_anchor=(1.,0.6))

# # All complexes
# fig, axs = plt.subplots(len(ad.Zkeys)-1, figsize=[8,24], sharex=True)
# zmin = 1
# for zzz in range(zmin,len(ad.Zkeys)):
#     [axs[zzz-zmin].plot(KT, sat_avg[eidx,:,rhoidx,zzz-eidx],
#               color='C{0:d}'.format(eidx),
#               label='Z*={0:s}, exc={1:d}'.format(ad.Zkeys[zzz-eidx], eidx))
#          for eidx in exc_list if (zzz-eidx)>=0]
    
#     # Line-complex resolved line centers
#     axs[zzz-zmin].scatter(KT, hnu_avg[:,rhoidx,zzz], label='Averaged', color='k',
#                 facecolor='None')
#     axs[zzz-zmin].scatter(KT, hnu_avg[:,rhoidx,zzz], color='k',
#                 alpha=pgf[:,rhoidx,zzz]/np.nanmax(pgf[:,rhoidx,zzz]))
    
# axs[zzz-zmin].set(xlabel='kT (eV)',
#           ylabel='hnu (eV)',
#           # xscale='log',
#             xlim=[0,1100],
#           )



# %% Plot: opacity
#### Plot opacity image versus T at one rho
plt.figure(figsize=[4,3])
plt.pcolormesh(ad.KT, hnu_axis, ad.kappa[:,rhoidx,:].T, shading='nearest',
               # cmap='viridis')
                cmap='gist_earth_r')
plt.gca().set(aspect='auto',
              xlabel='T (eV)',
              ylabel='hnu (eV)',
              title=r'$\kappa$ (cm$^2$/g) at {0:0.1f} g/cm$^3$'.format(rho_grid[rhoidx])
              )
plt.colorbar()

#### Plot opacity traces at one rho
Tbounds = np.array([100, 1000])
Tidxs = np.where((KT>Tbounds[0]) * (KT<Tbounds[1]))[0]

plt.figure(figsize=[4,3])
norm = mpl.colors.Normalize(vmin=Tbounds.min(), vmax=Tbounds.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
[plt.plot(hnu_axis,ad.kappa[Tidx,rhoidx,:], c=cmap.to_rgba(KT[Tidx])) 
 for Tidx in Tidxs[::3]]
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'))
plt.gca().set(title=r'$\kappa$ (cm$^2$/g) at {0:0.1f} g/cm$^3$'.format(rho_grid[rhoidx]),
              xlabel='hnu (eV)',
              ylabel=r'$\kappa_\nu$ (cm$^2$/g)',
              # xlim=[5525,5545],
              )


plt.figure(figsize=[4,3])
Tidx = np.where(KT > 500)[0][0]
plt.plot(hnu_axis,ad.kappa[Tidx,rhoidx,:],label='kT = {0:0.0f} eV'.format(KT[Tidx]))
plt.gca().set(title=r'$\kappa$ (cm$^2$/g) at {0:0.1f} g/cm$^3$'.format(rho_grid[rhoidx]),
              xlabel='hnu (eV)',
              ylabel=r'$\kappa_\nu$ (cm$^2$/g)',
              # xlim=[5525,5545],
              )
plt.legend()

