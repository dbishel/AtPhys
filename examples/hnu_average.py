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

Zbar_min = ZZ - 10
nmax = 5 # Maximum allowed shell
exc_list = [0,1,2,3] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
# exc_list = [0,1] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
pf = 1

# Run model
ad = AtDat(ZZ, A, Zbar_min, nmax, exc_list)
ad.get_atomicdata(vb=0,  DIR=DIR)
ad.get_hnu(np.array(ad.Zkeys).astype(int))
ad.tidy_arrays()

# Define T, Ne, rho grids for SB
Nn, NT = 10, 51 # Number of density, temperature gridpoints
KT = np.logspace(1,np.log10(2e3), num=NT) # eV, Temperature range, to find IPD

rho0 = np.logspace(-1,2, num=Nn) # g/cc
Zbar0 = 20 # Estimated Zbar
NE = rho0 / (A*ad.mp) * Zbar0 # 1/cm^3, Ne range

Nrho = 12
rho_grid = np.logspace(-1,3, num=Nrho)

# Estimate IPD for Saha and interpolate onto NE vector
Zgrid_rho, __, __, __, __, __, CLgrid_rho = dense_plasma(Z=ZZ, Zbar=1, A=A, Ts=KT, rhos=rho_grid,
                                                         CL='IS Atzeni')

Zgrid = []
CLgrid = [] #np.zeros(shape=[NT, Nrho])
for i,t in enumerate(KT):
    CLgrid.append(np.interp(NE, rho_grid / (A*ad.mp) * Zgrid_rho[i,:], CLgrid_rho[i,:]))
    Zgrid.append(np.interp(NE, rho_grid / (A*ad.mp) * Zgrid_rho[i,:], Zgrid_rho[i,:]))
CLgrid = np.array(CLgrid) #NE-indexed
Zgrid = np.array(Zgrid) # NE-indexed

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
ad.saha_boltzmann(KT, NE, IPD=0) # NE-indexed
ad.saha_boltzmann_rho(rho_grid) # rho_grid-indexed

# View Zbar
fig, ax = plt.subplots(figsize=[4,3])
im = ax.pcolormesh(NE,KT, ad.Zbar, shading='nearest', vmin=0, vmax=ZZ)
ax.set(xscale='log', yscale='log',
              xlabel='NE (1/cc)',
              ylabel='kT (eV)',
              title='Saha-Boltzmann Zbar')
plt.colorbar(im, ax=ax)

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
sat_avg = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]
# Line-complex resolved line centers
hnu_avg = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='line') # Shape: [excitation, NT, Nrho, ionization]

# %% Excitation-resolved lines
rhoidx = 0
zidx = 3 # 3 = N-like

# Single charge-state
plt.figure(figsize=[4,3])
[plt.semilogx(KT, sat_avg[eidx,:,rhoidx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d} new'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]
plt.gca().set(xlabel='kT (eV)',
              ylabel='hnu (eV)')
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
              ylim=[None,5800])
# plt.legend()

# %% T-dependence of <hnu>
rhoidx = -1

#### Single satellite complex
fig, axs = plt.subplots(2, figsize=[5,4], sharex=True)

axs[0].semilogx(KT, ad.Zbar_rho[:,rhoidx], color='k')
axs[0].set(ylabel='Zbar',
           title=r'$\rho$={0:0.1e} g/cm$^3$'.format(rho_grid[rhoidx]))

# Satellite resolved line centers
hnu_avg = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]

[axs[1].plot(KT, hnu_avg[eidx,:,rhoidx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d}'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]

# Line-complex resolved line centers
hnu_avg = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='line') # Shape: [NT, Nrho, ionization]

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


# %% Satellite resolved line centers
hnu_avg = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]

fig, axs = plt.subplots(2, figsize=[4,4], sharex=True)
axs[0].plot(KT, ad.Zbar_rho[:,rhoidx], color='k')
axs[0].set(ylabel='Zbar',
           title=r'Z*={1:s} satellites, $\rho=${0:0.2f} g/cm$^3$'.format(rho_grid[rhoidx],
                                                                        ad.Zkeys[zidx]))


[axs[1].plot(KT, hnu_avg[eidx,:,rhoidx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d}'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]

# Line-complex resolved line centers
hnu_avg, pgf = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='line',
                                  return_weight=True) # Shape: [NT, Nrho, ionization]

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

# %% Check populations
print('Sum over Saha != 1:')
print('    ',np.where(abs(ad.psaha.sum(-1)-1)>1e-6))
print('Sum over Boltz != 1:')
print('    ', np.where(abs(ad.pboltz.sum(-1)-1)>1e-6))
print('Sum over state populations + population of bare ion != 1:')
print('    ', np.where(abs(ad.pstate.sum(-1).sum(-1)+ad.psaha[:,:,-1]-1)>1e-2))

