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

Zbar_min = ZZ - 15
nmax = 5 # Maximum allowed shell
exc_list = [0,1,2,3] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
# exc_list = [0,1] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
pf = 1

# Run model
ad = AtDat(ZZ, A, Zbar_min, nmax, exc_list)
# breakpoint()
ad.get_atomicdata(vb=0,  DIR=DIR)
ad.get_hnu(np.array(ad.Zkeys).astype(int))
ad.tidy_arrays()

# Define T, Ne, rho grids for SB
Nn, NT = 10, 51 # Number of density, temperature gridpoints
KT = np.logspace(1,4, num=NT) # eV, Temperature range, to find IPD

rho0 = np.logspace(-1,2, num=Nn) # g/cc
Zbar0 = 20 # Estimated Zbar
NE = rho0 / (A*ad.mp) * Zbar0 # 1/cm^3, Ne range

Nrho = 12
rho_grid = np.logspace(-1,1, num=Nrho)

# Estimate IPD for Saha and interpolate onto NE vector
Zgrid_rho, __, __, __, __, __, CLgrid_rho = dense_plasma(Z=ZZ, Zbar=1, A=A, Ts=KT, rhos=rho_grid)

CLgrid = [] #np.zeros(shape=[NT, Nrho])
for i,t in enumerate(KT):
    CLgrid.append(np.interp(NE, rho_grid / (A*ad.mp) * Zgrid_rho[i,:], CLgrid_rho[i,:]))
CLgrid = np.array(CLgrid)

# Check
fig, axs = plt.subplots(2, figsize=[4,6])
im = axs[0].pcolormesh(NE,KT, CLgrid, shading='nearest');
axs[0].set(xscale='log', yscale='log',
              xlabel='Ne (1/cc)',
              ylabel='kT (eV)',)
plt.colorbar(im, ax=axs[0])

im = axs[1].pcolormesh(rho_grid,KT, CLgrid_rho, shading='nearest');
axs[1].set(xscale='log', yscale='log',
              xlabel='rho (g/cc)',
              ylabel='kT (eV)',)
plt.colorbar(im, ax=axs[1])

# Run Saha-Boltzmann
ad.saha_boltzmann(KT, NE, IPD=CLgrid)
ad.saha_boltzmann_rho(rho_grid)

# Get oscillator strengths
gf_old = ad.get_gf(1,0,2,1, return_gs=False, old=True)
gf_new = ad.get_gf(1,0,2,1, return_gs=False, old=False)

# Satellite resolved line centers
hnu_avg_new = ad.get_hnu_average(ad.pstate, gf=gf_new, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]
hnu_avg_old = ad.get_hnu_average(ad.pstate, gf=gf_old, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]

ridx = 0
zidx = 2 # N-like
plt.figure()
[plt.plot(KT, hnu_avg_new[eidx,:,ridx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d} new'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]
[plt.plot(KT, hnu_avg_old[eidx,:,ridx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          ls='--',
          label='Z*={0:s}, exc={1:d} old'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]
plt.legend()


# Line-complex resolved line centers
hnu_avg_new = ad.get_hnu_average(ad.pstate, gf=gf_new, resolve='line') # Shape: [excitation, NT, Nrho, ionization]
hnu_avg_old = ad.get_hnu_average(ad.pstate, gf=gf_old, resolve='line') # Shape: [excitation, NT, Nrho, ionization]

ridx = 0
zidx = 2 # N-like
plt.figure(figsize=[4,3])
plt.plot(KT, hnu_avg_new[:,ridx,:], label=ad.Zkeys)
plt.plot(KT, hnu_avg_old[:,ridx,:], ls='--')
plt.gca().set(ylim=[5350,5800])
plt.legend()

# %% Focused view of a singlecharge state
ridx = 0
zidx = 2 # 0 = Ne-like, 1 = F-like
fig, axs = plt.subplots(2, figsize=[5,4])

axs[0].plot(KT, ad.Zbar[:,ridx], color='k')
axs[0].set(ylabel='Zbar')

# Satellite resolved line centers
hnu_avg_new = ad.get_hnu_average(ad.pstate, gf=gf_new, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]
hnu_avg_old = ad.get_hnu_average(ad.pstate, gf=gf_old, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]

[axs[1].plot(KT, hnu_avg_new[eidx,:,ridx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d} new'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]
[axs[1].plot(KT, hnu_avg_old[eidx,:,ridx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          ls='--',
          # label='Z*={0:s}, exc={1:d} old'.format(ad.Zkeys[zidx-eidx], eidx),
          )
     for eidx in exc_list if (zidx-eidx)>=0]

# Line-complex resolved line centers
hnu_avg_new = ad.get_hnu_average(ad.pstate, gf=gf_new, resolve='line') # Shape: [NT, Nrho, ionization]
hnu_avg_old = ad.get_hnu_average(ad.pstate, gf=gf_old, resolve='line') # Shape: [NT, Nrho, ionization]

axs[1].plot(KT, hnu_avg_new[:,ridx,zidx], label='Averaged', color='k')
axs[1].plot(KT, hnu_avg_old[:,ridx,zidx], ls='--', color='k')

axs[1].set(xlabel='kT (eV)',
          ylabel='hnu (eV)')

plt.legend(bbox_to_anchor=(1.,1))

# %% 
ridx = -1
zidx = 7 # 3 = N-like
fig, axs = plt.subplots(2, figsize=[5,4], sharex=True)

axs[0].plot(KT, ad.Zbar[:,ridx], color='k')
axs[0].set(ylabel='Zbar',
           title='Ne={0:0.1e} 1/cc'.format(NE[ridx]))

# Satellite resolved line centers
hnu_avg = ad.get_hnu_average(ad.pstate, gf=gf_new, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]

[axs[1].plot(KT, hnu_avg[eidx,:,ridx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d}'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]

# Line-complex resolved line centers
hnu_avg, pgf = ad.get_hnu_average(ad.pstate, gf=gf_new, resolve='line',
                                  return_weight=True) # Shape: [NT, Nrho, ionization]

axs[1].scatter(KT, hnu_avg[:,ridx,zidx], label='Averaged', color='k',
            facecolor='None')
axs[1].scatter(KT, hnu_avg[:,ridx,zidx], color='k',
            alpha=pgf[:,ridx,zidx]/np.nanmax(pgf[:,ridx,zidx]))

axs[1].set(xlabel='kT (eV)',
          ylabel='hnu (eV)',
          xscale='log')

plt.legend(bbox_to_anchor=(1.,1))
