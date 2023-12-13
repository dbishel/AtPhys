#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:32:44 2023

@author: dbis
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

Zbar_min = ZZ - 10
nmax = 5 # Maximum allowed shell
exc_list = [0,1,2,3] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
pf = 1

# Run model
ad = AtDat(ZZ, A, Zbar_min, nmax, exc_list)
ad.get_atomicdata(vb=0,  DIR=DIR)
ad.get_hnu(np.array(ad.Zkeys).astype(int))
ad.tidy_arrays()

# Define T, Ne, rho grids for SB
Nn, NT = 10, 51 # Number of density, temperature gridpoints
KT = np.logspace(1,np.log10(2e3), num=NT) # eV, Temperature range, to find IPD

rho0 = np.logspace(-1,3, num=Nn) # g/cc
Zbar0 = 20 # Estimated Zbar
NE = rho0 / (A*ad.mp) * Zbar0 # 1/cm^3, Ne range

Nrho = 12
rho_grid = np.logspace(-1,2, num=Nrho)

# Estimate IPD for Saha and interpolate onto NE vector
Zgrid_rho, __, __, __, __, __, CLgrid_rho = dense_plasma(Z=ZZ, Zbar=1, A=A, Ts=KT, rhos=rho_grid)

Zgrid = []
CLgrid = [] #np.zeros(shape=[NT, Nrho])
for i,t in enumerate(KT):
    CLgrid.append(np.interp(NE, rho_grid / (A*ad.mp) * Zgrid_rho[i,:], CLgrid_rho[i,:]))
    Zgrid.append(np.interp(NE, rho_grid / (A*ad.mp) * Zgrid_rho[i,:], Zgrid_rho[i,:]))
CLgrid = np.array(CLgrid)
Zgrid = np.array(Zgrid)

# %% Populations and gf
# Run Saha-Boltzmann
ad.saha_boltzmann(KT, NE, IPD=CLgrid)
ad.saha_boltzmann_rho(rho_grid)

# Calculate gf
gf = ad.get_gf(1,0,2,1,)

# Calculate average hnu
sat_avg = ad.get_hnu_average(ad.pstate, gf=gf, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]
hnu_avg = ad.get_hnu_average(ad.pstate, gf=gf, resolve='line') # Shape: [excitation, NT, Nrho, ionization]

# View Zbar
if 0:
    fig, ax = plt.subplots(figsize=[4,3])
    im = ax.pcolormesh(NE,KT, ad.Zbar, shading='nearest', vmin=0, vmax=ZZ)
    ax.set(xscale='log', yscale='log',
                  xlabel='NE (1/cc)',
                  ylabel='kT (eV)',
                  title='Saha-Boltzmann Zbar')
    plt.colorbar(im, ax=ax)

# Plot average hnu
if 1:
    plt.figure(figsize=[4,3])
    plt.semilogx(KT, hnu_avg[:,0,:]-hnu_avg[0,0,:], label=ad.Zkeys)
    plt.gca().set(#ylim=[5350,5800],
                  xlabel='kT (eV)',
                  ylabel='hnu (eV)')
    plt.legend()
    
    plt.figure(figsize=[4,3])
    [plt.semilogx(KT, sat_avg[e,:,0,:], label=ad.Zkeys, color='C{0:d}'.format(e)) for e in exc_list]
    plt.gca().set(#ylim=[5350,5800],
                  xlabel='kT (eV)',
                  ylabel='hnu (eV)')
    # plt.legend()


# Generate spectra
ad.append_lineshape(3*np.ones(ad.pstate_rho.shape), 'G') # Gaussian lineshape
# ad.append_lineshape(np.ones(ad.pstate_rho.shape), 'L')
ad.sum_linewidths()
linecenter = ad.get_linecenter()

ad.get_line_opacity(1, 0, 2, 1)

hnu_minmax = [ad.hnuarrs.flatten()[ad.hnuarrs.flatten()>0].min(),
              ad.hnuarrs.max()]
hnu_axis = np.linspace(5400, 5800, num=2000)
ls = ad.generate_lineshapes(hnu_axis)

ad.generate_spectra(hnu_axis)

ad.print_table()

# Gifs
gifT = np.arange(0,len(KT))
gifrho = np.ones(len(KT), dtype=int)*-1

bins = np.arange(5400, 5800, 5)

# %% Plots
#### Plot opacity image versus T at one rho
rhoidx = -1 # Indexed against rho_grid
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


# Plot line opacity versus rho for one T, divided by state populations
if 0:
    Zidx = 0 # Charge state
    Tidx = 25
    labs = ['Zbar = {0:s}, state = {1:d}'.format(ad.Zkeys[Zidx], item)
            for item in range(ad.kappa_line.shape[-1])]
    fig, ax = plt.subplots(figsize=[4,3])
    ax.semilogy(ad.rho_grid[:,Ellipsis],
                 # (ad.kappa_line/ad.pstate_rho)[Tidx,:,Zidx,:])
                 (ad.kappa_line)[Tidx,:,Zidx,:], label=labs)
    plt.legend()
    
    ax2 = plt.twinx(ax)
    ax2.semilogy(ad.rho_grid[:,Ellipsis],
                  (ad.pstate_rho)[Tidx,:,Zidx,:], '--')
    
    ax.set(xlabel=r'$\rho$ (g/cm$^3$)',
           ylabel=r'Line $\kappa$ (cm$^2$/g)',
           title=r'$\kappa$ (solid) $\rho$-dependence due to state population (dashed)')
    ax2.set(ylabel='State population',
            ylim=[1e-5,1])

#### Plot hnu by excitation degree
fig, ax = plt.subplots(figsize=[4,3])
ad.plot_hnu([0,3], xaxis='Zbar+exc', fig=fig, ax=ax)

# ad.plot_zbar('ne')
# ad.plot_rho_interp()
# buttons = ad.plot_sb_buttons()

#### Histogram of populations within different spectral bands
if 0:
    bins = np.linspace(6400, 6800) # Full spectrum
    # bins = np.linspace(6515,6535, num=6) # N-like complex
    # breakpoint()
    ad.plot_pop_hist_trace(0, bins=bins)
    plt.gca().set(xscale='log')



#### Plot opacity traces at one rho
rhoidx = -2

Tbounds = np.array([100, 1000])
Tidxs = np.where((KT>Tbounds[0]) * (KT<Tbounds[1]))[0]

plt.figure()
norm = mpl.colors.Normalize(vmin=Tbounds.min(), vmax=Tbounds.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)


[plt.plot(hnu_axis,ad.kappa[Tidx,rhoidx,:], c=cmap.to_rgba(KT[Tidx])) 
 for Tidx in Tidxs[::3]]
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'))
plt.gca().set(title=r'$\kappa$ (cm$^2$/g) at {0:0.1f} g/cm$^3$'.format(rho_grid[rhoidx]),
              xlabel='hnu (eV)',
              ylabel=r'$\kappa_\nu$ (cm$^2$/g)')

# plt.plot(hnu_axis,ad.kappa[-7,rhoidx,:],color='k',ls='--')

#### Satellite resolved line centers
zidx = 3
hnu_avg = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]

fig, axs = plt.subplots(2, figsize=[5,4], sharex=True)
axs[0].plot(KT, ad.Zbar[:,rhoidx], color='k')
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
