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

Zbar_min = ZZ - 10
nmax = 3 # Maximum allowed shell
# exc_list = np.arange(8) # Excitation degrees to consider (lower state is ground state, singly excited, ...)
exc_list = [0,1,2,] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
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
                                                              CL='IS Hansen',)
    
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
# Averaged within an excitation degree
hnu_exc, pgf_exc = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='ionization',
                             return_weight=True) # Shape: [excitation, NT, Nrho, ionization]
# Average within a satellite complex
# breakpoint()
hnu_sat, pgf = ad.get_hnu_average(ad.pstate_rho, gf=gf, resolve='line',
                                  return_weight=True) # Shape: [NT, Nrho, satellite]

# %% Opacity
if 1:
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
Tidx = 30
rhoidx = 5
zidx = 3
print()
print('pgf-weights within one ionization state:')
print('{0:0.2f} eV, {1:0.2f} g/cc'.format(KT[Tidx], rho_grid[rhoidx]))
print('{0:5s} | {1:3s} | {2:10s} | {3:10s}'.format('Zbar','exc', 'pop x gf', 'pop x gf (norm)'))
[print('{0:5s} | {1:3d} | {2:10.1e} | {3:10.1e}'.format(
    ad.Zkeys[zidx], e,
    pgf_exc[e,Tidx,rhoidx,zidx],
    pgf_exc[e,Tidx,rhoidx,zidx]/pgf_exc[0,Tidx,rhoidx,zidx])) for e in exc_list]

print('Saha balance')
[print('{0:s} : {1:8.1e}'.format(ad.Zkeys[zidx], ad.psaha[Tidx,rhoidx,zidx])) for zidx in range(ad.psaha.shape[-1]-1)];

# %% Check individual states
cs = '18' # Charge state
csidx = np.where(np.isin(ad.Zkeys, cs))[0][0]
sidx = np.where(ad.sats==ad.Z-int(cs))[0][0] # Satellite index

print(ad.Zkeys[csidx])
for e in exc_list:
    locond = ad.excarrs['lo'][csidx]==e
    upcond = ad.excarrs['up'][csidx]==e
    # print('Exc =', e)
    # print('  ', np.array(locond).astype(int))
    # print('  ', np.array(upcond).astype(int))
    Z = ad.Zkeys[csidx-e]
    [print(Z, '  ', *p, *g, El, hnu) for p,g, El, hnu
                               in zip(ad.Pnarrs['lo'][csidx-e][locond],
                                      ad.gtot_lists['lo'][csidx-e][locond],
                                      ad.Earrs['lo'][csidx-e][locond]-ad.Earrs['lo'][csidx-e].min(),
                                      ad.hnuarrs[csidx-e][locond])]
# Saha balance at 400 eV
# Tidx = np.where(KT>400)[0][0]
# rhoidx = np.where(rho_grid<40)[0][0]
# print(ad.psaha[Tidx,rhoidx,csidx:csidx-3:-1])
# print(ad.psaha[Tidx,rhoidx])

# T-dependent table at 400 eV
Tidx = np.where(KT>400)[0][0]
rhoidx = np.where(rho_grid<40)[0][-1]

hnu_minmax = [ad.hnuarrs.flatten()[ad.hnuarrs.flatten()>0].min(),
              ad.hnuarrs.max()]
hnu_axis = np.linspace(5400, 5800, num=2000)

# Vary nmax
ad3 = AtDat(ZZ, A, Zbar_min, nmax=3, exc_list=exc_list)
ad3.get_atomicdata(vb=0,  DIR=DIR, pop_method='inline')
ad3.get_hnu(np.array(ad.Zkeys).astype(int))
ad3.tidy_arrays()
ad3.saha_boltzmann(KT, NE, IPD=CLgrid) # NE-indexed
ad3.saha_boltzmann_rho(rho_grid) # rho_grid-indexed
# Generate spectra
ad3.append_lineshape(3*np.ones(ad3.pstate_rho.shape), 'G') # Gaussian lineshape
ad3.sum_linewidths()
ad3.get_line_opacity(1, 0, 2, 1)
ad3.generate_spectra(hnu_axis)



ad4 = AtDat(ZZ, A, Zbar_min, nmax=4, exc_list=exc_list)
ad4.get_atomicdata(vb=0,  DIR=DIR, pop_method='inline')
ad4.get_hnu(np.array(ad.Zkeys).astype(int))
ad4.tidy_arrays()
ad4.saha_boltzmann(KT, NE, IPD=CLgrid) # NE-indexed
ad4.saha_boltzmann_rho(rho_grid) # rho_grid-indexed
ad4.append_lineshape(3*np.ones(ad4.pstate_rho.shape), 'G') # Gaussian lineshape
ad4.sum_linewidths()
ad4.get_line_opacity(1, 0, 2, 1)
ad4.generate_spectra(hnu_axis)

ad5 = AtDat(ZZ, A, Zbar_min, nmax=5, exc_list=exc_list)
ad5.get_atomicdata(vb=0,  DIR=DIR, pop_method='inline')
ad5.get_hnu(np.array(ad.Zkeys).astype(int))
ad5.tidy_arrays()
ad5.saha_boltzmann(KT, NE, IPD=CLgrid) # NE-indexed
ad5.saha_boltzmann_rho(rho_grid) # rho_grid-indexed
ad5.append_lineshape(3*np.ones(ad5.pstate_rho.shape), 'G') # Gaussian lineshape
ad5.sum_linewidths()
ad5.get_line_opacity(1, 0, 2, 1)
ad5.generate_spectra(hnu_axis)

ad_list = [ad3,ad4,ad5]
nmax_list = [3,4,5,]

print('            || {0:36s} || {1:36s} || {2:36s} |'.format('n<=3', 'n<=4', 'n<=5'))
tmp = '| {0:10s} | {1:10s} |{2:12s}|'.format('sum(g)', 'pS', 'pS sum pB g')
print('| Iso | Exc |'+tmp+tmp+tmp)
for e in [0,1,2,3]:
    Z = ad.Zkeys[csidx-e]
    
    gsum = []
    ps = []
    op = []

    # Loop over each calculation    
    for adn in [ad3,ad4,ad5]:
        locond = adn.excarrs['lo'][csidx]==e
        gsum.append(adn.gtot_lists['lo'][csidx-e][locond].prod(axis=1).sum())
        ps.append(adn.psaha_rho[Tidx,rhoidx,csidx-e]) # Saha balance of this charge state
        
        op.append(np.sum(ps[-1] * adn.gtot_lists['lo'][csidx-e][locond].prod(axis=1)
                    *adn.pboltz_rho[Tidx,rhoidx, csidx-e, locond])) # (g x Saha x Boltzmann) ~ opacity        
    
    lead = '| {0:3.0f} | {1:3.0f} |'.format(ad.Z-int(cs)+e, e)
    tmp = ''.join(['| {0:10.0f} | {1:10.2e} | {2:10.2e} |'.format(g,p,o) for g,p,o in zip(gsum, ps, op)])
    print(lead+tmp)

hnu_exc_n = []
pgf_n = []
hnu_sat_n = []
pgf_exc_n = []

for adn in ad_list:
    gfn = adn.get_gf(1,0,2,1, return_gs=False)
    tmp = adn.get_hnu_average(adn.pstate_rho, gf=gfn, resolve='line',
                                  return_weight=True)
    hnu_sat_n.append(tmp[0])
    pgf_n.append(tmp[1])

    tmp = adn.get_hnu_average(adn.pstate_rho, gf=gfn, resolve='ionization',
                                  return_weight=True)
    hnu_exc_n.append(tmp[0])
    pgf_exc_n.append(tmp[1])
                     
hnu_sat_n = np.array(hnu_sat_n)
plt.figure()
plt.plot(KT, hnu_sat_n[:,:,rhoidx, sidx].T)
plt.legend([3,4,5])

print('{3:12s}|| {0:36.1f} || {1:36.1f} || {2:36.1f} |'.format(*hnu_sat_n[:,Tidx,rhoidx,sidx],
                                                            '<hnu> (eV)'))

    
 # %% Plot: Generate level diagram
# Energy level diagram is constructed from total atom energy of different configurations,
# NOT hydrogenic "levels" of a single configuration
cs = '18' # Charge state
csidx = np.where(np.isin(ad.Zkeys, cs))[0][0]

# Plot by excitation degree
cond = np.where(ad.Earrs['lo'][csidx]<0)[0] # Valid bound states
cond = ad.Earrs['lo'][csidx]<0 # Valid bound states
E = np.array([ad.Earrs['lo'][csidx]
              -ad.Earrs['lo'][csidx+1][0]]).squeeze()[cond] # Total atom energies, referenced to ionization potential of charge state
xrange = [(ad.excarrs['lo'][csidx]-0.45)[cond],
          (ad.excarrs['lo'][csidx]+0.45)[cond]] # Line extents sorted by excitation degree

fig, ax = plt.subplots(figsize=[4,3])

ax.hlines(E, *xrange, 'k', lw=1) # Plot energy levels
ax.hlines(0, xrange[0].min() , xrange[1].max(), 'k', ls='--', lw=1) # Plot ionization potential

ax.set_xticks(np.arange(ad.excarrs['lo'][csidx].max()+1))
xlabs = ['Ground\nstate', 'Singly\nexcited', 'Doubly\n excited', 'Triply\nexcited']
xlabs.extend(['{0:d}x\nexcited'.format(item)
              for item in range(4,ad.excarrs['lo'][csidx].max()+1)])
ax.set_xticklabels(xlabs)
# ax.set_xlim([-0.5,None])

ax.set(ylabel='E (eV)')
# print('stat.weight of ground state:', gi[csidx][0]*gj[csidx][0])
# print('stat.weight of first excited state:', gi[csidx][1]*gj[csidx][1])

# cond = ad.excarrs['lo'][csidx]==3
# for g,e in zip(gi[csidx][cond], (E-E[0])[cond]):
#     print('{0:10.0f} {1:10.0f}'.format(g,e))

# %% Plot: Excitation-resolved lines
# rhoidx = 1
rhoidx = 7

# Define satellite to zoom in on
sat = 7 # Isoelectronic
sidx = np.where(ad.sats==sat)[0][0] # Satellite index
zidx = np.where(np.isin(ad.Zkeys, '{0:0.0f}'.format(ad.Z-sat)))[0][0] # Corresponding ionization index

#### Single charge-state
plt.figure(figsize=[4,3])
[plt.semilogx(KT, hnu_exc[eidx,:,rhoidx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d} new'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]
plt.gca().set(xlabel='kT (eV)',
              ylabel='hnu (eV)',)
plt.legend()

#### All charge states
plt.figure(figsize=[4,3])
[plt.semilogx(KT, hnu_exc[eidx,:,rhoidx,:],
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
[axs[1].plot(KT, hnu_exc[eidx,:,rhoidx,zidx-eidx],
          color='C{0:d}'.format(eidx),
          label='Z*={0:s}, exc={1:d}'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]

# Line-complex resolved line centers
axs[1].plot(KT, hnu_sat[:,rhoidx,sidx], label='Averaged', color='k')

axs[1].set(xlabel='kT (eV)',
          ylabel='hnu (eV)')

plt.legend(bbox_to_anchor=(1.,1))

#### All satellite complexes, average only
# plt.figure(figsize=[4,3])
# plt.semilogx(KT, hnu_sat[:,rhoidx,:], label=ad.Zkeys)
# plt.gca().set(ylim=[5350,5800],
#               xlabel='kT (eV)',
#               ylabel='hnu (eV)')
# plt.legend()


# %% Plot: Satellite resolved line centers
nmax_idx = 0
fig, axs = plt.subplots(2, figsize=[4,4], sharex=True)
axs[0].plot(KT, ad.Zbar_rho[:,rhoidx], color='k')
axs[0].set(ylabel='Zbar',
           title=r'Z*={1:s} satellites, $\rho=${0:0.2f} g/cm$^3$'.format(rho_grid[rhoidx],
                                                                        ad.Zkeys[zidx]))
# Plot all, and zoom in appropriately
[axs[1].plot(KT, hnu_exc[eidx,:,rhoidx,:],
          color='C{0:d}'.format(eidx),
          )
      for eidx in exc_list]
axs[1].set(ylim= [np.nanmin(hnu_sat[:,rhoidx,sidx]),
                    # np.nanmax(hnu_sat[:,rhoidx,si])
                    hnu_exc[0,-1,-1,zidx] + 1
                    ])

# [axs[1].plot(KT, hnu_exc[eidx,:,rhoidx,zidx-eidx],
#           color='C{0:d}'.format(eidx),
#           label='Z*={0:s}, exc={1:d}'.format(ad.Zkeys[zidx-eidx], eidx))
#      for eidx in exc_list if (zidx-eidx)>=0]

#### Line-complex resolved line centers
axs[1].scatter(KT, hnu_sat[:,rhoidx,sidx], label='Averaged', color='k',
            facecolor='None')
axs[1].scatter(KT, hnu_sat[:,rhoidx,sidx], color='k',
            alpha=pgf[:,rhoidx,sidx]/np.nanmax(pgf[:,rhoidx,sidx]))

axs[1].set(xlabel='kT (eV)',
          ylabel='hnu (eV)',
          # xscale='log',
            xlim=[0,1100],
          )


axs[1].legend(bbox_to_anchor=(1.,0.6))

#### Each satellite
smin = 1
fig, axs = plt.subplots(len(ad.sats)-smin, figsize=[8,24], sharex=True)
for axi,sss in enumerate(ad.sats[smin:][-1::-1]):
    si = np.where(ad.sats==sss)[0][0]
    zi = np.where(np.isin(ad.Zkeys, '{0:0.0f}'.format(ad.Z-sss)))[0][0]
    cond = np.where(ad.satarrs==sss)
    # for i,e in zip(*cond):
    #     axs[axi].plot(KT, ad.hnuarrs[i,e], color='C{0:0.0f}'.format(ad.excarrs['lo'][i,e]))
    [axs[axi].plot(KT, hnu_exc[eidx,:,rhoidx,:],
              color='C{0:d}'.format(eidx),
              )
          for eidx in exc_list]
    
    # Line-complex resolved line centers
    axs[axi].scatter(KT, hnu_sat[:,rhoidx,si], label='Averaged', color='k',
                facecolor='None')
    axs[axi].scatter(KT, hnu_sat[:,rhoidx,si], color='k',
                alpha=pgf[:,rhoidx,si]/np.nanmax(pgf[:,rhoidx,si]))
    
    axs[axi].set(ylim= [np.nanmin(hnu_sat[:,rhoidx,si]),
                        # np.nanmax(hnu_sat[:,rhoidx,si])
                        hnu_exc[0,-1,-1,zi] + 1
                        ])
    
axs[-1].set(xlabel='kT (eV)',
          ylabel='hnu (eV)',
          # xscale='log',
            xlim=[0,1100],
          )

# %% Plot: hnu shift wrt parent transition
# Get Z indices corresponding to satellite ground-states
cond = [np.where(np.isin(ad.Zkeys,'{0:0.0f}'.format(ad.Z-z)))[0][0] for z in ad.sats]
# cond = [np.isin(ad.Zkeys,'{0:0.0f}'.format(ad.Z-z)) for z in ad.sats]
shift = np.array([hnu_sat[:,:,i] - hnu_exc[0,:,:,c] for i,c in enumerate(cond)])  # Satellite average minus Parent. 
shift = np.moveaxis(shift, [0,1,2],[2,0,1]) # Shape [satellite, T, rho] -> [T,rho, satellite]

#### Shift, satellite average - parent
labs = ['Iso={0:0.0f}'.format(s) for s in ad.sats]
plt.figure(figsize=(4,3))
plt.plot(KT, shift[:,rhoidx,:], label=labs)
plt.gca().set(xlabel='kT (eV)',
          ylabel='dhnu (eV)',
          # xscale='log',
          title='{0:0.1f} g/cm$^3$, nmax={1:0.0f}, Iso max={2:d}'.format(rho_grid[rhoidx],
                                                                         ad.nmax,
                                                                         ad.Z-int(ad.Zkeys[0])),
            xlim=[0,1100],
            ylim=[-60,None]
          )
plt.legend()


#### Absolute hnu of all excitations and satellite averages
plt.figure()
[plt.plot(KT, hnu_exc[eidx,:,rhoidx,:],
          color='C{0:d}'.format(eidx),
          )
      for eidx in exc_list]
# [plt.scatter(KT, hnu_sat[:,rhoidx,sidx], label='Averaged', color='k',
#             facecolor='None') for sidx in range(hnu_sat.shape[-1])]
[plt.plot(KT, hnu_sat[:,rhoidx,sidx], label='Averaged', color='k', ls='--')
     for sidx in range(hnu_sat.shape[-1])]
# .set(ylim= [np.nanmin(hnu_sat[:,rhoidx,sidx]),
#                     # np.nanmax(hnu_sat[:,rhoidx,si])
#                     hnu_exc[0,-1,-1,zidx] + 1
#                     ])

Tidxs = [np.where(KT<300)[0][-1],
         np.where(KT>400)[0][0]]

# print('Shift for T = {0:0.0f} to {1:0.0f} eV'.format(*[KT[i] for i in Tidxs]))
# [print('Iso {0:d} : {1:0.1f} eV'.format(ad.sats[i], 
#                                         hnu_sat[Tidxs[1],rhoidx,i] - hnu_sat[Tidxs[0],rhoidx,i]))
#          for i in range(len(ad.sats))];

print('Shift for rho = {0:0.1f} g/cc'.format(rho_grid[rhoidx]))
print('{0:5s} | {1:10s} | {2:10s} | {3:10s}'.format('Iso',
                                                    'T = {0:0.0f} eV'.format(KT[Tidxs[0]]),
                                                    'T = {0:0.0f} eV'.format(KT[Tidxs[1]]),
                                                    'Difference'))    

[print('{0:5d} | {1:10.1f} | {2:10.1f} | {3:10.1f}'.format(ad.sats[i], 
                                                           shift[Tidxs[0],rhoidx,i],
                                                           shift[Tidxs[1],rhoidx,i],
                                                           shift[Tidxs[1],rhoidx,i]-shift[Tidxs[0],rhoidx,i]))
         for i in range(len(ad.sats))];

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

# %% nmax: Satellite-integrated opacity
for adn in [ad3,ad4,ad5]:
    line_area = np.sqrt(2*np.pi*adn.lineshape_tot['G']**2)
    sat_op = adn.kappa_line*line_area
    
    op_int = []
    for sss in adn.sats:
        cond = adn.satarrs==sss
        
        op_int.append(sat_op[:,:,cond].sum(axis=-1))
        
    op_int = np.array(op_int)
    op_int = np.moveaxis(op_int, [0,1,2], [2,0,1])
    
    # Shift
    gf = adn.get_gf(1,0,2,1, return_gs=False)

    # Keep indexed to rho_grid throughout
    # Averaged within an excitation degree
    hnu_exc, pgf_exc = adn.get_hnu_average(adn.pstate_rho, gf=gf, resolve='ionization',
                                 return_weight=True) # Shape: [excitation, NT, Nrho, ionization]
    # Average within a satellite complex
    # breakpoint()
    hnu_sat, pgf = adn.get_hnu_average(adn.pstate_rho, gf=gf, resolve='line',
                                      return_weight=True) # Shape: [NT, Nrho, satellite]
    
    # Get Z indices corresponding to satellite ground-states
    cond = [np.where(np.isin(adn.Zkeys,'{0:0.0f}'.format(adn.Z-z)))[0][0] for z in adn.sats]
    shift = np.array([hnu_sat[:,:,i] - hnu_exc[0,:,:,c] for i,c in enumerate(cond)])  # Satellite average minus Parent. 
    shift = np.moveaxis(shift, [0,1,2],[2,0,1]) # Shape [satellite, T, rho] -> [T,rho, satellite]
    
    
    fig, axs = plt.subplots(3, figsize=[6,8], sharex=True)
    for z in range(14,23):
        zidx = np.where(np.isin(adn.Zkeys, str(z)))[0][0]
        axs[0].plot(KT,adn.psaha_rho[:,rhoidx,zidx],label=adn.Z-z)
    axs[0].set(ylabel='Ion fraction',
               title=adn.nmax)
    axs[0].legend()
    
    axs[1].plot(KT,op_int[:,rhoidx,:])
    axs[1].set(ylabel=r'Integrated opacity (cm$^2$ eV / g)')
    
    [axs[2].plot(KT, shift[:,rhoidx,sidx], color='C{0:d}'.format(c), ls='--',
                # alpha=pgf[:,rhoidx,sidx]/np.nanmax(pgf[:,rhoidx,sidx])
                )
         for c,sidx in enumerate(range(shift.shape[-1]))
         if any(pgf[:,rhoidx,sidx])] # Skip cases with all pgf==0
    
    [axs[2].scatter(KT, shift[:,rhoidx,sidx], color='C{0:d}'.format(c),
                alpha=pgf[:,rhoidx,sidx]/np.nanmax(pgf[:,rhoidx,sidx]))
         for c,sidx in enumerate(range(shift.shape[-1]))
         if any(pgf[:,rhoidx,sidx])] # Skip cases with all pgf==0
    axs[2].set(xlabel='T (eV)',
               ylabel=r'$\Delta$h$\nu$ (eV)',
               ylim=[-20,0])

# %% nmax: Single satellite complex
for he, hs, adn in zip(hnu_exc_n, hnu_sat_n, ad_list):
    
    fig, axs = plt.subplots(2, figsize=[5,4], sharex=True)
    
    axs[0].semilogx(KT, adn.Zbar_rho[:,rhoidx], color='k')
    axs[0].set(ylabel='Zbar',
               title=r'$\rho$={0:0.1e} g/cm$^3$'.format(rho_grid[rhoidx]))
    
    # Satellite resolved line centers
    [axs[1].plot(KT, he[eidx,:,rhoidx,zidx-eidx],
              color='C{0:d}'.format(eidx),
              label='Z*={0:s}, exc={1:d}'.format(adn.Zkeys[zidx-eidx], eidx))
         for eidx in exc_list if (zidx-eidx)>=0]
    
    # Line-complex resolved line centers
    axs[1].plot(KT, hs[:,rhoidx,sidx], label='Averaged', color='k')
    
    axs[1].set(xlabel='kT (eV)',
              ylabel='hnu (eV)')
    
    plt.legend(bbox_to_anchor=(1.,1))

