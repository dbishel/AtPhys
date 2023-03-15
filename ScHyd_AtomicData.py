#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 19:54:52 2023

@author: dbis

This script generates atomic data from the Screened Hydrogenic Model (v4).
Isolated atoms are constructed and evaluated for the desired upper state of all
charge states of a given atom. E.g. for 1-2, the complex will have a vacancy in the 
1s and various degrees of excitation.

Should be converted to a method to be more flexible / callable
"""

# Python modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
from matplotlib.lines import Line2D # For custom legend
from matplotlib.widgets import Button # For widgets

import os                       # Used to e.g. change directory
import sys                      # Used to put breaks/exits for debugging
import re
import itertools as it
from copy import deepcopy

from ScHyd import get_ionization, AvIon
from saha_boltzmann_populations import saha, boltzmann

# %% Atomic Data
vb = 0 # Verbose flag
ZZ = 26 # Nuclear charge
A = 51.996 # Nucleon number
Zbar_min = 16
nmax = 5 # Maximum allowed shell
exc_list = [0,1,2,3] # Excitation degrees to consider (lower state is ground state, singly excited, ...)

# Constants
mp = 1.67262192e-30 # g, proton mass

# Ionization potential
Ip, En_0 = get_ionization(ZZ, return_energy_levels=True)

# Get quantities associated with all excited configurations of each charge state
# Dict of dict of dict: En['up' or'lo'][Zbar][excitation degree]
dict_base = {'{0:d}'.format(item): {} for item in range(ZZ)}
En = {item:deepcopy(dict_base) for item in ['up','lo']}  # Shell energies
Pn = {item:deepcopy(dict_base) for item in ['up','lo']} # Shell populations
gn = {item:deepcopy(dict_base) for item in ['up','lo']}  # Shell statistical weights
Etot = {item:deepcopy(dict_base) for item in ['up','lo']}  # Total ion energy – used for transition energies

# Dict of dict
hnu = deepcopy(dict_base) # Transition energy 

for uplo in['up','lo']:
    for Zbar in range(Zbar_min, ZZ): # Start with neutral, end with H-like
        Zbar_str = '{0:d}'.format(Zbar)

        Nele = ZZ - Zbar
        # Exlcude unphysical excitations, e.g. Li can only be 0- and 1-excited
        valid_exc_list = [item for item in exc_list if item<(Nele-1)]
        for exc in valid_exc_list:
            # Save quantities as list of lists for current charge state and excitation degree
            Enx = [] # Shell energies
            Pnx = [] # Shell populations
            gnx = [] # Shell statistical weights
            Etotx = [] # Total ion energy
            fn = 'complexes/fac_{0:d}_{1:d}_{2:d}_{3:s}.txt'.format(Nele, nmax,
                                                                    exc, uplo)
            with open(fn, 'r') as file:
                l = True
                while l:
                    # Read each line. Files have one complex per line
                    l = file.readline()
                    
                    # Parse shell and population. First number is always shell, second population
                    p = re.compile('([0-9]+)\*([0-9]+)')
                    m = re.findall(p,l)
                    
                    # Skip remainder if end-of-file (m is empty)
                    if not m:
                        continue                
                    
                    # Initiate populations as all 0's
                    Pni = np.zeros(nmax) # populations of current complex
                    for shell, pop in np.array(m).astype(int): # Read one shell of the current complex at a time
                        
                        Pni[shell-1] = pop
                    
                    # Get energy levels from Average Ion
                    sh = AvIon(ZZ, Zbar=(ZZ-Nele), nmax=nmax)      
    
                    sh.Pn = Pni # Pass Pn manually
                    sh.get_Qn()
                    
                    sh.get_Wn()
                    sh.get_En()
                    sh.get_statweight()
                    sh.get_Etot()
                    
                    Enx.append(sh.En)
                    Pnx.append(Pni)
                    gnx.append(sh.statweight)
                    Etotx.append(sh.Etot)
                    
                    if vb: # Print if Verbose
                        print('\n----------')
                        print('----------\n')
                        print('Zbar: ', Zbar)
                        print('Nele: ', Nele)
                        print('Exc: ', exc)
                        print('FAC: ', l)
                        print('Parse: ', m)
                        print('Pops: ', Pni)
                        print('Stat.weight: ', sh.statweight)
    
                        print(sh.En)
            En[uplo][Zbar_str]['{0:d}'.format(exc)] = Enx
            Pn[uplo][Zbar_str]['{0:d}'.format(exc)] = Pnx
            gn[uplo][Zbar_str]['{0:d}'.format(exc)] = gnx
            Etot[uplo][Zbar_str]['{0:d}'.format(exc)] = Etotx
      
Zs = [Zkey for Zkey in list(En['up'].keys()) if list(En['up'][Zkey].keys())] # Keep only calculated charge states

# Check all entries are same length -- biggg assumption is that up/lo files are identical
# Convert to bool which raises a warning if populations don't differ correctly
if vb:
    for Z in Zs:
        for exc in list(En['up'][Z].keys()):
            # print('Lengths: ', len(En['lo'][Z][exc]), len(En['up'][Z][exc]))
            print('Pn_up - Pn_lo: ', np.array(Pn['up'][Z][exc]) - np.array(Pn['lo'][Z][exc]))
        
# %% hnu Plots

Zbar_plot = np.array(Zs).astype(int)
# Zbar_plot = [20,21,22,23,24]

# #### Plot energy levels by excitation degree for each charge state
# fig, axs = plt.subplots(2,2)
# axs = axs.flatten()

# for i, Zbar in enumerate(Zbar_plot):
#     valid_exc = list(En['{0:d}'.format(Zbar)].keys())
#     for exc in valid_exc:
#         tmp = En[str(Zbar)][str(exc)]
#         axs[i].plot(tmp, int(exc)*np.ones(len(tmp)), '.',
#                     color='C{0:d}'.format(int(exc)),
#                     alpha=0.3,
#                     label=exc)
# labels = np.arange(3).astype(str)
# custom_lines = [Line2D([0], [0], linestyle='',
#                        marker='.', markersize=3,
#                        color='C{0:s}'.format(i), lw=1,) for i in labels]
# plt.legend(custom_lines, ['Exc: {0:s}'.format(item) for item in labels])
# axs[2].set(xlabel='E (eV)',
#               ylabel='Excitation degree')

# #### Plot 1-2 transition energy by excitation degree for each charge state
# fig, axs = plt.subplots(2,2)
# axs = axs.flatten()

# tidx = [1,0] # Initial, final indices of transition to calculate

# for i, Zbar in enumerate(Zbar_plot):
#     valid_exc = list(En['{0:d}'.format(Zbar)].keys())
#     for exc in valid_exc:
#         Enx = En[str(Zbar)][str(exc)]
#         dE = [abs(item[tidx[1]] - item[tidx[0]]) for item in Enx] # Loop over each configuration
#         axs[i].plot(dE, [int(exc)]*len(dE), '.',
#                     color='C{0:d}'.format(int(exc)),
#                     alpha=0.3,
#                     label=exc)
# labels = np.arange(3).astype(str)
# custom_lines = [Line2D([0], [0], linestyle='',
#                        marker='.', markersize=3,
#                        color='C{0:s}'.format(i), lw=1,) for i in labels]
# plt.legend(custom_lines, ['Exc: {0:s}'.format(item) for item in labels])
# axs[2].set(xlabel='hnu (eV)',
#               ylabel='Excitation degree')

#### Plot hnu vs. Zbar

# Set color equal to excitation degree
exc_minmax = [min(exc_list), max(exc_list)]

cmap_name = 'rainbow'
cmap = mpl.cm.get_cmap(cmap_name)
norm = mpl.colors.Normalize(vmin=exc_minmax[0], vmax=exc_minmax[1])

fig, ax = plt.subplots(figsize=[4,3])
for i, Zbar in enumerate(Zbar_plot):
    Zbar_str = '{0:d}'.format(Zbar)
    valid_exc = list(En['up'][Zbar_str].keys())
    for exc in valid_exc:
        color = norm(int(exc)) # Get rgba from colorbar
        # Enx = En[str(Zbar)][str(exc)]
        hnu[Zbar_str][exc] = [abs(up - lo) for up,lo in zip(Etot['up'][Zbar_str][exc],
                                            Etot['lo'][Zbar_str][exc])] # Loop over each configuration
        plt.plot([Zbar]*len(hnu[Zbar_str][exc]), hnu[Zbar_str][exc], '.',
                    color=cmap(color),
                    alpha=1,
                    label=exc)
        
# Colorbar
cax = fig.add_axes([0.2, 0.85, 0.5, 0.05])
cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm = norm,
                                orientation='horizontal',)
cb.set_label('Excitation degree',)
plt.tight_layout()
ax.set(xlabel='Zbar',
       ylabel='hnu (eV)')

#### Plot hnu vs. Parent Zbar = Zbar+exc
tidx = [1,0] # Initial, final indices of transition to calculate

# Set color equal to excitation degree
cmap_name = 'rainbow'
cmap = mpl.cm.get_cmap(cmap_name)
norm = mpl.colors.Normalize(vmin=exc_minmax[0], vmax=exc_minmax[1])

fig, ax = plt.subplots(figsize=[4,3])
for i, Zbar in enumerate(Zbar_plot):
    Zbar_str = '{0:d}'.format(Zbar)

    valid_exc = list(En['up'][Zbar_str].keys())
    for exc in valid_exc:
        color = norm(int(exc)) # Get rgba from colorbar
        # Enx = En[str(Zbar)][str(exc)]
        dE = [abs(up - lo) for up,lo in zip(Etot['up'][Zbar_str][exc],
                                            Etot['lo'][Zbar_str][exc])] # Loop over each configuration
        plt.plot([Zbar+int(exc) - 0.1*int(exc)]*len(hnu[Zbar_str][exc]),
                  hnu[Zbar_str][exc], '.',
                  color=cmap(color),
                  alpha=1,
                  label=exc)
        
# Colorbar
cax = fig.add_axes([0.2, 0.85, 0.5, 0.05])
cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm = norm,
                                orientation='horizontal',)
cb.set_label('Excitation degree',)
plt.tight_layout()
ax.set(xlabel='Zbar + Excitation degree',
       ylabel='hnu (eV)')

# %% Saha-Boltzmann 
# on lower complexes energies

#### Grid and parse
# Generate KT, NE vectors
Nn, NT = 10, 11 # Number of density, temperature gridpoints

KT = np.logspace(1.5,3, num=NT) # eV, Temperature range, 

rho0 = np.logspace(0.5,2, num=Nn) # g/cc
Zbar0 = 20 # Estimated dZbar
NE = rho0 / (A*mp) * Zbar0 # 1/cm^3, Ne range

# Parse data for Saha-Boltzmann
Earrs = [] # Array of total energy of each complex, sorted by charge state
excarrs = [] # Excitation degree of each complex, sorted by charge state
glists = [] # Total statistical weight of each complex, sorted by charge state
hnuarrs = [] # Transition energy of each 

Zs = [Zkey for Zkey in list(Etot['lo'].keys()) if list(Etot['up'][Zkey].keys())] # Keep only calculated charge states
for Z in Zs:
    # Save off energy levels, excitation degree, and stat.weight of each complex,
    # grouped by ionization state
    tmpEtot = []
    tmpexc = []
    tmpgn = []
    tmphnu = []
    for exc in list(Etot['lo'][Z].keys()):
        N = len(Etot['lo'][Z][exc])
        tmpEtot.extend(Etot['lo'][Z][exc])
        [tmpexc.append(int(exc)) for item in range(N)]
        [tmpgn.append(np.prod(item)) for item in gn['lo'][Z][exc]]
        tmphnu.extend(hnu[Z][exc])
        
    Earrs.append(np.array(tmpEtot))
    excarrs.append(np.array(tmpexc))
    glists.append(tmpgn)
    hnuarrs.append(tmphnu)

# To enable slicing, 0-pad all state-resolved arrays
# 0 values in glists reslut in 0 contribution to partition functions, 0 effect on Saha-Boltzmann
# Pad no zeroes at beginning, out to max_length at end: pad_width = [(0, max_length-len(item))]
max_length = max([len(item) for item in excarrs]) # Longest length array
Earrs   = np.array([np.pad(item, [(0, max_length-len(item))], constant_values=int(0)) for item in Earrs]) 
excarrs = np.array([np.pad(item, [(0, max_length-len(item))], constant_values=int(0)) for item in excarrs])
glists  = np.array([np.pad(item, [(0, max_length-len(item))], constant_values=int(0)) for item in glists])
hnuarrs = np.array([np.pad(item, [(0, max_length-len(item))], constant_values=int(0)) for item in hnuarrs])

# Get ionization potentials
Iplist = [Ip[int(item)] for item in Zs]

Zbar = np.zeros(shape=[NT,Nn]) # Mean ionization state
psaha = np.zeros(shape=[NT,Nn,len(Zs)+1]) # Saha charge state populations. Shape T, Ne, Z+1
pboltz = np.zeros(shape=[NT,Nn,len(Zs), max_length]) # Boltzmann state populations. Shape T, Ne, Z, state
for idx, items in enumerate(it.product(KT,NE)):
    kT, ne = items
    i, j = np.unravel_index(idx, shape=(NT,Nn))
    
    #### Saha
    # Run Saha, with ne converted to m^-3. 
    out = saha(ne, kT, Earrs, glists, Iplist, returns='csd') # Returns: p     
    psaha[i,j] = out # Normalization: np.sum(psaha, axis=-1) should = 1 everywhere
    
    # Calculate Zbar
    Zbar[i,j] = np.sum(np.arange(int(Zs[0]), int(Zs[-1])+2,) * out)
    
    #### Boltzmann – Ne grid
    # Run Boltzmann on each charge state
    pb = []
    for Z,Earr,garr in zip(Zs, Earrs, glists):
        pb.append(boltzmann(Earr, garr, kT, normalize=True))
    
    pboltz[i,j] = np.array(pb)
    
rho = NE/Zbar * A * mp # g/cm^3. Ne in 1/cm^3, mp in g

#### Regrid in rho
# Construct regular grids in mass density
Nrho = 12
rho_grid = np.logspace(0.5,2, num=Nrho)
Zbar_rho = [] # Zbar regularly gridded against rho
psaha_rho = np.zeros(shape=[NT,Nrho,len(Zs)+1]) # Normalization: np.sum(psaha, axis=-1) should = 1 everywhere
for i,t in enumerate(KT):
    Zbar_rho.append(np.interp(rho_grid, rho[i], Zbar[i]))
    for k in range(psaha.shape[-1]):
        psaha_rho[i,:,k] = np.interp(rho_grid, rho[i], psaha[i,:,k]) 
Zbar_rho = np.array(Zbar_rho)

#### Boltzmann – rho grid
# Run Boltzmann on each charge state over rho grid
pboltz_rho = np.zeros(shape=[NT,Nrho,len(Zs), max_length]) # Shape T, Ne, Z, state
for idx, items in enumerate(it.product(KT,rho_grid)):
    kT, __ = items
    i, j = np.unravel_index(idx, shape=(NT,Nrho))
    p = []
    for Z,Earr,garr in zip(Zs, Earrs, glists):
        p.append(boltzmann(Earr, garr, kT, normalize=True))

    pboltz_rho[i,j] = np.array(p)

# Saha-Boltzmann pop of each state
pstate_rho = pboltz_rho * psaha_rho[Ellipsis,:-1, np.newaxis]  # Shape: T, rho, Z, state
    
#### Plots
# Zbar heatmap
fig, axs = plt.subplots(2, figsize=[4,5], sharex=True, sharey=True)
im = axs[0].pcolormesh(np.log10(rho), np.log10(KT), Zbar, shading='nearest',
               vmin=int(Zs[0]), vmax=float(Zs[-1]))
plt.colorbar(im, ax=axs[0])

im = axs[1].pcolormesh(np.log10(rho_grid), np.log10(KT), Zbar_rho, shading='nearest',
               vmin=int(Zs[0]), vmax=float(Zs[-1]))
plt.colorbar(im, ax=axs[1])

fig.suptitle(['Z={0:d}'.format(ZZ),
        ' exc=',exc_list,
        'Zbar = {0:s} to {1:s}'.format(Zs[0], Zs[-1])])
axs[0].set(xlabel='log10(rho (g/cm^3))',
              ylabel='log10(T (eV))',
              title='Regular Ne grid')
axs[1].set(xlabel='log10(rho (g/cm^3))',
              ylabel='log10(T (eV))',
              title='Interpolated onto rho grid')

# hnu vs. Zbar, with color equal to Boltzmann population
# Set color equal to population fraction
cmap_name = 'rainbow'
cmap = mpl.cm.get_cmap(cmap_name)
norm = mpl.colors.Normalize(vmin=0, vmax=1)

# %% S-B plots
#### Visualize populated transitions – widget
class updateIndex():
    '''Widget for changing 2-D indices governing face-color of scatter-plot points
    '''
    
    def __init__(self, scat, colors, Trho, ax=None):
        ''' Widget for changing T,rho indices governing face-color
            of given scatterplot objects (matplotlib.collections.PathCollection)
            
            Use mpl.cm.get_cmap and mpl.colors.Normalize to define colors

        Parameters
        ----------
        scat : list
            Scatter plot objects, plotted on ax. Length (Ns)
        colors : array
            Shape [Ni, Nj, Ns, Ndata, 4] array of the face-colors. \n
            - Ni, Nj are the number of T, rho gridpoints.
            - Ns is the number of scatter plot objects
            - Ndata is the (maximm) number of datapoints in each scatterplot
            - 4 corresonds to the elements of an RGBA color
        Trho : list
            Length-2 list of 1-D arrays of T and rho axes.
        ax : axes object, optional
            Axes to update title for displaying current T,rho.
            The default is None, where current axes is used.

        Returns
        -------
        None.

        '''
        self.i = 0
        self.j = 0
        self.scat = scat # List of scatterplot objects
        self.colors = colors
        self.labels = Trho
        
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
    
    def inc_i(self, event):
        '''Increments i-index and replots'''
        self.i += 1
        self.i = min(self.i, len(self.labels[0])-1)
        
        self.plot()

    def dec_i(self, event):
        '''Decrements i-index and replots'''
        self.i -= 1
        self.i = max(self.i, 0)
        
        self.plot()

    def inc_j(self, event):
        '''Increments j-index and replots'''
        self.j += 1
        self.j = min(self.j, len(self.labels[1])-1)
        
        self.plot()

    def dec_j(self, event):
        '''Decrements j-index and replots'''
        self.j -= 1
        self.j = max(self.j, 0)
        
        self.plot()
    
    def plot(self,):
        ''' Sets scatter plot colors to current reflect current indices
        '''
        # Calculate and set colors
        colors = self.colors[self.i, self.j]
        [s.set_color(c) for s,c in zip(self.scat, colors)] 
        self.update_title()
        
        plt.draw()
    
    def update_title(self):
        self.ax.set(title='Te = {0:0.0f} eV, rho = {1:0.1f} g/cc'.format(self.labels[0][self.i],
                                                                 self.labels[1][self.j]))


Tidx, rhoidx=[9,5]
fig, ax = plt.subplots(figsize=[10,6])
scat = [] # Keep scatter plots for later

cmap_name = 'rainbow'
cmap = mpl.cm.get_cmap(cmap_name)

scale = 'log'
if scale=='log':
    # Set min/max of colorbar
    norm = mpl.colors.Normalize(vmin=-6, vmax=0) # Converts value to linearly interpolte [0,1] between bounds
    
    # Calculate colorbars of populations
    colors = cmap(norm(np.log10(pstate_rho)))
    
elif scale=='lin':
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    colors = cmap(norm(pstate_rho))

    

for i in range(len(Zbar_plot)):
    # Parse values for current charge state
    Z = Zbar_plot[i]
    exc = excarrs[i]
    hnu_i = hnuarrs[i]
    p = pstate_rho[Tidx, rhoidx, i]
    
    cond = hnu_i>0
    scat.append(plt.scatter((Z+exc - 0.1*exc)[cond],
                        hnu_i[cond],
                        c=colors[i,Tidx,rhoidx, cond, :],
                        # norm=norm,
                        alpha=1,
                        s=10, # Marker size
                        label=exc))

# Colorbar
cax = fig.add_axes([0.2, 0.85, 0.5, 0.05])
cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm,
                                orientation='horizontal',)
cb.set_label('lgog10(State population)',)
plt.tight_layout()
ax.set(xlabel='Zbar + Excitation degree',
       ylabel='hnu (eV)')

# Buttons to change populations 
# colors = cmap(pstate_rho) # Shape T, rho, Z, state, RBGA
# colors = cmap(np.log10(pstate_rho)) # Shape T, rho, Z, state, RBGA
callback = updateIndex(scat, colors, Trho=[KT,rho_grid])

# Define axes
axmi = fig.add_axes([0.7, 0.05, 0.1, 0.075]) # axmi = "Axes minus i" – i-decrementing axes
axpi = fig.add_axes([0.81, 0.05, 0.1, 0.075]) # axpi = "Axes plus i" – i-incrementing axes
bpi = Button(axpi, 'T+') # bpi - i-inrecementing button
bpi.on_clicked(callback.inc_i)
bmi = Button(axmi, 'T-')
bmi.on_clicked(callback.dec_i)

axmj = fig.add_axes([0.15, 0.05, 0.1, 0.075]) # Axes minus i
axpj = fig.add_axes([0.26, 0.05, 0.1, 0.075])
bpj = Button(axpj, 'rho+')
bpj.on_clicked(callback.inc_j)
bmj = Button(axmj, 'rho-')
bmj.on_clicked(callback.dec_j)

# Histogram population-weighted transition energy - ignore 0-eV trans
bins = np.linspace(6400, 6800) # Full spectrum
bins = np.linspace(6515, 6535, num=6) # N-like complex

pop_hist = []
for ii in range(len(KT)):
    pop_hist.append(np.histogram(a=hnuarrs.flatten(),
                            weights=pstate_rho[ii,rhoidx].flatten(),
                            bins=bins)[0])
                    
plt.figure()
plt.plot(KT, pop_hist, label=np.arange(len(bins)-1))
plt.legend()

plt.gca().set(yscale='log')


