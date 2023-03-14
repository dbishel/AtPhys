#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 19:54:52 2023

@author: dbis

This script generates atomic data from the Screened Hydrogenic Model (v4).
Isolated atoms are constructed and evaluated for the desired upper state of all
charge states of a given atom. E.g. for 1-2, the complex will have a vacancy in the 
1s and various degrees of excitation.
"""

# Python modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
from matplotlib.lines import Line2D # For custom legend

import os                       # Used to e.g. change directory
import sys                      # Used to put breaks/exits for debugging
import re
import itertools as it

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

# Get shell energies and pops associated with all excited configurations of each charge state
En = {'{0:d}'.format(item): {} for item in range(ZZ)} # Dict of dict: En[Zbar][excitation degree]
Pn = {'{0:d}'.format(item): {} for item in range(ZZ)} # Dict of dict: En[Zbar][excitation degree]
for Zbar in range(Zbar_min, ZZ): # Start with neutral, end with H-like

    Nele = ZZ - Zbar
    # Exlcude unphysical excitations, e.g. Li can only be 0- and 1-excited
    valid_exc_list = [item for item in exc_list if item<(Nele-1)]
    for exc in valid_exc_list:
        Enx = [] # list of lists for current charge state and excitation degree
        Pnx = [] # List of lists for current shell populations
        fn = 'complexes/fac_{0:d}_{1:d}_{2:d}_lo.txt'.format(Nele,nmax,
                                                                           exc)
        with open(fn, 'r') as file:
            l = True
            while l:
                # Read each line. Files have one complex per line
                l = file.readline()
                
                # Parse shell and population. First number is always shell, second population
                # p = re.compile('[0-9]+\*[0-9]+')
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
                Enx.append(sh.En)
                Pnx.append(Pni)
                
                if vb: # Print if Verbose
                    print('\n----------')
                    print('----------\n')
                    print('Zbar: ', Zbar)
                    print('Nele: ', Nele)
                    print('Exc: ', exc)
                    print('FAC: ', l)
                    print('Parse: ', m)
                    print('Pops: ', Pn)
                    print(sh.En)
        En['{0:d}'.format(Zbar)]['{0:d}'.format(exc)] = Enx
        Pn['{0:d}'.format(Zbar)]['{0:d}'.format(exc)] = Pnx
        

# %% hnu Plots

Zs = [Zkey for Zkey in list(En.keys()) if list(En[Zkey].keys())] # Keep only calculated charge states
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
tidx = [1,0] # Initial, final indices of transition to calculate

# Set color equal to excitation degree
exc_minmax = [0,3]

cmap_name = 'rainbow'
cmap = mpl.cm.get_cmap(cmap_name)
norm = mpl.colors.Normalize(vmin=exc_minmax[0], vmax=exc_minmax[1])

fig, ax = plt.subplots(figsize=[4,3])
for i, Zbar in enumerate(Zbar_plot):
    valid_exc = list(En['{0:d}'.format(Zbar)].keys())
    for exc in valid_exc:
        color = norm(int(exc)) # Get rgba from colorbar
        Enx = En[str(Zbar)][str(exc)]
        dE = [abs(item[tidx[1]] - item[tidx[0]]) for item in Enx] # Loop over each configuration
        plt.plot([Zbar]*len(dE), dE, '.',
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
    valid_exc = list(En['{0:d}'.format(Zbar)].keys())
    for exc in valid_exc:
        color = norm(int(exc)) # Get rgba from colorbar
        Enx = En[str(Zbar)][str(exc)]
        dE = [abs(item[tidx[1]] - item[tidx[0]]) for item in Enx] # Loop over each configuration
        plt.plot([Zbar+int(exc) - 0.1*int(exc)]*len(dE), dE, '.',
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

# %% Saha-Boltzmann on complex energies - not yet implemented

# Generate KT, NE vectors
Nn, NT = 10, 11 # Number of density, temperature gridpoints

KT = np.logspace(1.5,3, num=NT) # eV, Temperature range, 

rho0 = np.logspace(0.5,2, num=Nn) # g/cc
Zbar0 = 20 # Estimated dZbar
NE = rho0 / (A*mp) * Zbar0 # 1/cm^3, Ne range

# Parse data for Saha-Boltzmann
Earrs = []
excarrs = []
glists = []
g = 2*np.arange(1,nmax+1)**2 # Statistical weights of each ScHyd model. See Cowan
Zs = [Zkey for Zkey in list(En.keys()) if list(En[Zkey].keys())] # Keep only calculated charge states

for Zkey in Zs:
    # Save off energy levels, excitation degree, and stat.weight of each complex
    tmpEn = []
    tmpexc = []
    tmpg = []
    for exc in list(En[Zkey].keys()):
        N = len(En[Zkey][exc])
        [tmpEn.extend(item) for item in En[Zkey][exc]]
        [tmpexc.extend(exc) for item in range(N)]
        
        # Construct stat weight for each
        [tmpg.extend(g) for item in range(N)] # NOTE: INCORRECT. SEE COWAN
    Earrs.append(np.array(tmpEn))
    excarrs.append(np.array(tmpexc))
    glists.append(tmpg)

# Get ionization potentials
Iplist = [Ip[int(item)] for item in Zs]

Zbar = np.zeros(shape=[NT,Nn])
for idx, items in enumerate(it.product(KT,NE)):
    kT, ne = items
    i, j = np.unravel_index(idx, shape=(NT,Nn))
    
    # Run Saha, with ne converted to m^-3. 
    out = saha(ne, kT, Earrs, glists, Iplist, returns='csd') # Returns: p     
    
    tmp = np.sum(np.arange(int(Zs[0]), int(Zs[-1])+2,) * out)
    Zbar[i,j] = tmp
    
    # Run Boltzmann on each charge state
    p = []
    for Z,Earr,garr in zip(Zs, Earrs, glists):
        p.append(boltzmann(Earr, garr, kT))

rho = NE/Zbar * A * mp # g/cm^3. Ne in 1/cm^3, mp in g

plt.figure(figsize=[5,3])
plt.pcolormesh(np.log10(rho), np.log10(KT), Zbar, shading='nearest',
               vmin=int(Zs[0]), vmax=float(Zs[-1]))
plt.colorbar()

plt.gca().set(xlabel='log10(rho (g/cm^3))',
              ylabel='log10(T (eV))',
              title=['Z={0:d}'.format(ZZ),
                      ' exc=',exc_list,
                      'Zbar = {0:s} to {1:s}'.format(Zs[0], Zs[-1])])

# # %% Saha-Boltzmann on shell energies – I think this is wrong...

# # Generate KT, NE vectors
# Nn, NT = 10, 11 # Number of density, temperature gridpoints

# KT = np.logspace(1.5,3, num=NT) # eV, Temperature range, 

# rho0 = np.logspace(0.5,2, num=Nn) # g/cc
# Zbar0 = 20 # Estimated dZbar
# NE = rho0 / (A*mp) * Zbar0 # 1/cm^3, Ne range

# # Parse data for Saha-Boltzmann
# Earrs = []
# excarrs = []
# glists = []
# g = 2*np.arange(1,nmax+1)**2 # Statistical weights of each ScHyd model. See Cowan
# Zs = [Zkey for Zkey in list(En.keys()) if list(En[Zkey].keys())] # Keep only calculated charge states

# for Zkey in Zs:
#     # Save off energy levels, excitation degree, and stat.weight of each set of energy levels
#     tmpEn = []
#     tmpexc = []
#     tmpg = []
#     for exc in list(En[Zkey].keys()):
#         N = len(En[Zkey][exc])
#         [tmpEn.extend(item) for item in En[Zkey][exc]]
#         [tmpexc.extend(exc) for item in range(N)]
        
#         # Construct stat weight for each
#         [tmpg.extend(g) for item in range(N)] # NOTE: INCORRECT. SEE COWAN
#     Earrs.append(np.array(tmpEn))
#     excarrs.append(np.array(tmpexc))
#     glists.append(tmpg)

# # Get ionization potentials
# Iplist = [Ip[int(item)] for item in Zs]

# Zbar = np.zeros(shape=[NT,Nn])
# for idx, items in enumerate(it.product(KT,NE)):
#     kT, ne = items
#     i, j = np.unravel_index(idx, shape=(NT,Nn))
    
#     # Run Saha, with ne converted to m^-3. 
#     out = saha(ne, kT, Earrs, glists, Iplist, returns='csd') # Returns: p     
    
#     tmp = np.sum(np.arange(int(Zs[0]), int(Zs[-1])+2,) * out)
#     Zbar[i,j] = tmp
    
#     # Run Boltzmann on each charge state
#     p = []
#     for Z,Earr,garr in zip(Zs, Earrs, glists):
#         p.append(boltzmann(Earr, garr, kT))

# rho = NE/Zbar * A * mp # g/cm^3. Ne in 1/cm^3, mp in g

# plt.figure(figsize=[5,3])
# plt.pcolormesh(np.log10(rho), np.log10(KT), Zbar, shading='nearest',
#                vmin=int(Zs[0]), vmax=float(Zs[-1]))
# plt.colorbar()

# plt.gca().set(xlabel='log10(rho (g/cm^3))',
#               ylabel='log10(T (eV))',
#               title=['Z={0:d}'.format(ZZ),
#                       ' exc=',exc_list,
#                       'Zbar = {0:s} to {1:s}'.format(Zs[0], Zs[-1])])

