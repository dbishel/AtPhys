#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:33:51 2023

@author: dbis
"""

# Python modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # For custom legend
import scipy as sp

import os                       # Used to e.g. change directory
import sys                      # Used to put breaks/exits for debugging

import itertools as it
import pandas as pd

def saha(ne, kT, Earrs, glists, Iplist, IPD=0, returns='csd'):
    """ Calculates Saha distribution of the given ion for a single ne, kT.
        Energy levels from a subset of charge states can be given
        Bare ion G (internal partition function) =1 is always appended.
    
    Parameters
    ----------
    ne : float
        Electron density (m^-3)
    kT : float
        Temperature (eV)
    Earrs : list of arrays
        Energy levels (eV) of each charge state. First entry is energy levels of most neutral ion,
        last entry is for most ionized.
    glists : list of arrays
        Multiplicities of each level. First entry is multiplicities of most neutral,
        last entry is for most ionize
    Iplist : list
        Ionization potentials (eV) of each charge state. First is of most neutral,
        last is of most ionized.
    include_bare : bool
        Option to append bare ion
    IPD : float
        Ionization potential depression, in eV. \
        Effectively reduces ionization potential in Saha ratio.
    returns : str ('csd', 'Zbar', 'abs', 'all')
        Choice of what to return: \n
        - 'csd' : fractional populations n_j/n_tot
        - 'zbar' : mean ionization state. Requires neutral atom through bare ion are included.
        - 'abs' : absolute number density n_j of each charge state. Requires neutral atom through bare ion are included.
        - 'all' : scd, zbar, and abs as a list. Requires neutral atom through bare ion are included.
        
    Returns
    -------
    p : list
        Fractional population n_j/n_tot of each charge state
    zbar : float
        Mean ionization state
    n : list
        Number density of each charge state
        
    Note
    ----
        - Zbar = np.sum(np.arange(len(f)) * f) \n
        - ntot = ne/Zbar
    """
    hbar = 1.0545606529268985e-34 # J*s
    me = 9.109e-31 # electron mass, kg
    el = 1.60217662e-19 # J per eV
    
    lam_th = np.sqrt(2*np.pi*hbar**2 / me / (kT * el)) # m. All variables in SI. kT -> J
    
    # Caulcate internal partition functions for each ion species
    G = [np.sum(gs * np.exp((-(Es - Es[0])/kT))) for Es, gs in zip(Earrs, glists)]
    
    # Append G of bare ion = 1??
    G.append(1)
    
    # Calculate Saha ratios r_j = n_j/n_j-1
    # r = [2/(ne*lam_th**3) * G[i]/G[i-1] * np.exp(-(Iplist[i-1] - IPD)/kT)
    r = [2/(ne*lam_th**3) * G[i]/G[i-1] * np.exp(-np.max(Iplist[i-1] - IPD,0)/kT)
         for i in range(1,len(Earrs)+1)] # Start from i=0 for n_1/n_0
    
    # Calculate cumulative product of r_j = n_j/n_0 = c_j
    c = np.cumprod(r)
    
    # Calculate fractional abundance of ground state f0 = n_0/n_tot = (1+ sum(c_j))^(-1)
    p = [(1 + np.sum(c))**(-1)] # One element – ground state only so far
    
    # Calculate fractional abundance of other states f_j = f_0 c_j
    p.extend([p[0] * item for item in c])
    
    p = np.array(p)
    
    # Return block
    if returns.lower()=='csd':
        return p
    
    Zbar = np.sum(np.arange(len(p)) * p) # Mean ionization state
    
    if returns.lower()=='zbar':
        return Zbar
    
    ntot = ne / Zbar # Total ion density
    n = p*ntot # Ion density of each charge state
    
    if returns.lower()=='abs':
        return n
    elif returns.lower()=='all':
        return [p, Zbar, n]

def boltzmann(Earr, garr, kT, normalize=True):
    """ Calculate Boltzmann populations (or factor) for states of given energy.
    
        Multiplicities are included in partition function (denominator) but not populations
        because oscillator strengths are calculated for individual quantum states
        and afterwards weighted by the multiplicty.
        
        Parameters
        ----------
        
        Earr : array
            Energy levels (eV) of given charge state. First entry is ground state.
        garr : array
            Multiplicities of each level. First entry is of ground state
        kT : float
            Temperature (eV)
        normalize : bool
            If False, return Boltzmann factors p
            If True, return fractional populations p / sum(p)

    """
    
    dE = Earr - Earr[0] # Energy difference with respect to ground state, including of the ground states
    p = np.exp(-dE/kT) / np.sum(garr * np.exp(-dE/kT)) # Ground state included in sum, i.e. sum starts at s=0, not at s=1 as in notes
    
    if normalize:
        return p / np.sum(p) # Normalized Boltzmann
    else:
        return p

# %% Main
if __name__=='__main__':
    # %% Hydrogen Zbar
    if 1:
        #### Hand-written H data
        nmax = 1
        Ryd = 13.6 # eV
        shells = np.arange(1,nmax+1) # pqn
        
        Earrs = [-Ryd / shells**2] # List of arrays. Hydrogenic shell energy
        glists = [2*shells**2] # List of arrays. Multiplicity of hydrogen shells. No angular momentum splitting
        Iplist = [13.6] # Ionization of hydrogen
            
        Nn = 100
        NT = 110
        
        NE = np.logspace(20,32, num=Nn) # electrons/m^3
        KT = np.logspace(-1.5,3, num=NT) # eV
        
        # f0 = np.zeros(shape=(Nn,NT))
        Zbar = np.zeros(shape=(NT,Nn))
        # n0 = np.zeros(shape=(Nn,NT))
        
        for idx, items in enumerate(it.product(KT,NE)):
            kT, ne = items
            i, j = np.unravel_index(idx, shape=(NT,Nn))
            out = saha(ne, kT, Earrs, glists, Iplist, returns='all') # p, Zbar, nj        
            
            Zbar[i,j] = out[1]
            
        mp = 1.66054e-27 # kg
        Ni = NE / Zbar # ion/m^3
        rho = NE / Zbar * mp # kg/m^3
        
        fig,axs = plt.subplots(2,2, sharey=True)
        axs = axs.flatten()
        
        im = axs[0].pcolormesh(np.log10(NE), np.log10(KT), Zbar, shading='nearest')
        plt.colorbar(im, ax=axs[0])
        axs[0].set(aspect='auto',
                      ylabel='log(T (eV))',
                      # xlabel='log(ne (1/m3))',
                      xlabel='log(ne (e-/m3))',
                      title='Saha Zbar of H')
    
        im = axs[1].pcolormesh(np.log10(Ni), np.log10(KT), Zbar, shading='nearest')
        plt.colorbar(im, ax=axs[1])
        axs[1].set(aspect='auto',
                      ylabel='log(T (eV))',
                      # xlabel='log(ne (1/m3))',
                      xlabel='log(ni (ion/m3))',
                      title='Saha Zbar of H',
                      xlim=(24,30))
    
        im = axs[2].pcolormesh(np.log10(rho), np.log10(KT), Zbar, shading='nearest')
        plt.colorbar(im, ax=axs[2])
        axs[2].set(aspect='auto',
                      ylabel='log(T (eV))',
                      xlabel='log(rho (kg/m3))',
                      title='Saha Zbar of H',
                      xlim=(-2,10))
        
        plt.figure(figsize=[5,3])
        plt.pcolormesh(np.log10(rho), np.log10(KT), Zbar, shading='nearest',
                       vmin=0, vmax=1)
        plt.colorbar()
        plt.contour(np.log10(rho), np.array([np.log10(KT)]*rho.shape[1]).T,
                    Zbar, levels=[0.1,0.5,0.9],
                    colors='grey')
        plt.gca().set(aspect='auto',
                      ylabel='log(T (eV))',
                      xlabel='log(rho (kg/m3))',
                      # title='Saha Zbar of H',
                      xlim=(-2, 10),
                      ylim=(0,None))
        
        #### Ethan's PRism
        # Compare against Ethan's H at 0.1 g/cc
        # Load Ethan's data
        fn = '/Users/dbis/Documents/99_General/AtomicPhysics/T_Zbar_H_0.1gcc.dat'
        esmi = np.genfromtxt(fn, skip_header=4)
        
        # Interpolate onto regular rho grid
        rho_grid = np.logspace(-2,10, num=50)
        zbar_rho = [] # Zbar regularly gridded against rho
        for i,t in enumerate(KT):
            zbar_rho.append(np.interp(rho_grid, rho[i], Zbar[i]))
        zbar_rho = np.array(zbar_rho)
                
        rho_grid = [100] # kg/m^3
        zbar_rho = [] # Zbar regularly gridded against rho
        for i,t in enumerate(KT):
            zbar_rho.append(np.interp(rho_grid, rho[i], Zbar[i]))
        zbar_rho = np.array(zbar_rho)
            
        #### FAC+SB, DCA
        from pfac import const, rfac # rfac is for READING filess
        
        # Load FAC data
        fn = '/Users/dbis/Documents/99_General/Databases/FAC/ex/hydrogen/h_1_5_0'
        lh, lbs = rfac.read_lev(fn+'.lev') # header, blocks
        IP = -lh['E0'] # eV, energy of ground state, negative
        
        # Concatenate all blocks into a single DataFrame
        df = pd.concat([pd.DataFrame.from_dict(item) for item in lbs])
        
        # Calculate multiplicity of each level
        g = np.ones(df.shape[0]) # df['2J'] # Place-hodler. 2J+1? 
        
        # Get Saha balance
        Zbar_fac = np.zeros(Zbar.shape)

        for idx, items in enumerate(it.product(KT,NE)):
            kT, ne = items
            i, j = np.unravel_index(idx, shape=(NT,Nn))
            out = saha(ne, kT, [df['ENERGY']], [g], [IP]) # E, g must be list of arrays
            # out = saha(ne, kT, Earrs, glists, Iplist, returns='all') # p, Zbar, nj        
            
            Zbar_fac[i,j] = out[1]
            
        rho_fac = NE / Zbar_fac * mp # kg/m^3
        
        # Interpolate to compare to Ethan's
        rho_interp = [100] # kg/m^3
        zbar_fac_interp = [] # Zbar regularly gridded against rho
        for i,t in enumerate(KT):
            zbar_fac_interp.append(np.interp(rho_interp, rho_fac[i], Zbar_fac[i]))
        zbar_fac_interp = np.array(zbar_fac_interp)

        #### FAC+SB, UTA
        sys.path.append('/Users/dbis/Documents/99_General/Databases/FAC/ex')
        from read_fac import read_tr_UTA

        # Load FAC data
        fn = '/Users/dbis/Documents/99_General/Databases/FAC/ex/hydrogen/h_UTA_1_5_0'
        lh, lbs = rfac.read_lev(fn+'.lev') # header, blocks
        IP = abs(lh['E0']) # eV, energy of ground state, positive
        
        # Concatenate all blocks into a single DataFrame
        df = pd.concat([pd.DataFrame.from_dict(item) for item in lbs])
        
        # Calculate multiplicity of each level
        g = np.ones(df.shape[0]) # df['2J'] # Place-hodler. 2J+1? 
        
        # Get Saha balance
        Zbar_UTA = np.zeros(Zbar.shape)

        for idx, items in enumerate(it.product(KT,NE)):
            kT, ne = items
            i, j = np.unravel_index(idx, shape=(NT,Nn))
            out = saha(ne, kT, [df['ENERGY']], [g], [IP]) # E, g must be list of arrays
            # out = saha(ne, kT, Earrs, glists, Iplist, returns='all') # p, Zbar, nj        
            
            Zbar_UTA[i,j] = out[1]
            
        rho_UTA = NE / Zbar_fac * mp # kg/m^3
                
        #### Plot Zbar heatmap and contours
        fig, axs = plt.subplots(2,2, sharex=True, sharey=True,
                                figsize=[6,5])
        axs = axs.flatten()
        
        # 1/n^2 Rydberg energies
        im = axs[0].pcolormesh(np.log10(rho), np.log10(KT), Zbar, shading='nearest',
                       vmin=0, vmax=1)
        plt.colorbar(im, ax=axs[0])
        axs[0].contour(np.log10(rho), np.array([np.log10(KT)]*rho.shape[1]).T,
                    Zbar, levels=[0.1,0.5,0.9],
                    colors='grey')
        axs[0].set(ylabel='log(T (eV))',
                    # xlabel='log(rho (kg/m3))',
                     title='Rydberg atom, nmax={0:d}'.format(nmax),
                    xlim=(-2, 10),
                    ylim=(0,None))
        
        # FAC DCA
        im = axs[1].pcolormesh(np.log10(rho_fac), np.log10(KT), Zbar_fac, shading='nearest',
                       vmin=0, vmax=1)
        plt.colorbar(im, ax=axs[1])
        axs[1].contour(np.log10(rho_fac), np.array([np.log10(KT)]*rho.shape[1]).T,
                    Zbar_fac, levels=[0.1,0.5,0.9],
                    colors='grey')
        axs[1].set(title='FAC, DCA',)
        
        # FAC UTA
        im = axs[2].pcolormesh(np.log10(rho_UTA), np.log10(KT), Zbar_UTA, shading='nearest',
                       vmin=0, vmax=1)
        plt.colorbar(im, ax=axs[2])
        axs[2].contour(np.log10(rho_UTA), np.array([np.log10(KT)]*rho.shape[1]).T,
                    Zbar_UTA, levels=[0.1,0.5,0.9],
                    colors='grey')
        axs[2].set(title='FAC, UTA',
                    ylabel='log(T (eV))',
                    xlabel='log(rho (kg/m3))',)
        
        # Contours only
        im = axs[3].contour(np.log10(rho), np.array([np.log10(KT)]*rho.shape[1]).T,
                    Zbar, levels=[0.1,0.5,0.9],
                    colors='C0')
        axs[3].contour(np.log10(rho_fac), np.array([np.log10(KT)]*rho.shape[1]).T,
                    Zbar_fac, levels=[0.1,0.5,0.9],
                    colors='C1')
        axs[3].contour(np.log10(rho_UTA), np.array([np.log10(KT)]*rho.shape[1]).T,
                    Zbar_UTA, levels=[0.1,0.5,0.9],
                    linestyles='--',
                    colors='C2')
        plt.colorbar(im, ax=axs[3])
        
        custom_lines = [Line2D([0], [0], color='C{0:d}'.format(i), lw=1) for i in range(3)]

        axs[3].legend(custom_lines,
                      ['Rydberg',
                       'FAC, DCA',
                       'FAC, UTA'])
        axs[3].set(ylabel='log(T (eV))',
                   xlabel='log(rho (kg/m3))',)
        
        # Plot Ethan's Prism and my Rydberg
        fig, AX = plt.subplots(figsize=[4,3])
        AX.plot(esmi[:,0],esmi[:,1], label='Prism', color='k')
        AX.semilogx(KT, zbar_rho, label='Rydberg SB, nmax={0:d} (no CL)'.format(nmax))
        AX.semilogx(KT, zbar_fac_interp, label='FAC DCA (no CL)'.format(nmax))
        
        # Plot Ethan's data
        AX.set(aspect='auto',
                      xlabel='T (eV)',
                      ylabel='Zbar',
                      title='Zbar of H, rho={0:0.1e} g/cc'.format(rho_grid[0]/1e3),
                      xlim=[0.1,1000])
        plt.legend()

# %% Fe Zbar


# %% Spectrum
    if 0:
        # Extract data from each block for current charge state
        header, blocks = rfac.read_tr(fn+'.tr')
        hnu  = [] # eV, energy of each transition
        j2 = [] # 2J
        rate = [] # 1/s, transition rate (A-rate?)
        for b in blocks:
            hnu.extend(b['Delta E'])
            j2.extend(b['upper_2J'])
            rate.extend(b['rate'])

# Run for FAC dataset of Li- to C-like of Fe

# Append with Boltzmann from Boltzmann_populations