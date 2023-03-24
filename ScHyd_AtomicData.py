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
from scipy.special import comb
from matplotlib.lines import Line2D # For custom legend
from matplotlib.widgets import Button # For widgets

import os                       # Used to e.g. change directory
import sys                      # Used to put breaks/exits for debugging
import re
import itertools as it
from copy import deepcopy

from ScHyd import get_ionization, AvIon
from saha_boltzmann_populations import saha, boltzmann


class AtDat():
    
    def __init__(self, Z, A, Zbar_min=0, nmax=5, exc_list=[0,1]):
        
        ##### Constants #####
        self.mp = 1.67262192e-24 # g .Proton mass
        self.rbohr = 0.519e-8 # cm. Bohr radius
        self.re = 2.81974e-13 # cm. Classical electron radius
        self.c = 2.998e10 # cm/s. Speed of light
        
        ##### Inputs #####
        self.Z = Z
        self.A = A
        self.Zbar_min = Zbar_min
        self.nmax = nmax
        self.exc_list = exc_list
        
        ##### Initialize dictionaries #####
        # Dict of dict of dict: En['up' or'lo'][Zbar][excitation degree]
        dict_base = {'{0:d}'.format(item): {} for item in range(ZZ)}
        self.En = {item:deepcopy(dict_base) for item in ['up','lo']}  # Shell energies
        self.Pn = {item:deepcopy(dict_base) for item in ['up','lo']} # Shell populations
        self.gn = {item:deepcopy(dict_base) for item in ['up','lo']}  # Shell statistical weights
        self.Etot = {item:deepcopy(dict_base) for item in ['up','lo']}  # Total ion energy – used for transition energies
        
        # Dict of dict: hnu[Zbar][excitation degree]
        self.hnu = deepcopy(dict_base) # Transition energy 
                        
    def get_atomicdata(self, DIR='complexes/', vb=False):
        ''' Calculates atomic data for each excited complex in given directory.
        
        Screened Hydrogenic model is run for each complex.
        
        Energies (eV), populations, and statistical weights of each shell and total
        energy (eV) of each complex are calculated.
        
        Complex files must be formatted for input to FAC. See write_configs.py for details
        
        Complexes included in files dictates what transitions are considered
        
        Parameters
        ----------
        DIR : str
            Location of FAC-formatted complex files
        '''

        for uplo in['up','lo']:
            for Zbar in range(Zbar_min, self.Z): # Start with neutral, end with H-like
                Zbar_str = '{0:d}'.format(Zbar)
        
                Nele = self.Z - Zbar
                # Exlcude unphysical excitations, e.g. Li can only be 0- and 1-excited
                valid_exc_list = [item for item in exc_list if item<(Nele-1)]
                for exc in valid_exc_list:
                    # Save quantities as list of lists for current charge state and excitation degree
                    Enx = [] # Shell energies
                    Pnx = [] # Shell populations
                    gnx = [] # Shell statistical weights
                    Etotx = [] # Total ion energy
                    fn = DIR+'fac_{0:d}_{1:d}_{2:d}_{3:s}.txt'.format(Nele, nmax,
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
                            sh = AvIon(self.Z, Zbar=(self.Z-Nele), nmax=self.nmax)      
            
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
                    self.En[uplo][Zbar_str]['{0:d}'.format(exc)] = Enx
                    self.Pn[uplo][Zbar_str]['{0:d}'.format(exc)] = Pnx
                    self.gn[uplo][Zbar_str]['{0:d}'.format(exc)] = gnx
                    self.Etot[uplo][Zbar_str]['{0:d}'.format(exc)] = Etotx
        
        # Keep list of calculated charge states
        self.Zkeys = [Zkey for Zkey in list(self.En['up'].keys()) if list(self.En['up'][Zkey].keys())] 
        
    def get_hnu(self, Zs):
        
        for i, Zbar in enumerate(Zs):
            Zbar_str = '{0:d}'.format(Zbar)
            valid_exc = list(self.En['up'][Zbar_str].keys())
            for exc in valid_exc:
                # color = self.norm(int(exc)) # Get rgba from colorbar
                # Calculate, save hnu for each configuration up/lo pair
                self.hnu[Zbar_str][exc] = [abs(up - lo) for up,lo in zip(self.Etot['up'][Zbar_str][exc],
                                                                         self.Etot['lo'][Zbar_str][exc])] 
        return
    
    def tidy_arrays(self, uplo='lo'):
        ''' Parses saved dictionaries into zero-padded arrays for use in Saha-Boltzmann
        calculation
        
        Parameters
        ----------
        uplo : str
            Option to save arrays of upper 'up' or lower 'lo' state.
        '''
        
        # Parse data into arrays
        Earrs = [] # Array of total energy of each complex, sorted by charge state
        excarrs = [] # Excitation degree of each complex, sorted by charge state
        glists = [] # Total statistical weight of each complex, sorted by charge state
        hnuarrs = [] # Transition energy of each 
        Pnarrs = []

        # Zs = [Zkey for Zkey in list(Etot['lo'].keys()) if list(Etot['up'][Zkey].keys())] # Keep only calculated charge states
        for Z in self.Zkeys:
            # Save off energy levels, excitation degree, and stat.weight of each complex,
            # grouped by ionization state
            tmpEtot = []
            tmpexc = []
            tmpgn = []
            tmphnu = []
            tmpPn = []
            for exc in list(self.Etot[uplo][Z].keys()):
                N = len(self.Etot[uplo][Z][exc])
                tmpEtot.extend(self.Etot[uplo][Z][exc])
                [tmpexc.append(int(exc)) for item in range(N)]
                [tmpgn.append(np.prod(item)) for item in self.gn[uplo][Z][exc]]
                tmphnu.extend(self.hnu[Z][exc])

                tmpPn.extend(self.Pn[uplo][Z][exc])
                
            Earrs.append(np.array(tmpEtot))
            Pnarrs.append(np.array(tmpPn))
            excarrs.append(np.array(tmpexc))
            glists.append(tmpgn)
            hnuarrs.append(tmphnu)

        # To enable slicing, 0-pad all state-resolved arrays
        # 0 values in glists reslut in 0 contribution to partition functions, 0 effect on Saha-Boltzmann
        # Pad no zeroes at beginning, out to max_length at end: pad_width = [(0, max_length-len(item))]
        max_length = max([len(item) for item in excarrs]) # Longest length array
        self.max_length = max_length
        self.Earrs   = np.array([np.pad(item, [(0, max_length-len(item))], constant_values=int(0)) for item in Earrs]) 
        self.excarrs = np.array([np.pad(item, [(0, max_length-len(item))], constant_values=int(0)) for item in excarrs])
        self.glists  = np.array([np.pad(item, [(0, max_length-len(item))], constant_values=int(0)) for item in glists])
        self.hnuarrs = np.array([np.pad(item, [(0, max_length-len(item))], constant_values=int(0)) for item in hnuarrs])

        # Pad 1st dimension of Pnarrs only. Shape (Z, max_length, nmax)
        self.Pnarrs  = np.array([np.pad(item, [(0, max_length-len(item)), (0,0)], constant_values=int(0)) for item in Pnarrs]) 

        return
    
    def saha_boltzmann(self, KT, NE):
        
        self.KT = KT
        self.NE = NE
        
        self.NT = len(KT)
        self.Nn = len(NE)
        
        Ip = get_ionization(self.Z, return_energy_levels=False)
        self.Iplist = [Ip[int(item)] for item in self.Zkeys]

        Zbar = np.zeros(shape=[self.NT,self.Nn]) # Mean ionization state
        psaha = np.zeros(shape=[self.NT,self.Nn,len(self.Zkeys)+1]) # Saha charge state populations. Shape T, Ne, Z+1
        pboltz = np.zeros(shape=[self.NT,self.Nn,len(self.Zkeys), self.max_length]) # Boltzmann state populations. Shape T, Ne, Z, state
        for idx, items in enumerate(it.product(KT,NE)):
            kT, ne = items
            i, j = np.unravel_index(idx, shape=(self.NT,self.Nn))
            
            #### Saha
            # Run Saha, with ne converted to m^-3. 
            out = saha(ne, kT, self.Earrs, self.glists, self.Iplist, returns='csd') # Returns: p     
            psaha[i,j] = out # Normalization: np.sum(psaha, axis=-1) should = 1 everywhere
            
            # Calculate Zbar
            Zbar[i,j] = np.sum(np.arange(int(self.Zkeys[0]), int(self.Zkeys[-1])+2,) * out)
            
            #### Boltzmann – Ne grid
            # Run Boltzmann on each charge state
            pb = []
            for Z,Earr,garr in zip(self.Zkeys, self.Earrs, self.glists):
                pb.append(boltzmann(Earr, garr, kT, normalize=True))
            
            pboltz[i,j] = np.array(pb)
        
        # Save out values
        self.psaha = psaha
        self.pboltz = pboltz
        self.Zbar = Zbar
        
        self.rho = NE/Zbar * self.A * self.mp # g/cm^3. Ne in 1/cm^3, mp in g
        return

    def saha_boltzmann_rho(self,rho_grid):
        #### Regrid in rho
        # Construct regular grids in mass density
        self.rho_grid = rho_grid
        self.Nrho = len(rho_grid)
        
        Zbar_rho = [] # Zbar regularly gridded against rho
        psaha_rho = np.zeros(shape=[self.NT,self.Nrho,len(self.Zkeys)+1]) # Normalization: np.sum(psaha, axis=-1) should = 1 everywhere
        for i,t in enumerate(KT):
            Zbar_rho.append(np.interp(rho_grid, self.rho[i], self.Zbar[i]))
            for k in range(self.psaha.shape[-1]):
                psaha_rho[i,:,k] = np.interp(rho_grid, self.rho[i], self.psaha[i,:,k]) 
        self.Zbar_rho = np.array(Zbar_rho)

        #### Boltzmann – rho grid
        # Run Boltzmann on each charge state over rho grid
        pboltz_rho = np.zeros(shape=[self.NT,self.Nrho,len(self.Zkeys), self.max_length]) # Shape T, Ne, Z, state
        for idx, items in enumerate(it.product(self.KT,rho_grid)):
            kT, __ = items
            i, j = np.unravel_index(idx, shape=(self.NT,self.Nrho))
            p = []
            for Z,Earr,garr in zip(self.Zkeys, self.Earrs, self.glists):
                p.append(boltzmann(Earr, garr, kT, normalize=True))

            pboltz_rho[i,j] = np.array(p)

        # Saha-Boltzmann pop of each state
        self.pboltz_rho = pboltz_rho
        self.psaha_rho = psaha_rho
        self.pstate_rho = pboltz_rho * psaha_rho[Ellipsis,:-1, np.newaxis]  # Shape: T, rho, Z, state
        return

    # def get_osc_str(self, ni,li,nj,lj):
    #     # Get hydrogenic oscillator strength for given transition. From Table
    #     faH_dict = {'10':{'21':0.4162, # Key format: ni li, nj lj. ['10']['21'] is 1s -> 2p
    #                  '31':0.0791,
    #                  '41':0.0290,
    #                  '51':0.0139,
    #                  '61':0.0078,
    #                  '71':0.0048,
    #                  '81':0.0032
    #                  },
    #            }
    #     faH = faH_dict['{0:d}{1:d}'.format(ni, li)]['{0:d}{1:d}'.format(nj, lj)]
    #     fa_bar = wi*(4*lj + 3-wj)/(4*lj + 2) * faH
        
        
    #     # Range of possible initial and final configuration (nl-resolved) occupations for absorption
    #     wi = np.arange(max(0, self.Pn[ni-1] - 2*ni**2 + (4*li + 2)),
    #                    min(self.Pn[ni-1], 4*li + 2)+1) # +1 for inclusive of endpoint
        
    #     wj = np.arange(max(0, self.Pn[nj-1]+1 - 2*nj**2 + (4*lj + 2)),
    #                    min(self.Pn[nj-1]+1, 4*lj + 2)+1) # Pn+1 because final state has one more electron in nj lj compared to initial state
        
    #     # Given a set of Pn, wi and wj are independent. Mesh product
    #     wi, wj = np.meshgrid(wi, wj)
        
    #     Fa_n = np.sum(wi*(4*lj + 3-wj)/(4*lj + 2) * faH) # Osc. str. summed over final levels, and summed over initial states. Modified from Griem (3.31)
        
    #     self.fsum = Fa_n # THIS is gf

    #     return

    def get_opacity(self,ni,li,nj,lj):
        ''' Converts atomic data and SB populations into spectra.
        

        Returns
        -------
        None.

        '''
        width = 10 * 2.41799e14 # eV -> Hz to match units of prefactor in xsec
        amp = 1/2/np.pi / width # 1/Hz. Value of normalized line shape at line center. Used in xsec
        # Get hydrogenic oscillator strength
        fH_dict = {'10':{'21':0.4162, # Key format: ni li, nj lj. ['10']['21'] is 1s -> 2p
                     '31':0.0791,
                     '41':0.0290,
                     '51':0.0139,
                     '61':0.0078,
                     '71':0.0048,
                     '81':0.0032
                     },
               }
        fH = fH_dict['{0:d}{1:d}'.format(ni, li)]['{0:d}{1:d}'.format(nj, lj)]
        
        # Weight osc. str. pre-factor over configurations which actually allow the transition

        # Lower state prefactor: w
        # Permissible populations of initial lower active state.
        # Range is from 1 (for at least one available) OR all other sub-shells full,
        # to shell pop OR full sub-shell
        r0 = np.maximum(1, self.Pnarrs[:,:,ni-1] - (2*ni**2-(4*li+2))).astype(int) # Range minimum
        r1 = np.minimum(self.Pnarrs[:,:,ni-1], 4*li+2).astype(int) + 1 # range maximum. +1 for inclusive
        gi = [] # Not truly statistical weight
        for rr in zip(r0.flatten(),r1.flatten()):
            tmp = []
            for w in range(*rr):
                tmp.append(w*comb(4*li+2, w))
            gi.append(np.sum(tmp))
        gi = np.array(gi).reshape(self.Pnarrs.shape[:-1])
        
        # Upper state prefactor: (4*lj + 2 - w) / (4*lj + 2)
        # Permissible populations of initial upper active state.
        # Range is from 0 (for at least one hole) OR all other sub-shells full,
        # to shell pop OR full sub-shell-1
        r0 = np.maximum(0, self.Pnarrs[:,:,nj-1] - (2*nj**2-(4*lj+2))).astype(int) # Range minimum
        r1 = np.minimum(self.Pnarrs[:,:,nj-1], 4*lj+2-1).astype(int) + 1 # range maximum. +1 for inclusive
        gj = []
        for rr in zip(r0.flatten(),r1.flatten()):
            tmp = []
            for w in range(*rr):
                tmp.append((4*lj+2 - w) / (4*lj+2) * comb(4*lj+2, w))
            gj.append(np.sum(tmp))
        gj = np.array(gj).reshape(self.Pnarrs.shape[:-1])

        # Total number of transitions
        g_tot = comb(2*ni**2, self.Pnarrs[:,:,ni-1]) \
              * comb(2*nj**2, self.Pnarrs[:,:,nj-1]) # Total possible transitions
              
        prefactor = gi*gj / g_tot
        
        # Sum over final states, and average over initial states
        gf = prefactor * fH
        
        # Calculate cross-section of each transition. See Perez-Callejo JQSRT 202
        xsec = 2.6553e-06 * gf * 1e4 # cm^2. Prefactor = e^2 / (4 epsilon_0) / me / c in mks
        
        breakpoint()
        # "State" population is fractional population * Ntot, on rho_grid
        # Note: pstate_rho is shape [T, rho, Zbar, complex]
        Ni = self.rho_grid / (self.A * self.mp) # cm^-3, ion density
        alpha_line = xsec * Ni[np.newaxis,:,np.newaxis,np.newaxis] \
                    * self.pstate_rho * amp # cm^-1. "Opacity" at line center, given state populations
                    
        # Calculate opacity
        self.kappa_line = alpha_line / rho_grid[None,:,None,None] # cm^2 / g
        
        # # Calculate cross-section of each transition. pstate_rho is shape [T, rho, Zbar, complex]
        # xsec = amp*(2*np.pi**2 * self.re * self.c * gf) # cm^2. Line cross-section (no lineshape)
        
        # # Calculate linear attenuation coefficient. pstate_rho is shape [T, rho, Zbar, complex]
        # # "State" population is fractional population * Ntot, on rho_grid
        # alpha = xsec * self.pstate_rho \
        #              * (self.rho_grid /self.A/self.mp)[np.newaxis,:,np.newaxis,np.newaxis]
                     
        # # Calculate opacity, cm^2/g
        # kappa = alpha / self.rho_grid[np.newaxis,:,np.newaxis,np.newaxis] # cm^2 / g
        # self.kappa = kappa

        # # Given shell populations, calculate fraction of all possible configurations
        # # which would admit desired transition

        # # Construct array of all population ranges to sum
        # # Permissible populations of initial lower active state.
        # # Range is from 1 (for at least one available) OR all other sub-shells full,
        # # to shell pop OR full sub-shell
        # r0 = np.maximum(1, self.Pnarrs[:,:,ni-1] - (2*ni**2-(4*li+2))).astype(int) # Range minimum
        # r1 = np.minimum(self.Pnarrs[:,:,ni-1], 4*li+2).astype(int) + 1 # range maximum. +1 for inclusive
        # gi = [] 
        # for rr in zip(r0.flatten(),r1.flatten()):
        #     tmp = []
        #     for w in range(*rr):
        #         tmp.append(comb(4*li+2, w))
        #     gi.append(np.sum(tmp))
        # gi = np.array(gi).reshape(self.Pnarrs.shape[:-1])

        # # Permissible populations of initial upper active state.
        # # Range is from 0 (for at least one hole) OR all other sub-shells full,
        # # to shell pop OR full sub-shell-1
        # r0 = np.maximum(0, self.Pnarrs[:,:,nj-1] - (2*nj**2-(4*lj+2))).astype(int) # Range minimum
        # r1 = np.minimum(self.Pnarrs[:,:,nj-1], 4*lj+2-1).astype(int) + 1 # range maximum. +1 for inclusive
        # gj = []
        # for rr in zip(r0.flatten(),r1.flatten()):
        #     tmp = []
        #     for w in range(*rr):
        #         tmp.append(comb(4*lj+2, w))
        #     gj.append(np.sum(tmp))
        # gj = np.array(gj).reshape(self.Pnarrs.shape[:-1])
        
        # # # Multiplicity of allowed transitions is product of initial and final
        # g_allowed = gi*gj
        # g_tot = comb(2*ni**2, self.Pnarrs[:,:,ni-1]) \
        #       * comb(2*nj**2, self.Pnarrs[:,:,nj-1]) # Total possible transitions
        
        # # frac = g_allowed / g_tot # Fraction of allowed transitions
        
        return
    
    def plot_hnu(self, exc_minmax, Zbar_plot=None, xaxis='Zbar',
                 fig=None, ax=None, cmap_name='rainbow'):
        ''' Plots photon energies of transitions
        

        Parameters
        ----------
        exc_minmax : list
            Two-element list giving the minimum and maximum excitation degrees to plot.
        xaxis : str, optional
            Choice to plot versus ionization state 'Zbar' or against 
            ionization state + excitation degree 'Zbar+exc' to emphasize satellite structure.
            The default is 'Zbar'.
        fig : figure object, optional
            Figure on which to place colorbar. The default is None.
        ax : axes object, optional
            Axes on which to plot data. The default is None.
        cmap_name : str, optional
            Colormap for scatter plot face colors. The default is 'rainbow'.
        '''
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = plt.gca()
        if Zbar_plot is None:
            Zbar_plot = np.array(self.Zkeys).astype(int)
            
        cmap = mpl.cm.get_cmap(cmap_name)
        norm = mpl.colors.Normalize(vmin=exc_minmax[0], vmax=exc_minmax[1])
        
        for i, Zbar in enumerate(Zbar_plot):
            Zbar_str = '{0:d}'.format(Zbar)
            valid_exc = list(self.En['up'][Zbar_str].keys())
            for exc in valid_exc:

                # Construct desired x axis
                if xaxis=='Zbar':
                    x = [Zbar]*len(self.hnu[Zbar_str][exc])
                    xlab = 'Zbar'
                elif xaxis=='Zbar+exc':
                    x = [Zbar+int(exc) - 0.1*int(exc)]*len(self.hnu[Zbar_str][exc])
                    xlab = 'Zbar + 0.9*(Excitation Degree)'
                
                color = norm(int(exc)) # Get rgba from colorbar

                ax.plot(x, self.hnu[Zbar_str][exc], '.',
                            color=cmap(color),
                            alpha=1,
                            label=exc)
                
        # Add colorbar
        cax = fig.add_axes([0.2, 0.85, 0.5, 0.05])
        cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm = norm,
                                        orientation='horizontal',)
        cb.set_label('Excitation degree',)
        fig.tight_layout()
        ax.set(xlabel=xlab,
               ylabel='hnu (eV)')
        return
        
    def plot_rho_interp(self):
        fig, axs = plt.subplots(2, figsize=[4,5], sharex=True, sharey=True)
        im = axs[0].pcolormesh(np.log10(self.rho),
                               np.log10(self.KT),
                               self.Zbar,
                               shading='nearest',
                               vmin=int(self.Zkeys[0]),
                               vmax=float(self.Zkeys[-1]))
        plt.colorbar(im, ax=axs[0])

        im = axs[1].pcolormesh(np.log10(self.rho_grid),
                               np.log10(self.KT),
                               self.Zbar_rho,
                               shading='nearest',
                               vmin=int(self.Zkeys[0]),
                               vmax=float(self.Zkeys[-1]))
        plt.colorbar(im, ax=axs[1])

        fig.suptitle(['Z={0:d}'.format(self.Z),
                ' exc=',self.exc_list,
                'Zbar = {0:s} to {1:s}'.format(self.Zkeys[0], self.Zkeys[-1])])
        axs[0].set(xlabel='log10(rho (g/cm^3))',
                      ylabel='log10(T (eV))',
                      title='Regular Ne grid')
        axs[1].set(xlabel='log10(rho (g/cm^3))',
                      ylabel='log10(T (eV))',
                      title='Interpolated onto rho grid')
        return
    
    def plot_sb_buttons(self, Zbar_plot=None):
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
        
        if Zbar_plot is None:
            Zbar_plot = self.Zkeys
        fig, ax = plt.subplots(figsize=[10,6])
        scat = [] # Keep scatter plots for later

        cmap_name = 'rainbow'
        cmap = mpl.cm.get_cmap(cmap_name)

        scale = 'log'
        if scale=='log':
            # Set min/max of colorbar
            norm = mpl.colors.Normalize(vmin=-6, vmax=0) # Converts value to linearly interpolte [0,1] between bounds
            
            # Calculate colorbars of populations
            colors = cmap(norm(np.log10(self.pstate_rho)))
            
        elif scale=='lin':
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            colors = cmap(norm(self.pstate_rho))

            
        Tidx, rhoidx = 0,0 # Initial populations to plot
        for i in range(len(Zbar_plot)):
            # Parse values for current charge state
            Z = int(Zbar_plot[i])
            exc = self.excarrs[i]
            hnu_i = self.hnuarrs[i]
            p = self.pstate_rho[Tidx, rhoidx, i]
            
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
        callback = updateIndex(scat, colors, Trho=[self.KT,self.rho_grid])

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
        return bpi, bmi, bpj, bmj

    def plot_pop_hist(self, rhoidx, bins):
        ''' Plots population histogram over hnu as a function of T for a single rho
        

        Parameters
        ----------
        rhoidx : int
            Index of rho condition to plot.
        bins : array
            1-D array of hnu edges of bins.

        Returns
        -------
        None.

        '''
        # Histogram population-weighted transition energy - ignore 0-eV trans
        # bins = np.linspace(6400, 6800) # Full spectrum
        # bins = np.linspace(6515, 6535, num=6) # N-like complex

        pop_hist = []
        for ii in range(len(KT)):
            pop_hist.append(np.histogram(a=self.hnuarrs.flatten(),
                                    weights=self.pstate_rho[ii,rhoidx].flatten(),
                                    bins=bins)[0])
        
        labels = [r'$\bar{h\nu}$ = %0.0f' % (item) for item in bins[:-1] + np.diff(bins)/2]
                            
        plt.figure()
        plt.plot(self.KT, pop_hist, label=labels)
        plt.legend()

        plt.gca().set(yscale='log',
                      xlabel='T (eV)',
                      ylabel='Population fraction')
        return

# %% Main
if __name__=='__main__':
        
    ZZ = 26 # Nuclear charge
    A = 51.996 # Nucleon number
    Zbar_min = 16
    nmax = 5 # Maximum allowed shell
    exc_list = [0,1,2,3] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
    pf = 0
    
    # Run model
    ad = AtDat(ZZ, A, Zbar_min, nmax, exc_list,)
    ad.get_atomicdata(vb=0)
    ad.get_hnu(np.array(ad.Zkeys).astype(int))
    ad.tidy_arrays('lo')
    
    # Define T, Ne, rho grids for SB
    Nn, NT = 10, 11 # Number of density, temperature gridpoints
    KT = np.logspace(1.5,3, num=NT) # eV, Temperature range, 
    rho0 = np.logspace(0.5,2, num=Nn) # g/cc
    Zbar0 = 20 # Estimated dZbar
    NE = rho0 / (A*ad.mp) * Zbar0 # 1/cm^3, Ne range
    
    Nrho = 12
    rho_grid = np.logspace(0.5,2, num=Nrho)
    
    # Run Saha-Boltzmann
    ad.saha_boltzmann(KT, NE)
    ad.saha_boltzmann_rho(rho_grid)
    
    ad.get_opacity(1, 0, 2, 1)
    
    if pf:
        fig, ax = plt.subplots(figsize=[4,3])
        ad.plot_hnu([0,3], xaxis='Zbar+exc', fig=fig, ax=ax)
        
        ad.plot_rho_interp()
        buttons = ad.plot_sb_buttons()
        
        bins = np.linspace(6400, 6800) # Full spectrum
        bins = np.linspace(6515,6535, num=6) # N-like complex
        ad.plot_pop_hist(0, bins=bins)

