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
import sys                      
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
        
        ##### Initialize data structures #####
        # Dict of dict of dict: En['up' or'lo'][Zbar][excitation degree]
        dict_base = {'{0:d}'.format(item): {} for item in range(Z)}
        self.En = {item:deepcopy(dict_base) for item in ['up','lo']}  # Shell energies
        self.Pn = {item:deepcopy(dict_base) for item in ['up','lo']} # Shell populations
        self.gn = {item:deepcopy(dict_base) for item in ['up','lo']}  # Shell statistical weights
        self.Etot = {item:deepcopy(dict_base) for item in ['up','lo']}  # Total ion energy – used for transition energies
        
        # Dict of dict: hnu[Zbar][excitation degree]
        self.hnu = deepcopy(dict_base) # Transition energy 
        
        # Dict of lists: lineshape_dict['L'] or ['G'] = []
        self.lineshape_dict = {'G':[],
                               'L':[]}
                        
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
            for Zbar in range(self.Zbar_min, self.Z): # Start with neutral, end with H-like
                Zbar_str = '{0:d}'.format(Zbar)
        
                Nele = self.Z - Zbar
                # Exlcude unphysical excitations, e.g. Li can only be 0- and 1-excited
                valid_exc_list = [item for item in self.exc_list if item<(Nele-1)]
                for exc in valid_exc_list:
                    # Save quantities as list of lists for current charge state and excitation degree
                    Enx = [] # Shell energies
                    Pnx = [] # Shell populations
                    gnx = [] # Shell statistical weights
                    Etotx = [] # Total ion energy
                    fn = DIR+'fac_{0:d}_{1:d}_{2:d}_{3:s}.txt'.format(Nele, self.nmax,
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
                            Pni = np.zeros(self.nmax) # populations of current complex
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
    
    def saha_boltzmann(self, KT, NE, IPD=0):
        
        self.KT = KT
        self.NE = NE
        
        self.NT = len(KT)
        self.Nn = len(NE)
        
        Ip = get_ionization(self.Z, return_energy_levels=False) - IPD
        self.Iplist = [Ip[int(item)] for item in self.Zkeys]

        Zbar = np.zeros(shape=[self.NT,self.Nn]) # Mean ionization state
        psaha = np.zeros(shape=[self.NT,self.Nn,len(self.Zkeys)+1]) # Saha charge state populations. Shape T, Ne, Z+1
        pboltz = np.zeros(shape=[self.NT,self.Nn,len(self.Zkeys), self.max_length]) # Boltzmann state populations. Shape T, Ne, Z, state
        for idx, items in enumerate(it.product(KT,NE)):
            kT, ne = items
            i, j = np.unravel_index(idx, shape=(self.NT,self.Nn))
            
            #### Saha
            # Run Saha, with ne (in cm^-3) converted to m^-3. 
            out = saha(ne*1e6, kT, self.Earrs, self.glists, self.Iplist, returns='csd') # Returns: p     
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
        for i,t in enumerate(self.KT):
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

    def append_lineshape(self, width, ltype):
        ''' Appends linewidth (eV) of given type ('G' or 'L')
        to lineshape_dict
        

        Parameters
        ----------
        width : array
            Linewidth (eV) for each state. Shape [T, rho, Z, complex]
        ltype : str
            Linewidth type. 'G' for Gaussian or 'L' for Lorentzian.
        '''
        
        self.lineshape_dict[ltype.upper()].append(width)
        return

    def append_doppler_width(self):
        ''' Calculates Doppler width (eV) of each line at each (T,rho) and appends to 
        lineshape list. Gaussian.
        

        Returns
        -------
        None.

        '''
        width = []
        self.append_lineshape(width, 'G')
        return

    def append_natural_width(self):
        ''' Calculates natural width (eV) of each line at each (T,rho) and appends to 
        lineshape list. Loretnzian
        

        Returns
        -------
        None.

        '''
        
        width = []
        self.append_lineshape(width, 'L')
        return
    
    def lorentz(self, x, x0, gamma, norm=False):
        ''' Returns Lorentzian profile from given width parameter gamma.
            gamma = HWHM
            FWHWM = 2*gamma.
        '''
        
        if norm: # Return area-normalized
            return 1/np.pi * gamma / ( (x-x0)**2 + gamma**2 )
        else:  # Return maximum = 1
            return gamma**2 / ( (x-x0)**2 + gamma**2)
    
    def gauss(self, x, x0, sigma, norm=False):
        ''' Returns normalized Gaussian profile from given width parameter sigma.
            sigma = FWHM / (2*sqrt(2 ln 2)) ~ FWHM / 2.3548200
            FWHWM = 2.3548200 * sigma.
        '''
        
        if norm: # Return area-normalized
            return 1/np.sqrt(2*np.pi) / sigma * np.exp( - (x-x0)**2 / 2 / sigma**2)
        else: # Return maximum = 1
            return np.exp( - (x-x0)**2 / 2 / sigma**2)
        
        
    def sum_linewidths(self, width_min=1e-100):
        ''' Calculates total linewidths (eV).
        Gaussian widths are quadrature-summed, and Lorentzian widths are summed.        

        Returns
        -------
        None.

        '''
        
        # Sum over Lorentzian width sources, denoted by ltype = 'L' or 'l'
        # Sum axis=0
        lor_width = np.sum(self.lineshape_dict['L'], axis=0)

        # @ Quadrature-sum over Gaussian widths, denoted by ltype = 'G' or 'g'
        # Sum axes is the same as lor_width
        gau_width = np.sqrt(np.sum(np.array(self.lineshape_dict['G'])**2, axis=0))
        
        # Set zero-width Lorentzian lines to minimum width
        if type(lor_width)==np.ndarray:
            lor_width[lor_width==0]=width_min
        else:
            if lor_width==0:
                lor_width = width_min
        
        # Set zero-width Gaussian lines to minimum width
        if type(gau_width)==np.ndarray:
            gau_width[gau_width==0]=width_min
        else:
            if gau_width==0:
                gau_width = width_min
            
        # Total Gaussian and Lorentzian lineshapes of each state. Shape[T,rho,Z,complex]
        self.lineshape_tot = {'G': gau_width,
                              'L': lor_width}
    
    def get_linecenter(self, method='pseudo'):
        ''' Returns the value (1/eV) at line center of each normalized lineshape.
            Widths must be in eV and cannot be 0
        '''
        
        if method=='pseudo':
            # Pseudo-Voigt
            fg = self.lineshape_tot['G'] * 2.3548200 # Gaussian FWHM
            fl = self.lineshape_tot['L'] * 2 # Lorentzian FWHM
            ftot = (fg**5
                    + 2.69269 * fg**4 * fl \
                    + 2.42843 * fg**3 * fl**2 \
                    + 4.47163 * fg**2 * fl**3 \
                    + 0.07842 * fg * fl**4 \
                    + fl**5 )**(1/5) # Total FWHM
            
            eta = (1.36603 * (fl/ftot) \
                    - 0.47719 * (fl/ftot)**2 \
                    + 0.11116 * (fl/ftot)**3) # pseudo-Voigt mixing parameter
            
            return (eta / (np.pi*self.lineshape_tot['L']) \
                    + (1-eta) / np.sqrt(2*np.pi) / self.lineshape_tot['G'])       
    
    def get_gf(self, ni, li, nj, lj, return_gs=False):
        ''' Calculates weighted oscillator strength, averaged over initial states
            and summed over final states
        

        Parameters
        ----------
        ni : int
            Initial principal quantum number.
        li : int
            Initial angular momentum quantum number.
        nj : int
            Final principal quantum number.
        lj : int
            Final angular  momentumquantum number.

        Returns
        -------
        gf : array
            Array of weighted oscillator strengths.

        '''
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
        
        if return_gs:
            return gf, gi, gj
        else:
            return gf

    def get_line_opacity(self,ni,li,nj,lj):
        ''' Converts atomic data and SB populations into opacity at linecenter of each line.
        

        Returns
        -------
        None.

        '''
        linecenter = self.get_linecenter() # 1/eV. Value of normalized line shape at line center. Used in xsec
        
        # Get hydrogenic oscillator strength
        gf = self.get_gf(ni, li, nj, lj)
        
        # Calculate cross-section of each transition. See Perez-Callejo JQSRT 202
        xsec = (2.6553e-06 / 2.41799e14) * gf * 1e4 # cm^2 eV. Prefactor = (e^2 / (4 epsilon_0) / me / c) * (eV/Hz) in mks
        
        # "State" population is fractional population * Ntot, on rho_grid
        # Note: pstate_rho is shape [T, rho, Zbar, complex]
        Ni = self.rho_grid / (self.A * self.mp) # cm^-3, ion density
        alpha_line = xsec * Ni[np.newaxis,:,np.newaxis,np.newaxis] \
                    * self.pstate_rho * linecenter # cm^-1. "Opacity" at line center, given state populations
                    
        # Calculate opacity
        # self.kappa_line = alpha_line / rho_grid[None,:,None,None] # cm^2 / g
        self.kappa_line = xsec * 1/(self.A * self.mp) \
                        * self.pstate_rho * linecenter # cm^2 / g
        
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
    
    def generate_lineshapes(self, hnu_axis, method='pseudo', norm=False):
        ''' Generates line profiles from total Gaussian and Lorentzian lineshapes
        according to the given method, for each state.
        

        Parameters
        ----------
        hnu_axis : array
            1-D array of hnu values (eV) at which to calculate lineprofile.
        method : str, optional
            Method to generate lineshapes. The default is 'pseudo'. \n
            - pseudo : Pseudo-Voigt profile is used
        norm : bool
            Option to return area-normalized (True) or unit-height (False) lineshapes.
            Unit-height is used to multiply against opacity at linecenter
            (from get_line_opacity) to obtiain opacity spectrum.

        Returns
        -------
        lines : array
            Shape [T, rho, Z, complex, hnu_axis] array of the 
            lineprofile of each state.

        '''
        # Expand dimensions. All must be shape [NT,Nn,NZ, Ncomplex, Nhnu]
        # Add empty dimensions to compensate
        hnu_axis = np.expand_dims(hnu_axis, axis=list(range(0,len(self.pstate_rho.shape)))) # Add T,n,Z,complex dimensions
        hnu0 = np.expand_dims(self.hnuarrs, axis=(0,1,-1)) # Add T,n, hnu dimensions
        
        # Sum over lines to get kappa spectrum. Shape[NT, Nn, Nhnu]
        # kappa_line.shape [NT, Nn, NZbar, Ncomplex]
        Nhnu = len(hnu_axis)
        spec = np.zeros([self.NT, self.Nn, Nhnu])
        
        if method=='pseudo':
            fg = self.lineshape_tot['G'] * 2.3548200 # Gaussian FWHM
            fl = self.lineshape_tot['L'] * 2 # Lorentzian FWHM
            ftot = (fg**5
                    + 2.69269 * fg**4 * fl \
                    + 2.42843 * fg**3 * fl**2 \
                    + 4.47163 * fg**2 * fl**3 \
                    + 0.07842 * fg * fl**4 \
                    + fl**5 )**(1/5) # Total FWHM
            
            eta = (1.36603 * (fl/ftot) \
                    - 0.47719 * (fl/ftot)**2 \
                    + 0.11116 * (fl/ftot)**3) # pseudo-Voigt mixing parameter
                
            # Add hnu dimension.
            ftot = np.expand_dims(ftot, axis=-1)
            eta = np.expand_dims(eta, axis=-1)

            # Lorentzian requires HWHM, Gaussian requires sigma = FWHM / 2.3548200
            lines = eta * self.lorentz(hnu_axis, hnu0, gamma=(ftot/2), norm=False) \
                    + (1-eta) * self.gauss(hnu_axis, hnu0, sigma=(ftot/2.3548200)) # pseudo-Voigt profile of each line

        # # Sum lines over Z and complex.  Shape [T, rho, Z, complex, hnu] -> [T,rho,hnu]
        # spec = np.sum(lines, axis=(2,3))     
            
        return lines
    
    
    def generate_spectra(self, hnu_axis, method='pseudo'):
        ''' Calculates opacity spectrum.
        For each T,rho, sums over spectral opacity (line opacity * lineshape)
        of each charge state and complex.
        '''
        
        # Get UNIT-HEIGHT lineshapes
        lines = self.generate_lineshapes(hnu_axis, method=method, norm=False)
        
        # Multiply against opacity at line center
        opac = lines * np.expand_dims(self.kappa_line, axis=-1)
        
        # Sum over Zbar and complexes, axis=(2,3)
        self.kappa = np.sum(opac, axis=(2,3))
        
        return
    
    
    def plot_hnu(self, exc_minmax, Zbar_plot=None, xaxis='Zbar',
                 fig=None, ax=None, cmap_name='rainbow', show_colorbar=True):
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
        if show_colorbar:
            cax = fig.add_axes([0.2, 0.85, 0.5, 0.05])
            cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm = norm,
                                            orientation='horizontal',)
            cb.set_label('Excitation degree',)
        fig.tight_layout()
        ax.set(xlabel=xlab,
               ylabel='hnu (eV)')
        return

    def plot_zbar(self, xaxis='rho_grid'):
        f, ax = plt.subplots()
        if xaxis=='ne':
            x = self.NE
            Z = self.Zbar
            xlab = 'Ne (1/cm^3)'
        elif xaxis=='rho':
            x = self.rho
            xlab = 'rho (g/cm^3)'
            Z = self.Zbar
        elif xaxis=='rho_grid':
            x = self.rho_grid
            xlab = 'Interpolated rho (g/cm^3)'
            Z = self.Zbar_rho
    
        im = ax.pcolormesh(np.log10(x),
                               np.log10(self.KT),
                               Z,
                               shading='nearest',
                               vmin=int(self.Zkeys[0]),
                               vmax=float(self.Zkeys[-1]))
        plt.colorbar(im, ax=ax)
        ax.set(xlabel='log10({0:s})'.format(xlab),
                      ylabel='log10(T (eV))',
                      title='Zbar')
        
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
                                c=colors[Tidx,rhoidx, i, cond, :], # Shape T, rho, Zbar, hnu, RGBA
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
        axmi = fig.add_axes([0.7, 0.005, 0.1, 0.04]) # axmi = "Axes minus i" – i-decrementing axes
        axpi = fig.add_axes([0.81, 0.005, 0.1, 0.04]) # axpi = "Axes plus i" – i-incrementing axes
        bpi = Button(axpi, 'T+') # bpi - i-inrecementing button
        bpi.on_clicked(callback.inc_i)
        bmi = Button(axmi, 'T-')
        bmi.on_clicked(callback.dec_i)

        axmj = fig.add_axes([0.15, 0.005, 0.1, 0.04]) # Axes minus i
        axpj = fig.add_axes([0.26, 0.005, 0.1, 0.04])
        bpj = Button(axpj, 'rho+')
        bpj.on_clicked(callback.inc_j)
        bmj = Button(axmj, 'rho-')
        bmj.on_clicked(callback.dec_j)
        return bpi, bmi, bpj, bmj
    
    def plot_pop_hist_bar(self, Tidx, rhoidx, bins, ax=None):
        
        if ax is None:
            ax = plt.gca()
            
        ax.hist(self.hnuarrs.flatten(), weights=self.pstate_rho[Tidx, rhoidx].flatten(),
                 bins=bins, color='w', edgecolor='k')
    
    def plot_pop_hist_trace(self, rhoidx, bins):
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
    
    def gif_pop_hist_bar(self, Ts, rhos, bins, Zbar_plot=None, 
                         xbds=[None,None], ybds=[None,None],
                         savename='./hist_bar.gif'):
        import imageio

        if Zbar_plot is None:
            Zbar_plot = self.Zkeys

        frames = []
        files = []
        for Tidx, rhoidx in zip(Ts,rhos):
            plt.figure(figsize=[5,3])
            self.plot_pop_hist_bar(Tidx, rhoidx, bins)
            
            plt.gca().set(xlabel=r'h$\nu$ (eV)',
                          ylabel='Population',
                          xlim=xbds,
                          ylim=ybds,
                          title=r'T = {0:0.0f} eV, $\rho$ = {1:0.1f} g/cm$^3$'.format(self.KT[Tidx],
                                                                                     self.rho_grid[rhoidx]))
            
            # Save frame to movie stack
            files.append(f'./img_{Tidx}_{rhoidx}.png')
            plt.savefig(f'./img_{Tidx}_{rhoidx}.png', 
                        transparent = False,  
                        facecolor = 'white'
                       )    
            plt.close()
        for f in files:
            im = imageio.imread(f)
            frames.append(im)
            if os.path.isfile(f):
                os.remove(f)

        
        imageio.mimsave(savename, # output gif
                frames,          # array of input frames
                fps = 5)         # optional: frames per second

   
    def gif_pops(self, Ts, rhos, Zbar_plot=None, scale='lin', cmap_name='rainbow',
                 log_vminmax=[-3,0], xaxis='Zbar', savename='./example.gif'):
        import imageio

        if Zbar_plot is None:
            Zbar_plot = self.Zkeys

        cmap = mpl.cm.get_cmap(cmap_name)
        if scale=='log':
            # Set min/max of colorbar
            norm = mpl.colors.Normalize(vmin=log_vminmax[0], vmax=log_vminmax[1]) # Converts value to linearly interpolte [0,1] between bounds
            
            # Calculate colorbars of populations
            colors = cmap(norm(np.log10(self.pstate_rho)))
            cblab = r'log$_{10}$(State population)'
            
        elif scale=='lin':
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            colors = cmap(norm(self.pstate_rho))
            cblab = r'State population'
            
            
        frames = []
        files = []
        for Tidx, rhoidx in zip(Ts, rhos):
            plt.figure(figsize=[5,3])
            for i in range(len(Zbar_plot)):
                # Parse values for current charge state
                Z = int(Zbar_plot[i])
                exc = self.excarrs[i]
                hnu_i = self.hnuarrs[i]
                p = self.pstate_rho[Tidx, rhoidx, i]
                
                if xaxis=='Zbar':
                    x = np.array([Z]*len(hnu_i))
                    xlab = r'$\bar{Z}$'
                elif xaxis=='Zbar+exc':
                    x = Z+exc - 0.1*exc
                    xlab = r'$\bar{Z} + 0.9 $\times$ Excitation Degree'
                cond = hnu_i>0
                plt.scatter(x[cond],
                                    hnu_i[cond],
                                    c=colors[Tidx,rhoidx, i, cond, :],
                                    # norm=norm,
                                    alpha=1,
                                    s=10, # Marker size
                                    label=exc)
                
            plt.gca().set(xlabel=xlab,
                          ylabel=r'h$\nu$ (eV)',
                          title=r'T = {0:0.0f} eV,  $\rho$ = {1:0.1f} g/cm$^3$'.format(self.KT[Tidx],self.rho_grid[rhoidx]))
            cax = plt.gcf().add_axes([0.2, 0.8, 0.5, 0.05])
            cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm = norm,
                                            orientation='horizontal',)
            cb.set_label(cblab,)
            
            # Save frame to movie stack
            files.append(f'./img_{Tidx}_{rhoidx}.png')
            plt.savefig(f'./img_{Tidx}_{rhoidx}.png', 
                        transparent = False,  
                        facecolor = 'white'
                       )    
            plt.close()
        
        for f in files:
            im = imageio.imread(f)
            frames.append(im)
            if os.path.isfile(f):
                os.remove(f)

        imageio.mimsave(savename, # output gif
                frames,          # array of input frames
                fps = 5)         # optional: frames per second
        
        return
    
    def print_table(self):
        ''' Prints formatted table of transition energies, oscillator strength, and upper/lower states
        

        Returns
        -------
        None.

        '''
        
        # Get weighted oscillator strengths
        gf = self.get_gf(1,0,2,1)
        
        print('{0:5s} | {1:10s} | {2:10s} | {3:25s} | {4:25s} '.format(
                'Zbar', 'hnu (eV)', 'gf', 'lo cfg', 'up cfg'))
        for i,Zstr in enumerate(self.Zkeys):
             # i,j = np.unravel_index(idx, ad.hnuarrs.shape)
             
             # Format configurations
             lo_pops = []
             [lo_pops.extend(item) for item in self.Pn['lo'][Zstr].values()]
             # breakpoint()
        
             up_pops = []
             [up_pops.extend(item) for item in self.Pn['up'][Zstr].values()]
             
             for j in range(len(lo_pops)):
                 lo = lo_pops[j]
                 up = up_pops[j]
                 
                 # Keep only shells with non-zero population
                 lo_config = ''.join(['{0:d}*{1:0.0f} '.format(shell+1, pop) for shell, pop in enumerate(lo) if pop])
                 up_config = ''.join(['{0:d}*{1:0.0f} '.format(shell+1, pop) for shell, pop in enumerate(up) if pop])
                 
                 print('{0:5s} | {1:10.1f} | {2:10.2e} | {3:25s} | {4:25s} '.format(Zstr, self.hnuarrs[i,j], gf[i,j], lo_config, up_config))
                
            

        

# %% Main
if __name__=='__main__':
        
    # ZZ = 26 # Nuclear charge
    # A = 55.845 # Nucleon number
    
    ZZ = 24 # Nuclear charge
    A = 51.996 # Nucleon number

    Zbar_min = ZZ - 10
    nmax = 5 # Maximum allowed shell
    exc_list = [0,1,2,3] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
    # exc_list = [0,1] # Excitation degrees to consider (lower state is ground state, singly excited, ...)
    pf = 1
    
    # Run model
    ad = AtDat(ZZ, A, Zbar_min, nmax, exc_list,)
    ad.get_atomicdata(vb=0)
    ad.get_hnu(np.array(ad.Zkeys).astype(int))
    ad.tidy_arrays('lo')
    
    # Define T, Ne, rho grids for SB
    # Nn, NT = 10, 11 # Number of density, temperature gridpoints
    # KT = np.logspace(1.5,3, num=NT) # eV, Temperature range, 

    Nn, NT = 10, 51 # Number of density, temperature gridpoints
    # KT = np.logspace(1.8,2, num=NT) # eV, Temperature range, 
    # KT = np.linspace(70,200, num=NT) # eV, Temperature range, 
    KT = np.linspace(50,1000, num=NT) # eV, Temperature range, to find IPD

    rho0 = np.logspace(-1,2, num=Nn) # g/cc
    Zbar0 = 20 # Estimated dZbar
    NE = rho0 / (A*ad.mp) * Zbar0 # 1/cm^3, Ne range
    
    Nrho = 12
    rho_grid = np.logspace(-1,1, num=Nrho)
    
    # Run Saha-Boltzmann
    ad.saha_boltzmann(KT, NE, IPD=0)
    ad.saha_boltzmann_rho(rho_grid)
    
    # Generate spectra
    # ad.append_lineshape(np.ones(ad.pstate_rho.shape), 'G')
    ad.append_lineshape(3*np.ones(ad.pstate_rho.shape), 'G')
    # ad.append_lineshape(np.ones(ad.pstate_rho.shape), 'L')
    # ad.append_lineshape(np.ones(ad.pstate_rho.shape), 'L')
    ad.sum_linewidths()
    linecenter = ad.get_linecenter()
    
    ad.get_line_opacity(1, 0, 2, 1)
    
    hnu_minmax = [ad.hnuarrs.flatten()[ad.hnuarrs.flatten()>0].min(),
                  ad.hnuarrs.max()]
    hnu_axis = np.linspace(5400, 5800, num=2000)
    # hnu_axis = np.linspace(6400, 6800, num=2000)
    # hnu_axis = np.linspace(6665, 6680, num=1000)
    # ls = ad.generate_lineshapes(hnu_axis)
    
    # ad.generate_spectra(hnu_axis)
    
    ad.print_table()
    
    # Gifs
    gifT = np.arange(0,len(KT))
    gifrho = np.ones(len(KT), dtype=int)*-1
    
    bins = np.arange(5400, 5800, 5)
    # ad.gif_pops(Ts=gifT, rhos=gifrho, scale='log')
    
    # f0 = mpl.rcParams['font.size']
    # mpl.rcParams['font.size'] = 14
    # ad.gif_pop_hist_bar(Ts=gifT, rhos=gifrho, bins=bins,
    #                     ybds=[0,1],
    #                     savename='./hist_bar_test.gif')
    # mpl.rcParams['font.size'] = f0

    
    if pf:
        # Plot Saha at one condition
        plt.figure()
        plt.bar(x=ad.Zkeys, height=ad.psaha_rho[26,-1,:-1], color='w', edgecolor='k')
        plt.gca().set(xlabel='Ionization',
                      ylabel='Population')
        
        # Plot transitions – He to B only
        # ad.plot_hnu([0], Zbar_plot=['24'], xaxis='Zbar')
        Zbar_plot = [ZZ-2,ZZ-3,ZZ-4,ZZ-5]
        ad.plot_hnu([0,3], Zbar_plot=Zbar_plot, show_colorbar=False)
        plt.gcf().set(figwidth=3,
                      figheight=2.8)
        plt.gcf().axes[0].set(xlim=[ZZ-5.2,ZZ-1.8],
                       ylim=[5600,5800],
                      xticks=Zbar_plot,
                      )
        # Plot transitions – All available
        # ad.plot_hnu([0], Zbar_plot=['24'], xaxis='Zbar')
        ad.plot_hnu([0,3], Zbar_plot=None, show_colorbar=False)
        plt.gcf().set(figwidth=4,
                      figheight=3)
        plt.gcf().axes[0].set(#xlim=[ZZ-5.2,ZZ-1.8],
                       ylim=[5425,5800],
                      )
        
        # Plot Gif of population flow
        # gifT = np.arange(0,len(KT))
        # gifrho = np.ones(len(KT), dtype=int)*-1
        # ad.gif_pops(gifT, gifrho, scale='log', xaxis='Zbar', savename='cr_gif_pops.gif')
        
        
        # Plot ionization at one rho
        rho_idx = 0
        fig, ax = plt.subplots(figsize=[4,3])
        ax.plot(ad.KT, ad.Zbar_rho[:,rho_idx])
        ax.set(xlabel='T (eV)',
               ylabel='Zbar',
               title='rho={0:0.1e} g/cm^3'.format(ad.rho_grid[rho_idx]))
        
        # Plot opacity at one rho
        rhoidx = -1
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

        sys.exit('here')

        # Plot opacity at one T - not constant due to changing state populations
        Tidx = NT//2 # Something in the middle
        fig, ax = plt.subplots(figsize=[4,3])
        im = ax.pcolormesh(ad.rho_grid, hnu_axis, ad.kappa[Tidx,:,:].T, shading='nearest')
        ax.set(xlabel=r'$\rho$ (g/cm$^3$)',
               ylabel=r'h$\nu$ (eV)',
               title='rho={0:0.1e} g/cm^3'.format(ad.rho_grid[rho_idx]))
        plt.colorbar(im, ax=ax)
        
        # Plot line opacity versus rho for one T, divided by state populations
        Zidx = 0 # Charge state
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
        
        fig, ax = plt.subplots(figsize=[4,3])
        ad.plot_hnu([0,3], xaxis='Zbar+exc', fig=fig, ax=ax)
        
        ad.plot_zbar('ne')
        ad.plot_rho_interp()
        buttons = ad.plot_sb_buttons()
        
        bins = np.linspace(6400, 6800) # Full spectrum
        bins = np.linspace(6515,6535, num=6) # N-like complex
        ad.plot_pop_hist_trace(0, bins=bins)
        
        

