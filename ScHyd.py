#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 09:21:54 2022

@author: dbis

Screened Hydrogenic model from Atzeni and Meyer-ter-vehn Chapter 10.
Capable of calculating atomic data for isolated atoms and prescribed shell populations,
or calculating charge state as a function of plasma conditions in an average ion model.
"""
# Python modules
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import pdb                      # Python debugging
import pandas as pd             # Used to import multi-column datasets
import os                       # Used to e.g. change directory
import sys                      # Used to put breaks/exits for debugging
from scipy import signal        # Used to apply median filter to smooth before interpolating loaded data
import itertools as it

#################

class AvIon():
    ''' Screened Hydrogenic average ion model with shell structure only (no term-splitting).
        Can be used generate basic atomic data for isolated atoms (e.g. ionization potential)
        as well as for ions in a dense plasma via formalisms for continuum lowering,
        pressure ionization, chemical potential.
        
        To-do: add radiative properties
    '''
    def __init__(self, Z, Zbar, A=None, nmax=10):
        
        # Constants - lengths in cm, masses in grams, energies in J
        self.a0 = 5.291772e-09 # Bohr radius, cm
        self.mp = 1.6726231e-24 # Proton mass, grams
        self.hbar = 1.0545606529268985e-34 # J*s
        self.me = 9.1093897e-28 # grams

        
        self.Z = Z
        self.Zbar = Zbar
        self.A = A
        self.nmax = nmax
        self.Ea = 27.2 # eV
        self.dEC = 0 # Assume no CL from the outset
        
        self.n = np.arange(1,nmax+1) # Define number of shells considered
    
        # Load screening coefficients
        fn = '/Users/dbis/Documents/More1982_ScreeningCoeff.xlsx'
        sc = pd.read_excel('/Users/dbis/Documents/More1982_ScreeningCoeff.xlsx',
                           skiprows=1, nrows=10, usecols=np.arange(11), header=None, index_col=0)
        sc = sc.values
        #print(sc)
        
        # Convert to Atzeni's notation
        np.fill_diagonal(sc, 1/2*sc.diagonal())
        self.sc = sc[:nmax,:nmax].T # Transpose required to match Fig. 10.1 in Atzeni
    
    def get_Pn(self, kT=None,C=None):
        ''' Calculates shell populations Pn. 
        
            Maximum electron number enforced by normalizing sum(Pn) to the nuclear charge
            when sum(Pn) exceeds the nuclear charge. Removes many discontinuities in Zbar from 
            previous versions.
        
        Parameters
        ----------
        kT : float
            Temperature in eV. If None, fill lowest shells first (for isolated atom)
        C : float
            Step size towards new value of Pn, i.e. NEW = (1-C)*OLD + C*(CALCULATED)
        '''
        
        if kT is None: # Run isolated atom case – fill lowest shells first
            Pn_full = 2*self.n**2 # Shell populations if all full
            epop_full = np.cumsum(Pn_full) # Total electrons inside and interior to shell
            nmax_idx = np.where(epop_full>(self.Z - self.Zbar))[0][0] # First shell to not be full
            
            # Start with empty shells
            Pn = np.zeros_like(Pn_full)
            Pn[:nmax_idx] = Pn_full[:nmax_idx] # Assign full shells their proper value
            Pn[nmax_idx] = (self.Z - self.Zbar) - np.sum(Pn) # Populate first unfilled shell with remaining electrons
            
            self.Pn = Pn
        else: # Run dense plasma case – populate according to Fermi distribution
            Pn = self.gn/(1+np.exp((self.En+self.dEc-self.mu)/kT))
            
            # If more electrons are assigned than available, reduce all populations equally
            Pn *= 1/max(Pn.sum()/self.Z, 1) # Divide by 1 or the over-assignment of electrons
            if C:
                # breakpoint()
                self.Pn = (1-C)*self.Pn + C*Pn
            else:
                self.Pn = Pn
        return
            
    def get_Qn(self, C=None):
        ''' Calculates effetive charge of shell n after screening of nuclear Z
            by inner electrons and electons in the same shell
        
        Parameters
        ----------
        C : float
            Step size towards new value of Pn, i.e. NEW = (1-C)*OLD + C*(CALCULATED)
        '''
        
        # Calculate sum over sigma_nm*P_m, for m<=n (sum over column, for column <= row)
        MAT  = self.sc * self.Pn[np.newaxis,:]
        SUM = np.sum(np.tril(MAT), axis=1) # tril takes lower triangle (column<=row), axis=1 sums over columns
        Qn = self.Z - SUM
        
        if C:
            self.Qn = (1-C)*self.Qn + C*Qn
        else:
            self.Qn = Qn
        return
    
    def get_Wn(self, C=None):
        ''' Parameters
            ----------
            C : float
                Step size towards new value of Pn, i.e. NEW = (1-C)*OLD + C*(CALCULATED)
        '''  

        MAT  = (self.Pn*self.Qn/self.n**2)[:,np.newaxis] * self.sc 
        
        # Calculate sum for MAT_mn, m>=n (sum over row, for row >= column)
        SUM = np.sum(np.tril(MAT), axis=0) # tril takes lower triangle (row>=column), axis=0 sums over rows
        
        if C:
            self.Wn = (1-C)*self.Wn + C*SUM
        else:
            self.Wn = SUM
        return
    
    def get_En(self, C=None):
        '''Calculates energy levels. Needs Wn, Qn
        
            Parameters
            ----------
            C : float
                Step size towards new value of Pn, i.e. NEW = (1-C)*OLD + C*(CALCULATED)
        '''
        
        En = self.Ea*(self.Wn - self.Qn**2/2/self.n**2)
        if C:
            self.En = (1-C) * self.En + C*En
        else:
            self.En = En
        return
    
    def get_Zion(self):
        print(self.Z - np.sum(self.Pn))
        return
    
    def get_Etot(self):
        '''Calculates total energy of ion. Needs Qn, Pn.
            Uses Atzeni's Equation for Ei0 in ionization potentials,
            and applies to any complex
        '''
        self.Etot = self.Ea * np.sum(-self.Qn**2/2/self.n**2 * self.Pn)
        return
    
    def get_statweight(self):
        '''Calculates shell-resolve dstatistical weight of the given complex with populations self.Pn
            Stat. weight of given complex is the product of stat. weight
            of each shell, np.prod(self.statweight).
            For stat.weights of transitions, some configurations will not be allowed,
            and will need to be subtracted from the total statweight.
            Ex: For 1s-2p absorption, the lower state cannot have a full 2p orbital,
            as is the case for one configuration implied by 2^6. That configuration's
            statweight (g=1) must be subtracted from shell's stat weight before multiplication.
        Returns
        -------
        None.

        '''
        self.statweight = sp.special.comb(2*self.n**2, self.Pn)
        return 
        
        
        
    #### Functions for dense plasma
    def get_gn(self, rho, C=None, CLmodel=None):
        '''Calculates gn from Zimmerman's empirical pressure ionization. \
            Calculates and saves Hydrogenic orbitaltpta radius Rn as a prerequisite

            Parameters
            ----------
            C : float
                Step size towards new value of Pn, i.e. NEW = (1-C)*OLD + C*(CALCULATED)
                
            CLmodel : str or None
                Continuum lowering model used. If None, use 2n^2 occupancy. \
                For all continuum lowering models, reduce 2n^2 by Zimmerman's pressure ionization
        
        '''
        
        self.Rn = self.a0 * self.n**2/self.Qn # cm
        if CLmodel is None:
            # If CL model is None, don't reduce occupancies
            gn = 2*self.n**2
            
        else:
            # If a CL model is used, reduce occupancies with Zimmerman's pressure ionization
            a, b = [3, 4] # Parameters for Zimmerman's Pressure Ionization occupation function
            R0 = (3*self.A*self.mp/4/np.pi/rho)**(1/3) # cm

            gn = 2*self.n**2 / (1+ (a*self.Rn/R0)**b)
        
        if C:
            self.gn = (1-C)*self.gn + C*gn
        else:
            self.gn = gn
        return
    
    def get_CL(self, rho, model='IS Atzeni', C=None):
        '''Calculates reduction in free energy due to continuum lowering

            Parameters
            ----------
            rho : float
                Mass density in g/cc
            C : float
                Step size towards new value of Pn, i.e. NEW = (1-C)*OLD + C*(CALCULATED)
        '''
        e_squared = 1.44e-9 * 100 # eV*cm. x100 converts from m to cm
        R0 = (3*self.A*self.mp/4/np.pi/rho)**(1/3) # cm
        
        if model=='IS Atzeni': # Ion-sphere CL presented in Atzeni Ch. 10
            dEc = 3/2 * self.Zbar * e_squared/R0
            
        elif model=='IS Hansen': # Ion-sphere (high-density Stewart-Pyatt) CL presented in Hansen HEDP 2017
            dEc = 3/2 * (self.Zbar+1) * e_squared/R0  
            
        elif model=='EK Hansen': # Ecker-Kroll CL presented in Hansen HEDP 2017
            dEc = (self.Zbar+1)**(4/3) * e_squared/R0
            
        elif model==None: # No CL. Emulates isolated atom ionization, Saha balance
            dEc = 0
            
        if C:
            self.dEc = (1-C)*self.dEc + C*dEc
        else:
            self.dEc = dEc
        return

    def get_mu(self, kT,rho, C=None):
        '''Parametrized form of mu given by Atzeni 10.35

            Parameters
            ----------
            kT : float
                Temperature in eV
            rho : float
                Mass density in g/cc
            C : float
                Step size towards new value of Pn, i.e. NEW = (1-C)*OLD + C*(CALCULATED)
        '''
        A = 0.250254
        B = 0.072
        b = 0.858
        
        # Calculate T_Fermi in eV, converting me to kg and rho/mp to 1/m^3
        kTFermi = (self.hbar**2/2/(self.me/1e3) * \
                   (3*np.pi**2* (rho*1e6)/self.A/self.mp*self.Zbar)**(2/3)
                    ) / 1.602e-19 # eV
        TH = kT/kTFermi
        
        mu = kT * (-3/2 * np.log(TH) + np.log(4/3/np.sqrt(np.pi))
                        + (A * TH**( -(b+1) ) + B * TH**( -(b+1)/2 )) / (1 + A * TH**(-b))
                       )
        if C:
            self.mu = (1-C)*self.mu + C*mu
        else:
            self.mu = mu
        return
    
    def get_Zbar(self, C=None):
        '''
            Parameters
            ----------
            C : float
                Step size towards new value of Pn, i.e. NEW = (1-C)*OLD + C*(CALCULATED)
        '''
        cond = np.where(self.En<self.dEc)[0] # Bound states with energy less than continuum
        SUM = np.sum(self.Pn[cond])
        
        Zbar = self.Z - SUM
        
        if C:
            self.Zbar = (1-C)*self.Zbar + C*Zbar
        else:
            self.Zbar = Zbar
            
    def get_abs_osc_str(self, ni, li, nj, lj):
        ''' Calculates absorption oscillator strength of the given nl transition,
        summed over all initial and final states given shell populations.
            Due to the double sum, this gives a statistically weighted osc. str. gf
            
            Current populations are taken to be the lower state.
            
        
        Parameters
        ----------
            ni, li : int
                Initial principal and angular quantum numbers
            nj, lj : int
                Final principal and angular quantum numbers
                
        '''
        
        # Get hydrogenic oscillator strength for given transition. From Table
        faH_dict = {'10':{'21':0.4162, # Key format: ni li, nj lj. ['10']['21'] is 1s -> 2p
                     '31':0.0791,
                     '41':0.0290,
                     '51':0.0139,
                     '61':0.0078,
                     '71':0.0048,
                     '81':0.0032
                     },
               }
        faH = faH_dict['{0:d}{1:d}'.format(ni, li)]['{0:d}{1:d}'.format(nj, lj)]
                     
        
        # Range of possible initial and final configuration (nl-resolved) occupations
        # Are wi and wj guaranteed to be same length?
        # Are wi and wj independent given only the Pn? Yes if delta_n != 0
        wi = np.arange(max(0, self.Pn[ni-1] - 2*ni**2 + (4*li + 2)),
                       min(self.Pn[ni-1], 4*li + 2)+1) # +1 for inclusive of endpoint
        
        wj = np.arange(max(0, self.Pn[nj-1]+1 - 2*nj**2 + (4*lj + 2)),
                       min(self.Pn[nj-1]+1, 4*lj + 2)+1) # Pn+1 because final state has one more electron in nj lj compared to initial state
        
        # Given a set of Pn, wi and wj are independent. Mesh product
        wi, wj = np.meshgrid(wi, wj)
        
        Fa_n = np.sum(wi*(4*lj + 3-wj)/(4*lj + 2) * faH) # Osc. str. summed over final levels, and summed over initial states. Modified from Griem (3.31)
        
        self.fsum = Fa_n # THIS is gf --- check notes
        return
        

def get_ionization(Z, return_energy_levels=False):
    ''' Calculates ionization potential for isolated atoms.
    '''
    
    Ei0 = []
    En = []
    
    for Zbar in range(Z+1):
        sh = AvIon(Z,Zbar)

        sh.get_Pn()
        sh.get_Qn()
        Ei0.append(sh.Ea * np.sum(-sh.Qn**2/2/sh.n**2 * sh.Pn))
        
        if return_energy_levels:
            sh.get_Wn()
            sh.get_En()
            En.append(sh.En)
    if return_energy_levels:
        return np.diff(Ei0), np.array(En)
    else:
        return np.diff(Ei0)

def dense_plasma(Z,Zbar,A,Ts, rhos, nmax=10, iter_scheme='123', step=[0.5,0.5,0.5],
                 CL='IS Atzeni',
                 vb=0, pf=1, return_last=False,):
    ''' Calculates Zbar for a plasma along the given temperatures and densities.
        Iteration and convergence options are available.

    Parameters
    ----------
    Z, A: floats
        Nuclear charge and mass number
    Zbar : int, float, array, or 'auto'
        Initial guesses for charge state.
            If int or float, all conditions are initialized at the same Zbar\n
            If array, must be shape [Ts, rhos] and each condition is given its own initial guess\n
            If 'auto', an empirical fit to the TF model is used.
    Ts : 1-D array
        Array of temperatures in eV.
    rhos : 1-D array
        Arry of densities in g/cc.
    nmax : int
        Highest principal quantum number to consider. Maximum allowed is 10.
    iter_scheme : str, optional
        Sets which loops are iterated. The default is '123', iterating over all three loops.
            '1': g-P-Q\n
            '2': P-Q-W-E\n
            '3': Zbar
        Ex: If '12', only loops 1 and 2 are iterated. If ' ', no iteration.
    step : list, optional
        Determines how quickly the iteration converges for each loop.
        New values are NEW = (1-C)*OLD + C*(CALCULATED).
        The default is [0.5,0.5,0.5].
    CL : str or None
        Denotes Continuum Lowering model to use. Options: {'IS Atzeni', 'IS Hansen', 'EK Hansen', None}. \
        If None, CL (IPD) is set to 0, and Zimmerman's pressure ionization occupancy factor is ignored,
        i.e. gn is just 2n^2.
    vb : bool
        Verbose flag. If True, outputs extra information to console.
    pf : bool
        Plot flag. If True, plots Zbar.
    return_last : bool
        If True, return last instance of AvIon object. Corresponds to Ts[-1], rhos[-1]

    Returns
    -------
    Zgrid : 2-D array
        Array of calculated Zbar. Shape is [T,rho]
    Engrid, Pngrid, Rngrid, Wngrid, Qngrid: 3-D arrays
        Arrays of calculated energy levels, shell populations,
        hydrogenic orbital radii, outer screening, and 
        screening from inner and same shells. Shapes are [nmax,T,rho]
    CLgrid : 2-D array
        Array of continuum lowering values, in eV. Shape is [T,rho]

    '''
    title_dict = {'1':'g-P-Q',
                  '2':'P-Q-W-E',
                  '3':'Zbar'}
    TITLE = 'Iterating '
    TITLE = [title_dict[s] for s in iter_scheme]
    TITLE = 'Iterating ' + ', '.join(TITLE)
    
    nT, nrho = len(Ts), len(rhos)
    
    # Initialize Zbar
    if (type(Zbar) is float) or (type(Zbar) is int):
        Zgrid = Zbar * np.ones([nT,nrho])
    elif Zbar=='auto':
        # Estimate Zbar uisng empirical fit to TF from Atzeni
        Zgrid = get_TF_Zbar(Ts,rhos, Z, A)
    else:
        Zgrid = Zbar
        
    # Intitialize other output grids
    Engrid = np.zeros([nmax, nT, nrho])
    Pngrid = np.zeros([nmax, nT, nrho])
    Rngrid = np.zeros([nmax, nT, nrho])
    Wngrid = np.zeros([nmax, nT, nrho])
    Qngrid = np.zeros([nmax, nT, nrho])
    CLgrid = np.zeros([nT, nrho])
    
    for ij,Trho in enumerate(it.product(Ts,rhos)):
        # Unwrap T,rho indices and values
        Tidx,rhoidx = np.unravel_index(ij, [nT,nrho])
        kT,rho = Trho
        # print(Tidx,rhoidx, kT,rho)
        
        # Initialize with isolated atom
        dp = AvIon(Z=Z, Zbar=Zbar, A=A, nmax=nmax)
        dp.get_Pn()
        dp.get_Qn()
        dp.get_Wn()
        dp.get_En()
        
        # General iteration scheme
        zfrac = 1 # Initialize to trigger while loop
        zct = 0 # Number of times Zbar loop is iterated
        while (zfrac > 1e-3):
            ''' Zbar depends on EVERYTHING, and is an ENCOMPASSING loop
            Iterate until fractional standard deviation of old and new Z's
            is less than 0.1%
            If iteration on Zbar is not desired, just set zfrac=0 before end of loop
            '''
            zct+=1
            z = dp.Zbar

            dp.get_CL(rho, model=CL)
            dp.get_mu(kT,rho)
            dp.get_gn(rho, CLmodel=CL)
            
            dp.get_Pn(kT)
            dp.get_Qn()

            if '1' in iter_scheme:
                ''' g,P,Q depend on each other cyclically. Only P and Q are called by other quantities.
                Iterate until fractional standard deviation of old and new P's and Q's
                are less than 0.1%
                '''
                pfrac, qfrac = 1, 1 # Set to trigger while loop on first pass
                cond = True
                ct = 0
                # while (pfrac > 1e-3) and (qfrac > 1e-3):
                # while cond:
                while (pfrac > 1e-3) or (qfrac > 1e-3):
                    ct+=1

                    gn, pn, qn = dp.gn, dp.Pn, dp.Qn
                    dp.get_gn(rho, C=step[0], CLmodel=CL)
                    dp.get_Pn(kT, C=step[0])
                    dp.get_Qn(C=step[0])
                    
                    # Repeat if variance of differences is too large
                    pfrac = np.var((dp.Pn-pn)/pn)**(1/2)
                    qfrac = np.var((dp.Qn-qn)/qn)**(1/2)
                    
                    # Repeat if ANY difference is too large. 
                    # pfrac = (dp.Pn-pn)/pn
                    # qfrac = (dp.Qn-qn)/qn
                    # cond = np.any(np.greater([pfrac,qfrac],1e-3)) # Repeat condition
                    
                    
                    if vb and ct==30:
                        print('gPQ failed at (kT={0:0.3f},rho={1:0.3f})'.format(kT, rho))
                    if vb and (ct>30):
                        print('Pn: ', pfrac)
                        print('Qn: ', qfrac)
                    if ct>40:
                        break

            dp.get_Wn()
            dp.get_En()
            
            if '2' in iter_scheme:
                ''' P,Q,W,E depend on each other cyclically. Only P and E are called by other quantities.
                Iterate until fractional standard deviation of old and new P's and E's
                are less than 0.1%
                '''
                pfrac, efrac = 1, 1 # Set to trigger while loop on first pass
                ct = 0
                while (pfrac > 1e-3) and (efrac > 1e-3):
                # while (pfrac > 1e-3) or (efrac > 1e-3):
                    ct+=1
                    
                    # Save previous results
                    pn, en = dp.Pn, dp.En
                    
                    # Get new results
                    dp.get_Pn(kT,C=step[1])
                    dp.get_Qn(C=step[1])
                    dp.get_Wn(C=step[1])
                    dp.get_En(C=step[1])
                    
                    
                    pfrac = np.var((dp.Pn-pn)/pn)**(1/2)
                    efrac = np.var((dp.En-en)/en)**(1/2)
                    
                    if vb and ct==30:
                        print('  PQWE failed at (kT={0:0.3f},rho={1:0.3f})'.format(kT, rho))
                    if vb and (ct>30):
                        print('Pn: ', pfrac)
                        print('En: ', efrac)
                    if ct>40:
                        
                        break
            
            dp.get_Zbar(C=step[2])
            if dp.Zbar<0:
                dp.Zbar = 1e-99
        
            if '3' in iter_scheme:
                zfrac = abs((dp.Zbar-z)/z)
                # if int(dp.Zbar)==13:
                #     pdb.set_trace()
                
                if vb and zct==30:
                    print('    Zbar failed at (kT={0:0.3f},rho={1:0.3f})'.format(kT, rho))
                if vb and (zct>30):
                    print('    Zvar: ', zfrac)
                if zct>40:
                    break
            else:
                break # Skips while loop
                        
        Zgrid[Tidx,rhoidx] = dp.Zbar
        Engrid[:,Tidx,rhoidx] = dp.En
        Pngrid[:,Tidx,rhoidx] = dp.Pn
        Rngrid[:,Tidx,rhoidx] = dp.Rn
        Wngrid[:,Tidx,rhoidx] = dp.Wn
        Qngrid[:,Tidx,rhoidx] = dp.Qn
        CLgrid[Tidx,rhoidx] = dp.dEc

    if pf: # option to plot
        if len(rhos)==1:
            plt.figure(figsize=[4,3])
            plt.semilogx(Ts, Zgrid.flatten(), '.')
            plt.title(TITLE)
            plt.xlabel('kT (eV)')
            plt.ylabel('Charge state')
        else:
            plt.figure(figsize=[4,3])
            plt.imshow(Zgrid, aspect='auto')
            plt.colorbar()
            
            xt = plt.xticks()[0]
            xt = xt[(xt>=0) * (xt<len(rhos))]
            xl = ['{0:0.1e}'.format(rhos[int(s)]) for s in xt]
            plt.xticks(xt, labels=xl, rotation=45)
    
            yt = plt.yticks()[0]
            yt = yt[(yt>=0) * (yt<len(Ts))]
            yl = ['{0:0.1e}'.format(Ts[int(s)]) for s in yt]
            plt.yticks(yt, labels=yl, rotation=45)
            
            plt.xlabel(r'$\rho$ (g/cc)')
            plt.ylabel(r'$T_e$ (eV)')
    
            plt.gca().invert_yaxis()
    
    # pdb.set_trace()
    if return_last:
        return Zgrid, Engrid, Pngrid, Rngrid, dp
    else:
        return Zgrid, Engrid, Pngrid, Rngrid, Wngrid, Qngrid, CLgrid
           

def get_TF_Zbar(T,rho, Z,A):
    ''' Calculates charge state Zbar from an empirical fit to Thomas-Fermi model for any 
        ion at any temperature and density.
    
        2/1/22 – Not working well. Doesn't reproduce TF curve in Fig. 10.2
        
    Parameters
    ----------
    T,rho : arrays
        Temperature (eV) and density (g/cc) arrays. Must be same size

    Returns
    -------
    Zbar : array
        Array of calculated charge state.

    '''
    
    # Constants
    alpha, beta = 14.3139, 0.6624
    a1, a2, a3, a4 = 0.003323, 0.9718, 9.26148e-5, 3.10165
    b0, b1, b2 = -1.7630, 1.43175, 0.31546
    c1, c2 = -0.366667, 0.983333
    
    # Calculate intermediate terms
    rho1 = rho/(A*Z)
    T1 = T / (Z)**(4/3)
    Tf = T1 / (1+T1)
    A = a1*T1**a2 + a3*T1**a4
    B = -np.exp(b0 + b1*Tf + b2*Tf**7)
    C = c1*Tf + c2
    Q1 = A*rho1**beta 
    Q = (rho1**C + Q1**C)**(1/C)
    x=alpha*Q**beta
    
    # Calculate Zbar
    Zbar = Z*x/(1+x+np.sqrt(1+2*x))
    
    return Zbar

# def saha(Z,A,rho,kT, pf=0):
#     ''' Calculate Saha distribution of plasma at given T,rho
#         Ported from old ScrHyd model. Not currently working.
#     '''
#     mp = 1.6726e-24 # grams
#     a0 = 5.291772e-11 # m
#     hbar = 1.0545606529268985e-34 # J*s
#     me = 9.109e-31 # kg
#     el = 1.60217662e-19 # C

#     ni = rho / (A*mp) * 1e6
    
#     # Get ionization potentials
#     Ei = get_ionization(Z)
    
#     # Parameters for  Pressure ionized DoS of shell n
#     a, b = 3,4
#     R0 = (3/4/np.pi/ni)**(1/3)

#     Gi = list() # Internal partition function of ion i
#     for i in range(Z+1): # 0 to Z inclusive
#         # Get energy levels and other quantities of ion Z-i
#         shi = AvIon(Z,i,A)
        
#         shi.get_Pn()
#         shi.get_Qn()
        
#         shi.get_Wn()
#         shi.get_En()

#         # Pressure ionized DoS of shell n
#         Rn = a0*shi.n**2/shi.Qn
#         gn = 2*shi.n**2 /(1+ (a*Rn/R0)**b)
#         tmp = np.sum(gn* np.exp(-(shi.En - shi.En[0])/kT) )
#         Gi.append(tmp)

#     Gi.reverse()   #Flip left/right to make neutral atom the first entry to agree with remainder of Saha CSD calcualtion
#     Gi = np.array(Gi)    

#     Grat = Gi[1:]/Gi[:-1]
    
#     # Calculate Zbar for T,rho
#     sh = dense_plasma(Z, Zbar=1, A=A, Ts=[kT], rhos=[rho], return_last=True, pf=0)
#     sh = sh[-1]
    
#     ne = ni*sh.Zbar
    
#     # Calculate ratio between ions: n(i+1)/n(i)
#     # Order of ratios is in order of ionization energy differences.
#     # Here, we use idx=0 to correspond to neutral atom. So, lowest POSITIVE du should be first
#     # du = np.flip(np.diff(Ei))
#     du = Ei
#     rats = 2/ne/ (2*np.pi*hbar**2/me/(kT*el))**(3/2) *Grat *np.exp(-du/kT) # Ratios between adjacent CS

#     # Calculate cumulative product of this. ith term is n(i)/n(0)
#     c = np.cumprod(rats)
    
#     # 1/(1 + SUM over cumulative product) is  n_tot / n(0) = f_0
#     f = (1+np.sum(c))**(-1)
#     # Then f_j = f0*c_j
#     f = np.append(f, f*c)  # Normalized CSD! First entry is neutral atom

# #    pdb.set_trace()
#     if pf:
#         plt.figure()
#         plt.bar(np.arange(Z+1), f)
#         plt.xlabel('Ion charge (0 = neutral atom)')
#     return f, sh


# %% Main
if __name__=='__main__':
    #### Examples
    if 0: # Ionization only. Compare against Fig. 10.1
        Z = 13
        Ii = get_ionization(Z)
        
        fig, axs = plt.subplots(1,2,sharey=True)
        axs[0].semilogy(np.arange(1,Z+1), Ii[-1::-1], '.')

        Z = 79
        Ii = get_ionization(Z)
        
        axs[1].semilogy(np.arange(1,Z+1), Ii[-1::-1], '.')
        axs[1].set_xlabel('Charge state')
        axs[0].set_ylabel('Ionization potential (eV)')
        axs[0].set_title('Al')
        axs[1].set_title('Au')
    
    if 0:
        Z = 24
        # isol = AvIon(36,0, nmax=10)
        isol = AvIon(Z,0, nmax=10)
        isol.get_Pn()
        isol.get_Qn()
        isol.get_Wn()
        isol.get_En()
        
        # Check results
        print('Pn: ', isol.Pn)
        print('Zbar: ')
        isol.get_Zion()
        print('En: ', isol.En)
        
        # Check ionization
        Ei0, En = get_ionization(Z, return_energy_levels=True)

        
        plt.figure()
        plt.semilogy(Z-np.arange(1,Z+1), Ei0[-1::-1], '.')
        # plt.gca().tick_params(which='minor', length=2)
        plt.xlabel('Charge state')
        plt.ylabel('Ionization Potential (eV)')
        
        plt.figure()
        plt.plot(np.arange(0,En.shape[0]-1), En[:-1,1:4]-En[:-1,0][:,np.newaxis],'.')
        
        CLOSED = np.array([2,10,18,36])
        [plt.axvline(e) for e in Z-CLOSED if e>0]
        
        # Es = [5414, 5947, 5932, 7016]
        # [plt.axhline(E) for E in Es]
        
    if 0: # Test oscillator strengths
        Z = 1
        # isol = AvIon(36,0, nmax=10)
        isol = AvIon(Z,Zbar=0, nmax=10)
        isol.get_Pn()
        isol.get_Qn()
        isol.get_Wn()
        isol.get_En()
        
        isol.get_abs_osc_str(1, 0, 2, 1)
        print(isol.fsum)
   
    
    #### Dense plasma      
    if 0: # Aluminum to reproduce Atzeni's Fig. 10.3      
        iter_scheme = '123' # Loop 1=gPQ, 2=PQWE, and/or 3=Z*
        Z, Zbar, A = 13,1,26.9
        Ts = np.logspace(0,3, num=21)
        rhos = np.logspace(-3,4, num=21)
        grids = dense_plasma(Z, Zbar, A, Ts, rhos,
                             iter_scheme=iter_scheme, step=[0.5,0.5,0.5])
    if 0: # Test different continuum lowering models      
        iter_scheme = '123' # Loop 1=gPQ, 2=PQWE, and/or 3=Z*
        Z, Zbar, A = 13,1,26.9
        Ts = np.logspace(0,3, num=21)
        rhos = np.logspace(-3,4, num=21)
        
        CLs = ['IS Atzeni',
               'IS Hansen',
               'EK Hansen']
        for CL in CLs:
            grids = dense_plasma(Z, Zbar, A, Ts, rhos,CL=CL,
                                 iter_scheme=iter_scheme, step=[0.5,0.5,0.5])
            plt.title(CL)
        
        
    if 1: # Test other Z
        iter_scheme = '123' # Loop 1=gPQ, 2=PQWE, and/or 3=Z*
        # Z, Zbar, A = 8, 1, 16 # O
        # Z, Zbar, A = 24, 1, 52 #   cCr
        # Z, Zbar, A = 32, 1, 72 # Ge
        # Z, Zbar, A = 6, 1, 12 # C
        # Z, Zbar, A = 1, 1, 1 # H
        # Z, Zbar, A = 13, 1, 27 # Al 
        # Z, Zbar, A = 47, 1, 107.87 # Ag
        Z, Zbar, A = 79, 1, 196.97 # Au
        
        # Standard grid
        Ts = np.logspace(0,3.5, num=20) # eV
        rhos = np.logspace(-4,4, num=21) # g/cc

        # Zaire grid
        # Ts = np.linspace(1e4,10e4, num=20)/11604 # V
        # rhos = np.linspace(1,10, num=21) # g/cc    
        
        # Single point
        # Ts = [500]
        # rhos = [50]
        
        grids = dense_plasma(Z, Zbar, A, Ts, rhos,
                             iter_scheme=iter_scheme, step=[0.5,0.5,0.5]) # Zbar, ...
        
        plt.figure(figsize=[3,2.5])
        plt.pcolormesh(rhos,Ts, grids[0], shading='nearest')
        plt.colorbar()
        plt.contour(rhos, Ts, grids[0], levels=[Z-36,Z-18,Z-10,Z-2], colors='grey')
        plt.gca().set(xscale='log',
                      yscale='log',
                      xlabel=r'$\rho$ (g/cm$^3$)',
                      ylabel= '$T_e$ (eV)',
                      )
        
    if 0: # Saha not operational
        # Ts = np.array([500,1000])
        # rhos = np.array([10,20])
        saha(32, 73, 10, 2000, pf=1)

