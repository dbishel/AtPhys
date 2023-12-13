#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:15:21 2023

@author: dbis

This script calculates and prints the statweights summed over a given excited complex.
Used to ensure correct calculation.
"""

# Python modules
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import os                       # Used to e.g. change directory
import sys                      # Used to put breaks/exits for debugging
import re

sys.path.append('../')
from ScHyd import get_ionization, AvIon
from saha_boltzmann_populations import saha, boltzmann
from ScHyd_AtomicData import AtDat


# %% Total statweights
Z = 24
nmax = 5
Nele = 6
uplo = 'lo'
vb = False

DIR = '../complexes/'

for exc in [0,1,2]:
    all_sw =[] # Collect stat weights of this excitation degree, to sum later.
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
            sh = AvIon(Z, Zbar=(Z-Nele), nmax=nmax)      
    
            sh.Pn = Pni # Pass Pn manually
            sh.get_Qn()
            
            sh.get_Wn()
            sh.get_En()
            sh.get_statweight()
            sh.get_Etot()
            
            all_sw.append(np.product(sh.statweight))
            
            if vb:
                print()
                print(l)
                print(Pni)
                print(np.product(sh.statweight))
                
    print('Sum: ', np.sum(all_sw)) # Print sum over all stat.weights in this excitation degree
        
# %% Transition-valid statweights    
ZZ = 24 # Nuclear charge
A = 51.996 # Nucleon number

Zbar_min = ZZ - 10
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

# Print zero, positive, and negative elements in gtot
print('-------------------------------------')
print('{0:3s} | {1:5s} | {2:5s} | {3:5s} | {4:5s}'.format('cfg', '>0', '=0', '<0', 'size'))
[print('{0:3s} | {1:5.0f} | {2:5.0f} | {3:5.0f} | {4:5.0f}'.format(item,
                                          len(np.where(ad.gtot_lists[item]>0)[0]),
                                          len(np.where(ad.gtot_lists[item]==0)[0]),
                                          len(np.where(ad.gtot_lists[item]<0)[0]),
                                          ad.gtot_lists[item].size
                                          ))
    for item in ['lo','up']];

# breakpoint()
gf, gi, gj = ad.get_gf(1,0,2,1,True,False)

# Print zero, positive, and negative elements in gtot
print('-------------------------------------')
print('{0:3s} | {1:5s} | {2:5s} | {3:5s} | {4:5s}'.format('cfg', '>0', '=0', '<0', 'size'))
[print('{0:3s} | {1:5.0f} | {2:5.0f} | {3:5.0f} | {4:5.0f}'.format(item,
                                          len(np.where(ad.gtot_lists[item]>0)[0]),
                                          len(np.where(ad.gtot_lists[item]==0)[0]),
                                          len(np.where(ad.gtot_lists[item]<0)[0]),
                                          ad.gtot_lists[item].size
                                          ))
    for item in ['lo','up']];

# Ground-state Li, F
print('State  g_lower g_upper')
print('ground, Li:', gi[-2,0], gj[-2,0])
print('ground, F:', gi[2,0], gj[2,0])

# Excited, Li
print('excited, Li:', gi[-2,1:4].sum(), gj[-2,1:4].sum())
