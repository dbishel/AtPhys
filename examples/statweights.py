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
            
