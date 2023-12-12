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
import scipy as sp

import os                       # Used to e.g. change directory
import sys                      # Used to put breaks/exits for debugging
import re

sys.path.append('../')
from ScHyd import get_ionization, AvIon
from saha_boltzmann_populations import saha, boltzmann
from ScHyd_AtomicData import AtDat

# %% Run AD
DIR = '../complexes/'

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

# Define T, Ne, rho grids for SB
Nn, NT = 10, 51 # Number of density, temperature gridpoints

# KT = np.linspace(50,1000, num=NT) # eV, Temperature range, to find IPD
KT = np.logspace(0.5,3, num=NT) # eV, Temperature range, to find IPD

rho0 = np.logspace(-1,2, num=Nn) # g/cc
Zbar0 = 20 # Estimated Zbar
NE = rho0 / (A*ad.mp) * Zbar0 # 1/cm^3, Ne range

Nrho = 12
rho_grid = np.logspace(-1,1, num=Nrho)

# Run Saha-Boltzmann
ad.saha_boltzmann(KT, NE, IPD=0)
ad.saha_boltzmann_rho(rho_grid)

# Satellite resolved line centers
hnu_avg = ad.get_hnu_average(ad.pboltz, gf=1, resolve='ionization') # Shape: [excitation, NT, Nrho, ionization]

ridx = 0
zidx = 2 # N-like
plt.figure()
[plt.plot(KT, hnu_avg[eidx,:,ridx,zidx-eidx],
          label='Z*={0:s}, exc={1:d}'.format(ad.Zkeys[zidx-eidx], eidx))
     for eidx in exc_list if (zidx-eidx)>=0]
plt.legend()

# Line-complex resolved line centers
hnu_avg = ad.get_hnu_average(ad.pboltz, gf=1, resolve='line') # Shape: [NT, Nrho, ionization]

ridx = 0
zidx = 2 # N-like
plt.figure()
plt.plot(KT, hnu_avg[:,ridx,:], label=ad.Zkeys)
plt.gca().set(ylim=[5350,5800])
plt.legend()