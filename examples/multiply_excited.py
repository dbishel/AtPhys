#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:31:28 2023

@author: dbis

This script tests the writing and use of multiply excited states.
"""

# Python modules
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import os                       # Used to e.g. change directory
import sys                      # Used to put breaks/exits for debugging

sys.path.append('../FAC')
from fac_write_configs import generate_complexes, all_counts

# %% Writing
NE = 10
nmax = 5

for e in [0,1,2]:
    lo, up = generate_complexes(NE, nmax, e)
    
    print(lo, up)
    
# %% In ScHyd
