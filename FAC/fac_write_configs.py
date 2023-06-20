#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:50:36 2023

@author: dbis

Assigns each of N (n) indistinguishable electrons to R (k) distinguishable shells.
Used to construct unique shell occupations (complexes) to be read in by FAC
"""

# Python modules
import numpy as np

from math import comb, ceil

# %% Functions
def gen_list(n, k): # Assign each of n indistinguishable balls to one of k distinguishable buckets
    # Returns generator with bucket assignment of each ball, corresponding to unique bucket populations
    lst = [0] * n
    lst[0] = -1
    while True:
        index = 0
        lst[index] += 1
        while lst[index] == k:
            lst[index] = 0
            index += 1
            if index >= n:
                return None
            lst[index] += 1
        if np.all(np.sort(lst)==lst):
            # Assign if sorted
            yield lst[:]
        else:
            # Skip if unassigned
            continue

def count_balls(lst, k):
    # Counts and returns the number of balls in each of the k buckets of the single combination lst
    return [len(np.where(np.equal(lst, item))[0]) for item in range(k)]

def all_counts(n, k):
    # Counts the number of balls in each bucket
    combos = gen_list(n, k)
    for elem in combos:
        yield count_balls(elem, k)

def write_kalpha(nmax, Nele, exc, fn='fac_cfg_ka.txt', overwrite=True):

    """Number of electrons to distribute.
             2 electrons are already accounted for:
             - 1s*1 2p*1 for upper state,
             - 1s*2 for lower state"""
    N = Nele - 2 
            
    """Number of available shells.
      n=1 shell is already fully specified"""
    R = nmax - 1
    print(Nele, N, nmax, R)
    counts = all_counts(N,R) # [pi for i in range(2,R+1)], ignoring core 1s and active 2p electrons
    
    # [print(c) for c in counts]
    if overwrite:
        open_mode = 'w'
    else:
        open_mode = 'a'
    with open(fn, open_mode) as file: # Use file to refer to the file object
        for c in counts:
            # Skip if excitation degree is allowed by user
            if not(np.sum(c[1:]) in exc):
                continue
            
            # Construct upper, lower state n=1 and n=2 
            up = ['1*1 2*{0:d};2p>0'.format(c[0]+1)]
            if c[0]>0:
                lo = ['1*2 2*{0:d}'.format(c[0])]
            else:
                lo = ['1*2']
            
            # Construct upper state, lower state n>2 only if populated
            app = ['{0:d}*{1:d}'.format(shell+3, pop)
                       for shell,pop in enumerate(c[1:]) if pop>0]
            up.extend(app)
            lo.extend(app)
            
            up = " ".join(up)
            lo = " ".join(lo)
            
            # Write upper, lower states to file
            file.write(up)
            file.write('\n')
            file.write(lo)
            file.write('\n')

# def write_corevac_complex(nmax, Nele, exc, fn='fac_cfg.txt', 
#                   active_shell=2,
#                   overwrite=True,
#                   ):
#     """ Writes electron complexes for FAC to use as input.
#         Includes upper states with a core vacancy and corresponding lower states.

#     Parameters
#     ----------
#     nmax : int
#         Maximum principal quantum number considered.
#     Nele : int
#         Number of electrons in the ion.
#     exc : list
#         List of integers to specify what fold-excitation complexes are considered \n
#         - 0 denotes resonant transition to ground state.
#         - 1 denotes satellite transition to singly excited state, 2 to doubly-excited, ...
#     fn : str, optional
#         File into which complexes are written. The default is 'fac_cfg.txt'.
#     overwrite : bool, optional
#         Flag of whether to overwrite file or append to file. The default is True.
#     active_shell : int, optional
#         Active shell of desired transition. X yields n=1 to n=X transitions. The default is 2.
#         Upper state is required to have at least one p electron

#     Returns
#     -------
#     None.

#     """

#     """Number of electrons to distribute.
#              2 electrons are already accounted for:
#              - 1s*1 2p*1 for upper state,
#              - 1s*2 for lower state"""
#     N = Nele - 2 
            
#     """Number of available shells.
#       n=1 shell is already fully specified"""
#     R = nmax - 1
    
#     counts = all_counts(N,R) # [pi for i in range(2,R+1)], ignoring core 1s and active 2p electrons
    
#     HOS_ref = np.cumsum(2*np.arange(1,nmax+1)**2) # Reference to determine
#                                         # highest occupied shell in ground state
#     HOS = np.where(Nele <= HOS_ref)[0][0] + 1 # Highest occupied shell 
#                                         # in ground state. Add 1 due to 0-register
#     LUS = HOS + 1  # Lowest unoccupied shell
#     LUS_idx = LUS - 2 # -1 for 0-register, -1 for ignoring n=1 in counts later

#     if overwrite:
#         open_mode = 'w'
#     else:
#         open_mode = 'a'
#     with open(fn, open_mode) as file: # Use file to refer to the file object
#         for c in counts:
#             # Skip if excitation degree is allowed by user
#             # Excitation degree defined by number of electrons in shells unoccupied in ground state
#             exc_deg = np.sum(c[LUS_idx:]) # Degree of excitation (resonant, singly-excited, doubly-...)
#             if exc_deg in exc: 
#                 bf = False
#                 # Construct upper, lower state n=1
#                 up = ['1*1']
#                 lo = ['1*2']
    
#                 # # Skip configuration if any population exceeds maximum occupation 2n^2
#                 # occ = c
#                 # occ[active_shell-2] += 1
                
#                 # if occ > (2*range(2,len(shell)+2)**2):
#                 #     break
#                 # elif shell==active_shell and ((pop+1) > (2*shell**2)):
#                 #     continue
#                 # Construct n>1
#                 for shell, pop in enumerate(c):
#                     shell+=2 # Zero indexing and n=1 shell skipped
                    
#                     if pop>(2*shell**2):
#                         bf = True
#                         break # and continue outside this loop to prevent saving
#                     elif (shell==active_shell) and ((pop+1)>(2*shell**2)):
#                         bf = True
#                         break # and force continue outside this loop to prevent saving
                        
                    
#                     # Upper state is treated differently if active
#                     if shell==active_shell:
#                         # Upper: population +1 and require p>0
#                         up.append('{0:d}*{1:d};{0:d}p>0'.format(shell, pop+1))
#                     else:
#                         # For non-active shell, save population only if non-zero
#                         if pop>0:
#                             up.append('{0:d}*{1:d}'.format(shell, pop))
#                         else:
#                             continue
                        
#                     # Lower state is irrespective of active shell
#                     if pop>0:
#                         lo.append('{0:d}*{1:d}'.format(shell, pop))
#                     elif pop==0:
#                         [] # Append nothing to lower state if empty
#                 if bf:
#                     continue
#                 up = " ".join(up)
#                 lo = " ".join(lo)
                
#                 # Write upper, lower states to file
#                 file.write(up)
#                 file.write('\n')
#                 file.write(lo)
#                 file.write('\n')
#             else:
#                 continue

def write_hhe(nmax):
    # Writes 2-1 levels for Hydrogen and Helium
    
    # Hydrogen - resonant
    fn = 'fac_1_{0:d}_0_lo.txt'.format(nmax)
    with open(fn,'w') as file:
        file.write('1*1\n')
    
    fn = 'fac_1_{0:d}_0_up.txt'.format(nmax)
    with open(fn,'w') as file:
        [file.write('{0:d}*1\n'.format(n)) for n in range(2,nmax+1)]
        
    # Helium - resonant
    fn = 'fac_2_{0:d}_0_lo.txt'.format(nmax)
    with open(fn,'w') as file:
        file.write('1*2\n')
    
    fn = 'fac_2_{0:d}_0_up.txt'.format(nmax)
    with open(fn,'w') as file:
        [file.write('1*1 {0:d}*1\n'.format(n)) for n in range(2,nmax+1)]

    # Helium - singly-excited ground state
    fn = 'fac_2_{0:d}_1_lo.txt'.format(nmax)
    with open(fn,'w') as file:
        [file.write('1*1 {0:d}*1\n'.format(n)) for n in range(2,nmax+1)]
    
    fn = 'fac_2_{0:d}_1_up.txt'.format(nmax)
    with open(fn,'w') as file:
        file.write('2*2\n')
        [file.write('2*1 {0:d}*1\n'.format(n)) for n in range(3,nmax+1)]

def write_corevac_complex(nmax, Nele, exc, upper_state=True,
                          fn='fac_cfg.txt', 
                          active_shell=2,
                          overwrite=True,
                          ):
    """ Writes electron complexes for FAC to use as input.
        Includes upper states with a core vacancy and corresponding lower states.
        Only works for 2 < Nele <=10

    Parameters
    ----------
    nmax : int
        Maximum principal quantum number considered.
    Nele : int
        Number of electrons in the ion.
    exc : list
        List of integers to specify what fold-excitation complexes are considered \n
        - 0 denotes resonant transition to ground state.
        - 1 denotes satellite transition to singly excited state, 2 to doubly-excited, ...
    upper_state : bool
        Define whether to calculate upper states (with core hole) or lower states (no core hole).
        FAC requires separate groups for lower and upper to calculate TransitionTable
    fn : str, optional
        File into which complexes are written. The default is 'fac_cfg.txt'.
    overwrite : bool, optional
        Flag of whether to overwrite file or append to file. The default is True.
    active_shell : int, optional
        Active shell of desired transition. X yields n=1 to n=X transitions. The default is 2.
        Upper state is required to have at least one p electron

    Returns
    -------
    None.

    """

    """Number of electrons to distribute.
             2 electrons are already accounted for:
             - 1s*1 2p*1 for upper state,
             - 1s*2 for lower state"""
    N = Nele - 2 
            
    """Number of available shells.
      n=1 shell is already fully specified"""
    R = nmax - 1
    
    counts = all_counts(N,R) # [pi for i in range(2,R+1)], ignoring core 1s and active 2p electrons
    
    
    ## Old excitation degree
    HOS_ref = np.cumsum(2*np.arange(1,nmax+1)**2) # Reference to determine
                                        # highest occupied shell in ground state
    HOS = np.where(Nele <= HOS_ref)[0][0] + 1 # Highest occupied shell 
                                        # in ground state. Add 1 due to 0-register
    LUS = HOS + 1  # Lowest unoccupied shell
    LUS_idx = LUS - 2 # -1 for 0-register, -1 for ignoring n=1 in counts later
    
    ## New excitation degree. Reference against ground state
    # Determine first shell which is not full in ground state
    Pn_full = 2*np.arange(1,nmax+1)**2 # Shell populations if all full
    epop_full = np.cumsum(Pn_full) # Total electrons inside and interior to shell
    nmax_idx = np.where(epop_full>(Nele))[0][0] # First shell to not be full
            
    # Start with empty shells
    GS = np.zeros_like(Pn_full)
    GS[:nmax_idx] = Pn_full[:nmax_idx] # Assign full shells their proper value
    GS[nmax_idx] = (Nele) - np.sum(GS) # Populate first unfilled shell with remaining electrons

    if overwrite:
        open_mode = 'w'
    else:
        open_mode = 'a'
    with open(fn, open_mode) as file: # Use file to refer to the file object
        for c in counts:
            # Skip if excitation degree is allowed by user
            # Excitation degree defined by number of electrons in shells unoccupied in ground state
            exc_deg = np.sum(c[LUS_idx:]) # Degree of excitation (resonant, singly-excited, doubly-...)
            
            ## New excitation degree. Defined by number of electrons different from ground state GS
            dpop = c - GS[1:] # Population difference wrt GS
            exc_deg = np.sum(dpop[np.where(dpop>0)[0]]) # NEED exception for upper state
            
            if exc_deg in exc: 
                bf = False
                # Construct upper, lower state n=1
                if upper_state:
                    cmplx = ['1*1'] # Upper state – core hole
                else:
                    cmplx = ['1*2'] # Lower state – filled 1s
    
                # Construct population of each shell n>1
                for shell, pop in enumerate(c):
                    shell+=2 # +1 for zero indexing and +1 for n=1 shell skipped
                    
                    # Ignore complex if this shell exceeds maximum occupation 2n^2
                    if pop>(2*shell**2):
                        bf = True # Set break flag to 'continue' outside this loop to prevent saving
                        break
                    
                    # Ignore complex if active shell in lower state is full, thus having no allowable upper state
                    elif (not(upper_state)) and (shell==active_shell) and (pop==(2*shell**2)):
                        bf = True
                        break
                    
                    # Ignore complex if active shell in upper state exceeds maximum occupancy.
                    # pop+1 because active shell of upper state is given +1 later
                    elif (upper_state) and (shell==active_shell) and ((pop+1)>(2*shell**2)):
                        bf = True # Set break flag
                        break 
                        
                    
                    # Upper state is treated differently if active
                    if upper_state:
                        if shell==active_shell:
                            # Upper: population +1 and require p>0
                            cmplx.append('{0:d}*{1:d};{0:d}p>0'.format(shell, pop+1))
                        else:
                            # For non-active shell, save population only if non-zero
                            if pop>0:
                                cmplx.append('{0:d}*{1:d}'.format(shell, pop))
                            else:
                                continue
                    
                    # Lower state is irrespective of active shell
                    else:
                        if pop>0:
                            cmplx.append('{0:d}*{1:d}'.format(shell, pop))
                        elif pop==0:
                            [] # Append nothing to lower state if empty
                if bf:
                    continue
                cmplx = " ".join(cmplx)
                
                # Write complexes to file
                file.write(cmplx)
                file.write('\n')
            else:
                continue

# %%
if __name__=='__main__':

    # Filename: fac_NE_NMAX_EXC_lo/up.txt
    nmax = 5  # largest principal quantum number shell to consider
    # exc = np.arange(6) # Accepted degree of excitations. 0=resonance, 1=singly-excited, ...
    exc_list = [0,1,2,3]
    # exc_list = [3]

    Nlist = np.arange(10,19)
    # Nlist = [12]
    # Nlist = [3,4,5,9,10,11,]
    for Nele in Nlist:
        
        fn = '../complexes/fac_{0:d}_{1:d}'.format(Nele,nmax)
        # fn = '/Users/dbis/Desktop/complexes/fac_{0:d}_{1:d}'.format(Nele,nmax)
        
        for exc in exc_list:
            # breakpoint()
            write_corevac_complex(nmax, Nele, [exc],
                                  fn=fn+'_{0:d}_lo.txt'.format(exc),
                                  upper_state=False, overwrite=True,
                                  active_shell=2)
            
            write_corevac_complex(nmax, Nele, [exc],
                                  fn=fn+'_{0:d}_up.txt'.format(exc),
                                  upper_state=True, overwrite=True,
                                  active_shell=2)
    
    
# %%
# NNN = 11
# nvec = np.arange(1, nmax+1)
# pvec = 2*nvec**2
# HOS_ref = np.cumsum(pvec) # Reference to determine
#                                     # highest occupied shell in ground state
# HOS = np.where(NNN <= HOS_ref)[0][0] + 1 # Index of highest occupied shell 

# print(nvec)
# print(pvec)
# print(HOS_ref)
# print(Nele)
# print(Nele <= HOS_ref)
# print(np.where(Nele <= HOS_ref))
# print(HOS)
