#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:39:12 2023

@author: dbis

Reads and plots useful things from FAC outputs
Uses pfac.rfac.read_* modules to load FAC outputs
"""

import numpy as np
import matplotlib.pyplot as plt
# from math import *

# from pfac.table import *
# from pfac.spm import *
import pprint

from pfac import const, rfac # rfac is for READING filess
from pfac.rfac import _get_header, _read_value # Used to read UTA values

def read_tr_UTA(filename):
    """ read *a.tr file with SetUTA(m=1). """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # header
    header, lines = _get_header(lines)
    if header['NBlocks'] == 0:
        return header,()
    lines = lines[1:]
    def read_blocks(lines):
        block = {}
        block['NELE'], lines = _read_value(lines, int)
        ntrans, lines = _read_value(lines, int)
        block['MULTIP'], lines = _read_value(lines, int)
        block['GAUGE'], lines = _read_value(lines, int)
        block['MODE'], lines = _read_value(lines, int)
        # read the values
        block['lower_index'] = np.zeros(ntrans, dtype=int)
        block['lower_2J-1'] = np.zeros(ntrans, dtype=int)
        block['upper_index'] = np.zeros(ntrans, dtype=int)
        block['upper_2J-1'] = np.zeros(ntrans, dtype=int)
        block['Delta E'] = np.zeros(ntrans, dtype=float) # Transition energy including energy correction
        block['sigma'] = np.zeros(ntrans, dtype=float) # Gaussian standard deviation. See 'DB_TR' in FAC manual
        block['gf'] = np.zeros(ntrans, dtype=float)
        block['rate'] = np.zeros(ntrans, dtype=float)
        block['multipole'] = np.zeros(ntrans, dtype=float)
        block['CI_mult'] = np.zeros(ntrans, dtype=float) # Configuration interaction multiplier. See 'TR_EXTRA' in FAC manual

        for i, line in enumerate(lines):
            if line.strip() == '':  # if empty
                blocks = read_blocks(lines[i+1:])
                return (block, ) + blocks
            items = line.split()
            
            for kidx, key in enumerate(list(block.keys())[4:8]):
                block[key][i] = int(items[kidx])
            for kidx, key in enumerate(list(block.keys())[8:]):
                block[key][i] = float(items[kidx+4])

        return (block, )

    return header, read_blocks(lines)


def summarize(header, blocks):
    """ Prints summary of given header and blocks given by rfac.read_*
        Assumes all blocks use same keys
        Assumes number of entries in a block is given by length of last key 
    """
    
    print('------')
    print('Header')
    print('------')
    [print(k,header[k]) for k in list(header.keys())];
    
    print()
    print('-------------')
    print('Block summary')
    print('-------------')
    print('Number of blocks: {0:0.0f}'.format(len(blocks)))
    print('Length of each block:')
    [print(' ',len(d[list(d.keys())[-1]])) for d in blocks]
    
    print('Keys:')
    [print(' ',k) for k in list(blocks[0].keys())];
    
    return None

if __name__=='__main__':
    DIR = 'hydrogen/'
    FILE = 'H_1_5_0'
    
    # %% Transition rate table
    
    FN = DIR + FILE + '.tr'
    
    header, blocks = rfac.read_tr(FN) # header dictionary, tuple of block dictionaries
    
    summarize(header,blocks)
    
    plt.figure()
    for b in blocks:
        plt.plot(b['Delta E'], b['rate'], '.', alpha=0.1, label=b['NELE'])
    plt.legend()
    
    # %% Levels table
    
    FN = DIR + FILE + '.lev'
    
    header, blocks = rfac.read_lev(FN) # header dictionary, tuple of block dictionaries
    
    summarize(header,blocks)
    
    plt.figure()
    for b in blocks:
        plt.hist(b['ENERGY'], label=b['NELE'], bins=100)
    plt.legend()
    plt.gca().set(xlabel='hnu (eV)',
                  ylabel='# lines',
                  title='Histogram of transitions')
    
    # %% UTA Transition rate table
    FILE = 'h_UTA_1_5_0'
    FN = DIR + FILE + '.tr'
    
    header, blocks = read_tr_UTA(FN) # header dictionary, tuple of block dictionaries
    
    summarize(header,blocks)
    
    gauss = lambda x,x0,sig,A: A / (np.sqrt(2*np.pi)*sig) * np.exp(-(x-x0)**2 / (2*sig**2))
    # hnugrid = np.linspace(6575,6700, num=6000)
    hnugrid = np.linspace(9,13, num=300)
    fig, axs = plt.subplots(2, sharex=True)
    for b in blocks:
        # Show energy, width, and rate
        hnu = b['Delta E'] # eV
        wid = b['sigma'] # eV
        rate = b['rate'] # 1/s – per atom (?) i.e. Einstein A coefficient
        gf = b['gf'] # 1/s – per atom (?) i.e. Einstein A coefficient
        wid[wid==0] = 0.1
        
        axs[0].errorbar(x=hnu, y=rate, xerr=wid, fmt='.', alpha=0.1)
    
        # Show model spectrum - plot each spec individuually...
        spec = np.sum(gauss(x=hnugrid[:,None], 
                            x0=hnu[None,:],
                            sig=wid[None,:],
                            A=rate[None,:]),
                      axis=1)
        axs[1].plot(hnugrid, spec)
        # plt.plot(hnugrid, gauss(x=hnugrid, x0=hnu[0], sig=wid[0], A=rate[0]))
        
        # Check relation between gf and rate: 
        el = 1.60217663-19 # J/eV
        me = 9.1093837e-31 # kg
        hc = 1.98644586e-25 # J*m
        c = 2.998e8 # m/s
        g = 8*np.pi**2 * el**2 / me / c * (hnu*el / hc)**2 * gf / rate
        print(g) # Off by 1/2.22e-10
        print(g/np.nanmin(g))
        # print(2 * (1/137)**3 * (hnu/2)**2 * gf / rate )
    
    axs[0].set(ylabel='Rate/trans (unts)')
    axs[1].set(ylabel='Rate tot (units)',
               xlabel='hnu (eV)')
    # plt.figure()
    # xx = np.linspace(-10,10, num=200)
    # plt.plot(xx, gauss(xx, 0, 0.3, 1))
