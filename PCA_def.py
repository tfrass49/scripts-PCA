#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:13:13 2022

@author: thomas
"""

import sys
# sys.path.insert(1,"../stagyy")
import numpy as np
import matplotlib.pyplot as plt
import shtns
from PCA import *
# from tools import Field
import cartopy.crs as ccrs
from pathlib import Path

class PCA:
    def __init__(self, fname, t_start, t_end, lmax, 
                 input_path = "/media/thomas/Data/these/data/AGE_TOT_ROT/",
                 fig_path = "/home/thomas/Documents/work/figures/PCA/",
                 output_path = "/home/thomas/Documents/work/PCA"):
        self.input_path = Path(input_path)
        self.fig_path = Path(fig_path)
        self.output_path = Path(output_path)
        self.fname = fname
        self.ts, self.te = t_start, t_end
        self.lmax = lmax
        self.casename = f"{fname}_{t_start}-{t_end}_{lmax}lmax"
        self.output_path_full = self.output_path / self.casename
        try:
            self.W = np.load(self.output_path_full / "weights.npy")
            self.S = np.load(self.output_path_full / "sing_val.npy")
            self.P = np.load(self.output_path_full / "patterns.npy")
            self.g = np.load(self.output_path_full / "avg_pattern.npy")
        except FileNotFoundError:
            print("PCA results not found, try to compute it")
            self.g, self.W, self.S, self.P = PCA_computation(self.fname, 
                                                             self.ts, 
                                                             self.te, 
                                                             self.lmax, 
                                                             self.input_path,
                                                             self.fig_path,
                                                             self.output_path,
                                                             plot=False,
                                                             store=True)
            
class Component(PCA):
    def __init__(self, pca, k):
        self.pca = pca
        self.k = k
        if k == 0:
            self.pat_sh = pca.g
        else:
            self.pat_sh = pca.P[k-1]
            self.vp = pca.S[k-1]
            self.amp = pca.W[:,k-1]
        self.SHspec = SHspectrum(reshape_coeffs(self.pat_sh, self.pca.lmax))
        
    def SH_lm_power(self, ls, ms):
        power, power_lm, sign_lm = SHspectrum(self.pat_sh, ls, ms)
        return(power_lm, sign_lm)
    
    def to_spat(self):
        sh = shtns.sht(self.pca.lmax)
        nlat, nphi = sh.set_grid(512,1024,shtns.sht_reg_fast)
        self.pat_spat = sh.synth(reshape_coeffs(self.pat_sh, self.pca.lmax))
        return(self.pat_spat)
    
    def plot_pat(self):
        dp = 2*np.pi/1024 #longitude step
        dt = np.pi/512 #latitude step
        liste_p_tot = np.arange(-np.pi, np.pi - dp/2, dp) + dp/2#longitude list
        liste_t_tot = np.arange(0, np.pi - dt/2, dt) + dt/2 #colatitude list
        lats = 90 - liste_t_tot*180/np.pi #latitudes in degrees
        lons = liste_p_tot*180/np.pi #longitude in degrees
        fig = plt.figure()
        ax = fig.add_subplot(111,projection=ccrs.Mollweide())
        plt.pcolormesh(lons,lats,self.pat_spat,transform=ccrs.PlateCarree(),
                       cmap='bwr',shading='nearest')
        plt.grid(linestyle='--',alpha=0.4)
        return(fig)
    
    def plot_wght(self):
        if self.k == 0:
            sys.exit("Average pattern ==> no weights!")
        else:
            plt.figure()
            fig = plt.plot(np.arange(self.pca.ts, self.pca.te + 1),
                           self.amp, c='black')
        return(fig)
        
def SHspectrum(coeffs, norm='ortho', ls=None, ms=None):
    delta_m0 = lambda m: 1 if (m==0) else 0
    #determine the maximum sh degree
    lmax = int((-3 + np.sqrt(1 + 8*len(coeffs))) / 2)
    #initialization of the power spectrum
    power = np.zeros(lmax + 1)
    #loop on SH orders
    sh = shtns.sht(lmax)
    N = 1
    if norm == 'fourpi':
        N = 4*np.pi
    for l in range(lmax+1):
        power_l = 0
        if norm == 'schmidt':
            N = 4*np.pi / (2*l + 1)
        for m in range(l+1):
            mask_lm = np.logical_and(sh.l==l,sh.m==m)
            power_l += (2 - delta_m0(m)) * N * np.abs(coeffs[mask_lm])**2
        power[l] = power_l
    if (ls != None and ms != None):
        mask_lm = np.logical_and(sh.l==ls,sh.m==ms)
        power_lm = (2 - delta_m0(m)) * N * np.abs(coeffs[mask_lm])**2
        power_lm /= np.sum(power)
        sign_lm = np.sign(coeffs[mask_lm].real)
        return(power, power_lm, sign_lm)
    else:
        return(power)