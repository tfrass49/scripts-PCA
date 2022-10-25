#!/ust = r/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:17:15 2021

@author: thomas
"""
###############################################################################
#Script meant to compute a Principal Component Analysis (PCA) of 2d scalar
#fields time series. The PCA is computed on the spherical harmonic coefficient 
#of the field. The sh decomposition is computed using the shtns code.
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import shtns
from pathlib import Path
import cartopy.crs as ccrs
import argparse

#sys.argv.insert(1,'fbot_rot')
#sys.argv.insert(2,'50')
#sys.argv.insert(3, '300')
#sys.argv.insert(4, '1131')
#sys.argv.insert(5,'-s')
#sys.argv.insert(6,'-p')

"""Funtions definitions"""
###############################################################################
def get_arguments():
    """
    Read the arguments given to the code

    Returns
    -------
    args: argparse.Namespace
        list of arguments 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("fname" ,help = "Name of the field to decompose. "
                        "The data files must be called 'fname_XXXX.npy'")
    parser.add_argument("lmax" , help = "Maximum sh degree to consider",
                        type=int)
    parser.add_argument("start", help = "Number of the first snapshot to "
                        "consider",
                        type=int)
    parser.add_argument("end", help = "Number of the last time step to "
                        " consider", 
                        type=int)
    parser.add_argument("-s", "--store" ,
                        help = "Option to store the results",
                        action='store_true')
    parser.add_argument("-p", "--plot" ,
                        help = "Option to plot the results",
                        action='store_true')
    args = parser.parse_args()
    return(args)
###############################################################################

###############################################################################
def reshape_coeffs(coeffs,lmax):
    """
    Convert an array of sh coefficients with separated real and imaginary parts

    Parameters
    ----------
    coeffs: numpy.ndarray
        array of sh coefficients with separated real and imaginary parts.

    Returns
    -------
    coeffs_shaped: numpy.ndarray
        array of sh coefficients with combined real and imaginary parts.

    """
    #if the mean has been removed, add the l=m=0 coeff = 0
    if len(coeffs) == (lmax+1) * (lmax+2) - lmax - 2:
        coeffs = np.concatenate((np.array([0]), coeffs))
    #number of SH coeffs
    nbcoeffs = int((lmax+1)*(lmax+2)/2)
    #real part of the coefficients
    Rcoeffs = coeffs[:nbcoeffs]
    #imaginary part of the coefficients
    Icoeffs = coeffs[nbcoeffs:] 
    #initialization of the array
    coeffs_shaped = np.zeros((nbcoeffs), dtype='complex')
    #l=0 coefficients
    coeffs_shaped[:lmax + 1] = Rcoeffs[:lmax + 1] + 0j 
    #l!=0 coefficients
    coeffs_shaped[lmax + 1:] = Rcoeffs[lmax + 1:] + Icoeffs*1j
    return(coeffs_shaped)
###############################################################################

###############################################################################
def initialize_matrix(nb_snaps,lmax):
    """
    Initialize the matrix

    Parameters
    ----------
    nb_snaps: int
        number of snapshots to put in the matrix.
    lmax: int
        maximum degree of the sh decomposition.

    Returns
    -------
    A_init: numpy.ndarray
        initialized matrix with the expected shape depending onnb_snaps, lmax 
        and no_mean.

    """
    #number of degree in the sh decomposition:
    ndeg = lmax + 1
    #number of coefficients in the sh decomposition
    ncoeffs = int(ndeg**2)
    #Initialization of the matrix
    A_init = np.zeros((nb_snaps,ncoeffs))
    return(A_init)
###############################################################################

###############################################################################
def build_matrix(fname,directory_load,lmax,snap_deb,snap_fin,sh):
    """
    Build the matrix to be decomposed in singular value. 
    The lines of the matrix are the SH decomposition of the snapshots.

    Parameters
    ----------
    directory_load: pathlib.PosixPath
        path to the input data to load.
    lmax: int
        maximum degree of the sh decomposition.
    snap_deb: int
        number of the first used snapshot.
    snap_fin: int
        number of the last used snapshot.

    Returns
    -------
    A : numpy.ndarray
        filled matrix to be decomposed in singular values.

    """
    #number of snapshot to compute in the PCA:
    nb_snaps = snap_fin - snap_deb + 1
    #matrix initialization
    A = initialize_matrix(nb_snaps, lmax)
    print('Filling of the data matrix...')
    #the matrix is filled line by line
    for k in range(snap_deb, snap_fin + 1):
        #importation of data:
        data = np.load(directory_load / (fname 
                       + '_' + str(k).zfill(4) + '.npy'))
        #spherical harmonic decomposition using shtns
        coeffs = sh.analys(data) 
        #real and imaginary part are decomposed
        #real part
        Rcoeffs = coeffs.real
        #imaginary part
        Icoeffs = coeffs[lmax + 1:].imag
        #the matrix is filled
        A[k - snap_deb] = np.concatenate((Rcoeffs,Icoeffs))
    return(A)
###############################################################################
        
###############################################################################
def centre_gravite(A):
    """
    Compute the gravity center of a matrix

    Parameters
    ----------
    A: numpy.ndarray
        2d matrix containing the data set.

    Returns
    -------
    g: numpy.ndarray
        2d matrix containing the gravity center of the input matrix.

    """
    g = np.dot(A.T, 1/A.shape[0] * np.ones(A.shape[0]))
    return(g)
###############################################################################

###############################################################################
def plot_results(fname, snap_deb, snap_fin, L, S, R, U, g,
                 figdir, lmax, sh, nbpatterns = 31, deg_max_plot = 200):
    """
    Plot the patterns and amplitudes
    
    Parameters
    ----------
    fname: string
        name of the field.
    snap_deb: int
        number of the first used snapshot.
    snap_fin: int
        number of the last used snapshot.
    L: numpy.ndarray
        matrix containing the amplitudes of the PCA components (left 
        eigenvectors matrix).
    S: numpy.ndarray
        matrix containing the singular values.
    R: numpy.ndarray
        matrix containing the sh decompositions of the PCA components (right 
        eigenvectors matrix).
    U: numpy.ndarray
        matrix containing the patterns of the PCA components.
    g: numpy.ndarray
        matrix containing the averaged pattern of the data set.
    figdir: pathlib.PosixPath
        path to the directory where the figures will be stored
    lmax: int
        maximum degree of the sh decomposition.
    sh: shtns.sht
        spherical harmonic transform object from shtns
    nbpatterns: integer
        Number of patterns to plot.
    deg_max_plot: integer
        Maximum PCA degree to consider in the figures.

    Returns
    -------
    None.

    """
    print('Plot the figures')
    
    #longitude step
    dp = 2*np.pi/1024
    #latitude step
    dt = np.pi/512
    #list of longitudes in the grid
    lons = (np.arange(-np.pi, np.pi - dp/2, dp) + dp/2) * 180/np.pi
    #list of latitudes in the grid
    lats = 90 - (np.arange(0, np.pi - dt/2, dt) + dt/2) * 180/np.pi
    #get the plot specifications for the field
    # label, scale, cmap = field.label, field.scale, field.cmap
    ###Patterns and amplitudes###
    #Modifications of the plot parameters
    #list of times
    times = np.arange(snap_deb,snap_fin + 1,1)
    nbfigures = int(nbpatterns/3) - 1
    for k in range(nbfigures):
        plt.figure(figsize=(12,15))
        for l in range(3):
            index = int(3*k + l)
            plt.subplot(3,2,int(2*l + 1))
            plt.title('n = ' + str(index))
            maxi = np.max((U[index].max(),-U[index].min()))
            mini = - maxi
            plt.pcolormesh(lons, lats, U[index],
                           cmap = 'viridis', shading = 'nearest',
                           vmin = mini, vmax = maxi)
            plt.colorbar()
            plt.subplot(3,2,int(2*l + 2))
            plt.title('n = ' + str(index))
            plt.plot(times,L[:,index]*S[index],color='black')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.minorticks_on()
            plt.xlim(times[0], times[-1])
            plt.grid()
            plt.tight_layout()
        plt.savefig(figdir / (fname + '_patterns_amplitudes' 
                    + str(int(3*k)) + '-' + str(int(3*k+2)) + '.png'))
        plt.close()
        
    ###Patterns in a Mollweide projection###
    nbfigures = int(nbpatterns/3) - 1
    for k in range(nbfigures):
        fig = plt.figure(figsize=(10,3))
        for l in range(3):
            index = int(3*k + l)
            ax = fig.add_subplot(1,3,l+1,projection = ccrs.Mollweide())
            maxi = np.max((U[index].max(),-U[index].min()))
            mini = - maxi
            p = ax.pcolormesh(lons, lats, U[index],
                              transform = ccrs.PlateCarree(),shading='nearest',
                              cmap = 'viridis', vmin = mini, vmax = maxi)
            cbar = fig.colorbar(p, orientation = 'horizontal', ticks=[-1,0,1], 
                                label = '$r_{' + str(index) + '}$')
            cbar.ax.tick_params(labelsize=20) 
        plt.subplots_adjust(left=0.1,
                            bottom=0.25, 
                            right=0.9, 
                            top=0.95, 
                            wspace=0.4, 
                            hspace=0.3)
        plt.savefig(figdir / (fname + '_patterns' 
                     + str(int(3*k)) + '-' + str(int(3*k+2)) + '.png'))
        plt.close()
    
    ###averaged pattern###
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1,projection = ccrs.Mollweide())
    # convert the gravity center in the spatial domain
    g_spat = sh.synth(reshape_coeffs(g,lmax)) #* scale
    p = ax.pcolormesh(lons, lats, g_spat,
                      transform = ccrs.PlateCarree(), shading='nearest',
                      cmap = 'viridis')
    cbar = fig.colorbar(p, orientation = 'horizontal')#,label=label)
    cbar.ax.tick_params(labelsize=20) 
    ax.gridlines()
    plt.savefig(figdir / (fname + '_centre_gravi.png'))
    plt.close()
    
    return()
###############################################################################

###############################################################################
def PCA_computation(fname, snap_deb, snap_fin, lmax, input_dir, fig_dir,
                    output_dir, plot, store):
    """
    

    Parameters
    ----------
    fname : string
        name of the field.
    snap_deb: int
        number of the first used snapshot.
    snap_fin: int
        number of the last used snapshot.
    lmax: int
        maximum degree of the sh decomposition.
    input_dir: pathlib.PosixPath
        path to the directory where the input data are stored.
    fig_dir: pathlib.PosixPath
        path to the directory where the figure will be ploted.
    output_dir: pathlib.PosixPath
        path to the directory where the results will be stored.
    plot : bool
        boolean defining wether to plot the figures or not.
    store : bool
        boolean defining wether to save the results or not.

    Returns
    -------
    None.

    """
    t0 = time.time()
    #directory where the data are stored
    input_dir_full = input_dir / fname
    #directory where the figures will be stored
    fig_dir_full = fig_dir / (fname + '_'\
             + str(snap_deb) + '-' + str(snap_fin) + '_' \
             + str(lmax) + 'lmax')
    #directory where the npy tables will be stored
    output_dir_full = output_dir / (fname\
             + '_' + str(snap_deb) + '-' + str(snap_fin) + '_'\
             + str(lmax) + 'lmax')
    #define the maximum order of the SH decomposition (same as maximum degree)
    mmax = lmax 
    #initialize shtns
    sh = shtns.sht(lmax, mmax)
    #initialize the grid
    nlat, nphi = sh.set_grid(512, 1024, shtns.sht_reg_fast)
    #Print informations concerning the PCA
    print('field : ',fname)
    print('lmax : ',lmax)
    print('start : ',snap_deb)
    print('end : ',snap_fin)
    print('==================================================================')
    #construction de la matrice sur laquelle la PCA est realisee:
    A = build_matrix(fname, input_dir_full, lmax, 
                   snap_deb, snap_fin, sh)
    print('==================================================================')
    #compute the gravity center
    g = centre_gravite(A)
    #remove the gravity center
    A = A - np.outer(np.ones(A.shape[0]), g.T)
    #singular value decomposition of A:
    svd = np.linalg.svd(A, full_matrices = False)
    print('svd done')
    print('==================================================================')
    L,S,R = svd[0],svd[1],svd[2]

    #Compute the 200 first patterns in the spatial domain
    if len(R) > 200:
        U = np.zeros((200,512,1024))
    else:
        U = np.zeros((len(R),512,1024))
    for k in range(U.shape[0]):
        U[k] = sh.synth(reshape_coeffs(R[k],lmax))

    #PLot the figures
    if plot:
        #make sure the directory exist
        Path(fig_dir_full).mkdir(parents=True, exist_ok=True) 
        deg_max_plot = np.min((200, snap_fin - snap_deb))
        plot_results(fname, snap_deb, snap_fin, L, S, R, U, g,  
                     fig_dir_full, lmax, sh,
                     deg_max_plot=deg_max_plot) 
        print('==============================================================')

    #Save the patterns, amplitudes and eigenvalues
    if store:
        print('save the results in the result directory')
        print('==============================================================')
        #make sure the directory exist
        Path(output_dir_full).mkdir(parents=True, exist_ok=True) 
        #number of components to save (max=len(R))
        nb_saved = len(R)
        if len(R) > nb_saved:
            R_save = R[:nb_saved]
            L_save = L[:,:nb_saved]
            S_save = S
        else:
            R_save = R[:len(R)]
            L_save = L[:len(R)]
            S_save = S[:len(R)]
        np.save(output_dir_full / 'patterns.npy', R_save)
        np.save(output_dir_full / 'weights.npy', L_save)
        np.save(output_dir_full / 'sing_val.npy', S_save)
        np.save(output_dir_full / 'avg_pattern.npy', g)
    print(fname + ' PCA ended, execution took ' + 
          str(time.time() - t0) + ' s')
    return(g,L,S,R)
    
    
if __name__ == '__main__':
    #Definition of the paths
    ###########################################################################
    input_dir = Path("/media/thomas/Data/these/data/AGE_TOT_ROT/")
    output_dir = Path("/home/thomas/Documents/work/PCA/")
    fig_dir = Path("/home/thomas/Documents/work/figures/PCA/")
    ###########################################################################
    args = get_arguments()
    #name of the field
    fname = args.fname
    #number of the first snapshot to consider
    snap_deb = args.start
    #number of the last snapshot to consider
    snap_fin = args.end
    #maximum SH degree of the decompositions
    lmax = args.lmax
    #flags for plots
    plot = args.plot
    #flag for storing results
    store = args.store
    #deinfition of the field
    g,L,S,R = PCA_computation(fname,snap_deb,snap_fin,lmax,input_dir,
                              fig_dir,output_dir,plot,store)
