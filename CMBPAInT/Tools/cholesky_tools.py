#!/usr/bin/env python
# coding: utf-8

"""
Created on Wed Apr 26 09:35:10 2023

@author: Christian Gimeno Amo (gimenoc@ifca.unican.es)

This module store all the necessary functions to read a stack of npy format files in a single dask array, and the ones
used to compute Cholesky decomposition.

Dependencies:

- numpy
- dask
- os

"""

import numpy as np 
import dask.array as da
import dask
import os

def Read_external_cov_mat(Path, **kwargs):

    """Read an external covariance matrix saved in a HDF5 format.

    Parameters
    ----------

    path : str
        Path to the .hdf5 file

    Returns
    -------
    x : dask.array.core.Array
        Dask array with the external covariance matrix.

    """

    return dask.dataframe.read_hdf(pattern = Path, **kwargs)


def Read_from_Dask(Pol, T_Pol, Input_PATH, NJ, lmax, Chunks):

    """Read a stack of npy format files in a single dask array.
    
    Parameters
    ----------

    Pol : bool
        If True, the input files also contains polarization matrices. If False, the input files are just the temperature matrices.
    T_Pol : bool
        If True, the input files are both the temperature and polarization matrices. If False, the input files depends on the value of Pol.
    Input_PATH : str
        Path to the input files.
    NJ : int
        Number of files.
    lmax : int
        Maximum multipole moment.
    Chunks : int
        Size of the chunks.

    Returns
    -------
    x_TT : dask.array.core.Array
        Dask array with the temperature matrices. If Pol and TPol are False.
    x_Pol : dask.array.core.Array
        Dask array with the Q and U polarization matrices. If Pol is True and TPol is False.
    x_TPol : dask.array.core.Array
        Dask array with the temperature and polarization matrices. If TPol is True.

    """

    load = dask.delayed(np.load)
    if (Pol == False):
        globals()["filenames_TT"] = [str(Input_PATH)+"/Cov_TT_lmax_"+str(lmax)+"_"+str(i)+".npy" if NJ!=1 else str(Input_PATH)+"/Cov_TT_lmax_"+str(lmax)+".npy" for i in range(NJ)]
        globals()["lazy_arrays_TT"] = [load(path) for path in filenames_TT]
        sample = lazy_arrays_TT[0].compute()
        globals()["x_TT"] = da.concatenate([da.from_delayed(lazy_arrays, dtype = sample.dtype, shape = sample.shape) for lazy_arrays in lazy_arrays_TT])
        globals()["x_TT"] = globals()["x_TT"]+da.tril(globals()["x_TT"], -1).T
        globals()["x_TT"] = da.rechunk(globals()["x_TT"], chunks = (Chunks, Chunks))
        return (x_TT)
    else:
        if (T_Pol == False):
            Dic = ["QQ", "UU", "QU", "UQ"]

            Cross_1 = [0, 1, 2, 3]
            Cross_2 = [0, 1, 3, 2]
            
            for i in range(4):
                globals()["filenames_"+Dic[i]] = [str(Input_PATH)+"/Cov_"+Dic[i]+"_lmax_"+str(lmax)+"_"+str(j)+".npy" if NJ!=1 else str(Input_PATH)+"/Cov_"+Dic[i]+"_lmax_"+str(lmax)+".npy" for j in range(NJ)]
                globals()["lazy_arrays_"+Dic[i]] = [load(path) for path in globals()["filenames_"+Dic[i]]]
                if (i==0):
                    sample = lazy_arrays_QQ[0].compute()

                globals()["x_"+Dic[i]] = da.concatenate([da.from_delayed(lazy_arrays, dtype = sample.dtype, shape = sample.shape) for lazy_arrays in globals()["lazy_arrays_"+Dic[i]]], axis = 0)

            for i in range(4):
                globals()["x_"+Dic[Cross_1[i]]] = globals()["x_"+Dic[Cross_1[i]]]+da.tril(globals()["x_"+Dic[Cross_2[i]]], -1).T

            x_Pol = da.concatenate([da.concatenate([globals()["x_QQ"], globals()["x_QU"]], axis = 1), da.concatenate([globals()["x_UQ"], globals()["x_UU"]], axis = 1)], axis = 0)
            x_Pol = da.rechunk(x_Pol, chunks = (Chunks, Chunks))
            return x_Pol
        else:
            Dic = ["TT", "TQ", "TU", "QT", "UT", "QQ", "UU", "QU", "UQ"]

            Cross_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            Cross_2 = [0, 3, 4, 1, 2, 5, 6, 8, 7]

            for i in range(9):
                globals()["filenames_"+Dic[i]] =  [str(Input_PATH)+"/Cov_"+Dic[i]+"_lmax_"+str(lmax)+"_"+str(j)+".npy" if NJ!=1 else str(Input_PATH)+"/Cov_"+Dic[i]+"_lmax_"+str(lmax)+".npy" for j in range(NJ)]
                globals()["lazy_arrays_"+Dic[i]] = [load(path) for path in globals()["filenames_"+Dic[i]]]

                if (i == 0):
                    sample = lazy_arrays_TT[0].compute()

                globals()["x_"+Dic[i]] = da.concatenate([da.from_delayed(lazy_arrays, dtype = sample.dtype, shape = sample.shape) for lazy_arrays in globals()["lazy_arrays_"+Dic[i]]], axis = 0)

            for i in range(9):
                globals()["x_"+Dic[Cross_1[i]]] = globals()["x_"+Dic[Cross_1[i]]]+da.tril(globals()["x_"+Dic[Cross_2[i]]], -1).T

            x_TPol = da.concatenate([da.concatenate([globals()["x_TT"], globals()["x_TQ"], globals()["x_TU"]], axis = 1), da.concatenate([globals()["x_QT"], globals()["x_QQ"], globals()["x_QU"]], axis = 1), da.concatenate([globals()["x_UT"], globals()["x_UQ"], globals()["x_UU"]], axis = 1)], axis = 0)
            x_TPol = da.rechunk(x_TPol, chunks = (Chunks, Chunks))
            return x_TPol
        
def Read_from_Dask_Numerical(Pol, T_Pol, Input_PATH, NJ, lmax, Chunks):

    """Read a stack of npy format files in a single dask array.
    
    Parameters
    ----------

    Pol : bool
        If True, the input files also contains polarization matrices. If False, the input files are just the temperature matrices.
    T_Pol : bool
        If True, the input files are both the temperature and polarization matrices. If False, the input files depends on the value of Pol.
    Input_PATH : str
        Path to the input files.
    NJ : int
        Number of files.
    lmax : int
        Maximum multipole moment.
    Chunks : int
        Size of the chunks.

    Returns
    -------
    x_TT : dask.array.core.Array
        Dask array with the temperature matrices. If Pol and TPol are False.
    x_Pol : dask.array.core.Array
        Dask array with the Q and U polarization matrices. If Pol is True and TPol is False.
    x_TPol : dask.array.core.Array
        Dask array with the temperature and polarization matrices. If TPol is True.

    """

    load = dask.delayed(np.load)
    if (Pol == False):
        globals()["filenames_TT"] = [str(Input_PATH)+"/Cov_TT_lmax_"+str(lmax)+"_"+str(i)+".npy" if NJ!=1 else str(Input_PATH)+"/Cov_TT_lmax_"+str(lmax)+".npy" for i in range(NJ)]
        globals()["lazy_arrays_TT"] = [load(path) for path in filenames_TT]
        sample = lazy_arrays_TT[0].compute()
        globals()["x_TT"] = da.concatenate([da.from_delayed(lazy_arrays, dtype = sample.dtype, shape = sample.shape) for lazy_arrays in lazy_arrays_TT])
        """
        globals()["x_TT"] = globals()["x_TT"]+da.tril(globals()["x_TT"], -1).T
        """
        globals()["x_TT"] = da.rechunk(globals()["x_TT"], chunks = (Chunks, Chunks))
        return (x_TT)
    else:
        if (T_Pol == False):
            Dic = ["QQ", "UU", "QU", "UQ"]

            Cross_1 = [0, 1, 2, 3]
            Cross_2 = [0, 1, 3, 2]
            
            for i in range(4):
                globals()["filenames_"+Dic[i]] = [str(Input_PATH)+"/Cov_"+Dic[i]+"_lmax_"+str(lmax)+"_"+str(j)+".npy" if NJ!=1 else str(Input_PATH)+"/Cov_"+Dic[i]+"_lmax_"+str(lmax)+".npy" for j in range(NJ)]
                globals()["lazy_arrays_"+Dic[i]] = [load(path) for path in globals()["filenames_"+Dic[i]]]
                if (i==0):
                    sample = lazy_arrays_QQ[0].compute()

                globals()["x_"+Dic[i]] = da.concatenate([da.from_delayed(lazy_arrays, dtype = sample.dtype, shape = sample.shape) for lazy_arrays in globals()["lazy_arrays_"+Dic[i]]], axis = 0)
            """
            for i in range(4):
                globals()["x_"+Dic[Cross_1[i]]] = globals()["x_"+Dic[Cross_1[i]]]+da.tril(globals()["x_"+Dic[Cross_2[i]]], -1).T
            """
            x_Pol = da.concatenate([da.concatenate([globals()["x_QQ"], globals()["x_QU"]], axis = 1), da.concatenate([globals()["x_UQ"], globals()["x_UU"]], axis = 1)], axis = 0)
            x_Pol = da.rechunk(x_Pol, chunks = (Chunks, Chunks))
            return x_Pol
        else:
            Dic = ["TT", "TQ", "TU", "QT", "UT", "QQ", "UU", "QU", "UQ"]

            Cross_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            Cross_2 = [0, 3, 4, 1, 2, 5, 6, 8, 7]

            for i in range(9):
                globals()["filenames_"+Dic[i]] =  [str(Input_PATH)+"/Cov_"+Dic[i]+"_lmax_"+str(lmax)+"_"+str(j)+".npy" if NJ!=1 else str(Input_PATH)+"/Cov_"+Dic[i]+"_lmax_"+str(lmax)+".npy" for j in range(NJ)]
                globals()["lazy_arrays_"+Dic[i]] = [load(path) for path in globals()["filenames_"+Dic[i]]]

                if (i == 0):
                    sample = lazy_arrays_TT[0].compute()

                globals()["x_"+Dic[i]] = da.concatenate([da.from_delayed(lazy_arrays, dtype = sample.dtype, shape = sample.shape) for lazy_arrays in globals()["lazy_arrays_"+Dic[i]]], axis = 0)
            """
            for i in range(9):
                globals()["x_"+Dic[Cross_1[i]]] = globals()["x_"+Dic[Cross_1[i]]]+da.tril(globals()["x_"+Dic[Cross_2[i]]], -1).T
            """
            x_TPol = da.concatenate([da.concatenate([globals()["x_TT"], globals()["x_TQ"], globals()["x_TU"]], axis = 1), da.concatenate([globals()["x_QT"], globals()["x_QQ"], globals()["x_QU"]], axis = 1), da.concatenate([globals()["x_UT"], globals()["x_UQ"], globals()["x_UU"]], axis = 1)], axis = 0)
            x_TPol = da.rechunk(x_TPol, chunks = (Chunks, Chunks))
            return x_TPol
    
def Permutations(x, Masks, Pol = False):
    
    """This function permutes the rows and columns of the input matrix according to the input masks. Rows and columns corresponding to masked pixels are
    placed in the final positions of the matrix. The function takes as input the matrix to be permuted, the masks and a boolean variable Pol. 
    If Pol is False, the mask is the temperature mask. If Pol is True, masks are the temperature and polarization masks. 
    
    Parameters
    ----------
    x : dask.array.core.Array
        Input matrix to be permuted.
    Masks : np.array or list
        Input masks.
    Pol : bool, optional
        If False, the mask is the temperature mask. If True, masks are the temperature and polarization masks. The default is False.
    
    Returns
    -------
    M3 : dask.array.core.Array
        Permuted matrix.
        
    """
    if (Pol == False):
        Int_Mask = Masks
        Ind1 = np.where(Int_Mask == 0)[0]
        Ind2 = np.where(Int_Mask == 1)[0]
        M3 = da.concatenate((x[Ind2, :], x[Ind1, :]), axis = 0)
        M3 = da.concatenate((M3[:, Ind2], M3[:, Ind1]), axis = 1)
        return M3
    else:
        if (len(Masks) == 2):
            Int_Mask = Masks[0]
            Pol_Mask = Masks[1]
            Ind_T_1 = np.where(Int_Mask == 0)[0]
            Ind_T_2 = np.where(Int_Mask == 1)[0]
            Ind_Pol_1 = np.where(Pol_Mask == 0)[0]
            Ind_Pol_2 = np.where(Pol_Mask == 1)[0]
            M3 = da.concatenate((da.concatenate((x[Ind_T_2], x[len(Int_Mask)+Ind_Pol_2], x[2*len(Int_Mask)+Ind_Pol_2]), axis = 0), da.concatenate((x[Ind_T_1], x[len(Int_Mask)+Ind_Pol_1], x[2*len(Int_Mask)+Ind_Pol_1]), axis = 0)), axis = 0)
            M3 = da.concatenate((da.concatenate((M3[:, Ind_T_2], M3[:, len(Int_Mask)+Ind_Pol_2], M3[:, 2*len(Int_Mask)+Ind_Pol_2]), axis = 1), da.concatenate((M3[:, Ind_T_1], M3[:, len(Int_Mask)+Ind_Pol_1], M3[:, 2*len(Int_Mask)+Ind_Pol_1]), axis = 1)), axis = 1)
            return M3
        else:
            Pol_Mask = Masks
            Ind_Pol_1 = np.where(Pol_Mask == 0)[0]
            Ind_Pol_2 = np.where(Pol_Mask == 1)[0]
            M3 = da.concatenate((da.concatenate((x[Ind_Pol_2], x[len(Pol_Mask)+Ind_Pol_2]), axis = 0), da.concatenate((x[Ind_Pol_1], x[len(Pol_Mask)+Ind_Pol_1]), axis = 0)), axis = 0)
            M3 = da.concatenate((da.concatenate((M3[:, Ind_Pol_2], M3[:, len(Pol_Mask)+Ind_Pol_2]), axis = 1), da.concatenate((M3[:, Ind_Pol_1], M3[:, len(Pol_Mask)+Ind_Pol_1]), axis = 1)), axis = 1)
            return M3
            
def Add_Reg_Noise(x, Noise_Level, Check = False):

    """This function adds regularize noise to the input matrix. 
    Covariance matrix should be invertible and definite positive. If not, small noise is added to the diagonal of the matrix.
    
    Parameters
    ----------
    x : dask.array.core.Array
        Input matrix.
    Noise_Level : float
        Noise level in percentage.
    Check : bool, optional 
        If True, the function checks if the input matrix is non-singular and definite positive. 
        For large matrices it takes a lot of time. If matrix is singular or not definite positive, small noise is added according to the input Noise_Level. 
        On the other hand, if matrix is non-singular and definite positive from the beggining, output is the matrix itself.
        If not, small noise is directly added to the diagonal of the matrix. The default is False.
    
    Returns
    -------
    x : dask.array.core.Array
        Matrix with added noise (if it is needed).

    """
    if (Check == True):
        eigenvalues = da.linalg.eigvalsh(x)
        is_positive_definite = da.all(eigenvalues > 0).compute()
        if (is_positive_definite == True):
            return x
        else:
            x = x + (Noise_Level/100)*da.tril(da.triu(x))
            return x
    else:
        x = x + (Noise_Level/100)*da.tril(da.triu(x))
        return (x)

def Cholesky(x):

    """ This function computes the Cholesky decomposition of the input matrix.

    Parameters
    ----------
    x : dask.array.core.Array
        Input matrix.
    
    Returns
    -------
    Chol : dask.array.core.Array
        Cholesky decomposition of the input matrix.
        
    """
    Chol = da.linalg.cholesky(x, lower = True)
    return Chol


__docformat__ = "numpy"
