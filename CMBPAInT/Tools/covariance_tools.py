#!/usr/bin/env python
# coding: utf-8

"""
Created on Wed Apr 26 09:35:10 2023

@author: Christian Gimeno Amo (gimenoc@ifca.unican.es)

This module store all the necessary functions to generate the pixel covariance matrices given an lmax and the power spectra.

Dependencies:

- numpy
- scipy

"""

import numpy as np
import scipy

def F_10_v(x, lmax):
    """Find the value of F_10_v for a given lmax and distance between two pixels, x.
    
    Parameters
    ----------
    x : np.array
        Array with the distances between couple of pixels.
    lmax : int
        Maximum multipole moment.

    Returns:
    --------
    F_10_v : np.array
        Array with the values of F_10_v for each multipole moment.

    """
    l = np.arange(2, lmax+1)
    if (abs(x) == 1.0):
        return (np.zeros(len(l)))
    else:
        return (np.ones(len(l))*(2*(((l*x/(1-x**2))*scipy.special.lpmv(0, l-1, x)-(l/(1-x**2)+(l*(l-1))/2)*scipy.special.lpmv(0, l, x))/((l-1)*l*(l+1)*(l+2))**(1/2))))
    
def F_10_v2(x, lmax, nint, nout):
    """Convert F_10_v Python function into a NumPy Universal Function (ufunc).

    Parameters  
    ----------
    x : float
        Distance between two pixels.
    lmax : int
        Maximum multipole moment.
    nint : int
        Number of input arguments.
    nout : int
        Number of output arguments.

    Returns
    -------
    F_10_v2 : ufunc
        NumPy Universal Function (ufunc) of F_10_v.

    """
    return np.frompyfunc(F_10_v, nint, nout)(x, lmax)

def F_12_v(x, lmax):
    """Find the value of F_12_v for a given lmax and distance between two pixels, x.
    
    Parameters
    ----------
    x : np.array
        Array with the distances between couple of pixels.
    lmax : int
        Maximum multipole moment.

    Returns:
    --------
    F_12_v : np.array
        Array with the values of F_12_v for each multipole moment.

    """
    l = np.arange(2, lmax+1)
    if (x == 1.0):
        return (0.5*np.ones(lmax-1))
    elif (x == -1.0):
        return (0.5*np.power((-1), l))
    else:
        return (np.ones(lmax-1)*(2*((((l+2)*x/(1-x**2))*scipy.special.lpmv(2, l-1, x)-((l-4)/(1-x**2)+(l*(l-1))/2)*scipy.special.lpmv(2, l, x))/((l-1)*l*(l+1)*(l+2)))))

def F_12_v2(x, lmax, nint, nout):
    """Convert F_12_v Python function into a NumPy Universal Function (ufunc).

    Parameters  
    ----------
    x : float
        Distance between two pixels.
    lmax : int
        Maximum multipole moment.
    nint : int
        Number of input arguments.
    nout : int
        Number of output arguments.

    Returns
    -------
    F_12_v2 : ufunc
        NumPy Universal Function (ufunc) of F_12_v.

    """
    return np.frompyfunc(F_12_v, nint, nout)(x, lmax)

def F_22_v(x, lmax):
    """Find the value of F_22_v for a given lmax and distance between two pixels, x.
    
    Parameters
    ----------
    x : np.array
        Array with the distances between couple of pixels.
    lmax : int
        Maximum multipole moment.
    
    Returns:
    --------
    F_22_v : np.array
        Array with the values of F_22_v for each multipole moment.
    
    """
    l = np.arange(2, lmax+1)
    if (x == 1.0):
        return ((-0.5)*np.ones(len(l)))
    elif (x == -1.0):
        return (0.5*np.power((-1), l))
    else:
        return (np.ones(lmax-1)*(4*(((l+2)*scipy.special.lpmv(2, l-1, x)-(l-1)*x*scipy.special.lpmv(2, l, x))/((l-1)*l*(l+1)*(l+2)*(1-x**2)))))

def F_22_v2(x, lmax, nint, nout):
    """Convert F_22_v Python function into a NumPy Universal Function (ufunc).
    
    Parameters
    ----------
    x : np.array
        Array with the distances between couple of pixels.
    lmax : int
        Maximum multipole moment.
    nint : int  
        Number of input arguments.
    nout : int  
        Number of output arguments.

    Returns
    -------
    F_22_v2 : ufunc
        NumPy Universal Function (ufunc) of F_22_v.

    """
    return np.frompyfunc(F_22_v, nint, nout)(x, lmax)


def Scalar_MatCov(Cls, lmax, x, Npix = None):
    """Find the pixel covariance matrix for a scalar field. Only the lower triangular part of the matrix is calculated.

    Parameters
    ----------
    Cls : np.array
        Array with the values of the angular power spectrum.
    lmax : int
        Maximum multipole moment.
    x : np.array   
        Array with the distances between couple of pixels.
    Npix : int, optional
        Number of pixels. The default is None. This parameter is necesary if the angular power spectrum is constant, as in the white noise case.
    
    Returns
    -------
    Cov_M : np.array
        Pixel covariance matrix.
    
    """

    if (np.all(Cls == Cls[2])):
        if (len(x) == 1):
            Cov_M = Cls[2]*Npix/(4*np.pi)
        else:
            Cov_M = np.zeros(len(x))
            Cov_M[len(x)-1] = (Cls[2]*Npix/(4*np.pi))
    else:
        ell = np.arange(lmax+1)
        Vec = ((2*ell+1)/(4*np.pi))*Cls[ell]
        Cov_M = np.polynomial.legendre.Legendre(Vec)(x)
    return Cov_M

def Only_Pol_MatCov(Cls, lmax, x):
    """Find the pixel covariance matrix for the polarization field, Q and U stokes parameters. Only the lower triangular part of the matrix is calculated.

    Parameters
    ----------
    Cls : np.array
        Array with the values of the angular power spectrum of E and B autospectra and EB cross spectra.
    lmax : int
        Maximum multipole moment.
    x : np.array
        Array with the distances between couple of pixels.
    
    Returns
    -------
    Cov_M : np.array
        Q and U pixel covariance matrix.

    """

    Cls_E = Cls[0]
    Cls_B = Cls[1]
    Cls_EB = Cls[2]

    a = F_12_v2(x, lmax, 2, 1)
    b = F_22_v2(x, lmax, 2, 1)

    l = np.arange(2, lmax+1)
    if (Cls_EB != None):
        Cov_QQ = ((2*l+1)/(4*np.pi))*(np.vstack(a)*Cls_E[l]-np.vstack(b)*Cls_B[l])
        Cov_UU = ((2*l+1)/(4*np.pi))*(np.vstack(a)*Cls_B[l]-np.vstack(b)*Cls_E[l])
        Cov_QU = ((2*l+1)/(4*np.pi))*(np.vstack(a)+np.vstack(b))*Cls_EB[l]
        return (np.sum(Cov_QQ, axis = 1), np.sum(Cov_UU, axis = 1), np.sum(Cov_QU, axis = 1))
    else:
        Cov_QQ = ((2*l+1)/(4*np.pi))*(np.vstack(a)*Cls_E[l]-np.vstack(b)*Cls_B[l])
        Cov_UU = ((2*l+1)/(4*np.pi))*(np.vstack(a)*Cls_B[l]-np.vstack(b)*Cls_E[l])
        return (np.sum(Cov_QQ, axis = 1), np.sum(Cov_UU, axis = 1))
    
def Intensity_Pol_MatCov(Cls, lmax, x, Npix):
    """Find the pixel covariance matrix for the intensity and polarization fields, TQU. Only lower triangular part is calculated.

    Parameters
    ----------
    Cls : np.array
        Array with the values of the angular power spectrum of T, E and B, together with the TE, TB and EB cross spectra.
    lmax : int
        Maximum multipole moment.
    x : np.array
        Array with the distances between couple of pixels.

    Returns
    -------
    Cov_M : np.array
        T, Q and U pixel covariance matrix.

    """
    Cls_T = Cls[0]
    Cls_E = Cls[1]
    Cls_B = Cls[2]
    Cls_TE = Cls[3]
    Cls_EB = Cls[4]
    Cls_TB = Cls[5]
    if (Cls_EB != None):
        if (Cls_TB != None):
            Cov_QQ, Cov_UU, Cov_QU = Only_Pol_MatCov((Cls_E, Cls_B, Cls_EB), lmax = lmax, x = x)
            Cov_TT = Scalar_MatCov(Cls_T, lmax = lmax, x = x, Npix = Npix)

            N = len(x)
            c = F_10_v2(x, lmax, nint = 2, nout = 1)
            l = np.arange(2, lmax+1)
            Cov_TQ = -np.sum(((2*l+1)/(4*np.pi))*(np.vstack(c)*Cls_TE[l]), axis = 1)
            Cov_TU = -np.sum(((2*l+1)/(4*np.pi))*(np.vstack(c)*Cls_TB[l]), axis = 1)
            return (Cov_TT, Cov_QQ, Cov_UU, Cov_TQ, Cov_TU, Cov_QU)
        else:
            Cov_QQ, Cov_UU, Cov_QU = Only_Pol_MatCov((Cls_E, Cls_B, Cls_EB), lmax = lmax, x = x)
            Cov_TT = Scalar_MatCov(Cls_T, lmax = lmax, x = x, Npix = Npix)

            N = len(x)
            c = F_10_v2(x, lmax, nint = 2, nout = 1)
            l = np.arange(2, lmax+1)
            Cov_TQ = -np.sum(((2*l+1)/(4*np.pi))*(np.vstack(c)*Cls_TE[l]), axis = 1)
            return (Cov_TT, Cov_QQ, Cov_UU, Cov_TQ, np.zeros(Cov_TQ.shape), Cov_QU)
    else:
        if (Cls_TB != None):
            Cov_QQ, Cov_UU = Only_Pol_MatCov((Cls_E, Cls_B, Cls_EB), lmax = lmax, x = x)
            Cov_TT = Scalar_MatCov(Cls_T, lmax = lmax, x = x, Npix = Npix)

            N = len(x)
            c = F_10_v2(x, lmax, nint = 2, nout = 1)
            l = np.arange(2, lmax+1)

            Cov_TQ = -np.sum(((2*l+1)/(4*np.pi))*(np.vstack(c)*Cls_TE[l]), axis = 1)
            Cov_TU = -np.sum(((2*l+1)/(4*np.pi))*(np.vstack(c)*Cls_TB[l]), axis = 1)
            return (Cov_TT, Cov_QQ, Cov_UU, Cov_TQ, Cov_TU, np.zeros(Cov_QQ.shape))
        else:
            Cov_QQ, Cov_UU = Only_Pol_MatCov((Cls_E, Cls_B, Cls_EB), lmax = lmax, x = x)
            Cov_TT = Scalar_MatCov(Cls_T, lmax = lmax, x = x, Npix = Npix)

            N = len(x)
            c = F_10_v2(x, lmax, nint = 2, nout = 1)
            l = np.arange(2, lmax+1)

            Cov_TQ = -np.sum(((2*l+1)/(4*np.pi))*(np.vstack(c)*Cls_TE[l]), axis = 1)
            return (Cov_TT, Cov_QQ, Cov_UU, Cov_TQ, np.zeros(Cov_TQ.shape), np.zeros(Cov_QQ.shape))

def Rotation_Angle(u, v):
    """ Find the rotation angle between two vectors.
    
    Parameters
    ----------
    u : np.array
        First vector.
    v : np.array
        Second vector.
    Returns
    -------
    Ind2 : np.array
        Indices of the vectors with non-zero norm.
    alfa_ij : np.array
        Rotation angle between the two vectors.
    alfa_ji : np.array
        Rotation angle between the two vectors.
    
    """
    Norms = np.round(np.linalg.norm(np.cross(u, v.T), axis = 1), decimals = 15)
    Ind1 = np.where(Norms == 0.0)[0]
    Ind2 = np.where(Norms != 0.0)[0]
    z = np.array((0,0,1))
    r_ij = np.cross(u, v[:, Ind2].T)/Norms[Ind2][:, np.newaxis]
    r_ji = -r_ij
    r_i = (np.cross(z, u)/np.linalg.norm(np.cross(z, u)))
    r_j = np.cross(z, v[:, Ind2].T)/np.linalg.norm(np.cross(z, v[:, Ind2].T), axis = 1)[:, np.newaxis]
    Sign_ij = np.sign(np.dot(r_ij, z))
    Sign_ji = np.sign(np.dot(r_ji, z))

    alfa_ij = (-1.0)*Sign_ij*np.arccos(np.round(np.dot(r_ij, r_i.T), decimals = 12))
    alfa_ji = (-1.0)*Sign_ji*np.arccos(np.round((r_ji*r_j).sum(axis = 1), decimals = 12))

    return (Ind2, alfa_ij, alfa_ji)


def Rotations_Pol(ele_mat, u, v):
    """ Rotate the polarization Q and U covariance matrices.

    Parameters
    ----------
    ele_mat : tuple
        Tuple with the Q and U covariance matrices.
    u : np.array
        First vector.
    v : np.array
        Second vector.
    Returns
    -------
    QQ : np.array
        Rotated Q covariance matrix.
    UU : np.array
        Rotated U covariance matrix.
    QU : np.array
        Rotated QU covariance matrix.
    UQ : np.array
        Rotated UQ covariance matrix.

    """
    N = len(v.T)
    Ind, alfa_ij, alfa_ji = Rotation_Angle(u, v)
    if (len(ele_mat) == 2):
        QQ = np.full(N, ele_mat[0], dtype = np.float64)
        UU = np.full(N, ele_mat[1], dtype = np.float64)
        QU = np.full(N, 0.0, dtype = np.float64)
        UQ = np.full(N, 0.0, dtype = np.float64)

        QQ[Ind] = np.cos(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[0][Ind])+np.sin(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[1][Ind])
        UU[Ind] = np.sin(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[0][Ind])+np.cos(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[1][Ind])

        UQ[Ind] = -np.cos(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[0][Ind])+np.sin(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[1][Ind])
        QU[Ind] = -np.sin(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[0][Ind])+np.cos(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[1][Ind])

    elif (len(ele_mat) == 3):
        QQ = np.full(N, ele_mat[0], dtype = np.float64)
        UU = np.full(N, ele_mat[1], dtype = np.float64)
        QU = np.full(N, ele_mat[2], dtype = np.float64)
        UQ = np.full(N, ele_mat[2], dtype = np.float64)

        QQ[Ind] = np.cos(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[0][Ind]+np.sin(2*alfa_ij)*ele_mat[2][Ind])+np.sin(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[2][Ind]+np.sin(2*alfa_ij)*ele_mat[1][Ind])
        UU[Ind] = np.sin(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[0][Ind]-np.cos(2*alfa_ij)*ele_mat[2][Ind])-np.cos(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[2][Ind]-np.cos(2*alfa_ij)*ele_mat[1][Ind])

        UQ[Ind] = -np.cos(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[0][Ind]-np.cos(2*alfa_ij)*ele_mat[2][Ind])-np.sin(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[2][Ind]-np.cos(2*alfa_ij)*ele_mat[1][Ind])
        QU[Ind] = -np.sin(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[0][Ind]+np.sin(2*alfa_ij)*ele_mat[2][Ind])+np.cos(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[2][Ind]+np.sin(2*alfa_ij)*ele_mat[1][Ind])

    return (np.round(QQ, decimals = 15), np.round(UU, decimals = 15), np.round(QU, decimals = 15), np.round(UQ, decimals = 15))


def Rotations_TPol(ele_mat, u, v):
    """ Rotate the polarization Q and U covariance matrices and also the TQ and TU matrices. 

    Parameters
    ----------
    ele_mat : tuple
        Tuple with the Q and U covariance matrices.
    u : np.array
        First vector.
    v : np.array
        Second vector.
    Returns
    -------
    TQ: np.array
        Rotated TQ covariance matrix.
    TU: np.array
        Rotated TU covariance matrix.
    QT : np.array
        Rotated QT covariance matrix.
    UT : np.array
        Rotated UT covariance matrix.  
    QQ : np.array
        Rotated Q covariance matrix.
    UU : np.array
        Rotated U covariance matrix.
    QU : np.array
        Rotated QU covariance matrix.
    UQ : np.array
        Rotated UQ covariance matrix.

    """
    N = len(v.T)
    (Ind, alfa_ij, alfa_ji) = Rotation_Angle(u, v)

    TQ = np.full(N, ele_mat[0], dtype = np.float64)
    TU = np.full(N, ele_mat[1], dtype = np.float64)
    QT = np.full(N, ele_mat[0], dtype = np.float64)
    UT = np.full(N, ele_mat[1], dtype = np.float64)
    QQ = np.full(N, ele_mat[2], dtype = np.float64)
    UU = np.full(N, ele_mat[3], dtype = np.float64)
    QU = np.full(N, ele_mat[4], dtype = np.float64)
    UQ = np.full(N, ele_mat[4], dtype = np.float64)

    TQ[Ind] = np.cos(2*alfa_ji)*ele_mat[0][Ind]+np.sin(2*alfa_ji)*ele_mat[1][Ind]
    QT[Ind] = np.cos(2*alfa_ij)*ele_mat[0][Ind]+np.sin(2*alfa_ij)*ele_mat[1][Ind]
    TU[Ind] = -np.sin(2*alfa_ji)*ele_mat[0][Ind]+np.cos(2*alfa_ji)*ele_mat[1][Ind]
    UT[Ind] = -np.sin(2*alfa_ij)*ele_mat[0][Ind]+np.cos(2*alfa_ij)*ele_mat[1][Ind]

    QQ[Ind] = np.cos(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[2][Ind]+np.sin(2*alfa_ij)*ele_mat[4][Ind])+np.sin(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[4][Ind]+np.sin(2*alfa_ij)*ele_mat[3][Ind])
    UU[Ind] = np.sin(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[2][Ind]-np.cos(2*alfa_ij)*ele_mat[4][Ind])-np.cos(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[4][Ind]-np.cos(2*alfa_ij)*ele_mat[3][Ind])

    UQ[Ind] = -np.cos(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[2][Ind]-np.cos(2*alfa_ij)*ele_mat[4][Ind])-np.sin(2*alfa_ji)*(np.sin(2*alfa_ij)*ele_mat[4][Ind]-np.cos(2*alfa_ij)*ele_mat[3][Ind])
    QU[Ind] = -np.sin(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[2][Ind]+np.sin(2*alfa_ij)*ele_mat[4][Ind])+np.cos(2*alfa_ji)*(np.cos(2*alfa_ij)*ele_mat[4][Ind]+np.sin(2*alfa_ij)*ele_mat[3][Ind])

    return (np.round(TQ, decimals = 15), np.round(QT, decimals = 15), np.round(TU, decimals = 15), np.round(UT, decimals = 15), np.round(QQ, decimals = 15), np.round(UU, decimals = 15), np.round(QU, decimals = 15), np.round(UQ, decimals = 15))


__docformat__ = "numpy"
