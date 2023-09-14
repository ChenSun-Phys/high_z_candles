###########################################
###    Code for cosmology functions     ###
###    by Manuel A. Buen-Abad, 2020     ###
###         and Chen Sun, 2020          ###
###########################################

from __future__ import division

import numpy as np

from numpy import pi, sqrt, log, log10, exp, power
from scipy.integrate import simps, quad

from ag_probs import omega_plasma, P0, Pnaive
from igm import igm_Psurv
from icm import ne_2beta, B_icm, icm_los_Psurv
from tools import treat_as_arr

# CONSTANTS:

_c_ = 299792458.  # [m/s]
_alpha_ = 1./137  # fine structure constant
_me_ = 510998.95  # electron mass in eV
_1_over_cm_eV_ = 1.9732698045930252e-5  # [1/cm/eV]


# FUNCTIONS:

# def Ekernel(a2, a3, z):
#     try:
#         res, _ = quad(lambda zp: 1 / sqrt(OmL + (1 - OmL) * (1 + zp)**3), 0, z)
#     except Warning:
#         print('OmL=%e, z=%e' % (OmL, z))
#         raise Exception
#     return res


def H_at_z_exact(z, h0, OmL, unit='Mpc'):
    """
    Hubble at z 

    :param z: redshift
    :param h0:  H in [100*km/s/Mpc]
    :param OmL: Omega_Lambda
    :param unit: flag to change the output unit
    :returns: H [1/Mpc] by default, or H [km/s/Mpc]

    """
    if unit == 'Mpc':
        res = h0*100.*sqrt(OmL + (1 - OmL) * (1 + z)**3)/(_c_/1000.)
    else:
        res = h0*100.*sqrt(OmL + (1 - OmL) * (1 + z)**3)
    return res


def H_at_z_naturallog(z, h0, a2, a3, a4, unit='Mpc'):
    """Hubble at z expanded with powers of ln(1+z).Rederived by hand for crosscheck. The conversion to 10 based expansion is 
    a2_naturalbase = a2_10base/ln(10), 
    a3_naturalbase = a3_10base/ln(10)**2.
    a4_naturalbase = a4_10base/ln(10)**3.

    :param z: redshift
    :param h0:  H in [100*km/s/Mpc]
    :param a2: the second coefficient of the log(1+z) expansion
    :param a3: the third coefficient of the log(1+z) expansion
    :param a4: the fourth coefficient of the log(1+z) expansion    
    :param unit: flag to change the output unit
    :returns: H [1/Mpc] by default, or H [km/s/Mpc]

    """
    x = np.log(1.+z)
    shape = (1.+z)**2 / (1. + (2.*a2-1.)*x + (3.*a3-a2)
                         * x**2 + (4.*a4-a3)*x**3-a4*x**4)
    if unit == 'Mpc':
        res = h0*100.*shape/(_c_/1000.)
    else:
        res = h0*100.*shape
    return res


def H_at_z_cross_check(z, h0, a2, a3, a4, unit='Mpc'):
    """Hubble at z expanded with powers of log10(1+z)

    :param z: redshift
    :param h0:  H in [100*km/s/Mpc]
    :param a2: the second coefficient of the log10(1+z) expansion
    :param a3: the third coefficient of the log10(1+z) expansion
    :param unit: flag to change the output unit
    :returns: H [1/Mpc] by default, or H [km/s/Mpc]

    """
    # a2-a3
    # shape = -(1.+z)**2*np.log(10)**2/(-np.log(10)**2+np.log(10)*(-2.*a2+np.log(10))
    #                                   * np.log(1+z)+(-3.*a3+a2*np.log(10))*np.log(1+z)**2+a3*np.log(1.+z)**3)
    # TODO: a2 a3 a4
    # if unit == 'Mpc':
    #     res = shape*(h0*100/(_c_/1000.))
    # else:
    #     # unit of (km/s)/Mpc
    #     res = shape*(h0*100.)
    raise Exception('this funciton is only for testing purpose. ')
    return res


def H_at_z(z, h0, a2, a3, a4, unit='Mpc'):
    """Hubble at z expanded with powers of log10(1+z)

    :param z: redshift
    :param h0:  H in [100*km/s/Mpc]
    :param a2: the second coefficient of the log10(1+z) expansion
    :param a3: the third coefficient of the log10(1+z) expansion
    :param a4: the fourth coefficient of the log10(1+z) expansion    
    :param unit: flag to change the output unit
    :returns: H [1/Mpc] by default, or H [km/s/Mpc]

    """
    x = np.log10(1.+z)
    # shape = -(1+z)**2/(-1+x*(-2.*a2-3.*a3*x+(1.+x*(a2+a3*x))*np.log(10)))
    shape = (1.+z)**2/(1.
                       + (2.*a2-np.log(10.))*x
                       + (3.*a3 - a2*np.log(10))*x**2
                       + (4.*a4 - a3*np.log(10))*x**3
                       - (a4*np.log(10)) * x**4)
    if unit == 'Mpc':
        res = shape*(h0*100/(_c_/1000.))
    else:
        # unit of (km/s)/Mpc
        res = shape*(h0*100.)
    return res


def dL_at_z(z, h0, a2, a3, a4):
    """compute the luminosity distance, return in Mpc

    :param z: redshfit
    :param h0: Hubble in 100 km/s/Mpc
    :param a2: the second coefficient of the log10(1+z) expansion
    :param a3: the third coefficient of the log10(1+z) expansion
    :param a4: the fourth coefficient of the log10(1+z) expansion    

    """
    x = np.log10(1.+z)
    res = np.log(10)*(_c_/1000.)/(h0*100.)*(x + a2*x**2 + a3*x**3 + a4*x**4)
    return res


def tau_at_z(z, h0, a2, a3, a4):
    """Compute the comoving distance, return in Mpc

    Parameters
    ----------
    :param z : scalar
        redshift
    :param h0 : scalar
        Hubble in 100 km/s/Mpc
    :param a2: the second coefficient of the log10(1+z) expansion
    :param a3: the third coefficient of the log10(1+z) expansion
    :param a4: the fourth coefficient of the log10(1+z) expansion    
    """
    res = dL_at_z(z, h0, a2, a3, a4)/(1.+z)
    return res


def tau_at_z_exact(z, h0, OmL):
    """
    Compute the comoving distance, return in Mpc

    Parameters
    ----------
    z : scalar
        redshift
    h0 : scalar
        Hubble in 100 km/s/Mpc
    OmL : scalar
        Omega_Lambda

    """
    try:
        res, _ = quad(lambda zp: 1. / sqrt(OmL +
                                           (1 - OmL) * (1 + zp)**3), 0., z)
    except Warning:
        print('OmL=%e, z=%e' % (OmL, z))
        raise Exception
    res = res * _c_/1e5/h0
    return res


def dA_at_z(z, h0, a2, a3, a4):
    """
    Angular distance [Mpc]

    :param z: redshift
    :param h0: H in [100*km/s/Mpc]
    :returns: angular distance [Mpc]

    """
    return dL_at_z(z, h0, a2, a3, a4)/(1.+z)**2


def muLCDM(z, h0, a2, a3, a4):
    """distance modulus defined as 5*log10(DL/10pc)
    """
    res = 5.*np.log10(dL_at_z(z, h0, a2, a3, a4)*1.e5)
    return res


def LumMod(ma, g, z, B, mg, h0, a2, a3, a4,
           s=1.,
           omega=1.,
           axion_ini_frac=0.,
           smoothed=False,
           redshift_dependent=True,
           method='simps',
           prob_func='norm_log',
           Nz=501):
    raise Exception('Not implemented!')


def ADDMod(ma, g, z, h0, a2, a3, a4,

           omegaX=1.e4,
           omegaCMB=2.4e-4,

           # IGM
           sIGM=1.,
           BIGM=1.,
           mgIGM=3.e-15,
           smoothed_IGM=False,
           redshift_dependent=True,
           method_IGM='simps',
           prob_func_IGM='norm_log',
           Nz_IGM=501,

           # ICM
           ICM_effect=False,
           r_low=0.,
           r_up=1800.,
           L=10.,
           smoothed_ICM=False,
           method_ICM='product',
           return_arrays=False,
           prob_func_ICM='norm_log',
           Nr_ICM=501,
           los_method='quad',
           los_use_prepared_arrays=False,
           los_Nr=501,

           # ICMdomain
           lst_r_Arr_raw=None,
           lst_L_Arr_raw=None,
           lst_sintheta_Arr_raw=None,
           varying_ICMdomain=None,
           galaxy_index=None,

           # B_icm
           B_ref=10.,
           r_ref=0.,
           eta=0.5,

           # ne_2beta
           ne0=0.01,
           rc_outer=100.,
           beta_outer=1.,
           f_inner=0.,
           rc_inner=10.,
           beta_inner=1.):
    """
    Function that modifies the ADDs from clusters, written in Eq. 12 of Manuel's notes.
    """

    raise Exception('Not implemented!')
