###########################################
###    Code for cosmology functions     ###
###  by Manuel A. Buen-Abad, 2020-2023  ###
###       and Chen Sun, 2020-2023       ###
###########################################

from __future__ import division

import numpy as np

from numpy import pi, sqrt, log, log10, exp, power
from scipy.integrate import simps, quad

from ag_probs import omega_plasma, P0, Pnaive
# from igm import igm_Psurv
# from icm import ne_2beta, B_icm, icm_los_Psurv
from tools import treat_as_arr

# CONSTANTS:

_c_ = 299792458.  # [m/s]
_alpha_ = 1./137  # fine structure constant
_me_ = 510998.95  # electron mass in eV
_1_over_cm_eV_ = 1.9732698045930252e-5  # [1/cm/eV]


# FUNCTIONS:

def Ekernel(Omega_L, z, w0=-1., wa=0.):
    """E(z) kernel, without integral. H = H0 * E(z)

    :param Omega_L: cosmological constant fractional density
    :param z: redshift
    :param w0: equation of state of the dark energy today (default: -1.)
    :param wa: parametrizes how w changes over time, w = w0 + wa*(1-a)  (default: 0.)

    """
    Omega_m = 1. - Omega_L

    return sqrt(Omega_L * (1.+z)**(3*(w0 + z/(1.+z)*wa)+3.)
                + Omega_m * (1.+z)**3)


def Ekernel_int(OmL, z, w0=-1., wa=0.):
    """The integral of 1/E(z) from 0 to z. 

    :param OmL: Omega_Lambda
    :param z: redshift
    :param w0: equation of state of the dark energy today (default: -1.)
    :param wa: parametrizes how w changes over time, w = w0 + wa*(1-a)  (default: 0.)

    """
    try:
        res, _ = quad(lambda zp: 1 / Ekernel(OmL, zp, w0, wa), 0, z)
    except Warning:
        print('OmL=%e, z=%e' % (OmL, z))
        raise Exception
    return res


def H_at_z(z, h0, OmL, w0=-1., wa=0., unit='Mpc'):
    """Hubble at z 

    :param z: redshift
    :param h0:  H in [100*km/s/Mpc]
    :param OmL: Omega_Lambda
    :param w0: equation of state of the dark energy today (default: -1.)
    :param wa: parametrizes how w changes over time, w = w0 + wa*(1-a)  (default: 0.)
    :param unit: flag to change the output unit. (Default: 'Mpc')
    :returns: H [1/Mpc] by default, or H [km/s/Mpc]
    """
    res = h0*100.*Ekernel(OmL, z, w0, wa)

    if unit == 'Mpc':
        res = res/(_c_/1000.)

    return res


def dL_at_z_a2a3(z, h0, a2, a3):
    """compute the luminosity distance by expanding ln(1+z) up to the second order, return in Mpc

    :param z: redshfit
    :param h0: Hubble in 100 km/s/Mpc
    :param a2: the second coefficient of the log(1+z) expansion
    :param a3: the third coefficient of the log(1+z) expansion

    """
    x = np.log10(1.+z)
    res = np.log(10)*(_c_/1000.)/(h0*100.)*(x + a2*x**2 + a3*x**3)
    return res


def tau_at_z(z, h0, OmL, w0=-1., wa=0.):
    """Compute the comoving distance, return in Mpc

    :param z: redshift
    :param h0: Hubble in 100 km/s/Mpc
    :param OmL: Omega_Lambda
    :param w0: equation of state of the dark energy today (default: -1.)
    :param wa: parametrizes how w changes over time, w = w0 + wa*(1-a)  (default: 0.)

    """
    # try:
    #     res, _ = quad(lambda zp: 1. / sqrt(OmL * (1+zp)**(3*(w0+zp/(1.+zp)*wa)+3.) +
    #                                        (1 - OmL) * (1 + zp)**3), 0., z)
    # except Warning:
    #     print('OmL=%e, z=%e, w0=%e, wa=%e' % (OmL, z, w0, wa))
    #     raise Exception
    res = Ekernel_int(OmL, z, w0, wa)
    res = res * _c_/1e5/h0
    return res


def dA_at_z(z, h0, OmL, w0=-1., wa=0.):
    """Angular distance [Mpc]

    :param z: redshift
    :param h0: H in [100*km/s/Mpc]
    :param OmL: Omega_Lambda
    :param w0: equation of state of the dark energy today (default: -1.)
    :param wa: parametrizes how w changes over time, w = w0 + wa*(1-a)  (default: 0.)

    :returns: angular distance [Mpc]
    """
    return tau_at_z(z, h0, OmL, w0=w0, wa=wa)/(1.+z)


def dL_at_z(z, h0, OmL, w0=-1., wa=0.):
    """Luminosity distance [Mpc]

    :param z: redshift
    :param h0: H in [100*km/s/Mpc]
    :param OmL: Omega_Lambda
    :param w0: equation of state of the dark energy today (default: -1.)
    :param wa: parametrizes how w changes over time, w = w0 + wa*(1-a)  (default: 0.)

    :returns: luminosity distance [Mpc]
    """

    return tau_at_z(z, h0, OmL, w0=w0, wa=wa)*(1.+z)


def distance_modulus(z, h0, OmL, w0=-1., wa=0.):
    """The distance modulus

    :param z: redshift
    :param h0: Hubble constant [100 km/s/Mpc]
    :param OmL: Omega_Lambda
    :param w0: equation of state of the dark energy today (default: -1.)
    :param wa: parametrizes how w changes over time, w = w0 + wa*(1-a)  (default: 0.)


    """
    return 5. * log10(dL_at_z(z, h0, OmL, w0, wa)) + 25
