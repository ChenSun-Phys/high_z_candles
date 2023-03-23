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
from igm import igm_Psurv
from icm import ne_2beta, B_icm, icm_los_Psurv
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


def LumMod(ma, g, z, B, mg, h, OmL, w0=-1., wa=0.,
           s=1.,
           omega=1.,
           axion_ini_frac=0.,
           smoothed=False,
           redshift_dependent=True,
           method='simps',
           prob_func='norm_log',
           Nz=501,
           skip_LumMod=False):
    """Here we use a simple function to modify the intrinsic luminosity of the SN

    :param ma:  axion mass [eV]
    :param g: axion photon coupling  [1/GeV]
    :param z: redshift, could be scalar or array. Array is preferred for fast vectorization. 
    :param B: magnetic field, today [nG]
    :param mg: photon mass [eV]
    :param h: Hubble [100 km/s/Mpc]
    :param OmL: Omega_Lambda
    :param w0: equation of state of the dark energy today (default: -1.)
    :param wa: parametrizes how w changes over time, w = w0 + wa*(1-a)  (default: 0.)
    :param s: domain size [Mpc]
    :param omega: energy [eV]
    :param axion_ini_frac: 
    :param smoothed: 
    :param redshift_dependent: 
    :param method: (simps, quad, old) for scalar z, or 'vectorize' if z is an array.
    :param prob_func: 
    :param Nz: 
    :param skip_LumMod: if switched on, return zero directly. This is useful for runs that do not involve axions. (Default: False)

    Returns
    -------
    res: scalar, delta M in the note
    """
    if not skip_LumMod:
        try:
            # 2.5log10(L/L(1e-5Mpc))
            res = 2.5 * log10(igm_Psurv(ma, g, z,
                                        s=s,
                                        B=B,
                                        omega=omega,
                                        mg=mg,
                                        h=h,
                                        Omega_L=OmL,
                                        w0=w0,
                                        wa=wa,
                                        axion_ini_frac=axion_ini_frac,
                                        smoothed=smoothed,
                                        redshift_dependent=redshift_dependent,
                                        method=method,
                                        prob_func=prob_func,
                                        Nz=Nz))

        except Warning:
            print('ma=%e, g=%e' % (ma, g))
            raise Exception('Overflow!!!')
    else:
        res = 0.

    return res


def ADDMod(ma, g, z, h, OmL, w0=-1., wa=0.,

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

    if ICM_effect:

        # ICMdomain
        if varying_ICMdomain:
            r_Arr_raw = lst_r_Arr_raw[galaxy_index]
            L_Arr_raw = lst_L_Arr_raw[galaxy_index]
            sintheta_Arr_raw = lst_sintheta_Arr_raw[galaxy_index]
        else:
            r_Arr_raw = None
            L_Arr_raw = None
            sintheta_Arr_raw = None

        PICM = icm_los_Psurv(ma, g, r_low, r_up, ne_2beta, B_icm,
                             L=L,
                             omega_Xrays=omegaX/1000.,
                             axion_ini_frac=0.,
                             smoothed=smoothed_ICM, method=method_ICM, return_arrays=return_arrays, prob_func=prob_func_ICM, Nr=Nr_ICM, los_method=los_method, los_use_prepared_arrays=los_use_prepared_arrays, los_Nr=los_Nr,
                             # B_icm
                             B_ref=B_ref, r_ref=r_ref, eta=eta,
                             # ne_2beta
                             ne0=ne0, rc_outer=rc_outer, beta_outer=beta_outer, f_inner=f_inner, rc_inner=rc_inner, beta_inner=beta_inner,
                             # ICMdomain
                             r_Arr_raw=r_Arr_raw,
                             L_Arr_raw=L_Arr_raw,
                             sintheta_Arr_raw=sintheta_Arr_raw,
                             varying_ICMdomain=varying_ICMdomain,
                             )

        Pg, Pa = PICM, 1.-PICM
        IaIg = Pa/Pg

    else:
        Pg = 1.
        IaIg = 0.

    Pgg_X = igm_Psurv(ma, g, z,
                      s=sIGM,
                      B=BIGM,
                      omega=omegaX,
                      mg=mgIGM,
                      h=h,
                      Omega_L=OmL,
                      w0=w0,
                      wa=wa,
                      axion_ini_frac=IaIg,
                      smoothed=smoothed_IGM,
                      redshift_dependent=redshift_dependent,
                      method=method_IGM,
                      prob_func=prob_func_IGM,
                      Nz=Nz_IGM)

    Pgg_CMB = igm_Psurv(ma, g, z,
                        s=sIGM,
                        B=BIGM,
                        omega=omegaCMB,
                        mg=mgIGM,
                        h=h,
                        Omega_L=OmL,
                        w0=w0,
                        wa=wa,
                        axion_ini_frac=0.,
                        smoothed=smoothed_IGM,
                        redshift_dependent=redshift_dependent,
                        method=method_IGM,
                        prob_func=prob_func_IGM,
                        Nz=Nz_IGM)

    return Pgg_CMB**2. / (Pgg_X * Pg)
