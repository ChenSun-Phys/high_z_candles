#######################################################
###    Code for probabilities for axion-photon      ###
###              conversion in the IGM              ###
###          by Manuel A. Buen-Abad, 2020           ###
###               and Chen Sun, 2020                ###
#######################################################

from __future__ import division

import numpy as np

from numpy import pi, sqrt, log, log10, exp, power
from scipy.integrate import simps, quad
from ag_probs import P0
from tools import treat_as_arr
from cosmo import tau_at_z, Ekernel

# CONSTANTS AND CONVERSION FACTORS:

c0 = 299792458.  # [m/s] speed of light
aem = 1./137  # QED coupling constant
meeV = (0.51099895 * 1.e6)  # [eV] electron mass
hbarc = 197.32698045930252e-18  # [GeV*cm]
GeV_over_eV = 1.e9  # GeV/eV

GeV_times_m = 1./hbarc  # GeV*m conversion
eV_times_cm = GeV_times_m * 1.e-11  # eV*cm conversion

Mpc_over_m = 3.085677581282e22  # Mpc/m conversion
Mpc_times_GeV = GeV_times_m * Mpc_over_m  # Mpc*GeV conversion
G_over_eV2 = 1.95e-2  # G/eV^2 conversion # MANUEL: NOTE: FIND MORE ACCURATE VALUE

Mpc_times_eV = Mpc_times_GeV/GeV_over_eV  # Mpc*eV conversion


# FUNCTIONS:

def igm_Psurv(ma, g, z,
              s=1.,
              B=1.,
              omega=1.,
              mg=3.e-15,
              h=0.7,
              Omega_L=0.7,
              w0=-1.,
              wa=0.,
              axion_ini_frac=0.,
              smoothed=False,
              redshift_dependent=True,
              method='simps',
              prob_func='norm_log',
              Nz=501):
    """Photon IGM survival probability.

    :param ma: axion mass [eV]
    :param g: axion-photon coupling [GeV^-2]
    :param z: redshift
    :param s: magnetic domain size, today [Mpc] (default: 1.)
    :param B: magnetic field, today [nG] (default: 1.)
    :param omega: photon energy, today [eV] (default: 1.)
    :param mg: photon mass [eV] (default: 3.e-15)
    :param h: reduced Hubble parameter H0/100 [km/s/Mpc] (default: 0.7)
    :param Omega_L: cosmological constant fractional density (default: 0.7)
    :param w0: equation of state of the dark energy today (default: -1.)
    :param wa: parametrizes how w changes over time, w = w0 + wa*(1-a)  (default: 0.)    
    :param axion_ini_frac: the initial intensity fraction of axions: I_axion/I_photon (default: 0.)
    :param smoothed: whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    :param redshift_dependent: whether the IGM background depends on redshift [bool] (default: True)
    :param method: the integration method 'simps'/'quad'/'old' (default: 'simps')
    :param prob_func: the form of the probability function: 'small_P' for the P<<1 limit, 'full_log' for log(1-1.5*P), and 'norm_log' for the normalized log: log(abs(1-1.5*P)) [str] (default: 'norm_log')
    :param Nz: number of redshift bins, for the 'simps' methods (default: 501)

    """
    z_arr, is_scalar = treat_as_arr(z)
    A = (2./3)*(1 + axion_ini_frac)  # equilibration constant
    dH = (c0*1.e-3)/(100.*h)  # Hubble distance [Mpc]

    if redshift_dependent:

        # z-dependent probability of conversion in one domain
        def Pga(zz): return P0(ma, g, s/(1+zz), B=B*(1+zz)**2.,
                               omega=omega*(1.+zz), mg=mg*(1+zz)**1.5, smoothed=smoothed)

        if method == 'vectorize':
            if is_scalar:
                raise Exception(
                    "'vectorize' only supports z array. Please choose 'simp' 'quad' or 'old' for scalar redshift")

            # fast vectorization
            zArr_raw = np.linspace(0., max(z_arr), int(Nz))
            zArr = sorted(np.concatenate((zArr_raw, z_arr)))
            zArr = np.unique(zArr)
            # constructing integrand
            if prob_func == 'norm_log':
                integrand = log(np.abs(1 - 1.5*Pga(zArr))) / \
                    Ekernel(Omega_L, zArr, w0=w0, wa=wa)
            elif prob_func == 'small_P':
                integrand = -1.5*Pga(zArr) / \
                    Ekernel(Omega_L, zArr, w0=w0, wa=wa)
            elif prob_func == 'full_log':
                integrand = log(1 - 1.5*Pga(zArr)) / \
                    Ekernel(Omega_L, zArr, w0=w0, wa=wa)
            else:
                raise ValueError(
                    "Argument 'prob_func'={} must be equal to either 'small_P', 'full_log', or 'norm_log'. It's neither.".format(prob_func))

            dzArr = np.concatenate(([0], np.diff(zArr)))
            integral_raw = np.cumsum(integrand*dzArr)  # integrating
            integral = np.interp(z_arr, zArr, integral_raw)  # picking input z
            argument = (dH/s)*integral  # argument of the exponential

        elif method == 'simps':
            if not is_scalar:
                raise Exception(
                    "only 'vectorize' supports z array for now and you chose 'simp'")

            # constructing array of redshifts
            if z <= 1.e-10:
                zArr = np.linspace(0., 1.e-10, int(Nz))
            else:
                zArr = np.linspace(0., z, int(Nz))

            # constructing integrand
            if prob_func == 'norm_log':
                integrand = log(np.abs(1 - 1.5*Pga(zArr))) / \
                    Ekernel(Omega_L, zArr, w0=w0, wa=wa)
            elif prob_func == 'small_P':
                integrand = -1.5*Pga(zArr) / \
                    Ekernel(Omega_L, zArr, w0=w0, wa=wa)
            elif prob_func == 'full_log':
                integrand = log(1 - 1.5*Pga(zArr)) / \
                    Ekernel(Omega_L, zArr, w0=w0, wa=wa)
            else:
                raise ValueError(
                    "Argument 'prob_func'={} must be equal to either 'small_P', 'full_log', or 'norm_log'. It's neither.".format(prob_func))

            integral = simps(integrand, zArr)  # integrating
            argument = (dH/s)*integral  # argument of the exponential

        elif method == 'quad':
            if not is_scalar:
                raise Exception(
                    "only 'vectorize' supports z array for now and you chose quad")
            # constructing integrand
            if prob_func == 'norm_log':
                def integrand(zz): return log(
                    np.abs(1 - 1.5*Pga(zz))) / Ekernel(Omega_L, zz, w0=w0, wa=wa)
            elif prob_func == 'small_P':
                def integrand(zz): return -1.5*Pga(zz) / \
                    Ekernel(Omega_L, zz, w0=w0, wa=wa)
            elif prob_func == 'full_log':
                def integrand(zz): return log(
                    1 - 1.5*Pga(zz)) / Ekernel(Omega_L, zz, w0=w0, wa=wa)
            else:
                raise ValueError(
                    "Argument 'prob_func'={} must be equal to either 'small_P', 'full_log', or 'norm_log'. It's neither.".format(prob_func))

            integral = quad(integrand, 0., z)[0]  # integrating
            argument = (dH/s)*integral  # argument of the exponential

        elif method == 'old':
            if not is_scalar:
                raise Exception(
                    "only 'vectorize' supports z array for now and you chose 'old'")

            # computing comoving distance
            y = tau_at_z(z, h=h, Omega_L=Omega_L, w0=w0, wa=wa)
            argument = -1.5*(y/s)*Pga(z)  # argument of exponential

        else:
            raise ValueError(
                "'method' argument must be either 'simps', 'quad', or 'old'.")

    else:
        if is_scalar:
            raise Exception(
                "only redshift+vectorize supports z array for now and you chose redshift independent scheme")

        # computing comoving distance
        y = tau_at_z(z, h=h, Omega_L=Omega_L, w0=w0, wa=wa)
        # z-independent probability conversion in one domain
        P = P0(ma, g, s, B=B, omega=omega, mg=mg, smoothed=smoothed)
        argument = -1.5*(y/s)*P  # argument of exponential

    return A + (1-A)*exp(argument)


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
