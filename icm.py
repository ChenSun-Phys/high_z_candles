#######################################################
###    Code for probabilities for axion-photon      ###
###              conversion in the ICM              ###
###          by Manuel A. Buen-Abad, 2020           ###
###               and Chen Sun, 2020                ###
#######################################################

from __future__ import division

import numpy as np

from numpy import pi, sqrt, log, log10, exp, power, cumprod
from scipy.integrate import simps, quad
from scipy.interpolate import interp1d
from inspect import getargspec
from ag_probs import omega_plasma, P0
from igm import igm_Psurv
from scipy.stats import rv_continuous
import builtins
from tqdm import tqdm

# FUNCTIONS:


def L_dist(L, L_low=3.5, L_up=10., n=-1.2):
    """
    Power law distribution of magnetic domain sizes.

    L : domain size [kpc]
    L_low : domain size lower bound [kpc] (default: 3.5)
    L_up : domain size upper bound [kpc] (default: 10.)
    n : power law (default: -1.2)
    """

    normal = (L_up**(n+1.))/(1+n) - (L_low**(n+1.))/(1+n)

    return L**n / normal


L_avg = quad(lambda l: L_dist(l) * l, 3.5, 10.)[0]


# def ne_2beta(r, ne0=0.01, rc_outer=100., beta_outer=1., f_inner=0., rc_inner=10., beta_inner=1.):
def ne_2beta(r, ne0, rc_outer, beta_outer, f_inner, rc_inner, beta_inner):
    """
    Electron number density [cm^-3] in the double-beta profile of the hydrostratic equilibrium model.

    r : distance from the center of the cluster [kpc]
    ne0 : central electron number density [cm^-3]
    rc_outer : core radius from the outer component [kpc] (default: 100.)
    beta_outer : slope from the outer component (default: 1.)
    f_inner : fractional contribution from inner component (default: 0.)
    rc_inner : core radius from the inner component [kpc] (default: 10.)
    beta_inner : slope from the inner component (default: 1.)
    """

    def outer(rr): return (1. + rr**2./rc_outer **
                           2.)**(-1.5*beta_outer)  # outer contribution
    def inner(rr): return (1. + rr**2./rc_inner **
                           2.)**(-1.5*beta_inner)  # inner contribution

    return ne0*(f_inner*inner(r) + (1.-f_inner)*outer(r))


def B_icm(r, ne_fn, B_ref=10., r_ref=0., eta=0.5, **kwargs):
    """
    Magnetic field [muG] in the ICM, proportional to a power of the electron number density.

    r : distance from the center of the cluster [kpc]
    ne_fn : function for the electron number density [cm^-3]
    B_ref : reference value of the magnetic field [muG] (default: 10.)
    r_ref : reference value of the radius [kpc] (default: 0.)
    eta : power law of B_icm as a function of ne (default: 0.5)
    kwargs : other keyword arguments of the function 'ne_fn'
    """

    return B_ref*(ne_fn(r, **kwargs)/ne_fn(r_ref, **kwargs))**eta


def icm_Psurv(ma, g, r_ini, r_fin, ne_fn, B_fn,
              L=10.,
              omega_Xrays=10.,
              axion_ini_frac=0.,
              smoothed=False,
              method='product',
              # if method=='product':
              return_arrays=False,
              # if method=='simps'/'quad':
              prob_func='norm_log',
              # if method=='simps':
              Nr=501,
              # ICMdomain
              r_Arr_raw=None,
              L_Arr_raw=None,
              sintheta_Arr_raw=None,
              varying_ICMdomain=False,
              **kwargs):
    """
    ICM survival probability for photons originating at a distance r from the cluster's center.

    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-2]
    r_ini : photon initial radial distance to the cluster center [kpc]
    r_fin : photon final radial distance to the cluster center [kpc]
    ne_fn : function for the electron number density [cm^-3]
    B_fn : function for the ICM magnetic field [muG]
    L : ICM magnetic field domain size [kpc] (default: 10.)
    omega_Xrays : photon energy [keV] (default: 10.)
    axion_ini_frac : the initial intensity fraction of axions: I_axion/I_photon (default: 0.)
    smoothed : whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    method : the integration method 'simps'/'quad'/'product' (default: 'product')

    # if method=='product':
    return_arrays : whether we return the partial products and radii arrays (useful for icm_los_Psurv) [bool] (default: False)

    # if method=='simps'/'quad':
    prob_func : the form of the probability function: 'small_P' for the P<<1 limit, 'full_log' for log(1-1.5*P), and 'norm_log' for the normalized log: log(abs(1-1.5*P)) [str] (default: 'norm_log')

    # if method=='simps':
    Nr : number of radius bins (default: 501)

    kwargs : other keyword arguments of the functions 'ne_fn' and 'B_fn'
    """

    if (return_arrays and (method != 'product')):
        raise ValueError(
            "If you use return_arrays = True you need method='product'.")

    A = (2./3)*(1 + axion_ini_frac)  # equilibration constant

    # reading the parameter names of ne_fn and B_fn
    ne_pars = getargspec(ne_fn)[0]
    B_pars = getargspec(B_fn)[0]

    # building the kwargs for ne_fn and B_fn
    ne_kwargs = {}
    B_kwargs = {}

    for key, val in kwargs.items():
        if key in ne_pars:
            ne_kwargs[key] = val
        if key in B_pars:
            B_kwargs[key] = val

    # # check if the kwargs that feeds into ne is also considerred as B_kwargs
    # # indeed **kwargs part in B_ICM is not considerred as B_pars from getargspec
    # print("ne_kwargs=%s" % ne_kwargs)
    # print("B_kwargs=%s" % B_kwargs)

    # defining functions of r
    # ICM electron number density [cm^-3]
    def ne(rr): return ne_fn(rr, **ne_kwargs)
    def mg(rr): return omega_plasma(ne(rr))  # photon plasma mass [eV]
    def Bicm(rr): return B_fn(rr, ne_fn, **kwargs)  # ICM magnetic field [muG]

    def P(rr): return P0(ma, g, L/1000., B=Bicm(rr)*1000., omega=omega_Xrays*1000., mg=mg(rr),
                         smoothed=smoothed)  # conversion probability in domain located at radius rr from center of cluster

    if method == 'product':

        # ICMdomain
        if varying_ICMdomain is True:

            def P(rr, L, sintheta):
                res = P0(ma, g, L/1000., B=Bicm(rr)*1000.*sintheta, omega=omega_Xrays*1000., mg=mg(rr),
                         smoothed=smoothed)  # conversion probability in domain located at radius rr from center
                return res

            # pass r_Arr_raw and L_Arr_raw from kwargs
            r_Arr = r_Arr_raw[np.where(r_Arr_raw < r_fin)]
            L_Arr = L_Arr_raw[:len(r_Arr)]
            sintheta_Arr = sintheta_Arr_raw[:len(r_Arr)]
            P_Arr = P(r_Arr, L_Arr, sintheta_Arr)
        else:
            N = int(round((r_fin - r_ini)/L))  # number of magnetic domains
            # array of r-values of the domains' centers
            r_Arr = (r_ini + L/2.) + L*np.arange(N)
            P_Arr = P(r_Arr)  # array of conversion probabilities

        factors = 1. - 1.5*P_Arr  # the factors in each domain
        total_prod = factors.prod()
        partial_prods = cumprod(factors[::-1])[::-1]

        if return_arrays:  # we are asked to return the arrays of partial products and radii for later use

            return (A + (1.-A)*total_prod, A + (1.-A)*partial_prods, r_Arr)

        else:  # we are asked to simply give the survival probability and nothing else

            return A + (1.-A)*total_prod

    elif method == 'simps':
        # ICMdomain
        if varying_ICMdomain is True:
            raise Exception("simps method doesn't support varying domain size")
        rArr = np.linspace(r_ini, r_fin, Nr)

        if prob_func == 'norm_log':
            integrand = log(np.abs(1. - 1.5*P(rArr)))
        elif prob_func == 'small_P':
            integrand = -1.5*P(rArr)
        elif prob_func == 'full_log':
            integrand = log(1. - 1.5*P(rArr))
        else:
            raise ValueError(
                "Argument 'prob_func'={} must be equal to either 'small_P', 'full_log', or 'norm_log'. It's neither.".format(prob_func))

        integral = simps(integrand, rArr)
        argument = integral/L

        return A + (1.-A)*exp(argument)

    elif method == 'quad':
        # ICMdomain
        if varying_ICMdomain is True:
            raise Exception("quad method doesn't support varying domain size")

        if prob_func == 'norm_log':
            def integrand(rr): return log(np.abs(1. - 1.5*P(rr)))
        elif prob_func == 'small_P':
            def integrand(rr): return -1.5*P(rr)
        elif prob_func == 'full_log':
            def integrand(rr): return log(1. - 1.5*P(rr))
        else:
            raise ValueError(
                "Argument 'prob_func'={} must be equal to either 'small_P', 'full_log', or 'norm_log'. It's neither.".format(prob_func))

        integral = quad(integrand, r_ini, r_fin)[0]
        argument = integral/L

        return A + (1.-A)*exp(argument)

    else:
        raise ValueError(
            "Argument 'method'={} must be equal to either 'simps', 'quad', or 'product'. It's neither.".format(method))


def icm_los_Psurv(ma, g, r_low, r_up, ne_fn, B_fn,
                  L=10.,
                  omega_Xrays=10.,
                  axion_ini_frac=0.,
                  smoothed=False,
                  method='product',
                  # if method=='product':
                  return_arrays=False,
                  # if method=='simps'/'quad':
                  prob_func='norm_log',
                  # if method=='simps':
                  Nr=501,
                  # for l.o.s. integration:
                  los_method='quad',
                  # if los_method=='simps' && method=='product' && return_arrays=True:
                  los_use_prepared_arrays=False,
                  # if los_method=='simps' && los_use_prepared_arrays=False:
                  los_Nr=501,
                  # ICMdomain
                  r_Arr_raw=None,
                  L_Arr_raw=None,
                  sintheta_Arr_raw=None,
                  varying_ICMdomain=False,
                  **kwargs):
    """
    Line-of-sight average of the photons ICM survival probability.

    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-2]
    r_low : lower end of the integration [kpc]
    r_up : upper end of the integration [kpc]
    ne_fn : function for the electron number density [cm^-3]
    B_fn : function for the ICM magnetic field [muG]
    L : ICM magnetic field domain size [kpc] (default: 10.)
    omega_Xrays : photon energy [keV] (default: 10.)
    axion_ini_frac : the initial intensity fraction of axions: I_axion/I_photon (default: 0.)
    smoothed : whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    method : the integration method 'simps'/'quad'/'product' (default: 'product')

    # if method=='product':
    return_arrays : whether we return the partial products and radii arrays (useful for icm_los_Psurv) [bool] (default: False)

    # if method=='simps'/'quad':
    prob_func : the form of the probability function: 'small_P' for the P<<1 limit, 'full_log' for log(1-1.5*P), and 'norm_log' for the normalized log: log(abs(1-1.5*P)) [str] (default: 'norm_log')

    # if method=='simps':
    Nr : number of radius bins, for the 'simps' methods (default: 501)

    # for l.o.s. integration:
    los_method : the integration method along the line of sight 'simps'/'quad' (default: 'simps')

    # if los_method=='simps' && method=='product' && return_arrays=True:
    los_use_prepared_arrays

    # if los_method=='simps' && los_use_prepared_arrays=False:
    los_Nr : number of radius bins along the line of sight, for the 'simps' methods (default: 501)

    kwargs : other keyword arguments of the functions 'ne_fn' and 'B_fn'
    """

    if los_use_prepared_arrays and (not return_arrays):
        raise ValueError(
            "You cannot pass los_use_prepared_arrays=True if you have return_arrays=False. You cannot use arrays that aren't there!")

    # reading the parameter names of ne_fn and B_fn
    ne_pars = getargspec(ne_fn)[0]
    B_pars = getargspec(B_fn)[0]

    # building the kwargs for ne_fn and B_fn
    ne_kwargs = {}
    B_kwargs = {}
    for key, val in kwargs.items():
        if key in ne_pars:
            ne_kwargs[key] = val
        if key in B_pars:
            B_kwargs[key] = val

    # defining functions of r
    def ne2(rr): return ne_fn(
        rr, **ne_kwargs)**2.  # square of the ICM electron number density [cm^-6]

    # ICM magnetic field [muG]
    def Bicm(rr): return B_fn(rr, ne_fn, **kwargs)

    if return_arrays:

        _, pArr, rArr = icm_Psurv(ma, g, r_low, r_up, ne_fn, B_fn,
                                  L=L,
                                  omega_Xrays=omega_Xrays,
                                  axion_ini_frac=axion_ini_frac,
                                  smoothed=smoothed,
                                  method=method,
                                  return_arrays=return_arrays,  # should be True
                                  prob_func=prob_func,
                                  Nr=Nr,
                                  # ICMdomain
                                  r_Arr_raw=r_Arr_raw,
                                  L_Arr_raw=L_Arr_raw,
                                  sintheta_Arr_raw=sintheta_Arr_raw,
                                  varying_ICMdomain=varying_ICMdomain,
                                  **kwargs)

        pfn = interp1d(rArr, pArr, fill_value='extrapolate')
        def Pgg_ne2(rr): return ne2(rr) * pfn(rr)
        # print('g=%e' % g)
        # # print("pArr=", pArr)
        # print("ma=%g, g=%g, r_low=%g, r_up=%g, ne=%g, B=%g"
        # % (ma, g, r_low, r_up, ne2((r_low+r_up)/2), Bicm((r_low+r_up)/2)))

    else:

        def Pgg_ne2(rr): return ne2(rr) * icm_Psurv(ma, g, rr, r_up, ne_fn, B_fn,
                                                    L=L,
                                                    omega_Xrays=omega_Xrays,
                                                    axion_ini_frac=axion_ini_frac,
                                                    smoothed=smoothed,
                                                    method=method,
                                                    return_arrays=return_arrays,  # should be False
                                                    prob_func=prob_func,
                                                    Nr=Nr,
                                                    # ICMdomain
                                                    r_Arr_raw=r_Arr_raw,
                                                    L_Arr_raw=L_Arr_raw,
                                                    sintheta_Arr_raw=sintheta_Arr_raw,
                                                    varying_ICMdomain=varying_ICMdomain,
                                                    **kwargs)

    if los_method == 'quad':  # this method requires functions

        num = quad(Pgg_ne2, r_low, r_up)[0]
        den = quad(ne2, r_low, r_up)[0]

    elif los_method == 'simps':  # this method requires arrays

        if los_use_prepared_arrays:  # in this case we already have arrays prepared, and we will reuse them for the simps integration

            # finding the array index closest to the lower end of the l.o.s. integration
            low_idx = np.abs(rArr - r_low).argmin()
            # finding the array index closest to the upper end of the l.o.s. integration
            up_idx = np.abs(rArr - r_up).argmin()

            los_rArr = rArr[low_idx:up_idx+1]  # the radii array
            ne2_Arr = ne2(los_rArr)  # the ne2 array
            Pgg_ne2_Arr = ne2_Arr * pArr[low_idx:up_idx+1]  # the ne2*Pgg array

            del low_idx, up_idx

        else:  # we need to prepare the arrays for the simps integration

            los_rArr = np.linspace(r_low, r_up, los_Nr)  # the radii array
            ne2_Arr = ne2(los_rArr)  # the ne2 array

            Pgg_ne2_Arr = []  # the ne2*Pgg array

            for r in los_rArr:

                if not np.isnan(Pgg_ne2(r)):
                    Pgg_ne2_Arr.append(Pgg_ne2(r))
                else:
                    Pgg_ne2_Arr.append(0.)

            Pgg_ne2_Arr = np.array(Pgg_ne2_Arr)

        num = simps(Pgg_ne2_Arr, los_rArr)
        den = simps(ne2_Arr, los_rArr)

        del los_rArr, ne2_Arr, Pgg_ne2_Arr

    else:
        raise ValueError(
            "Argument 'los_method'={} must be equal to either 'simps' or 'quad'. It's neither.".format(los_method))

    return num/den


def gen_power_law(n=-1.2, Lmin=3.5, Lmax=10):
    """Generating a truncated power law random number generator 

    :param n: power
    :param Lmin: lower bound of the range
    :param Lmax: upper bound of the range
    :returns: a random number generator

    """

    # define unnormalized distribution
    def f(x):
        return x**n

    # # get the norm
    # norm = quad(f, Lmin, Lmax)
    # print(norm)

    # compute the norm
    norm = (Lmax**(n+1) - Lmin**(n+1))/(n+1)

    # define truncated pdf
    def pdf(x):
        return f(x)/norm

    # define the random number generator
    class powerlaw_gen(rv_continuous):
        "Truncated power law distribution"

        def _pdf(self, x):
            return pdf(x)

    # initialize the rnd
    p = powerlaw_gen(name='powerlaw', a=Lmin, b=Lmax)
    return p


def L_ICM_draw(n, Lmin, Lmax, size):
    """Make a draw using the truncated power law randum number generator

    :param n: power law
    :param Lmin: lower bound
    :param Lmax: upper bound
    :param size: size of the draw
    :returns: array of values with the length of size

    """
    try:
        p = gen_power_law(n=n, Lmin=Lmin, Lmax=Lmax)
    except:
        print('Lmin=%s' % Lmin)
        print('Lmax=%s' % Lmax)
        raise
    return p.rvs(size=size)


def sintheta_ICM_draw(size, n=0, thetamin=0., thetamax=np.pi):
    try:
        p = gen_power_law(n=n, Lmin=thetamin, Lmax=thetamax)
    except:
        print('thetamin=%s' % thetamin)
        print('thetamax=%s' % thetamax)
        raise
    theta_arr = p.rvs(size=size)
    return np.sin(theta_arr)


def check_DA_scattering(ma, g, galaxy_names, data, result, result_mean, result_z, number_of_sigma=2, grid=4, flg_integrate=False, idx_check=None):
    """Check the theoretical uncertainty of D_A given the internal scattering of the gal-cluster measurements

    :param ma: mass of axion [eV]
    :param g: a-g coupling [GeV**-1]
    :param galaxy_names: array of galaxy names to be checked 
    :param data: the Bonamente table
    :param result: dictionary that contains the P_{ag} probability for each galaxy
    :param result_mean: dictionary that contains the P_{ag} probability with mean value chosen for each galaxy
    :param result_z: redshift of each galaxy
    :param number_of_sigma: number of sigmas to be scanner over for each nuisance. (Default: 2)
    :param grid: grid size of each nuisance parameter. (Default: 4)
    :param flg_integrate: whether to marginalize. True: marginalize each nuisance by a weighted sum. False: show the scatterign directly. (Default: False)
    :param idx_check: the idx of the parameter to be checked
    :returns: None. (result saved into result, result_mean, and result_z)

    """
    # unpack
    (names,
     z_cls,
     DA_cls,
     err_cls,
     asymm_cls,
     ne0_cls,
     beta_cls,
     rc_out_cls,
     f_cls,
     rc_in_cls,
     Rvir_cls,
     ne0_err_cls,
     beta_err_cls,
     rc_out_err_cls,
     f_err_cls,
     rc_in_err_cls) = data

    # ma = 1.e-16
    # g = 6.e-13

    # result = {}
    # result_mean = {}
    for i in tqdm(range(len(names))):
        z = z_cls[i]
        DA = DA_cls[i]
        err = err_cls[i]
        asymm = asymm_cls[i]
        ne0 = ne0_cls[i]
        beta = beta_cls[i]
        rc_out = rc_out_cls[i]
        f = f_cls[i]
        rc_in = rc_in_cls[i]
        Rvir = Rvir_cls[i]
        ne0_err = ne0_err_cls[i]
        beta_err = beta_err_cls[i]
        rc_out_err = rc_out_err_cls[i]
        f_err = f_err_cls[i]
        rc_in_err = rc_in_err_cls[i]

        # do a little scan with each cluster
        ne0_arr = np.linspace(ne0 - number_of_sigma *
                              ne0_err, ne0 + number_of_sigma * ne0_err, grid)
        beta_arr = np.linspace(beta - number_of_sigma *
                               beta_err, beta + number_of_sigma * beta_err, grid)
        rc_in_arr = np.linspace(
            rc_in - number_of_sigma * rc_in_err, rc_in + number_of_sigma * rc_in_err, grid)
        f_arr = np.linspace(f - number_of_sigma * f_err,
                            f + number_of_sigma * f_err, grid)
        rc_out_arr = np.linspace(
            rc_out - number_of_sigma * rc_out_err, rc_out + number_of_sigma * rc_out_err, grid)

        los_Psurv_flat = []
        los_Psurv_mean = []

        varying_keys = ["ne0", "rc_outer", "beta_outer",
                        "f_inner", "rc_inner", "beta_inner"]
        varying_range = [ne0_arr, rc_out_arr,
                         beta_arr, f_arr, rc_in_arr, beta_arr]
        err_arr = [ne0_err, rc_out_err, beta_err, f_err, rc_in_err, beta_err]
        mean_arr = [ne0, rc_out, beta, f, rc_in, beta]

        # give kwargs the mean first
        kwargs = {
            # prepare ne_fn kw
            "ne0": ne0,
            "rc_outer": rc_out,
            "beta_outer": beta,
            "f_inner": f,
            "rc_inner": rc_in,
            "beta_inner": beta,

            # prepare B_fn
            "B_ref": 25.,
            "r_ref": 0.,
            "eta": 0.7
        }
        # compute mean
        los_Psurv = icm_los_Psurv(ma=ma,
                                  g=g,
                                  r_low=10,
                                  r_up=Rvir,
                                  ne_fn=ne_2beta,
                                  B_fn=B_icm,
                                  L=L_avg,
                                  omega_Xrays=5.,
                                  axion_ini_frac=0.,
                                  smoothed=False,
                                  method='product',
                                  return_arrays=True,
                                  los_method='simps',
                                  # los_method='quad',
                                  los_use_prepared_arrays=True,
                                  **kwargs
                                  )
        los_Psurv_mean.append(los_Psurv)

        for k in range(len(varying_keys)):
            # give kwargs the mean first
            kwargs = {
                # prepare ne_fn kw
                "ne0": ne0,
                "rc_outer": rc_out,
                "beta_outer": beta,
                "f_inner": f,
                "rc_inner": rc_in,
                "beta_inner": beta,

                # prepare B_fn
                "B_ref": 25.,
                "r_ref": 0.,
                "eta": 0.7
            }

            if idx_check is not None:
                if k != idx_check:
                    continue

            los_Psurv_this_param_arr = []
            dist_this_param_arr = []

            for j in range(len(varying_range[k])):
                val = varying_range[k][j]

                # update the param
                kwargs[varying_keys[k]] = val

                los_Psurv = icm_los_Psurv(ma=ma,
                                          g=g,
                                          r_low=10,
                                          r_up=Rvir,
                                          ne_fn=ne_2beta,
                                          B_fn=B_icm,
                                          L=L_avg,
                                          omega_Xrays=5.,
                                          axion_ini_frac=0.,
                                          smoothed=False,
                                          method='product',
                                          return_arrays=True,
                                          los_method='simps',
                                          # los_method='quad',
                                          los_use_prepared_arrays=True,
                                          **kwargs
                                          )
                if flg_integrate:
                    mean = mean_arr[k]
                    err = err_arr[k]
                    p_this_param = p(val, mean, err)

                    los_Psurv_this_param_arr.append(los_Psurv)
                    dist_this_param_arr.append(
                        p_this_param)  # assign the weight
                else:
                    los_Psurv_flat.append(los_Psurv)
            if flg_integrate:
                # integrate
                los_Psurv_this_param_arr = np.asarray(
                    los_Psurv_this_param_arr)  # the Psurv
                dist_this_param_arr = np.asarray(
                    dist_this_param_arr)  # the Gaussian weight
                los_Psurv_int = simps(
                    los_Psurv_this_param_arr*dist_this_param_arr, varying_range[k])  # the weighted sum
                los_Psurv_flat.append(los_Psurv_int)
            else:
                pass
                # save for i-th galaxy
        result[names[i]] = np.asarray(los_Psurv_flat)
        result_mean[names[i]] = np.asarray(los_Psurv_mean)
        result_z[names[i]] = z
    return


def p(x, mu, sigma):
    """ The 1D gaussian 

    """
    res = 1./np.sqrt(2.*np.pi)/sigma * np.exp(-1./2*(x-mu)**2/sigma**2)
    return res


# since it mainly depends on icm with small dependence on igm,
# I'm putting it here in the icm module

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
        # print("B_ref=", B_ref, ",r_ref=", r_ref, ",eta=", eta)
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
        # print(PICM)
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

    # print(Pg)
    # print("Pgg_CMB=%g, Pgg_X=%g, Pg=%g, ADDMod=%g" %
    #       (Pgg_CMB, Pgg_X, Pg, Pgg_CMB**2. / (Pgg_X * Pg)))
    # check Pgg_X
    # pring("ma=%g, g=%g, z=%g, s_IGM=%g, BIGM=%g, omegaCMB=%g, mgIGM=%g, h=%g, OmL=%g, w0=%g, wa=%g, axion_ini_frac=%g, smoothed=%g, redshift_dependent=%g, method=%g, prob_func=%g, Nz=%gNz_IGM")
    # print("IaIg")
    return Pgg_CMB**2. / (Pgg_X * Pg)
