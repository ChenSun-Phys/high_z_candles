#######################################################
###           Code for chi2 calculations            ###
###          by Manuel A. Buen-Abad, 2020           ###
###               and Chen Sun, 2020                ###
#######################################################

import numpy as np
import scipy.linalg as la
from numpy import pi, sqrt, log, log10, exp, power
from cosmo import H_at_z, tau_at_z, dA_at_z, distance_modulus
from igm import LumMod
from icm import ADDMod
from tools import flatten_tuples, smooth_step
import data

_Mpc_over_cm_ = 3.0857e+24
_Mpc_over_10pc_ = 1.e5

##########################
# auxiliary functions
##########################


def is_Out_of_Range(x, keys, params):
    """
    Returns a Boolean type indicating whether the current
    point is within the range

    Parameters
    ----------
    x : tuple
        the current point in the hyperspace to be checked
    keys: list
        each correspond to a dimension in the hyperspace,
        i.e. all the variables to be scanned
    """
    res = False

    for i in range(len(x)):
        if x[i] > params[keys[i]+' up'] or x[i] < params[keys[i]+' low']:
            res = True
            # print(i, keys[i], "params[keys[i]+' up']", params[keys[i]+' up'],
            #       "params[keys[i]+' low']", params[keys[i]+' low'], "x[i]", x[i])
            break
    return res


##########################
# chi2 functions
##########################


def chi2_SH0ES(M0, data=None):
    """
    Computes SH0ES chi2. data must be equal to (Anchor_SN, Anchor_SNsig, Anchor_Ceph, Anchor_Cephsig, Anchor_M, Anchor_Msig, aB, aBsig)
    """

    Anchor_SN, _, Anchor_Ceph, _, _, Anchor_Msig, _, _ = data

    chi2 = 0.

    for i in range(len(Anchor_SN)):
        chi2 += (Anchor_SN[i] - M0 - Anchor_Ceph[i])**2 / Anchor_Msig[i]**2

        if use_loglkl:
            # note this is no longer chi2. It's -2(log(lkl))
            # so it can be combined with quasars
            chi2 += np.log(2.*np.pi) + 2.*np.log(Anchor_Msig[i])

    return chi2


# def chi2_quasars_dist_mod(x, data=None, vectorize=True, full_output=False, **kwargs):
#     """
#     Computes quasars chi2 usign distance modulus. Note that this chi2 is only for testing purpose, as it uses the distance modulus given direclty in the data set of Lusso2020. This doesn't take into account the intrinsic scattering either.
#     """

#     # theory point
#     (ma, ga, OmL, h0, qso_gamma, qso_beta) = x

#     # Anchor_SN, _, Anchor_Ceph, _, _, Anchor_Msig, _, _ = data
#     (qso_name_arr,
#      qso_z_arr,
#      qso_logf2500_arr,
#      qso_dlogf2500_arr,
#      qso_logf2keV_arr,
#      qso_dlogf2keV_low_arr,
#      qso_dlogf2keV_up_arr,
#      qso_Gamma_arr,
#      qso_dist_mod_arr,
#      qso_ddist_mod_arr) = data

#     chi2 = 0.

#     kwargs_local = kwargs.copy()
#     omega_X = kwargs_local.pop('omega_X')
#     omega_UV = kwargs_local.pop('omega_UV')

#     if vectorize:

#         # LumMod_vec = np.vectorize(LumMod)
#         kwargs_local['method'] = 'vectorize'
#         tau_at_z_vec = np.vectorize(tau_at_z)

#         logPggX_arr = 1/2.5*LumMod(ma=ma,
#                                    g=ga,
#                                    z=qso_z_arr,
#                                    h=h0,
#                                    OmL=OmL,
#                                    omega=omega_X,
#                                    **kwargs_local)

#         logPggUV_arr = 1/2.5*LumMod(ma=ma,
#                                     g=ga,
#                                     z=qso_z_arr,
#                                     h=h0,
#                                     OmL=OmL,
#                                     omega=omega_UV,
#                                     **kwargs_local)
#         # print(np.sum(np.abs(logPggX_arr)))
#         # print(np.sum(np.abs(logPggUV_arr)))
#         DL_arr = tau_at_z_vec(qso_z_arr, h0, OmL) * \
#             (1.+qso_z_arr) * _Mpc_over_10pc_  # [10 pc]
#         mu_th_arr = 5.*np.log10(DL_arr)
#         # print("mu_th_arr:", mu_th_arr)
#         # print(np.sum(np.abs(mu_th_arr)))

#         # get the measurement
#         mu_exp_arr = qso_dist_mod_arr

#         # get the 1 sigma std deviation
#         sigma_arr = qso_ddist_mod_arr

#         chi2 = np.sum((mu_th_arr - mu_exp_arr)**2/sigma_arr**2)

#     else:
#         raise Exception('Only vectorize is implemented for now.')

#     if full_output and vectorize:
#         # used for debugging to plot out the data and the theory
#         return chi2, mu_th_arr, mu_exp_arr, sigma_arr, qso_z_arr
#     else:
#         return chi2


def chi2_quasars(x,
                 data=None,
                 vectorize=True,
                 full_output=False,
                 dm_output=False,
                 get_optical=False,
                 # quasars_delta=None,
                 **kwargs):
    """Computes quasars chi2.     **kwargs contain the arguments for LumMod.

    :param x: the theory point that contains (ma, ga, OmL, h0, w0, wa, qso_gamma, qso_beta)
    :param data: must be have certain structures. See source code for the structure needed.
    :param vectorize: whether to vectorize the computation
    :param full_output: whether to output other quantities besides chi2, useful for testing.

    """

    if not use_loglkl:
        raise Exception(
            'You asked for chi2 but quasars have to use log(likelihood) since qso_delta makes it otherwise unbounded')

    # theory point
    (ma, ga, OmL, h0, w0, wa, qso_gamma, qso_beta0,
     qso_beta1, qso_delz, qso_z0, qso_delta) = x

    # Anchor_SN, _, Anchor_Ceph, _, _, Anchor_Msig, _, _ = data
    (qso_name_arr,
     qso_z_arr,
     qso_logf2500_arr,
     qso_dlogf2500_arr,
     qso_logf2keV_arr,
     qso_dlogf2keV_low_arr,
     qso_dlogf2keV_up_arr,
     _) = data

    chi2 = 0.

    # Note: the 'vectorize' flag is extracted automatically
    # and set to 'vectorize' in chi2_quasars(),
    # so it's no longer in kwargs or kwargs_local
    kwargs_local = kwargs.copy()
    omega_X = kwargs_local.pop('omega_X')
    omega_UV = kwargs_local.pop('omega_UV')

    if vectorize:
        # LumMod_vec = np.vectorize(LumMod)
        kwargs_local['method'] = 'vectorize'
        tau_at_z_vec = np.vectorize(tau_at_z)

        logPggX_arr = 1/2.5*LumMod(ma=ma,
                                   g=ga,
                                   z=qso_z_arr,
                                   h=h0,
                                   OmL=OmL,
                                   w0=w0,
                                   wa=wa,
                                   omega=omega_X,
                                   **kwargs_local)

        logPggUV_arr = 1/2.5*LumMod(ma=ma,
                                    g=ga,
                                    z=qso_z_arr,
                                    h=h0,
                                    OmL=OmL,
                                    w0=w0,
                                    wa=wa,
                                    omega=omega_UV,
                                    **kwargs_local)

        if get_optical:
            # for debugging purpose to check the mod in optical band
            # relevant for PAN
            logPggOptical_arr = 1/2.5*LumMod(ma=ma,
                                             g=ga,
                                             z=qso_z_arr,
                                             h=h0,
                                             OmL=OmL,
                                             w0=w0,
                                             wa=wa,
                                             omega=1.,  # the optical energy PAN uses
                                             **kwargs_local)

        # print(np.sum(np.abs(logPggX_arr)))
        # print(np.sum(np.abs(logPggUV_arr)))
        DL_arr = tau_at_z_vec(qso_z_arr, h0, OmL, w0=w0, wa=wa) * \
            (1.+qso_z_arr) * _Mpc_over_cm_  # [cm]

        # make a beta array based on the redshift
        qso_beta_arr = smooth_step(
            qso_z_arr, qso_beta0, qso_beta1, qso_delz, qso_z0)
        # qso_beta_arr = np.zeros_like(qso_z_arr)
        # qso_beta_arr[qso_z_arr < qso_z0] = qso_beta0
        # qso_beta_arr[qso_z_arr >= qso_z0] = qso_beta1

        y_th_arr = 2.*(qso_gamma-1)*log10(DL_arr) + logPggX_arr - \
            qso_gamma*logPggUV_arr + qso_beta_arr + \
            (qso_gamma-1)*log10(4.*np.pi)
        # print("y_th_arr:", y_th_arr)
        # print(np.sum(np.abs(y_th_arr)))

        # get the measurement
        y_exp_arr = (qso_logf2keV_arr - qso_gamma*qso_logf2500_arr)
        # print("y_exp_arr:", y_exp_arr)
        # print(np.sum(np.abs(y_exp_arr)))

        # get the 1 sigma std deviation
        # using the symmetric error for now
        sigma_arr = np.sqrt(
            (qso_gamma * qso_dlogf2500_arr)**2 +
            (qso_dlogf2keV_low_arr + qso_dlogf2keV_up_arr)**2/4 +
            qso_delta**2)  # intrinsic scattering added here

        chi2 = np.sum((y_th_arr - y_exp_arr)**2/sigma_arr**2
                      + 2.*np.log(sigma_arr) + np.log(2.*np.pi))
        # added the log term, relevant when delta is a nuisance parameter

    else:
        raise Exception(
            'The non-vectorized quasar routine is no longer maintained. ')

        # for i in range(len(qso_z_arr)):
        #     # compute theoretical prediction
        #     z = qso_z_arr[i]

        #     # use corresponding energy
        #     # Note: the current LumMod has a 2.5 in the front.
        #     # We don't need it here.

        #     logPggX = 1/2.5*LumMod(ma=ma,
        #                            g=ga,
        #                            z=z,
        #                            h=h0,
        #                            OmL=OmL,
        #                            w0=w0,
        #                            wa=wa,
        #                            omega=omega_X,
        #                            **kwargs_local)

        #     logPggUV = 1/2.5*LumMod(ma=ma,
        #                             g=ga,
        #                             z=z,
        #                             h=h0,
        #                             OmL=OmL,
        #                             w0=w0,
        #                             wa=wa,
        #                             omega=omega_UV,
        #                             **kwargs_local)

        #     DL = tau_at_z(z, h0, OmL, w0, wa) * (1+z) * _Mpc_over_cm_  # [cm]
        #     y_th = 2.*(qso_gamma-1)*log10(DL) + logPggX - \
        #         qso_gamma*logPggUV + qso_beta + \
        #         (qso_gamma-1)*log10(4.*np.pi)

        #     # get the measurement
        #     y_exp = (qso_logf2keV_arr[i] - qso_gamma*qso_logf2500_arr[i])

        #     # get the 1 sigma std deviation
        #     # using the symmetric error for now
        #     sigma = np.sqrt(
        #         (0.6*qso_dlogf2500_arr[i])**2 + (qso_dlogf2keV_low_arr[i] + qso_dlogf2keV_up_arr[i])**2/4 + qso_delta**2)

        #     chi2 += (y_th - y_exp)**2/sigma**2 + np.log(sigma)

    if full_output and vectorize:
        # used for debugging to plot out the data and the theory
        return chi2, y_th_arr, y_exp_arr, sigma_arr, qso_z_arr

    elif dm_output and vectorize and (not get_optical):
        # output the distance modulus
        dm_th_arr = 5. * np.log10(DL_arr/(_Mpc_over_cm_/_Mpc_over_10pc_))
        dm_exp_arr = 5./2./(qso_gamma-1.) \
            * (qso_logf2keV_arr-qso_gamma*qso_logf2500_arr
               - logPggX_arr + qso_gamma*logPggUV_arr
               - qso_beta_arr - (qso_gamma-1)*np.log10(4.*np.pi))\
            - 5.*np.log10(_Mpc_over_cm_ / _Mpc_over_10pc_)

        return (chi2,
                dm_th_arr,
                dm_exp_arr,
                qso_z_arr,
                qso_gamma,
                qso_beta_arr,
                qso_delta,
                qso_logf2500_arr,
                qso_logf2keV_arr,
                qso_dlogf2500_arr,
                qso_dlogf2keV_low_arr,
                qso_dlogf2keV_up_arr,
                logPggX_arr,
                logPggUV_arr)

    elif dm_output and vectorize and get_optical:
        # output the distance modulus
        dm_th_arr = 5. * np.log10(DL_arr/(_Mpc_over_cm_/_Mpc_over_10pc_))
        dm_exp_arr = 5./2./(qso_gamma-1.) \
            * (qso_logf2keV_arr-qso_gamma*qso_logf2500_arr
               - logPggX_arr + qso_gamma*logPggUV_arr
               - qso_beta_arr - (qso_gamma-1)*np.log10(4.*np.pi))\
            - 5.*np.log10(_Mpc_over_cm_ / _Mpc_over_10pc_)

        return (chi2,
                dm_th_arr,
                dm_exp_arr,
                qso_z_arr,
                qso_gamma,
                qso_beta_arr,
                qso_delta,
                qso_logf2500_arr,
                qso_logf2keV_arr,
                qso_dlogf2500_arr,
                qso_dlogf2keV_low_arr,
                qso_dlogf2keV_up_arr,
                logPggX_arr,
                logPggUV_arr,
                logPggOptical_arr)

    else:
        return chi2


def chi2_BOSSDR12(x, data=None):
    """
    Computes BOSSDR12 chi2. data must be equal to (BOSS_rsfid, BOSS_meas_z, BOSS_meas_dM, BOSS_meas_Hz, BOSS_cov, BOSS_icov, BOSS_cov_logdet)
    """

    (OmL, h0, w0, wa, rs) = x
    BOSS_rsfid, BOSS_meas_z, BOSS_meas_dM, BOSS_meas_Hz, _, BOSS_icov, BOSS_cov_logdet = data

    chi2 = 0.
    data_array = np.array([], 'float64')

    for i, z in enumerate(BOSS_meas_z):

        DM_at_z = tau_at_z(z, h0, OmL, w0=w0, wa=wa)  # comoving
        H_at_z_val = H_at_z(z, h0, OmL, w0=w0, wa=wa, unit='SI')  # in km/s/Mpc

        theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rs * BOSS_rsfid
        theo_H_rd_by_rdfid = H_at_z_val * rs / BOSS_rsfid

        # calculate difference between the sampled point and observations
        DM_diff = theo_DM_rdfid_by_rd_in_Mpc - BOSS_meas_dM[i]
        H_diff = theo_H_rd_by_rdfid - BOSS_meas_Hz[i]

        # save to data array
        data_array = np.append(data_array, DM_diff)
        data_array = np.append(data_array, H_diff)

    chi2 += np.dot(np.dot(data_array, BOSS_icov), data_array)

    if use_loglkl:
        # note this is no longer chi2. It's -2(log(lkl))
        # so it can be combined with quasars
        chi2 += BOSS_cov_logdet + len(data_array) * np.log(2.*np.pi)

    return chi2


def chi2_BAOlowz(x, data=None):
    """
    Computes BAOlowz chi2. data must be equal to (BAOlowz_meas_exp, BAOlowz_meas_z, BAOlowz_meas_rs_dV, BAOlowz_meas_sigma, BAOlowz_meas_type)
    """

    (OmL, h0, w0, wa, rs) = x
    _, BAOlowz_meas_z, BAOlowz_meas_rs_dV, BAOlowz_meas_sigma, BAOlowz_meas_type = data

    chi2 = 0.
    for i, z in enumerate(BAOlowz_meas_z):
        da = dA_at_z(z, h0, OmL, w0=w0, wa=wa)
        dr = z / H_at_z(z, h0, OmL, w0=w0, wa=wa)
        dv = (da * da * (1 + z) * (1 + z) * dr)**(1. / 3.)

        if BAOlowz_meas_type[i] == 3:
            theo = dv / rs
        elif BAOlowz_meas_type[i] == 7:
            theo = rs / dv
        chi2 += ((theo - BAOlowz_meas_rs_dV[i]) / BAOlowz_meas_sigma[i]) ** 2

        if use_loglkl:
            # note this is no longer chi2. It's -2(log(lkl))
            # so it can be combined with quasars
            chi2 += np.log(2.*np.pi) + np.log(BAOlowz_meas_sigma[i]**2)

    return chi2


def chi2_Pantheon(x, data=None, vectorize=True, M0_low=None, M0_up=None, full_output=False, **kwargs):
    """
    Computes Pantheon chi2. data must be equal to (PAN_lkl, PAN_cov_sqrt, PAN_cov_logdet). **kwargs are the arguments for LumMod.
    :param x: the data point, to be unpacked
    :param data: the data dict that contains the Pantheon data
    :param vectorize: whether to vectorize in computing LumMod
    :param M0_low: debug flag used when M0 is not provided. In that case a fit over M0 will be performed with M0_low being the lower bound
    :param M0_up: debug flag used when M0 is not provided. In that case a fit over M0 will be performed with M0_up being the upper bound.

    return:
    (is_M0_provided==True) && (full_output==True):         # output format: chi2, muth_arr, muexp_arr
    (is_M0_provided==False) && (full_output==True):        # output format: chi2, muth_arr, muexp_arr, M0 best fit
    (is_M0_provided==True) && (full_output==False):        # output format: chi2
    (is_M0_provided==False) && (full_output==False):       # output format: chi2, M0 best fit


    """
    is_M0_provided = False
    try:
        (ma, ga, OmL, h0, w0, wa, M0) = x
        is_M0_provided = True
    except ValueError:
        # for the case of missing M0,
        (ma, ga, OmL, h0, w0, wa) = x

    # the core computation defined

    PAN_lkl, PAN_cov_sqrt, PAN_cov_logdet = data

    def compute_chi2(M0):
        chi2 = 0.

        # numerical scan

        if vectorize:
            # so that kwargs is not touched
            kwargs_local = kwargs.copy()
            # overwrite the kwargs that's fed to LumMod
            kwargs_local['method'] = 'vectorize'
            z_arr = PAN_lkl[:, 0]
            m_meas_arr = PAN_lkl[:, 1]
            change_arr = LumMod(ma, ga, z_arr, h=h0, OmL=OmL,
                                w0=w0, wa=wa, **kwargs_local)
            distance_modulus_vec = np.vectorize(distance_modulus)
            distance_modulus_arr = distance_modulus_vec(
                z_arr, h0, OmL, w0=w0, wa=wa)

            residuals = distance_modulus_arr - \
                m_meas_arr + [M0]*len(z_arr) - change_arr

            # for later plots
            if full_output:
                muth_arr = distance_modulus_arr - change_arr
                muexp_arr = m_meas_arr - [M0]*len(z_arr)
        else:
            # if is_M0_provided:
            #     raise Exception(
            #         "M0 fit is only available in vectorized module. Turn vectorize to True.")
            residuals = []

            if full_output:
                muth_arr = []
                muexp_arr = []

            for rec in PAN_lkl:
                z = rec[0]
                m_meas = rec[1]

                change = LumMod(ma, ga, z, h=h0, OmL=OmL,
                                w0=w0, wa=wa, **kwargs)

                residuals.append(
                    distance_modulus(z, h0, OmL, w0=w0, wa=wa) - m_meas + M0 - change)

                if full_output:
                    muth_arr.append(distance_modulus(
                        z, h0, OmL, w0=w0, wa=wa) - change)
                    muexp_arr.append(m_meas - M0)

        L_residuals = la.solve_triangular(
            PAN_cov_sqrt, residuals, lower=True, check_finite=False)
        chi2 = np.dot(L_residuals, L_residuals)

        if use_loglkl:
            # note this is no longer chi2. It's -2(log(lkl))
            # so it can be combined with quasars
            chi2 += PAN_cov_logdet + np.log(2.*np.pi) * len(L_residuals)

        if full_output:
            # also compute the error bar of muexp

            # from numpy.linalg import eig
            # sigma_arr, _ = eig(PAN_cov_sqrt)
            # discard the eigenvalue method as it changes the order of the points.

            sigma_arr = np.diag(PAN_cov_sqrt)
            return (chi2, muth_arr, muexp_arr, np.array(sigma_arr), z_arr)
        else:
            return chi2

    # determine if a fit over M0 is needed
    if is_M0_provided:
        res = compute_chi2(M0)
        # result either chi2 value or (chi2 value, M0 best fit)
        # depending on full_output flag
        return res

    else:
        # a fit over M0 is needed
        res_arr = []
        M0_arr = np.linspace(M0_low, M0_up)
        for M0 in M0_arr:
            res_arr.append(compute_chi2(M0))
        if full_output:
            # in this case res_arr is not just an array of chi2 valuees
            # so extract the chi2, i.e. first value of each entry in res_arr
            chi2_arr = np.array(res_arr, dtype=object)[:, 0]
        else:
            chi2_arr = res_arr
        min_idx = np.argmin(chi2_arr)

        return flatten_tuples((res_arr[min_idx], M0_arr[min_idx]))


def chi2_External(h0, data=None):
    """
    Computes h0 chi2. data must be equal to (h_TD, h_TD_sig).
    """
    h0_prior_mean, h0_prior_sig = data

    chi2 = 0.

    # add a Gaussian prior to H0

    chi2 += (h0 - h0_prior_mean)**2 / h0_prior_sig**2

    if use_loglkl:
        # note this is no longer chi2. It's -2(log(lkl))
        # so it can be combined with quasars
        chi2 += np.log(2.*np.pi) + 2.*np.log(h0_prior_sig)

    return chi2


def chi2_Gaussian(mu, data=None):
    """Computes the chi2 based on a Gaussian prior described by data. data must be equal to (mean, sig).
    """

    prior_mean, prior_sig = data

    chi2 = 0.

    # add a Gaussian prior to rs

    chi2 += (mu - prior_mean)**2 / prior_sig**2

    if use_loglkl:
        # note this is no longer chi2. It's -2(log(lkl))
        # so it can be combined with quasars
        chi2 += np.log(2.*np.pi) + 2.*np.log(prior_sig)

    return chi2


def chi2_clusters(pars, data=None, wanna_correct=True, fixed_Rvir=False, **kwargs):
    """
    Computes clusters chi2. data must be equal to (names, z_cls, DA_cls, err_cls, asymm_cls, ne0_cls, beta_cls, rc_out_cls, f_cls, rc_in_cls, Rvir_cls). **kwargs are the arguments of ADDMod.
    """

    (ma, ga, OmL, h0, w0, wa) = pars
    names, z_cls, DA_cls, err_cls, asymm_cls, ne0_cls, beta_cls, rc_out_cls, f_cls, rc_in_cls, Rvir_cls = data

    chi2 = 0.
    residuals = []

    for i in range(len(names)):

        z = z_cls[i]
        DA = DA_cls[i]

        ne0 = ne0_cls[i]
        rc_outer = rc_out_cls[i]
        beta_outer = beta_cls[i]
        f_inner = f_cls[i]
        rc_inner = rc_in_cls[i]
        beta_inner = beta_cls[i]

        if fixed_Rvir:
            r_up = 1800.  # [kpc] =  1.8 Mpc for all clusters, same as Perseus
        else:
            # each cluster has its own virial radius, already computed under some fiducial LCDM assumption
            r_up = Rvir_cls[i]

        factor = ADDMod(ma, ga, z, h0, OmL,
                        w0=w0, wa=wa,
                        ne0=ne0,
                        rc_outer=rc_outer,
                        beta_outer=beta_outer,
                        f_inner=f_inner,
                        rc_inner=rc_inner,
                        beta_inner=beta_inner,
                        r_up=r_up,
                        galaxy_index=i,
                        **kwargs)

        DA_th = dA_at_z(z, h0, OmL,
                        w0=w0, wa=wa) * factor

        # print('factor=', factor)
        residuals.append(DA - DA_th)

    residuals = np.array(residuals)

    correction = 1.

    if wanna_correct:
        correction += -2.*asymm_cls * \
            (residuals/err_cls) + 5.*asymm_cls**2. * (residuals/err_cls)**2.

    terms = ((residuals / err_cls)**2.)*correction

    if use_loglkl:
        # note this is no longer chi2. It's -2(log(lkl))
        # so it can be combined with quasars. Symmetric err bar is used here
        terms = terms + np.log(2.*np.pi) + np.log(err_cls**2)

    chi2 = terms.sum()

    return chi2


##########################
# total likelihood
##########################

def lnprob(x,
           keys=None, keys_fixed=None, params=None,
           use_SH0ES=False, shoes_data=None,
           use_BOSSDR12=False, boss_data=None,
           use_BAOlowz=False, bao_data=None,
           use_Pantheon=False, pan_data=None, pan_kwargs=None,
           use_quasars=False, quasars_data=None, quasars_kwargs=None,
           use_h0prior=False, ext_data=None,
           use_early=False, early_data=None,
           use_PlanckOmegaL=False, PlanckOmegaL_data=None,
           use_Planckw0=False, Planckw0_data=None,
           use_Planckwa=False, Planckwa_data=None,
           use_clusters=False, clusters_data=None, wanna_correct=True, fixed_Rvir=False, clusters_kwargs=None,
           verbose=False):
    """
    Computes the total likelihood, as well as that for each experiment
    """
    current_point = {}

    for ii in range(len(keys)):
        current_point[keys[ii]] = x[ii]
    for key in keys_fixed:
        current_point[key] = params[key+' fixed']

    global use_loglkl
    use_loglkl = params['use_loglkl']

    ma = 10**current_point['logma']
    ga = 10**current_point['logga']
    OmL = current_point['OmL']
    h0 = current_point['h0']
    w0 = current_point['w0']
    wa = current_point['wa']

    if use_Pantheon:
        M0 = current_point['M0']
    if use_quasars:
        qso_gamma = current_point['qso_gamma']
        qso_beta0 = current_point['qso_beta0']
        qso_beta1 = current_point['qso_beta1']
        qso_delz = current_point['qso_delz']
        qso_z0 = current_point['qso_z0']
        qso_delta = current_point['qso_delta']
    if use_BOSSDR12:
        rs = current_point['rs']

    # counting the number of experiments used
    experiments_counter = sum(
        [use_SH0ES, use_Pantheon, use_quasars, use_h0prior, use_early, use_PlanckOmegaL, use_Planckw0, use_Planckwa, use_BOSSDR12, use_BAOlowz, use_clusters])
    lnprob_each_chi2 = []

    if not is_Out_of_Range(x, keys, params):  # to avoid overflow
        chi2 = 0

        # Note: the following order needs to be consistent with the
        # main routine inside cosmo_axions_run.py

        # anchors
        if use_SH0ES:

            this_chi2 = chi2_SH0ES(M0, data=shoes_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('SHOES=%f' % this_chi2)

        # Pantheon
        if use_Pantheon:

            this_chi2 = chi2_Pantheon(
                (ma, ga, OmL, h0, w0, wa, M0), data=pan_data, **pan_kwargs)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('pantheon=%f' % this_chi2)

        # quasars
        if use_quasars:

            this_chi2 = chi2_quasars(
                (ma, ga, OmL, h0, w0, wa, qso_gamma,
                 qso_beta0, qso_beta1, qso_delz, qso_z0, qso_delta),
                data=quasars_data,
                **quasars_kwargs)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('quasars=%f' % this_chi2)

        # other H0 experiments
        if use_h0prior:

            this_chi2 = chi2_External(h0, data=ext_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('TDCOSMO=%f' % this_chi2)

        if use_early:

            this_chi2 = chi2_Gaussian(rs, data=early_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('early rs chi2=%f' % this_chi2)

        if use_PlanckOmegaL:

            this_chi2 = chi2_Gaussian(OmL, data=PlanckOmegaL_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('Planck OmegaL chi2=%f' % this_chi2)

        if use_Planckw0:
            this_chi2 = chi2_Gaussian(w0, data=Planckw0_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)
            if verbose > 2:
                print('Planck w0 chi2=%f' % this_chi2)

        if use_Planckwa:
            this_chi2 = chi2_Gaussian(wa, data=Planckwa_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)
            if verbose > 2:
                print('Planck wa chi2=%f' % this_chi2)

        # BOSS DR12
        if use_BOSSDR12:

            this_chi2 = chi2_BOSSDR12((OmL, h0, w0, wa, rs), data=boss_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('boss DR12=%f' % this_chi2)

        # BAOlowz (6DFs + BOSS DR7 MGS, called smallz in MontePython)
        if use_BAOlowz:

            this_chi2 = chi2_BAOlowz((OmL, h0, w0, wa, rs), data=bao_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('bao low z=%f' % this_chi2)

        # clusters
        if use_clusters:

            this_chi2 = chi2_clusters((ma, ga, OmL, h0, w0, wa), data=clusters_data,
                                      wanna_correct=wanna_correct, fixed_Rvir=fixed_Rvir, **clusters_kwargs)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('clusters=%f' % this_chi2)

    else:
        chi2 = np.inf
        lnprob_each_chi2 = [np.inf]*experiments_counter

        if verbose > 2:
            print("out of range... chi2 = np.inf")

    # determine output
    res = -1./2.*chi2

    lnprob_each_chi2.insert(0, res)
    lnprob_each_chi2 = tuple(lnprob_each_chi2)

    return lnprob_each_chi2
