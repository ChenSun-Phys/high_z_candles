#######################################################
###           Code for chi2 calculations            ###
###          by Manuel A. Buen-Abad, 2020           ###
###               and Chen Sun, 2020                ###
#######################################################

import numpy as np
import scipy.linalg as la
from numpy import pi, sqrt, log, log10, exp, power
from cosmo import H_at_z, tau_at_z, dA_at_z, muLCDM, LumMod, ADDMod
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
                 # quasars_delta=None,
                 **kwargs):
    """Computes quasars chi2.     **kwargs contain the arguments for LumMod. 

    :param x: the theory point that contains (ma, ga, OmL, h0, qso_gamma, qso_beta)
    :param data: must be have certain structures. See source code for the structure needed. 
    :param vectorize: whether to vectorize the computation
    :param full_output: whether to output other quantities besides chi2, useful for testing. 

    """

    # theory point
    (ma, ga, OmL, h0, qso_gamma, qso_beta, qso_delta) = x

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
                                   omega=omega_X,
                                   **kwargs_local)

        logPggUV_arr = 1/2.5*LumMod(ma=ma,
                                    g=ga,
                                    z=qso_z_arr,
                                    h=h0,
                                    OmL=OmL,
                                    omega=omega_UV,
                                    **kwargs_local)
        # print(np.sum(np.abs(logPggX_arr)))
        # print(np.sum(np.abs(logPggUV_arr)))
        DL_arr = tau_at_z_vec(qso_z_arr, h0, OmL) * \
            (1.+qso_z_arr) * _Mpc_over_cm_  # [cm]
        mu_th_arr = 2.*(qso_gamma-1)*log10(DL_arr) + logPggX_arr - \
            qso_gamma*logPggUV_arr + qso_beta + \
            (qso_gamma-1)*log10(4.*np.pi)
        # print("mu_th_arr:", mu_th_arr)
        # print(np.sum(np.abs(mu_th_arr)))

        # get the measurement
        mu_exp_arr = (qso_logf2keV_arr - qso_gamma*qso_logf2500_arr)
        # print("mu_exp_arr:", mu_exp_arr)
        # print(np.sum(np.abs(mu_exp_arr)))

        # get the 1 sigma std deviation
        # using the symmetric error for now
        # when computing the sigma error, assuming gamma to be 0.6 for logf2500
        # change on top of gamma=0.6 is of higher order
        sigma_arr = np.sqrt(
            (0.6*qso_dlogf2500_arr)**2 + (qso_dlogf2keV_low_arr + qso_dlogf2keV_up_arr)**2/4 + qso_delta**2)  # intrinsic scattering added here

        chi2 = np.sum((mu_th_arr - mu_exp_arr)**2/sigma_arr**2
                      + np.log(sigma_arr))
        # added the log term, relevant when delta is a nuisance parameter

    else:

        for i in range(len(qso_z_arr)):
            # compute theoretical prediction
            z = qso_z_arr[i]

            # use corresponding energy
            # Note: the current LumMod has a 2.5 in the front.
            # We don't need it here.

            logPggX = 1/2.5*LumMod(ma=ma,
                                   g=ga,
                                   z=z,
                                   h=h0,
                                   OmL=OmL,
                                   omega=omega_X,
                                   **kwargs_local)

            logPggUV = 1/2.5*LumMod(ma=ma,
                                    g=ga,
                                    z=z,
                                    h=h0,
                                    OmL=OmL,
                                    omega=omega_UV,
                                    **kwargs_local)

            DL = tau_at_z(z, h0, OmL) * (1+z) * _Mpc_over_cm_  # [cm]
            mu_th = 2.*(qso_gamma-1)*log10(DL) + logPggX - \
                qso_gamma*logPggUV + qso_beta + \
                (qso_gamma-1)*log10(4.*np.pi)

            # get the measurement
            mu_exp = (qso_logf2keV_arr[i] - qso_gamma*qso_logf2500_arr[i])

            # get the 1 sigma std deviation
            # using the symmetric error for now
            sigma = np.sqrt(
                (0.6*qso_dlogf2500_arr[i])**2 + (qso_dlogf2keV_low_arr[i] + qso_dlogf2keV_up_arr[i])**2/4 + qso_delta**2)

            chi2 += (mu_th - mu_exp)**2/sigma**2 + np.log(sigma)

    if full_output and vectorize:
        # used for debugging to plot out the data and the theory
        return chi2, mu_th_arr, mu_exp_arr, sigma_arr, qso_z_arr
    else:
        return chi2


def chi2_BOSSDR12(x, data=None):
    """
    Computes BOSSDR12 chi2. data must be equal to (BOSS_rsfid, BOSS_meas_z, BOSS_meas_dM, BOSS_meas_Hz, BOSS_cov, BOSS_icov)
    """

    (OmL, h0, rs) = x
    BOSS_rsfid, BOSS_meas_z, BOSS_meas_dM, BOSS_meas_Hz, _, BOSS_icov = data

    chi2 = 0.
    data_array = np.array([], 'float64')

    for i, z in enumerate(BOSS_meas_z):

        DM_at_z = tau_at_z(z, h0, OmL)  # comoving
        H_at_z_val = H_at_z(z, h0, OmL, unit='SI')  # in km/s/Mpc

        theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rs * BOSS_rsfid
        theo_H_rd_by_rdfid = H_at_z_val * rs / BOSS_rsfid

        # calculate difference between the sampled point and observations
        DM_diff = theo_DM_rdfid_by_rd_in_Mpc - BOSS_meas_dM[i]
        H_diff = theo_H_rd_by_rdfid - BOSS_meas_Hz[i]

        # save to data array
        data_array = np.append(data_array, DM_diff)
        data_array = np.append(data_array, H_diff)

    chi2 += np.dot(np.dot(data_array, BOSS_icov), data_array)

    return chi2


def chi2_BAOlowz(x, data=None):
    """
    Computes BAOlowz chi2. data must be equal to (BAOlowz_meas_exp, BAOlowz_meas_z, BAOlowz_meas_rs_dV, BAOlowz_meas_sigma, BAOlowz_meas_type)
    """

    (OmL, h0, rs) = x
    _, BAOlowz_meas_z, BAOlowz_meas_rs_dV, BAOlowz_meas_sigma, BAOlowz_meas_type = data

    chi2 = 0.
    for i, z in enumerate(BAOlowz_meas_z):
        da = dA_at_z(z, h0, OmL)
        dr = z / H_at_z(z, h0, OmL)
        dv = (da * da * (1 + z) * (1 + z) * dr)**(1. / 3.)

        if BAOlowz_meas_type[i] == 3:
            theo = dv / rs
        elif BAOlowz_meas_type[i] == 7:
            theo = rs / dv
        chi2 += ((theo - BAOlowz_meas_rs_dV[i]) / BAOlowz_meas_sigma[i]) ** 2

    return chi2


def chi2_Pantheon(x, data=None, vectorize=True, **kwargs):
    """
    Computes Pantheon chi2. data must be equal to (PAN_lkl, PAN_cov). **kwargs are the arguments for LumMod.
    """

    (ma, ga, OmL, h0, M0) = x
    PAN_lkl, PAN_cov = data

    chi2 = 0.

    # numerical scan

    if vectorize:
        # so that kwargs is not touched
        kwargs_local = kwargs.copy()
        # overwrite the kwargs that's fed to LumMod
        kwargs_local['method'] = 'vectorize'
        z_arr = PAN_lkl[:, 0]
        m_meas_arr = PAN_lkl[:, 1]
        change_arr = LumMod(ma, ga, z_arr, h=h0, OmL=OmL, **kwargs_local)
        muLCDM_vec = np.vectorize(muLCDM)
        muLCDM_arr = muLCDM_vec(z_arr, h0, OmL)

        residuals = muLCDM_arr - m_meas_arr + [M0]*len(z_arr) - change_arr

    else:
        residuals = []
        for rec in PAN_lkl:
            z = rec[0]
            m_meas = rec[1]

            change = LumMod(ma, ga, z, h=h0, OmL=OmL, **kwargs)

            residuals.append(muLCDM(z, h0, OmL) - m_meas + M0 - change)

    L_residuals = la.solve_triangular(
        PAN_cov, residuals, lower=True, check_finite=False)
    chi2 = np.dot(L_residuals, L_residuals)

    return chi2


def chi2_External(h0, data=None):
    """
    Computes h0 chi2. data must be equal to (h_TD, h_TD_sig).
    """
    h0_prior_mean, h0_prior_sig = data

    chi2 = 0.

    # add a Gaussian prior to H0

    chi2 += (h0 - h0_prior_mean)**2 / h0_prior_sig**2

    return chi2


def chi2_early(rs, data=None):
    """
    Computes rs chi2. data must be equal to (rsdrag_mean, rsdrag_sig).
    """

    rsdrag_prior_mean, rsdrag_prior_sig = data

    chi2 = 0.

    # add a Gaussian prior to rs

    chi2 += (rs - rsdrag_prior_mean)**2 / rsdrag_prior_sig**2

    return chi2


def chi2_clusters(pars, data=None, wanna_correct=True, fixed_Rvir=False, **kwargs):
    """
    Computes clusters chi2. data must be equal to (names, z_cls, DA_cls, err_cls, asymm_cls, ne0_cls, beta_cls, rc_out_cls, f_cls, rc_in_cls, Rvir_cls). **kwargs are the arguments of ADDMod.
    """

    (ma, ga, OmL, h0) = pars
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
                        ne0=ne0,
                        rc_outer=rc_outer,
                        beta_outer=beta_outer,
                        f_inner=f_inner,
                        rc_inner=rc_inner,
                        beta_inner=beta_inner,
                        r_up=r_up,
                        galaxy_index=i,
                        **kwargs)

        DA_th = dA_at_z(z, h0, OmL) * factor

        residuals.append(DA - DA_th)

    residuals = np.array(residuals)

    correction = 1.

    if wanna_correct:
        correction += -2.*asymm_cls * \
            (residuals/err_cls) + 5.*asymm_cls**2. * (residuals/err_cls)**2.

    terms = ((residuals / err_cls)**2.)*correction

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
           use_TDCOSMO=False, ext_data=None,
           use_early=False, early_data=None,
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

    ma = 10**current_point['logma']
    ga = 10**current_point['logga']
    OmL = current_point['OmL']
    h0 = current_point['h0']

    if use_Pantheon:
        M0 = current_point['M0']
    if use_quasars:
        qso_gamma = current_point['qso_gamma']
        qso_beta = current_point['qso_beta']
        qso_delta = current_point['qso_delta']
    if use_BOSSDR12:
        rs = current_point['rs']

    # counting the number of experiments used
    experiments_counter = sum(
        [use_SH0ES, use_Pantheon, use_quasars, use_TDCOSMO, use_early, use_BOSSDR12, use_BAOlowz, use_clusters])
    lnprob_each_chi2 = []

    if not is_Out_of_Range(x, keys, params):  # to avoid overflow
        chi2 = 0

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
                (ma, ga, OmL, h0, M0), data=pan_data, **pan_kwargs)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('pantheon=%f' % this_chi2)

        # quasars
        if use_quasars:

            this_chi2 = chi2_quasars(
                (ma, ga, OmL, h0, qso_gamma, qso_beta, qso_delta),
                data=quasars_data,
                **quasars_kwargs)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('quasars=%f' % this_chi2)

        # other H0 experiments
        if use_TDCOSMO:

            this_chi2 = chi2_External(h0, data=ext_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('TDCOSMO=%f' % this_chi2)

        if use_early:

            this_chi2 = chi2_early(rs, data=early_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('early=%f' % this_chi2)

        # BOSS DR12
        if use_BOSSDR12:

            this_chi2 = chi2_BOSSDR12((OmL, h0, rs), data=boss_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('boss=%f' % this_chi2)

        # BAOlowz (6DFs + BOSS DR7 MGS, called smallz in MontePython)
        if use_BAOlowz:

            this_chi2 = chi2_BAOlowz((OmL, h0, rs), data=bao_data)
            chi2 += this_chi2
            lnprob_each_chi2.append(this_chi2)

            if verbose > 2:
                print('bao=%f' % this_chi2)

        # clusters
        if use_clusters:

            this_chi2 = chi2_clusters((ma, ga, OmL, h0), data=clusters_data,
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
