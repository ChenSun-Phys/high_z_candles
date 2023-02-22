#######################################################
###          Code for binned chi2(ma, ga)           ###
###          by Manuel A. Buen-Abad, 2020           ###
###               and Chen Sun, 2020, 2022          ###
#######################################################


import os
import errno
import sys
import getopt
import warnings
import random
import h5py

import numpy as np
from numpy import pi, sqrt, log, log10, exp, power
from scipy.interpolate import interp2d
from scipy.interpolate import LinearNDInterpolator as lndi
from tqdm import tqdm
from cosmo_axions_run import pltpath, fill_mcmc_parameters


def parse(directory,
          chain_name,
          directory2=None,
          chain2_name=None,
          bins=25,
          x_name='logma',
          y_name='logga',
          x_upper=np.inf,
          x_lower=-np.inf,
          y_upper=np.inf,
          y_lower=-np.inf):
    """Parse the chains and return the binned chi2. Marginalize all but two variables, denoted x and y.

    :param directory: directory of the chain
    :param chain_name: name of the chain
    :param directory2: directory of the second chain, optional
    :param chain2_name: name of the second chain to be combined together, optional
    :param bins: number of bins (default 25)
    :param x_name: the name of variable x (default: 'logma')
    :param y_name: the name of variable y (default: 'logga')
    :param x_upper: the upper cut of the x value (default: np.inf)
    :param x_lower: the lower cut of the x value (default: -np.inf)
    :param y_upper: the upper cut of the y value (default: np.inf)
    :param y_lower: the lower cut of the y value (default: -np.inf)
    :returns: (bf_chi2, ma_mesh, ga_mesh, chi2_mins, idx_mins_global, ma_arr, ga_arr, delta_arr) for global best fit chi2, meshgrid of ma, meshgrid of ga, meshgrid of chi2 local minimum in each block, the corresponding global indices of the local chi2 minima, ma array, ga array, interpolated local chi2 minimum
    :rtype: tuple of (scalar, 2D array, 2D array, 2D array, 2D array, 1D array, 1D array, 1D array)

    """
    # reading chains

    path1 = os.path.join(directory, chain_name)
    f1 = h5py.File(path1, 'r')
    f1 = f1['mcmc']

    if directory2 is not None:
        # there are two chains provided
        path2 = os.path.join(directory2, chain2_name)
        f2 = h5py.File(path2, 'r')
        f2 = f2['mcmc']
        f = {}
        for h5_key in f1.keys():
            f[h5_key] = np.concatenate((f1[h5_key], f2[h5_key]))
    else:
        # there is only one chain
        f = f1

    h5_keys = f.keys()
    print(h5_keys)

    # continue parsing the chain and make boxes around x and y
    pts = np.array(f['chain'])  # the points
    print("pts shape is:", pts.shape)

    # make sure to reshape according to the number of param
    num_of_params = pts.shape[-1]
    pts = pts.reshape(-1, num_of_params)

    chi2_tot = np.array(f['log_prob'])
    print("chi2_tot shape is:", chi2_tot.shape)

    chi2_tot *= -2
    chi2_tot = chi2_tot.reshape(-1)

    blobs = f['blobs']
    experiments = dict(blobs.dtype.fields).keys()
    print("experiments:", experiments)

    del f

    #
    # define the box of x, y
    #
    # load the meta data from log.param
    # load log.param
    params, mcmc_keys, mcmc_keys_fixed = fill_mcmc_parameters(
        os.path.join(directory, 'log.param'))

    # find the index of x and y
    print('variables scanned:', mcmc_keys)
    print(x_name)
    x_idx = np.where(np.array(mcmc_keys) == x_name)[0][0]
    y_idx = np.where(np.array(mcmc_keys) == y_name)[0][0]
    print(x_idx)
    print(y_idx)

    # test data integrity
    if len(mcmc_keys) != len(pts[0]):
        raise Exception(
            'log.param and h5 files are not consistent. Data is compromised. Quit analyzing.')

    # # the best fit chi2 and where it is

    chain_y = pts[:, y_idx]  # the values of y (e.g. ga)
    chain_y_cut = chain_y[np.where(chain_y < y_upper)]  # y upper cut
    chain_y_cut = chain_y_cut[np.where(chain_y_cut > y_lower)]  # y lower cut
    _, edges_y = np.histogram(
        chain_y_cut, bins=bins+1)  # the edges of the bins
    print("edges_y:", edges_y)
    print("chain_y has the length of", len(chain_y))
    print("chain_y_cut has the length of", len(chain_y_cut))

    chain_x = pts[:, x_idx]  # the values of x (e.g. ma)
    chain_x_cut = chain_x[np.where(chain_x < x_upper)]  # upper cut
    chain_x_cut = chain_x_cut[np.where(chain_x_cut > x_lower)]  # lower cut
    _, edges_x = np.histogram(
        chain_x_cut, bins=bins)  # the edges of the bins
    print("edges_x:", edges_x)
    print("chain_x has the length of", len(chain_x))
    print("chain_x_cut has the length of", len(chain_x_cut))
    # # test
    # tmp1_arr = np.where(chain_ga == 0.)[0]
    # tmp2_arr = np.where(tmp1_arr > 3901099)[0]
    # print("len(tmp1_arr)", len(tmp1_arr))
    # print("len(tmp2_arr)", len(tmp2_arr))
    # print("the index of the zero", tmp1_arr)

    # the best fit chi2 and where it is
    # bf_chi2 = min(chi2_tot[np.where(chain_ga < 0)])
    # bf_idx = chi2_tot[np.where(chain_ga < 0)].argmin()
    bf_chi2 = min(chi2_tot)
    bf_idx = chi2_tot.argmin()

    # the sum of the chi2 from each experiment at the best fit point
    # the experiments' chi2s for each point
    print('experiments:', experiments)
    print('bf_m2loglkl:', bf_chi2, 'bf_idx:', bf_idx)
    each_chi2 = {
        exper: blobs[exper].reshape(-1) for exper in experiments}
    # each_chi2 = {
    #     exper: blobs[exper].reshape(-1)[np.where(chain_ga < 0)] for exper in experiments}
    chi2_arr = [each_chi2[exper][bf_idx] for exper in experiments]
    each_sum = sum(chi2_arr)
    print("Each m2loglkl:", chi2_arr)
    print("m2loglkl best fit: {} = {}".format(
        bf_chi2, each_sum))  # sanity check

    # the center values
    block_y = (edges_y[:-1] + edges_y[1:])/2.
    block_x = (edges_x[:-1] + edges_x[1:])/2.
    # ga_mesh, ma_mesh = np.meshgrid(block_ga, block_ma, indexing='ij')
    x_mesh, y_mesh = np.meshgrid(block_x, block_y, indexing='ij')

    # preparation for the computation of the chi2(ma, ga) function

    chi2_mins = []  # the min chi2
    idx_mins = []  # the index of the min chi2
    idx_mins_global = []  # the index of the min chi2 in the total chi2 chain
    # the triples (ma, ga, min_chi2) only for those bins where the value is well defined
    x_y_chi2 = []

    wheres = {}  # those indices that satisfy the conditions to be within the bin

    for i in tqdm(range(len(edges_x)-1)):
        for j in (range(len(edges_y)-1)):

            # those points with y, x values within the bin
            wheres[i, j] = np.where((chain_y > edges_y[j])
                                    & (chain_y <= edges_y[j+1])
                                    & (chain_x > edges_x[i])
                                    & (chain_x <= edges_x[i+1]))

            # print('ma=%.2g, ga=%.2g' % (edges_ma[j], edges_ga[i]))
            # print('(%d, %d) block size: %d' % (i, j, len(wheres[i, j][0])))

            # the chi2s in that bin
            chi2_block = chi2_tot[wheres[i, j]]
            # print("chi2_block shape: ", chi2_block.shape)

            # appending minima and indices
            if len(chi2_block) > 0:

                this_min_chi2 = min(chi2_block)  # the minimum chi2 of this bin
                # print('this_min_chi2:', this_min_chi2)

                # appending to the list
                chi2_mins.append(this_min_chi2)
                idx_mins.append(chi2_block.argmin())
                idx_mins_global.append(
                    np.where(chi2_tot == this_min_chi2)[0][0])
                # appending to the data
                x_y_chi2.append(
                    [x_mesh[i, j], y_mesh[i, j], this_min_chi2])

            else:
                chi2_mins.append(np.inf)
                # chi2_mins.append(256)
                idx_mins.append(-1)
                idx_mins_global.append(-1)

                continue

    # converting to numpy arrays
    chi2_mins = np.array(chi2_mins)
    idx_mins = np.array(idx_mins, dtype=int)
    idx_mins_global = np.array(idx_mins_global, dtype=int)

    chi2_mins = chi2_mins.reshape(x_mesh.shape)
    idx_mins = idx_mins.reshape(x_mesh.shape)
    idx_mins_global = idx_mins_global.reshape(x_mesh.shape)
    # print(idx_mins_global)

    x_y_chi2 = np.array(x_y_chi2)

    #
    # interpolating over the data
    #
    # since data is not a uniform grid, we need to use LinearNDInterpolator
    delta_chi2 = lndi(x_y_chi2[:, 0:2], x_y_chi2[:, 2]-bf_chi2)

    x_arr = np.linspace(edges_x[0], edges_x[-1], 201)
    y_arr = np.linspace(edges_y[0], edges_y[-1], 201)
    y_gr, x_gr = np.meshgrid(y_arr, x_arr, indexing='ij')
    delta_arr = delta_chi2(x_gr, y_gr)

    return (bf_chi2, x_mesh, y_mesh, chi2_mins, idx_mins_global, x_arr, y_arr, delta_arr, _, pts, blobs)


def query(x, y, x_mesh, y_mesh, target_mesh):
    """This function finds the minimum of the chi2 in the block that contains the point (ma, ga). 

    :param x: value of x, e.g. log10(mass of axion/eV)
    :param y: value of y, e.g. log10(axion-photon coupling/(1/GeV))
    :param x_mesh: the meshgrid of log10(ma)
    :param y_mesh: the meshgrid of log10(ga)
    :param target_mesh: the target meshgrid to be checked. If it's the chi2_mesh, it outputs the minimal chi2 of the given box; if it's the global index mesh, it will output the index of the the minimal chi2 in the given box. 

    """
    x_arr = x_mesh[:, 0]
    y_arr = y_mesh[0, :]

    # find the block
    x_idx = np.searchsorted(x_arr, x)
    y_idx = np.searchsorted(y_arr, y)
    print(x_idx)
    print(y_idx)

    return target_mesh[x_idx, y_idx]


if __name__ == '__main__':
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from datetime import datetime
    except:
        pass

    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    argv = sys.argv[1:]
    help_msg = 'python %s -c <chain> -a <another_chain> -b <bins>' % (
        sys.argv[0])
    try:
        opts, args = getopt.getopt(argv, 'h:c:b:a:')
    except getopt.GetoptError:
        raise Exception(help_msg)
    flgc = False
    flgc2 = False
    flgb = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception(help_msg)
        elif opt == '-c':
            directory = os.path.dirname(arg)
            chain_name = os.path.basename(arg)
            flgc = True
        elif opt == '-a':
            directory2 = os.path.dirname(arg)
            chain2_name = os.path.basename(arg)
            flgc2 = True
        elif opt == '-b':
            bins = int(arg)
            flgb = True

    if not (flgc and flgb):
        raise Exception(help_msg)
    if flgc and flgc2:
        (bf_chi2,
         ma_mesh,
         ga_mesh,
         chi2_mins,
         idx_mins_global,
         ma_arr,
         ga_arr,
         delta_arr) = parse(directory,
                            chain_name,
                            directory2,
                            chain_name2,
                            bins)
    if flgc and (not flgc2):
        (bf_chi2,
         ma_mesh,
         ga_mesh,
         chi2_mins,
         idx_mins_global,
         ma_arr,
         ga_arr,
         delta_arr) = parse(directory,
                            chain_name,
                            bins=bins)

    # output of plots and tables
    # # the points of the 2-sigma (95.45% C.L.) contour
    cs = plt.contour(ma_arr, ga_arr, delta_arr, levels=[6.15823])
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    np.savetxt(pltpath(directory, head='2sigma_pts', ext='.txt'), v)

    # # the points of the 95% C.L. contour
    cs2 = plt.contour(ma_arr, ga_arr, delta_arr, levels=[5.99146])
    p2 = cs2.collections[0].get_paths()[0]
    v2 = p2.vertices
    np.savetxt(pltpath(directory, head='95CL_pts', ext='.txt'), v2)

    #
    # final plot
    #
    plt.figure(101)
    plt.xlabel(r'$\log_{10} m_a$')
    plt.ylabel(r'$\log_{10} g_a$')
    # plt.xlim(-17., -11.)
    # plt.ylim(-13., -8.)
    plt.title(r'$\Delta \chi^2$ contours')

    # the delta_chi2 1- and 2-sigma contours, both straight out of the data and interpolated
    plt.contour(ma_mesh, ga_mesh, chi2_mins-bf_chi2,
                levels=[2.29141, 6.15823, 10, 100], colors=['b', 'r', 'C2', 'C3'])
    # plt.contour(ma_mesh, ga_mesh, chi2_mins-bf_chi2,
    #             levels=[2.29141, 6.15823], colors=['b', 'r'])
    # the interpolation result
    plt.contour(ma_arr, ga_arr, delta_arr, levels=[2.29141, 6.15823], colors=[
                'C0', 'C1'], linestyles=[':', ':'])
    plt.savefig(pltpath(directory, head='delta_chi2_contours'))
