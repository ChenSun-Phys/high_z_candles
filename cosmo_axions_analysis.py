#######################################################
###      Code for emcee cosmo_axions analysis       ###
###               by Chen Sun, 2020                 ###
###         and Manuel A. Buen-Abad, 2020           ###
#######################################################

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime
except:
    pass
import os
import numpy as np
import emcee
from emcee.autocorr import AutocorrError
import corner
import h5py
import sys
import getopt
from cosmo_axions_run import pltpath, fill_mcmc_parameters, dir_init


if __name__ == '__main__':
    argv = sys.argv[1:]
    help_msg = 'python %s -i <folder_of_chains>' % (
        sys.argv[0])
    try:
        opts, args = getopt.getopt(argv, 'hi:')
    except getopt.GetoptError:
        raise Exception(help_msg)
    flgi = False
    flgx = False
    flgy = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception(help_msg)
        elif opt == '-i':
            directory = arg
            flgi = True
        # elif opt == '-x':
        #     x = arg
        #     flgx = True
        # elif opt == '-y':
        #     y = arg
        #     flgy = True
    if not flgi:
        raise Exception(help_msg)

    # determine if a custom plot x vs y will be made
    # if not (flgx and flgy):
    #     skip_custom = True
    # else:
    #     skip_custom = False

    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            path = os.path.join(directory, filename)

            reader = emcee.backends.HDFBackend(path, read_only=True)
            # tau = reader.get_autocorr_time()
            try:
                tau = reader.get_autocorr_time()
                print('auto correlation time = %s' % tau)
            except AutocorrError as e:
                # this is the case the chain is shorter than 50*(autocorr time)
                print('%s' % e)
                # tau = [410., 100., 140, 140]
                tau = e.tau
                print('setting correlation time to the current estimate.')

            # use auto-correlation time to estimate burnin here
            # works only for long chains
            burnin = int(2*np.max(tau))
            thin = int(0.5*np.min(tau))
            samples = reader.get_chain(
                discard=burnin, flat=True, thin=thin)
            print("burn-in: {0}".format(burnin))
            print("thin: {0}".format(thin))
            print("flat chain shape: {0}".format(samples.shape))
            try:
                all_samples = np.append(all_samples, samples, axis=0)
            except:
                all_samples = samples
        else:
            continue

    # load log.param
    params, keys, keys_fixed = fill_mcmc_parameters(
        os.path.join(directory, 'log.param'))

    # test data integrity
    if len(keys) != len(samples[0]):
        raise Exception(
            'log.param and h5 files are not consistent. Data is compromised. Quit analyzing.')

    # compute mean
    dim_of_param = len(samples[0])
    mean = np.mean(samples, axis=0)
    print('mean = %s' % mean)

    # corner plot
    plt.figure(0)
    # labels = keys
    # labels = [r"$\Omega_\Lambda$", r"$h$", r"$\log\ m_a$", r"$\log\ g_a$"]
    # labels = [r"$\Omega_\Lambda$"]
    labels = []
    print("keys:", keys)
    if 'OmL' in keys:
        labels.append(r"$\Omega_\Lambda$")
    if 'h0' in keys:
        labels.append(r"$h$")
    if 'w' in keys:
        labels.append(r"$w$")
    if 'w0' in keys:
        labels.append(r"$w_0$")
    if 'wa' in keys:
        labels.append(r"$w_a$")
    if 'logma' in keys:
        labels.append(r"$\log\ m_a$")
    if 'logga' in keys:
        labels.append(r"$\log\ g_a$")
    if 'M0' in keys:
        labels.append(r"$M_0$")
    if 'rs' in keys:
        labels.append(r"$r_s^{drag}$")
    if 'qso_gamma' in keys:
        labels.append(r"$\gamma$")
    if 'qso_beta' in keys:
        labels.append(r"$\beta$")
    if 'qso_delta' in keys:
        labels.append(r"$\delta$")

    print("labels:", labels)
    figure = corner.corner(samples,
                           labels=labels,
                           quantiles=[0.16, 0.5, 0.84],
                           levels=(0.68, 0.95),
                           show_titles=True,
                           title_kwargs={"fontsize": 12})
    axes = np.array(figure.axes).reshape((dim_of_param, dim_of_param))

    plt.savefig(pltpath(directory))

    # if not skip_custom:
    def get_custom(x, y):
        # define your custom 2D posterior

        # # axion analysis ma-ga
        # reduced_labels = [r"$\log\ m_a$", r"$\log\ g_a$"]
        # # reduced_samples = samples[:, 2:4]
        # reduced_samples = samples[:, 2:4]

        # wCDM analysis
        # reduced_labels = [r"$\log\ m_a$", r"$\log\ g_a$"]
        # reduced_labels = [r"$\Omega_\Lambda$", r"$w$"]
        # reduced_samples = samples[:, 0:3:2]
        # reduced_samples = reduced_samples[:, 0:2]

        x_idx = np.where(np.array(keys) == x)[0][0]
        y_idx = np.where(np.array(keys) == y)[0][0]

        print(x_idx, y_idx)
        print(labels[x_idx], labels[y_idx])

        reduced_labels = np.concatenate(([labels[x_idx]], [labels[y_idx]]))
        reduced_samples = samples[:, [x_idx, y_idx]]

        reduced_dim = len(reduced_labels)
        print(reduced_dim)

        # focusing on one contour 2sigma
        plt.figure(1)
        figure = corner.corner(reduced_samples,
                               labels=reduced_labels,
                               quantiles=[0.16, 0.5, 0.84],
                               color='b', show_titles=True,
                               plot_datapoints=False,
                               plot_density=False,
                               fill_contours=True,
                               # levels=[1.-np.exp(-(2.)**2 /2.)],
                               levels=[0.68, 0.95, 0.997],
                               title_kwargs={"fontsize": 12},
                               label_kwargs={"fontsize": 7},
                               hist_kwargs={'color': None},
                               contour_kwargs={'alpha': 0},
                               labelpad=-0.1)

        # print("figure.axes type is:", type(figure.axes))
        # print("figure.axes:", np.shape(figure.axes))
        # figure.axes = (figure.axes[2],)
        # print("figure.axes:", np.shape(figure.axes))

        # remove 1D posterior
        axes = np.array(figure.axes).reshape((reduced_dim, reduced_dim))
        print("shape of axes", axes.shape)
        for ax in axes[np.triu_indices(reduced_dim)]:
            ax.remove()

        for ax in figure.get_axes():
            # shrink tick size
            ax.tick_params(labelsize=7)
            # the following will not have any effects as
            # it's the same as passing labelpad into label_kwargs.
            # According to the doc of corner.py,
            # "Note that passing the labelpad keyword in this
            # dictionary will not have the desired effect. Use
            # the labelpad keyword in this function instead.

            # ax.xaxis.labelpad = 0
            # ax.yaxis.labelpad = 0

        # # saving the points of the 1/2/3 sigma contours
        # np.savetxt(pltpath(directory, head='corner_pts_2sigma_', ext='.txt'), v)
        plt.savefig(pltpath(directory, head='./plt_extract/custom_%s_%s' % (x, y)),
                    bbox_inches='tight')  # , bbox_inches=0)

        # 95 CL
        plt.figure(2)
        figure = corner.corner(reduced_samples,
                               labels=reduced_labels,
                               quantiles=[0.16, 0.5, 0.84],
                               color='b', show_titles=True,
                               plot_datapoints=False,
                               plot_density=False,
                               # levels=[1.-np.exp(-(2.)**2 /2.)],
                               levels=[0.95],
                               title_kwargs={"fontsize": 12},
                               hist_kwargs={'color': None})
        axes = np.array(figure.axes).reshape((reduced_dim, reduced_dim))

        p = (figure.axes)[2].collections[0].get_paths()[0]
        v = p.vertices

        # saving the points of the 95% C.R. contour
        np.savetxt(pltpath(directory, head='./pts_extract/corner_%s_%s_pts_2sigma_' %
                   (x, y), ext='.txt'), v)

        # other CL
        # focusing on one contour 3sigma
        plt.figure(3)
        figure = corner.corner(reduced_samples,
                               labels=reduced_labels,
                               quantiles=[0.16, 0.5, 0.84],
                               color='r', show_titles=True,
                               plot_datapoints=False,
                               plot_density=False,
                               # levels=[1.-np.exp(-(2.)**2 /2.)],
                               levels=[0.997],
                               title_kwargs={"fontsize": 12},
                               hist_kwargs={'color': None})
        axes = np.array(figure.axes).reshape((reduced_dim, reduced_dim))

        p = (figure.axes)[2].collections[0].get_paths()[0]
        v = p.vertices

        # saving the points of the 99.7% C.R. contour
        np.savetxt(pltpath(directory, head='./pts_extract/corner_%s_%s_pts_3sigma_' %
                   (x, y), ext='.txt'), v)

        # other CL
        # focusing on one contour 1sigma
        plt.figure(4)
        figure = corner.corner(reduced_samples,
                               labels=reduced_labels,
                               quantiles=[0.16, 0.5, 0.84],
                               color='r', show_titles=True,
                               plot_datapoints=False,
                               plot_density=False,
                               # levels=[1.-np.exp(-(2.)**2 /2.)],
                               levels=[0.68],
                               title_kwargs={"fontsize": 12},
                               hist_kwargs={'color': None})
        axes = np.array(figure.axes).reshape((reduced_dim, reduced_dim))

        p = (figure.axes)[2].collections[0].get_paths()[0]
        v = p.vertices

        # saving the points of the 68% C.R. contour
        np.savetxt(pltpath(directory, head='./pts_extract/corner_%s_%s_pts_1sigma_' %
                   (x, y), ext='.txt'), v)

    # actual payload
    dir_init(os.path.join(directory, './plots/pts_extract'))
    dir_init(os.path.join(directory, './plots/plt_extract'))
    for i, x in enumerate(keys[:-1]):
        for y in keys[i+1:]:
            get_custom(x, y)
