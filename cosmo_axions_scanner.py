#######################################################
###       Code for emcee cosmo_axions chains        ###
###               by Chen Sun, 2022                 ###
###         and Manuel A. Buen-Abad, 2022           ###
#######################################################

"""This is a module to scan a fixed grid of (ma, ga). At each (ma, ga) point, it will launch an MCMC (using emcee package) to maximize the log-likelihood w.r.t. other parameters. When run through this script, the MCMC tool is used as a minimizer. 

DONE: factorize out the main() and solve the pickle error.
TODO: make changes to different ma.

"""
import os
import sys
import warnings
import getopt
import numpy as np
from cosmo_axions_run import dir_init, fill_mcmc_parameters


if __name__ == '__main__':
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    argv = sys.argv[1:]
    help_msg = 'python %s -N <number_of_steps> -o <output_folder> -L <likelihood_directory> -i <param_file> -w <number_of_walkers>' % (
        sys.argv[0])
    try:
        opts, args = getopt.getopt(argv, 'hN:o:L:i:w:')
    except getopt.GetoptError:
        raise Exception(help_msg)
    flgN = False
    flgo = False
    flgL = False
    flgi = False
    flgw = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception(help_msg)
        elif opt == '-N':
            chainslength = int(arg)
            flgN = True
        elif opt == '-o':
            directory = arg
            flgo = True
        elif opt == '-L':
            dir_lkl = arg
            flgL = True
        elif opt == '-i':
            path_of_param = arg
            flgi = True
        elif opt == '-w':
            number_of_walkers = int(arg)
            flgw = True
    if not (flgN and flgo and flgL and flgi and flgw):
        raise Exception(help_msg)

    # init the dir
    dir_init(directory)
    # check if there's a preexisting param file
    if os.path.exists(os.path.join(directory, 'log.param')):
        path_of_param = os.path.join(directory, 'log.param')
        # get the mcmc params from existing file
        params, keys, keys_fixed = fill_mcmc_parameters(
            path_of_param)
    else:
        # get the mcmc params
        params, keys, keys_fixed = fill_mcmc_parameters(
            path_of_param)
        # save the input file only after the params are legit
        from shutil import copyfile
        copyfile(path_of_param, os.path.join(directory, 'log.param'))

    # read out ma, ga
    logma_arr = np.arange(params['logma'][1],
                          params['logma'][2], params['logma'][3])
    logga_arr = np.arange(params['logga'][1],
                          params['logga'][2], params['logga'][3])

    logma_mesh, logga_mesh = np.meshgrid(logma_arr, logga_arr, indexing='ij')
    logma_flat, logga_flat = logma_mesh.reshape(-1), logga_mesh.reshape(-1)
    for i, _ in enumerate(logma_flat):
        logma = logma_flat[i]
        logga = logga_flat[i]

        # subfolder name
        subdirectory = os.path.join(
            directory, "logma_%.2f_logga_%.2f" % (logma, logga))

        dir_init(subdirectory)

        # print(subdirectory)
        # init subfolder
        # generate new param file
    raise
    # print("logma:", logma_arr)
    # print("logga:", logga_arr)
    # # print("logga:", params['logga'])
    # raise

    # edit the param to fix ma, ga

    # the loop
    # TODO: change directory/subdir
    # TODO: chagne path_of_param
    main(chainslength,
         directory,
         dir_lkl,
         path_of_param,
         number_of_walkers)
