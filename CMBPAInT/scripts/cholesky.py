def calculate_cholesky(Input_Path, NJ, Nside, Lmax, Pol, TPol, Noise_Level, I_Mask, P_Mask, Chunks, DP, external_m, **kwargs):
    
    """ This function computes the cholesky decomposition of the pixel covariance matrix.
        It does not return anything, but it creates a hdf5 file saving the cholesky decomposition.

    Parameters
    ----------

    Input_Path : str
        Path to the folder where the pixel covariance matrix is saved in numpy format.
        Matrix could be divided in different files if the code is parallelized.
    NJ : int
        Number of files.
    Nside : int
        Resolution (in HEALPix format).
    Lmax : int
        Maximum multipole.
    Pol : bool
        True if polarization is considered.
    TPol : bool
        True if intensity and polarization is considered.
    Noise_Level : float
        Regularization noise level.
    I_Mask : str
        Path to the intensity mask.
    P_Mask : str
        Path to the polarization mask.
    Chunks : int
        Chunk size. This is used by dask package to speed up the code. An optimal chunk size is needed,
        not too big (big chunks), not too small (too many chunks).
    DP : bool
        True if double precision is wanted.
    external_m : str
        Path to external covariance matrix if needed. None if just CMB signal is considered.

    """

    timer_start = time.time()
    
    Npix = hp.nside2npix(Nside)
    IM = hp.read_map(I_Mask)
    PM = hp.read_map(P_Mask) if P_Mask is not None else None

    File = os.path.isfile(Input_Path + "/Chol.hdf5")
    if File:
        os.remove(Input_Path + "/Chol.hdf5")
    x = cholesky_tools.Read_from_Dask(Pol, TPol, Input_Path, NJ, Lmax, Chunks)
    if external_m is not None:
        x += cholesky_tools.Read_external_cov_mat(external_m, **kwargs)
    x = [cholesky_tools.Permutations(x, (IM, PM), Pol = True) if Pol and TPol else cholesky_tools.Permutations(x, PM, Pol = False) if Pol and TPol == False else cholesky_tools.Permutations(x, IM, Pol = False)][0]
    x = cholesky_tools.Add_Reg_Noise(x, Noise_Level)
    x = da.rechunk(x, (Chunks, Chunks))
    x = cholesky_tools.Cholesky(x)
    x = da.rechunk(x, (Chunks, Chunks))
    f = h5py.File(Input_Path + "/Chol.hdf5", "w")
    if DP:
        print("Precision: double")
        dset = f.create_dataset("/data", shape = x.shape, chunks = (x.chunks[0][0], x.chunks[1][0]), dtype = "float64")
    else:
        print("Precision: single")
        dset = f.create_dataset("/data", shape = x.shape, chunks = (x.chunks[0][0], x.chunks[1][0]), dtype = "float32")
    
    da.store(x, dset)
    
    timer_finish = time.time()
    
    np.save(Input_Path + "/Timer_Cholesky", timer_finish-timer_start)

if __name__ == '__main__':
    import os
    import time
    import numpy as np 
    from Tools import cholesky_tools
    import dask.array as da
    import h5py
    import argparse
    import healpy as hp 
    import configparser

    def initial():
        """Function to read input arguments from command line and to offer help to user."""
    
        parser = argparse.ArgumentParser()
    
        parser.add_argument("--config", type=str, help='Config file path.')
    
        args = parser.parse_args()
        
        return args

    args = initial()
    config_file = args.config

    config = configparser.ConfigParser()
    config.read(config_file)

    Input_Path = config['SOFTWARE_PARAMS']['Out_Matrix']
    Local = [False if config['SOFTWARE_PARAMS']['Local'] == "False" else True][0]
    Nside = int(config['MODEL_PARAMS']['Nside'])
    Lmax = int(config['MODEL_PARAMS']['Lmax'])
    NJ = [1 if Local else int(config['SOFTWARE_PARAMS']['NJ_Cov'])][0]
    Pol = [False if config['MODEL_PARAMS']['Pol'] == "False" else True][0]
    TPol = [False if config['MODEL_PARAMS']['TPol'] == "False" else True][0]
    Noise_Level = float(config['SOFTWARE_PARAMS']['Noise_Level'])
    I_Mask = config['MODEL_PARAMS']['I_Mask']
    P_Mask = None if config['MODEL_PARAMS']['P_Mask'] == 'None' else config['MODEL_PARAMS']['P_Mask']
    Chunks = int(config['SOFTWARE_PARAMS']['Chunks'])
    DP = [False if config['SOFTWARE_PARAMS']['DP'] == "False" else True][0]
    external_matrix = None if config['MODEL_PARAMS']['external_matrix'] == 'None' else config['MODEL_PARAMS']['external_matrix']

    calculate_cholesky(Input_Path, NJ, Nside, Lmax, Pol, TPol, Noise_Level, I_Mask, P_Mask, Chunks, DP, external_m = external_matrix)
