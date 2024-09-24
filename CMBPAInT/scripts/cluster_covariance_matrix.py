def calculate_covariance_matrix_cluster(Cls, Nside, Lmax, Out_PATH, Pol, TPol, EB, TB, NJ, Counter):

    """This function computes the signal pixel-pixel covariance matrix.
       It allows to split the computation in different jobs, each of them
       parallelized using MPI standard. This script is developed to be used in a cluster.
       It does not return anything, but it generates a set of numpy files saving the covariance matrix
       and some information about computational time.

    Parameters
    ----------

    Cls: str
        Path to the fiducial Cls (the model). It could be saved in a text file or in a numpy format.
        Cls should be an array with 6 rows (TT, EE, BB, TE, TB, EB) and lmax+1 columns.
        First two columns should be 0 if monopole and dipole are not take into account.
    Nside: int
        Resolution (in HEALPix format).
    Lmax: int
        Maximum multipole.
    Out_PATH: str
        Path to the folder where the output files will be stored.
    Pol: bool
        True if polarization is considered.
    TPol: bool
        True if TE correlation is considered.
    EB: bool
        True if EB is considered. This allows not standard models where EB is zero.
    TB: bool
        True if TB correlation is considered. If both, TPol and TB, are zero, TQ and TU
        blocks in the covariance matrix will be zero before rotations are applied.
        On the other hand, if both are False, but Pol is True, only polarization
        covariance matrix will be saved.
    NJ: int
        Number of jobs.
    Counter: int
        Job ID.

    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    Npix = hp.nside2npix(Nside)

    if rank == 0:
        Indices = np.arange(Npix)
        if NJ != 1:
            quotient, remainder = divmod(Indices.size, NJ)
            counts = [quotient + 1 if p < remainder else quotient for p in range(NJ)]
            starts = [sum(counts[:p]) for p in range(NJ)]
            ends = [sum(counts[:p+1]) for p in range(NJ)]
            Indices = [Indices[starts[p]:ends[p]] for p in range(NJ)]
            Indices = Indices[Counter]
        u = np.zeros((len(Indices), 3))
        u[:, 0], u[:, 1], u[:, 2] = hp.pix2vec(nside = Nside, ipix = Indices)
        quotient, remainder = divmod(Indices.size, size)
        counts = [quotient + 1 if p < remainder else quotient for p in range(size)]
        starts = [sum(counts[:p]) for p in range(size)]
        ends = [sum(counts[:p+1]) for p in range(size)]
        Indices_U = [u[starts[p]:ends[p], :] for p in range(size)]
        Indices_Pix = [Indices[starts[p]:ends[p]] for p in range(size)]
    else:
        Indices_U = None
        Indices_Pix = None

    start_time = time.time()
    
    Indices_U = comm.scatter(Indices_U, root = 0)
    Indices_Pix = comm.scatter(Indices_Pix, root = 0)

    if rank == 0:
        print("Number of processes: ", size)

    v = np.zeros((3, Npix))
    v[0, :], v[1, :], v[2, :] = hp.pix2vec(nside = Nside, ipix = np.arange(Npix))

    # Read the Cls and compute the covariance matrix

    if type(Cls) == str:
        try:
            Cls = np.load(Cls)
        except:
            Cls = np.loadtxt(Cls)
    else:
        Cls = Cls

    if rank == 0:
        print("Pol: ", Pol)
        print("TPol: ", TPol)

    if Pol == False:
        Cls_TT = Cls[0, :]
        Cov_TT = np.zeros((len(Indices_U), Npix))
        N_Rows = len(Indices_U)

        if rank == 0:
            pbar = tqdm(total = N_Rows)

        for i in range(N_Rows):
            x = np.round(np.dot(Indices_U[i, :], v[:, :(Indices_Pix[i]+1)]), decimals = 15)
            Cov_TT[i, :(Indices_Pix[i]+1)] = covariance_tools.Scalar_MatCov(Cls_TT, Lmax, x, Npix = None)

            if rank == 0:
                pbar.update(1)
            
        memory_usage = get_memory_usage()
        print(f"MPI Task {rank}/{size}: Memory usage = {memory_usage:.2f} MB")

        comm.barrier()
    else:
        Cls_TT = Cls[0, :]
        Cls_EE = Cls[1, :]
        Cls_BB = Cls[2, :]
        Cls_TE, Cls_TB, Cls_EB = [Cls[3+lst, :] if [TPol, TB, EB][lst] == True else None for lst in range(3)]

        N_Rows = len(Indices_U)
            
        Cov_TT = [np.zeros((len(Indices_U), Npix)) if TPol == True else None][0]
        Cov_TQ_R = [np.zeros((len(Indices_U), Npix)) if TPol == True else None][0]
        Cov_TU_R = [np.zeros((len(Indices_U), Npix)) if TPol == True else None][0]
        Cov_QT_R = [np.zeros((len(Indices_U), Npix)) if TPol == True else None][0]
        Cov_UT_R = [np.zeros((len(Indices_U), Npix)) if TPol == True else None][0]
        Cov_QQ_R = np.zeros((len(Indices_U), Npix))
        Cov_UU_R = np.zeros((len(Indices_U), Npix))
        Cov_QU_R = np.zeros((len(Indices_U), Npix))
        Cov_UQ_R = np.zeros((len(Indices_U), Npix))

        if rank == size-1:
            pbar = tqdm(total = N_Rows)

        if TPol:
            for i in range(N_Rows):
                x = np.round(np.dot(Indices_U[i, :], v[:, :(Indices_Pix[i]+1)]), decimals = 15)
                Cov_TT[i, :(Indices_Pix[i]+1)], Cov_QQ, Cov_UU, Cov_TQ, Cov_TU, Cov_QU = covariance_tools.Intensity_Pol_MatCov((Cls_TT, Cls_EE, Cls_BB, Cls_TE, Cls_EB, Cls_TB), lmax = Lmax, x = x, Npix = Npix)
                Cov_TQ_R[i, :(Indices_Pix[i]+1)], Cov_QT_R[i, :(Indices_Pix[i]+1)], Cov_TU_R[i, :(Indices_Pix[i]+1)], Cov_UT_R[i, :(Indices_Pix[i]+1)], Cov_QQ_R[i, :(Indices_Pix[i]+1)], Cov_UU_R[i, :(Indices_Pix[i]+1)], Cov_QU_R[i, :(Indices_Pix[i]+1)], Cov_UQ_R[i, :(Indices_Pix[i]+1)] = covariance_tools.Rotations_TPol((Cov_TQ, Cov_TU, Cov_QQ, Cov_UU, Cov_QU), Indices_U[i, :], v[:, :(Indices_Pix[i]+1)])

                if rank == size-1:
                    pbar.update(1)

            memory_usage = get_memory_usage()
            print(f"MPI Task {rank}/{size}: Memory usage = {memory_usage:.2f} MB")

            comm.barrier()
        else:  
            for i in range(N_Rows):
                x = np.round(np.dot(Indices_U[i, :], v[:, :(Indices_Pix[i]+1)]), decimals = 15)
                Cov = covariance_tools.Only_Pol_MatCov((Cls_EE, Cls_BB, Cls_EB), lmax = Lmax, x = x)
                Cov_QQ_R[i, :(Indices_Pix[i]+1)], Cov_UU_R[i, :(Indices_Pix[i]+1)], Cov_QU_R[i, :(Indices_Pix[i]+1)], Cov_UQ_R[i, :(Indices_Pix[i]+1)] = covariance_tools.Rotations_Pol(Cov, Indices_U[i, :], v[:, :(Indices_Pix[i]+1)])
                    
                if rank == size-1:
                    pbar.update(1)

            memory_usage = get_memory_usage()
            print(f"MPI Task {rank}/{size}: Memory usage = {memory_usage:.2f} MB")

            comm.barrier()

    if Pol == False:
        newCov_TT = comm.gather(Cov_TT, root = 0)
        
        print("Rank: "+str(rank)+". Shape: "+str(Cov_TT.shape))
        
        end_time = time.time()
        execution_time = end_time-start_time
    
        exec_time = comm.gather(execution_time, root = 0)
        
        if rank == 0:
            New_Cov_TT = np.vstack(newCov_TT)

            if (os.path.exists(Out_PATH) == False):
                os.makedirs(Out_PATH)
            if (os.path.exists(Out_PATH+"/Time_Stats") == False):
                os.mkdir(Out_PATH+"/Time_Stats")
                
            if NJ != 1:                
                np.save(Out_PATH+"/Cov_TT_lmax_"+str(Lmax)+"_"+str(Counter)+".npy", New_Cov_TT)
                
                with open(Out_PATH+'/Time_Stats/covmat_timer_TT_'+str(Counter)+'.txt', 'w') as f:
                    f.write(f'Execution time: {np.mean(exec_time):.2f} seconds')
                np.save(Out_PATH+"/Time_Stats/covmat_timer_TT_"+str(Counter), np.array(exec_time))
            else:
                np.save(Out_PATH+"/Cov_TT_lmax_"+str(Lmax)+".npy", New_Cov_TT)
            
                with open(Out_PATH+'/Time_Stats/covmat_timer_TT.txt', 'w') as f:
                    f.write(f'Execution time: {np.mean(exec_time):.2f} seconds')
                np.save(Out_PATH+"/Time_Stats/covmat_timer_TT", np.array(exec_time))

        comm.barrier()

        np.save(Out_PATH + "/Time_Stats/covmat_timer_T_"+str(Counter)+"_rank_"+str(rank)+".npy", execution_time)
            
    else:
        newCov_TT = [comm.gather(Cov_TT, root = 0) if TPol == True else None][0]
        newCov_TQ_R = [comm.gather(Cov_TQ_R, root = 0) if TPol == True else None][0]
        newCov_TU_R = [comm.gather(Cov_TU_R, root = 0) if TPol == True else None][0]
        newCov_QT_R = [comm.gather(Cov_QT_R, root = 0) if TPol == True else None][0]
        newCov_UT_R = [comm.gather(Cov_UT_R, root = 0) if TPol == True else None][0]
        newCov_QQ_R = comm.gather(Cov_QQ_R, root = 0)
        newCov_UU_R = comm.gather(Cov_UU_R, root = 0)
        newCov_QU_R = comm.gather(Cov_QU_R, root = 0)
        newCov_UQ_R = comm.gather(Cov_UQ_R, root = 0)
        
        end_time = time.time()
        execution_time = end_time-start_time
        
        exec_time = comm.gather(execution_time, root = 0)
        
        if rank == 0:
            New_Cov_TT = [np.vstack(newCov_TT) if TPol == True else None][0]
            New_Cov_TQ_R = [np.vstack(newCov_TQ_R) if TPol == True else None][0]
            New_Cov_TU_R = [np.vstack(newCov_TU_R) if TPol == True else None][0]
            New_Cov_QT_R = [np.vstack(newCov_QT_R) if TPol == True else None][0]
            New_Cov_UT_R = [np.vstack(newCov_UT_R) if TPol == True else None][0]
            New_Cov_QQ_R = np.vstack(newCov_QQ_R)
            New_Cov_UU_R = np.vstack(newCov_UU_R)
            New_Cov_QU_R = np.vstack(newCov_QU_R)
            New_Cov_UQ_R = np.vstack(newCov_UQ_R)

            if (os.path.exists(Out_PATH) == False):
                os.makedirs(Out_PATH)
            if (os.path.exists(Out_PATH+"/Time_Stats") == False):
                os.mkdir(Out_PATH+"/Time_Stats")
            
            if NJ != 1:
                [np.save(Out_PATH+"/Cov_TT_lmax_"+str(Lmax)+"_"+str(Counter)+".npy", New_Cov_TT) if TPol == True else None][0]
                [np.save(Out_PATH+"/Cov_TQ_lmax_"+str(Lmax)+"_"+str(Counter)+".npy", New_Cov_TQ_R) if TPol == True else None][0]
                [np.save(Out_PATH+"/Cov_TU_lmax_"+str(Lmax)+"_"+str(Counter)+".npy", New_Cov_TU_R) if TPol == True else None][0]
                [np.save(Out_PATH+"/Cov_QT_lmax_"+str(Lmax)+"_"+str(Counter)+".npy", New_Cov_QT_R) if TPol == True else None][0]
                [np.save(Out_PATH+"/Cov_UT_lmax_"+str(Lmax)+"_"+str(Counter)+".npy", New_Cov_UT_R) if TPol == True else None][0]
                np.save(Out_PATH+"/Cov_QQ_lmax_"+str(Lmax)+"_"+str(Counter)+".npy", New_Cov_QQ_R)
                np.save(Out_PATH+"/Cov_UU_lmax_"+str(Lmax)+"_"+str(Counter)+".npy", New_Cov_UU_R)
                np.save(Out_PATH+"/Cov_QU_lmax_"+str(Lmax)+"_"+str(Counter)+".npy", New_Cov_QU_R)
                np.save(Out_PATH+"/Cov_UQ_lmax_"+str(Lmax)+"_"+str(Counter)+".npy", New_Cov_UQ_R)
                
                if TPol:
                    with open(Out_PATH+'/Time_Stats/covmat_timer_TPol_'+str(Counter)+'.txt', 'w') as f:
                        f.write(f'Execution time: {np.mean(exec_time):.2f} seconds')
                    np.save(Out_PATH+"/Time_Stats/covmat_timer_TPol_"+str(Counter), np.array(exec_time))
                else:
                    with open(Out_PATH+'/Time_Stats/covmat_timer_Pol_'+str(Counter)+'.txt', 'w') as f:
                        f.write(f'Execution time: {np.mean(exec_time):.2f} seconds')
                    np.save(Out_PATH+"/Time_Stats/covmat_timer_Pol_"+str(Counter), np.array(exec_time))
                
            else:
                [np.save(Out_PATH+"/Cov_TT_lmax_"+str(Lmax)+".npy", New_Cov_TT) if TPol == True else None][0]
                [np.save(Out_PATH+"/Cov_TQ_lmax_"+str(Lmax)+".npy", New_Cov_TQ_R) if TPol == True else None][0]
                [np.save(Out_PATH+"/Cov_TU_lmax_"+str(Lmax)+".npy", New_Cov_TU_R) if TPol == True else None][0]
                [np.save(Out_PATH+"/Cov_QT_lmax_"+str(Lmax)+".npy", New_Cov_QT_R) if TPol == True else None][0]
                [np.save(Out_PATH+"/Cov_UT_lmax_"+str(Lmax)+".npy", New_Cov_UT_R) if TPol == True else None][0]
                np.save(Out_PATH+"/Cov_QQ_lmax_"+str(Lmax)+".npy", New_Cov_QQ_R)
                np.save(Out_PATH+"/Cov_UU_lmax_"+str(Lmax)+".npy", New_Cov_UU_R)
                np.save(Out_PATH+"/Cov_QU_lmax_"+str(Lmax)+".npy", New_Cov_QU_R)
                np.save(Out_PATH+"/Cov_UQ_lmax_"+str(Lmax)+".npy", New_Cov_UQ_R)
                
                if TPol:
                    with open(Out_PATH+'/Time_Stats/covmat_timer_TPol.txt', 'w') as f:
                        f.write(f'Execution time: {np.mean(exec_time):.2f} seconds')
                    np.save(Out_PATH+"/Time_Stats/covmat_timer_TPol", np.array(exec_time))
                else:
                    with open(Out_PATH+'/Time_Stats/covmat_timer_Pol.txt', 'w') as f:
                        f.write(f'Execution time: {np.mean(exec_time):.2f} seconds')
                    np.save(Out_PATH+"/Time_Stats/covmat_timer_TPol", np.array(exec_time))

        comm.barrier()

        np.save(Out_PATH + "/Time_Stats/covmat_timer_Pol_"+str(Counter)+"_rank_"+str(rank)+".npy", execution_time)

def get_memory_usage():

    """
    Get the memory usage per MPI task.
    """

    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB
                    
if __name__ == '__main__':
    import os
    import time
    import psutil
    import argparse
    import numpy as np
    import configparser
    import healpy as hp
    from tqdm import tqdm
    from Tools import covariance_tools

    def initial():
        """Function to read input arguments from command line and to offer help to user."""
    
        parser = argparse.ArgumentParser()
    
        parser.add_argument("--config", type=str, help='Config File Path.')
        parser.add_argument("-C", type=int, help='Job ID from the stack.')
    
        args = parser.parse_args()
        
        return args

    args = initial()
    config_file = args.config
    Counter = args.C

    config = configparser.ConfigParser()
    config.read(config_file)

    Cls = config['MODEL_PARAMS']['Cls']
    Nside = int(config['MODEL_PARAMS']['Nside'])
    Lmax = int(config['MODEL_PARAMS']['Lmax'])
    Pol = [False if config['MODEL_PARAMS']['Pol'] == "False" else True][0]
    TPol = [False if config['MODEL_PARAMS']['TPol'] == "False" else True][0]
    EB = [False if config['MODEL_PARAMS']['EB'] == "False" else True][0]
    TB = [False if config['MODEL_PARAMS']['TB'] == "False" else True][0]
    Out_PATH = config['SOFTWARE_PARAMS']['Out_Matrix']
    NJ = int(config['SOFTWARE_PARAMS']['NJ_cov'])

    calculate_covariance_matrix_cluster(Cls, Nside, Lmax, Out_PATH, Pol, TPol, EB, TB, NJ = NJ, Counter = Counter)
