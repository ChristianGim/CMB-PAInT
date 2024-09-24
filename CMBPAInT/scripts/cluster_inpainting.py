def calculate_inpainting(Input_Path, Output_Path, Name, Chol_Path, I_Mask, P_Mask, Pol, TPol, Num_Sims, Single, No_Z, Cons_Uncons, Fields, Zbar_Path, NJ, Counter):

    """This function inpaints the input map (or maps).
       It allows to split the computation in different jobs, each of them
       parallelized using MPI standard. This script is developed to be used in a cluster.
       It does not return anything, but instead it saves the inpainted maps in fits format, together
       with some numpy files (information about computational time).

    Parameters
    ----------

    Input_Path: str
        Path to the folder where the input map (or maps) are stored.
    Output_Path: str
        Path to the folder where the inpainted maps will be stored.
    Name: str
        Name of input files. If more than one maps are given, the code
        will concatenate this with "_"+str(i)+".fits", where i index runs from 0 to Num_Sims-1.
    Chol_Path: str
        Path to the folder where cholesky decomposition is stored (in hdf5 format).
    I_Mask: str
        Path to the intensity mask.
    P_Mask: str
        Path to the polarization mask.
    Pol: bool
        True if polarization is considered.
    TPol: bool
        True if intensity and polarization are considered together.
    Num_Sims: int
        Number of simulations to be inpainted.
    Single: bool
        If True a single input map is going to be inpainted. In this case,
        Num_Sims parameter fix how many inpainted realizations are going to be generated.
        If False, an unique inpainted realizations per input map is generated.
    No_Z: bool
        If True Z variables are not computed. It assumes precomputed Z.
    Cons_Uncons: bool
        If True constrained and unconstrained parts will be stored also in the fits file (3 fields).
        The order will be: constrained, unconstrained, and the sum.
    Fields: int
        The fields to be readed from the input fits file (only for polarization case).
        If Fields = 2, it assumes that the fits file contains only QU maps, and it reads fields 0 and 1.
        If Fields = 3, it assumes that the fits file constains TQU maps, and it reads fields 1 and 2.
    Zbar_Path: str
        Path to the folder where Zbar random variables are storesv(including the name of the file, the code
        will concatenate this with "_"+str(i)+".npy").
        If None, new normal random variables are generated.
        This allows to reuse previously presaved seeds.
    NJ: int
        Number of jobs.
    Counter: int
        Job ID.

    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    time_global_start = time.time()
    
    if rank == 0:
        if Single:
            print("Single sky. Number of realizations: ", Num_Sims)
        else:
            print("Number of skies: ", Num_Sims)

    comm.barrier()

    if rank == 0:
        Indices = np.arange(Num_Sims)
        if NJ != 1:
            quotient, remainder = divmod(Indices.size, NJ)
            counts = [quotient + 1 if p < remainder else quotient for p in range(NJ)]
            starts = [sum(counts[:p]) for p in range(NJ)]
            ends = [sum(counts[:p+1]) for p in range(NJ)]
            Indices = [Indices[starts[p]:ends[p]] for p in range(NJ)]
            Indices = Indices[Counter]
        quotient, remainder = divmod(Indices.size, size)
        counts = [quotient + 1 if p < remainder else quotient for p in range(size)]
        starts = [sum(counts[:p]) for p in range(size)]
        ends = [sum(counts[:p+1]) for p in range(size)]
        Indices = [Indices[starts[p]:ends[p]] for p in range(size)]
    else:
        Indices = None

    Indices = comm.scatter(Indices, root = 0)

    read_timer_start = time.time()
    
    Mask_T = [hp.read_map(I_Mask) if I_Mask != None else None][0]
    Mask_Pol = [hp.read_map(P_Mask) if P_Mask != None else None][0]
    Ind_T = [np.where(Mask_T == 1)[0] if I_Mask != None else None][0]
    Ind_Pol = [np.where(Mask_Pol == 1)[0] if P_Mask != None else None][0]
    Chol = h5py.File(os.path.join(Chol_Path, 'Chol.hdf5'))
    
    if No_Z:
        pass
    else:
        L = [Chol['/data'][:(len(Ind_T)+2*len(Ind_Pol)), :(len(Ind_T)+2*len(Ind_Pol))] if TPol and Pol else Chol['/data'][:2*len(Ind_Pol), :2*len(Ind_Pol)] if TPol == False and Pol else Chol['/data'][:len(Ind_T), :len(Ind_T)]][0]

        L_read_timer = time.time()

        memory_usage = get_memory_usage()
        print("L block reading...")
        print(f"MPI Task {rank}/{size}: Memory usage = {memory_usage:.2f} MB")

        comm.barrier()
        
        if Single:
            if rank == 0:
                starts_Zs = time.time()
                print("Starting Z computation...")
                if (os.path.exists(os.path.join(Output_Path, "Z")) == False):
                    os.makedirs(os.path.join(Output_Path, "Z"))
                Map_Path = os.path.join(Input_Path, Name)
                Z = inpainting_tools.Z(Map_Path, L, I_Mask, Pol = Pol, TPol = TPol, Mask_Pol = P_Mask)
                [np.save(os.path.join(Output_Path, "Z/Z_TPol"), Z) if TPol and Pol else np.save(os.path.join(Output_Path, "Z/Z_Pol"), Z) if TPol == False and Pol else np.save(os.path.join(Output_Path, "Z/Z_T"), Z)][0]
                finishes_Zs = time.time()
            else:
                starts_Zs = None
                finishes_Zs = None

            comm.barrier()
        else:
            starts_Zs = np.zeros(len(Indices))
            finishes_Zs = np.zeros(len(Indices))

            if rank == 0:
                pbar = tqdm(total = len(Indices))
                if (os.path.exists(os.path.join(Output_Path, "Z")) == False):
                    os.makedirs(os.path.join(Output_Path, "Z"))
            
            comm.barrier()

            for i in range(len(Indices)):
                starts_Zs[i] = time.time()
                Map_Path = os.path.join(Input_Path, Name+"_"+str(Indices[i])+".fits")
                Z = inpainting_tools.Z(Map_Path, L, I_Mask, Pol = Pol, TPol = TPol, Mask_Pol = P_Mask)
                [np.save(os.path.join(Output_Path, "Z/Z_TPol_"+str(Indices[i])), Z) if TPol and Pol else np.save(os.path.join(Output_Path, "Z/Z_Pol_"+str(Indices[i])), Z) if TPol == False and Pol else np.save(os.path.join(Output_Path, "Z/Z_T_"+str(Indices[i])), Z)][0]

                if rank == 0:
                    pbar.update(1)
                
                finishes_Zs[i] = time.time()

            comm.barrier()

        del(L)

        if rank == 0:
            print("Z computation done!")

    comm.barrier()

    R_timer_start = time.time()
    
    R = [Chol['/data'][(len(Ind_T)+2*len(Ind_Pol)):, :(len(Ind_T)+2*len(Ind_Pol))] if TPol and Pol else Chol['/data'][2*len(Ind_Pol):, :2*len(Ind_Pol)] if TPol == False and Pol else Chol['/data'][len(Ind_T):, :len(Ind_T)]][0]
    Lbar = [Chol['/data'][(len(Ind_T)+2*len(Ind_Pol)):, (len(Ind_T)+2*len(Ind_Pol)):] if TPol and Pol else Chol['/data'][2*len(Ind_Pol):, 2*len(Ind_Pol):] if TPol == False and Pol else Chol['/data'][len(Ind_T):, len(Ind_T):]][0]

    R_timer_finish = time.time()
    
    starts_inp = np.zeros(len(Indices))
    finishes_inp = np.zeros(len(Indices))

    memory_usage = get_memory_usage()
    print("R and Lbar blocks reading...")
    print(f"MPI Task {rank}/{size}: Memory usage = {memory_usage:.2f} MB")

    comm.barrier()

    if rank == 0:
        print("Starting inpainting...")
        pbar = tqdm(total = len(Indices))
        if (os.path.exists(os.path.join(Output_Path, "Inpainted_Maps")) == False):
            os.makedirs(os.path.join(Output_Path, "Inpainted_Maps"))
        if (os.path.exists(os.path.join(Output_Path, "Zbar")) == False):
            os.makedirs(os.path.join(Output_Path, "Zbar"))

    comm.barrier()

    if Single:
        Map_Path = os.path.join(Input_Path, Name)
        Z = [np.load(os.path.join(Output_Path, "Z/Z_TPol.npy")) if TPol and Pol else np.load(os.path.join(Output_Path, "Z/Z_Pol.npy")) if TPol == False and Pol else np.load(os.path.join(Output_Path, "Z/Z_T.npy"))][0]
        
        for i in range(len(Indices)):
            starts_inp[i] = time.time()
            
            if Zbar_Path is not None:
                z_bar = Zbar_Path+"_"+str(Indices[i])+".npy"
            else:
                z_bar = None
            
            if TPol and Pol:
                Returns = inpainting_tools.Inpaint(Map_Path, R, Lbar, Z, I_Mask, z_bar = z_bar, Cons_Uncons = Cons_Uncons, Pol = True, Pol_Mask = P_Mask, TPol = True)
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_T_"+str(Indices[i])+".fits"), Returns[:3], dtype = np.float64, overwrite = True) if Cons_Uncons else hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_TQU_"+str(Indices[i])+".fits"), Returns[:3], dtype = np.float64, overwrite = True)][0]
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_Q_"+str(Indices[i])+".fits"), Returns[3:6], dtype = np.float64, overwrite = True) if Cons_Uncons else None][0]
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_U_"+str(Indices[i])+".fits"), Returns[6:9], dtype = np.float64, overwrite = True) if Cons_Uncons else None][0]
            elif TPol == False and Pol:
                Returns = inpainting_tools.Inpaint(Map_Path, R, Lbar, Z, I_Mask, z_bar = z_bar, T_Mask = None, Cons_Uncons = Cons_Uncons, Pol = True, Pol_Mask = P_Mask, TPol = False, Fields = Fields)
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_Q_"+str(Indices[i])+".fits"), Returns[:3], dtype = np.float64, overwrite = True) if Cons_Uncons else hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_QU_"+str(Indices[i])+".fits"), Returns[:2], dtype = np.float64, overwrite = True)][0]
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_U_"+str(Indices[i])+".fits"), Returns[3:6], dtype = np.float64, overwrite = True) if Cons_Uncons else None][0]
            else:
                Returns = inpainting_tools.Inpaint(Map_Path, R, Lbar, Z, I_Mask, z_bar = z_bar, Cons_Uncons = Cons_Uncons, Pol = False, Pol_Mask = None, TPol = False)
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_T_"+str(Indices[i])+".fits"), Returns[:3], dtype = np.float64, overwrite = True) if Cons_Uncons else hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_T_"+str(Indices[i])+".fits"), Returns[0], dtype = np.float64, overwrite = True)][0]
                
            [np.save(os.path.join(Output_Path, "Zbar/Zbar_TPol_"+str(Indices[i])), Returns[-1]) if TPol and Pol else np.save(os.path.join(Output_Path, "Zbar/Zbar_Pol_"+str(Indices[i])), Returns[-1]) if TPol == False and Pol else np.save(os.path.join(Output_Path, "Zbar/Zbar_T_"+str(Indices[i])), Returns[-1])][0]

            if rank == 0:
                pbar.update(1)

            finishes_inp[i] = time.time()

        comm.barrier()
    else:
        for i in range(len(Indices)):
            starts_inp[i] = time.time()
            Map_Path = os.path.join(Input_Path, Name+"_"+str(Indices[i])+".fits")
            Z = [np.load(os.path.join(Output_Path, "Z/Z_TPol_"+str(Indices[i])+".npy")) if TPol and Pol else np.load(os.path.join(Output_Path, "Z/Z_Pol_"+str(Indices[i])+".npy")) if TPol == False and Pol else np.load(os.path.join(Output_Path, "Z/Z_T_"+str(Indices[i])+".npy"))][0]
            
            if Zbar_Path is not None:
                z_bar = Zbar_Path+"_"+str(Indices[i])+".npy"
            else:
                z_bar = None
                
            if TPol and Pol:
                Returns = inpainting_tools.Inpaint(Map_Path, R, Lbar, Z, I_Mask, z_bar = z_bar, Cons_Uncons = Cons_Uncons, Pol = True, Pol_Mask = P_Mask, TPol = True)
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_T_"+str(Indices[i])+".fits"), Returns[:3], dtype = np.float64, overwrite = True) if Cons_Uncons else hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_TQU_"+str(Indices[i])+".fits"), Returns[:3], dtype = np.float64, overwrite = True)][0]
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_Q_"+str(Indices[i])+".fits"), Returns[3:6], dtype = np.float64, overwrite = True) if Cons_Uncons else None][0]
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_U_"+str(Indices[i])+".fits"), Returns[6:9], dtype = np.float64, overwrite = True) if Cons_Uncons else None][0]
            elif TPol == False and Pol:
                Returns = inpainting_tools.Inpaint(Map_Path, R, Lbar, Z, I_Mask, z_bar = z_bar, T_Mask = None, Cons_Uncons = Cons_Uncons, Pol = True, Pol_Mask = P_Mask, TPol = False, Fields = Fields)
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_Q_"+str(Indices[i])+".fits"), Returns[:3], dtype = np.float64, overwrite = True) if Cons_Uncons else hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_QU_"+str(Indices[i])+".fits"), Returns[:2], dtype = np.float64, overwrite = True)][0]
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_U_"+str(Indices[i])+".fits"), Returns[3:6], dtype = np.float64, overwrite = True) if Cons_Uncons else None][0]
            else:
                Returns = inpainting_tools.Inpaint(Map_Path, R, Lbar, Z, I_Mask, z_bar = z_bar, Cons_Uncons = Cons_Uncons, Pol = False, Pol_Mask = None, TPol = False)
                [hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_T_"+str(Indices[i])+".fits"), Returns[:3], dtype = np.float64, overwrite = True) if Cons_Uncons else hp.write_map(os.path.join(Output_Path, "Inpainted_Maps/Inpainted_T_"+str(Indices[i])+".fits"), Returns[0], dtype = np.float64, overwrite = True)][0]
                
            [np.save(os.path.join(Output_Path, "Zbar/Zbar_TPol_"+str(Indices[i])), Returns[-1]) if TPol and Pol else np.save(os.path.join(Output_Path, "Zbar/Zbar_Pol_"+str(Indices[i])), Returns[-1]) if TPol == False and Pol else np.save(os.path.join(Output_Path, "Zbar/Zbar_T_"+str(Indices[i])), Returns[-1])][0]

            if rank == 0:
                pbar.update(1)

            finishes_inp[i] = time.time()

        comm.barrier()

    del(R)
    del(Lbar)

    if rank == 0:
        print("Inpainting done!")

    comm.barrier()

    timer_global_finish = time.time()
        
    if No_Z == False:
        if Single:
            if rank == 0:
                time_Zs = finishes_Zs - starts_Zs
            else:
                time_Zs = None
        else:
            time_Zs = finishes_Zs - starts_Zs
                
    time_inp = finishes_inp - starts_inp
    
    if rank == 0:
        if (os.path.exists(Output_Path+"/Time_Stats") == False):
            os.makedirs(Output_Path+"/Time_Stats")

    comm.barrier()
    
    if NJ != 1:
        np.save(Output_Path+"/Time_Stats/Inpainting_inp_info_"+str(Counter)+"_rank_"+str(rank), time_inp)
        np.save(Output_Path+"/Time_Stats/Global_timer_"+str(Counter)+"_rank_"+str(rank), timer_global_finish-time_global_start)
        np.save(Output_Path+"/Time_Stats/L_Read_timer_"+str(Counter)+"_rank_"+str(rank), L_read_timer-read_timer_start)
        np.save(Output_Path+"/Time_Stats/R_Read_timer_"+str(Counter)+"_rank_"+str(rank), R_timer_finish-R_timer_start)
        if Single:
            if rank == 0:
                [np.save(Output_Path+"/Time_Stats/Inpainting_zs_info_"+str(Counter)+"_rank_0", time_Zs) if No_Z == False else None]
        else:
            [np.save(Output_Path+"/Time_Stats/Inpainting_zs_info_"+str(Counter)+"_rank_"+str(rank), time_Zs) if No_Z == False else None]
    else:
        np.save(Output_Path+"/Time_Stats/Inpainting_inp_info_rank_"+str(rank), time_inp)
        np.save(Output_Path+"/Time_Stats/Global_timer_rank_"+str(rank), timer_global_finish-time_global_start)
        np.save(Output_Path+"/Time_Stats/L_Read_timer_rank_"+str(rank), L_read_timer-read_timer_start)
        np.save(Output_Path+"/Time_Stats/R_Read_timer_rank_"+str(rank), R_timer_finish-R_timer_start)
        if Single:
            if rank == 0:
                [np.save(Output_Path+"/Time_Stats/Inpainting_zs_info_rank_0", time_Zs) if No_Z == False else None]
        else:
            [np.save(Output_Path+"/Time_Stats/Inpainting_zs_info_rank_"+str(rank), time_Zs) if No_Z == False else None]

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
    import h5py
    import psutil
    import argparse
    import numpy as np
    import configparser
    import healpy as hp
    from tqdm import tqdm
    from Tools import inpainting_tools

    def initial():
        """Function to read input arguments from command line and to offer help to user."""
    
        parser = argparse.ArgumentParser()
    
        parser.add_argument("--config", type=str, help='Config file path.')
        parser.add_argument("-C", type=int, help='Job ID from the stack.')
    
        args = parser.parse_args()
        
        return args

    args = initial()
    config_file = args.config
    Counter = args.C

    config = configparser.ConfigParser()
    config.read(config_file)

    Input_Path = config['SOFTWARE_PARAMS']['Inp_In_Path']
    Output_Path = config['SOFTWARE_PARAMS']['Inp_Out_Path']
    Chol_Path = config['SOFTWARE_PARAMS']['Out_Matrix']
    Name = config['SOFTWARE_PARAMS']['Name']
    Num_Sims = int(config['SOFTWARE_PARAMS']['Num_Sims'])
    Single = [False if config['SOFTWARE_PARAMS']['Single'] == "False" else True][0]
    No_Z = [False if config['SOFTWARE_PARAMS']['No_Z'] == "False" else True][0]
    Cons_Uncons = [False if config['SOFTWARE_PARAMS']['Cons_Uncons'] == "False" else True][0]
    Fields = int(config['SOFTWARE_PARAMS']['Fields'])
    Zbar_Path = [None if config['SOFTWARE_PARAMS']['Zbar_Path'] == "None" else config['SOFTWARE_PARAMS']['Zbar_Path']][0]
    Pol = [False if config['MODEL_PARAMS']['Pol'] == "False" else True][0]
    TPol = [False if config['MODEL_PARAMS']['TPol'] == "False" else True][0]
    I_Mask = [None if config['MODEL_PARAMS']['I_Mask'] == "None" else config['MODEL_PARAMS']['I_Mask']][0]
    P_Mask = [None if config['MODEL_PARAMS']['P_Mask'] == "None" else config['MODEL_PARAMS']['P_Mask']][0]
    NJ = int(config['SOFTWARE_PARAMS']['NJ_inp'])
    
    calculate_inpainting(Input_Path, Output_Path, Name, Chol_Path, I_Mask, P_Mask, Pol, TPol, Num_Sims, Single, No_Z, Cons_Uncons, Fields, Zbar_Path, NJ = NJ, Counter = Counter)
