"""

@author: Christian Gimeno Amo
Supervised by: R. Belén Barreiro & E. Martínez-Gonzáles

This program generates inpainted realizations of an input Cosmic Microwave Background (CMB) map(s). The methodology is based on
Gaussian constrained realizations.

"""

import os
import configparser

class CMB_PAInT():
    
    """ Class to compute the covariance matrix of the CMB, do cholesky decomposition and inpaint maps. Under Gaussian assumption, non-standard models
    are supported. This class support two ways to compute the covariance matrix and inpaint the maps: in local (also compatible with jupyter notebooks) or in a cluster.
    Both of them use mpi4py package to parallelize the computation and reduce the cost time. However, in a cluster we can add an extra parallelization level by
    splitting the computation in different jobs, each of them parallelized. Cholesky decomposition is managed by dask to improve the performance.
    Bash script is needed to submit a job to the cluster. An input .sh file is accepted (making compatible with different configurations), but the code can generate it
    for NERSC type cluster which use slurm job scheduling system.
    """
    
    def __init__(self, Cls, Nside, Lmax, Out_PATH, Inp_In_PATH, Inp_Out_PATH, In_Name, Num_Sims, Single, No_Z, Cons_Uncons, Fields, Zbar_PATH, I_Mask, Chunks, job_name, env = "cudaaware_chrisenv_2", P_Mask = None, Noise_Level = 0.0001, DP = False, Pol = False, TPol = False, EB = False, TB = False, local = False, sh_covariance_PATH = None, sh_chol_PATH = None, sh_inp_PATH = None, qos = "overrun", nodes_cov = 1, nodes_inp = 1, time_limit_cov = "00:10:00", time_limit_inp = "00:10:00", time_limit_chol = "00:30:00", ntasks_cov = 32, ntasks_inp = 4, cpus_per_task_cov = 5, cpus_per_task_inp = 20, email = None, NJ_cov = 1, NJ_inp = 1, constraint = "cpu", config_name = "config.ini", external_matrix = None):
        
        """ Initialise the class to compute the covariance matrix.

        Parameters
        ----------

        Cls: np.array
            Array with the Cls of the CMB. The shape of the array must be (N, Lmax+1), where N is the number of Cls 
            and Lmax is the maximum multipole.
            The Cls must be in the following order: TT, EE, BB, TE, (TB, EB, if non-standard model). 
            If Pol = False, then the Cls is just TT.
        Nside: int
            Covariance matrix is computed for Healpy type of pixelization. Nside is the parameter defining the resolution
            and is directly related with the number of pixels (12*Nside**2), and thus, the dimension of the matrix.
        Lmax: int
            Maximum multipole to consider.
        Out_PATH: str
            Path to the folder where covariance matrix (and cholesky decomposition) will be stored.
        Inp_In_PATH: str
            Path to the input maps.
        Inp_Out_PATH: str
            Path to the folder where inpainted maps will be saved.
        In_Name: str
            Name of the input maps.
        Num_Sims: int
            If single = False, number of simulations to be inpainted. If single = True, number of inpainted realizations for a single sky.
        Single: bool
            If True, only one sky will be inpainted. If False, Num_Sims skies will be inpainted.
        No_Z: bool
            If True, the code assummes that z variables are precomputed.
        Cons_Uncons: bool
            If True, constrained and unconstrained maps are also provided.
        Fields: int
            Number of fields of the input map(s). If TPol = False, it specify if input maps contains also T. If Fields = 3, then the code will read fields 1 and 2 (Q and U)
            If Fields = 2, input map only contains polarization, so the code will read fields 0 and 1.
        Zbar_PATH: str
            Path to the folder where zbar (normal random) variables are stored. This are the seeds of the inpainted maps. By default is None,
            and the code will generate and save them. This is useful as it allows the use of precomputed or presaved seeds.
        I_Mask: str
            Path to the intensity mask.
        Chunks: int
             Chunk size, number of elements per row (or column) to be included in the chunk. This is used by dask package to speed up the code.
             An optimal chunk size is needed, not too big (big chunks), not too small (too many chunks).
             Take into account that the chunk size must be a divisor of the number of pixels.
        job_name: str
            Name of the sh file.
        P_Mask: str
            Path to the polarization mask.
        Noise_Level: float
            Regularization noise level.
        DP: bool
            If True, precision = DOUBLE. If False, precision = SINGLE.
        Pol: bool
            If True, the covariance matrix will be computed for Q and U (or T+QU if TPol is True). If False, only temperature will be considered.
        TPol: bool
            If True, the covariance matrix will be computed for T+QU. If False, the covariance matrix will be computed for Q and U (or just T if Pol is False).
        EB: bool
            If True, the covariance matrix will be computed asumming a non standard model where EB is different from zero. False means EB = 0.
        TB: bool
            If True, the covariance matrix will be computed asumming a non standard model where TB is different from zero. False means TB = 0.
        sh_covariance_PATH: str
            Bash script to run the computation of the covariance matrix in a cluster. 
            If None, the code will generate it taking as reference NERSC system.
        sh_chol_PATH: str
            Bash script to run the computation of the Cholesky decomposition of the covariance matrix in a cluster.
            If None, the code will generate it taking as reference NERSC system.
        sh_inp_PATH: str
            Bash script to run the inpainting.
            If None, the code will generate it taking as reference NERSC system.
        qos: str
            QOS ("Quality of Service") parameter for the cluster. For instance, "overrun" is the free of charge option with very low priority.
        nodes_cov: int
            Number of nodes per job for the covariance matrix computation.
        nodes_inp: int
            Number of nodes per job for the inpainting step.
        time_limit_cov: str
            Time limit per job for the covariance matrix computation.
        time_limit_inp: str
            Time limit per job for the inpainting step.
        time_limit_chol: str
            Time limit per job for the cholesky decomposition.
        env: str
            Conda environment to be loaded.
        ntasks_cov: int
            Number of tasks per node for the covariance matrix computation.
        ntasks_inp: int
            Number of tasks per node for the inpainting step. Be careful with the memory, each task will read large matrices.
        cpus_per_task_cov: int 
            Number of cpus per task to use in the cluster for the covariance matrix computation. Take into account the number of CPUs per node in your cluster.
            cpus_per_task_cov times ntask_cov <= Total number of CPUs per node.
        cpus_per_task_inp: int 
            Number of cpus per task to use in the cluster for the inpainting. Take into account the number of CPUs per node in your cluster.
            cpus_per_task_inp times ntask_inp <= Total number of CPUs per node.
        email: str
            Email direction. NERSC will notify by email the status of the job (when it starts and finishs).
        constraint: str
            Constraint to be used. By default this will be "cpu".
        NJ_cov: int
            Number of jobs in which the computation of the covariance matrix will be distributed.
        NJ_inp: int
            Number of jobs in which the inpainting procedure will be distributed.
        Local: bool
            If True the code will be launched for a local system (not distributed along different jobs). If False, a cluster will be use.
        Config_Name: str
            Name of the configuration file.
        External_Matrix: str
            Path to an external covariance matrix to be added to the signal matrix.
        
        """

        self.cls = Cls
        self.nside = Nside
        self.lmax = Lmax
        self.out = Out_PATH
        self.inp_in_path = Inp_In_PATH
        self.inp_out_path = Inp_Out_PATH
        self.in_name = In_Name
        self.num_sims = Num_Sims
        self.single = Single
        self.no_z = No_Z
        self.cons_uncons = Cons_Uncons
        self.fields = Fields
        self.zbar_path = Zbar_PATH
        self.I_Mask = I_Mask
        self.P_Mask = P_Mask
        self.noise_level = Noise_Level
        self.chunks = Chunks
        self.DP = DP
        self.pol = Pol
        self.tpol = TPol
        self.eb = EB
        self.tb = TB
        self.sh_covariance = sh_covariance_PATH
        self.sh_chol = sh_chol_PATH
        self.sh_inp = sh_inp_PATH
        self.qos = qos
        self.nodes_cov = nodes_cov
        self.nodes_inp = nodes_inp
        self.time_limit_cov = time_limit_cov
        self.time_limit_inp = time_limit_inp
        self.time_limit_chol = time_limit_chol
        self.env = env
        self.ntasks_cov = ntasks_cov
        self.ntasks_inp = ntasks_inp
        self.cpus_per_task_cov = cpus_per_task_cov
        self.cpus_per_task_inp = cpus_per_task_inp
        self.email = email
        self.constraint = constraint
        self.nj_cov = NJ_cov
        self.nj_inp = NJ_inp
        self.local = local
        self.config_name = config_name
        self.job_name = job_name
        self.external_cov = external_matrix

    def generate_config_file(self):
        config = configparser.ConfigParser()
        config['MODEL_PARAMS'] = {'Cls': str(self.cls),
                            'Nside': str(self.nside),
                            'Lmax': str(self.lmax),
                            'Pol': str(self.pol),
                            'TPol': str(self.tpol),
                            'EB': str(self.eb),
                            'TB': str(self.tb),
                            'I_Mask': str(self.I_Mask),
                            'P_Mask': str(self.P_Mask),
                            'external_matrix': str(self.external_cov)}
        config['SOFTWARE_PARAMS'] = {'sh_covariance_PATH': str(self.sh_covariance),
                            'sh_chol_PATH': str(self.sh_chol),
                            'sh_inp_PATH': str(self.sh_inp),
                            'qos': str(self.qos),
                            'nodes_cov': str(self.nodes_cov),
                            'nodes_inp': str(self.nodes_inp),
                            'env': str(self.env),
                            'ntasks_cov': str(self.ntasks_cov),
                            'ntasks_inp': str(self.ntasks_inp),
                            'cpus_per_task_cov': str(self.cpus_per_task_cov),
                            'cpus_per_task_inp': str(self.cpus_per_task_inp),
                            'email': str(self.email),
                            'constraint': str(self.constraint),
                            'NJ_cov': str(self.nj_cov),
                            'NJ_inp': str(self.nj_inp),
                            'local': str(self.local),
                            'time_limit_cov': str(self.time_limit_cov),
                            'time_limit_chol': str(self.time_limit_chol),
                            'time_limit_inp': str(self.time_limit_inp),
                            'DP': str(self.DP), 
                            'Noise_Level': str(self.noise_level), 
                            'Chunks': str(self.chunks), 
                            'Inp_In_Path': str(self.inp_in_path), 
                            'Inp_Out_Path': str(self.inp_out_path), 
                            'Out_Matrix': str(self.out),
                            'Name': str(self.in_name),
                            'Num_Sims': str(self.num_sims),
                            'Single': str(self.single),
                            'No_Z': str(self.no_z),
                            'Cons_Uncons': str(self.cons_uncons),
                            'Fields': str(self.fields),
                            'Zbar_Path': str(self.zbar_path),
                            'config_name': str(self.config_name), 
                            'job_name': str(self.job_name)}

        if (os.path.exists(self.out) == False):
            os.makedirs(self.out)

        with open(self.out+"/"+str(self.config_name), "w") as configfile:
            config.write(configfile)

    def calculate_covariance_matrix(self, global_path = None):
        import subprocess
        self.generate_config_file()
        if global_path is not None:
            cmd = ["mpiexec", "-n", str(self.ntasks_cov), "python3", global_path+"/scripts/local_covariance_matrix.py", "--config", str(self.out)+"/"+str(self.config_name)]
        else:
            cmd = ["mpiexec", "-n", str(self.ntasks_cov), "python3", "./scripts/local_covariance_matrix.py", "--config", str(self.out)+"/"+str(self.config_name)]
        subprocess.call(cmd)

    def calculate_chol(self, global_path = None):
        import subprocess
        self.generate_config_file()
        if global_path is not None:
            cmd = ["python3", global_path+"/scripts/cholesky.py", "--config", str(self.out)+"/"+str(self.config_name)]
        else:
            cmd = ["python3", "./scripts/cholesky.py", "--config", str(self.out)+"/"+str(self.config_name)]
        subprocess.call(cmd)

    def calculate_inp(self, global_path = None):
        import subprocess
        self.generate_config_file()
        if global_path is not None:
            cmd = ["mpiexec", "-n", str(self.ntasks_inp), "python3", global_path+"/scripts/local_inpainting.py", "--config", str(self.out)+"/"+str(self.config_name)]
        else:
            cmd = ["mpiexec", "-n", str(self.ntasks_inp), "python3", "./scripts/local_inpainting.py", "--config", str(self.out)+"/"+str(self.config_name)]
        subprocess.call(cmd)
            
    def generate_cluster_script_covariance(self):
        script = "#!/bin/bash\n"
        script += "#SBATCH --qos=" + str(self.qos) + "\n"
        script += "#SBATCH --nodes=" + str(self.nodes_cov) + "\n"
        script += "#SBATCH --time=" + str(self.time_limit_cov) + "\n"
        script += "#SBATCH --cpus-per-task=" + str(self.cpus_per_task_cov) + "\n"
        script += "#SBATCH --mail-type=ALL\n"
        script += "#SBATCH --mail-user=" + str(self.email) + "\n"
        script += "#SBATCH --constraint=" + str(self.constraint) + "\n"

        script += "# run the application: \n"
        script += "module load python \n"
        script += "module load fast-mkl-amd \n"

        script += "conda activate " + self.env + "\n"
        script += "srun -n " + str(self.ntasks_cov) + " -c " + str(self.cpus_per_task_cov) + " python3 "+os.getcwd()+"/scripts/cluster_covariance_matrix.py --config " + str(self.out)+"/"+str(self.config_name) + " -C " + str(0) + "\n"
        
        script += "conda deactivate"

        with open(os.getcwd()+"/"+str(self.job_name)+".sh", "w") as f:
            f.write(script)

    def generate_multiple_cluster_script_covariance(self):
        script = "#!/bin/bash\n"

        script += "#SBATCH --qos=" + str(self.qos) + "\n"
        script += "#SBATCH --constraint=" + str(self.constraint) + "\n"

        script += "for i in {0.." +str(self.nj_cov-1) + "}; \n"
        script += "\tdo sed -i \"s/-C .*$/-C ${i}/\" " + str(os.getcwd()) + "/"+str(self.job_name)+".sh; \n"
        script += "\tmore " + str(os.getcwd()) + "/"+str(self.job_name)+".sh; \n"
        script += "\tsbatch " + str(os.getcwd()) + "/"+str(self.job_name)+".sh; \n"

        script += "done"

        with open(os.getcwd()+"/"+str(self.job_name)+"_2.sh", "w") as f:
            f.write(script)

    def generate_cluster_script_inpainting(self):
        script = "#!/bin/bash\n"
        script += "#SBATCH --qos=" + str(self.qos) + "\n"
        script += "#SBATCH --nodes=" + str(self.nodes_inp) + "\n"
        script += "#SBATCH --time=" + str(self.time_limit_inp) + "\n"
        script += "#SBATCH --cpus-per-task=" + str(self.cpus_per_task_inp) + "\n"
        script += "#SBATCH --mail-type=ALL\n"
        script += "#SBATCH --mail-user=" + str(self.email) + "\n"
        script += "#SBATCH --constraint=" + str(self.constraint) + "\n"

        script += "# run the application: \n"
        script += "module load python \n"
        script += "module load fast-mkl-amd \n"

        script += "conda activate " + self.env + "\n"
        script += "srun -n " + str(self.ntasks_inp) + " -c " + str(self.cpus_per_task_inp) + " python3 "+os.getcwd()+"/scripts/cluster_inpainting.py --config " + str(self.out)+"/"+str(self.config_name) + " -C " + str(0) + "\n"
        
        script += "conda deactivate"

        with open(os.getcwd()+"/"+str(self.job_name)+".sh", "w") as f:
            f.write(script)

    def generate_multiple_cluster_script_inpainting(self):
        script = "#!/bin/bash\n"

        script += "#SBATCH --qos=" + str(self.qos) + "\n"
        script += "#SBATCH --constraint=" + str(self.constraint) + "\n"

        script += "for i in {0.." +str(self.nj_inp-1) + "}; \n"
        script += "\tdo sed -i \"s/-C .*$/-C ${i}/\" " + str(os.getcwd()) + "/"+str(self.job_name)+".sh; \n"
        script += "\tmore " + str(os.getcwd()) + "/"+str(self.job_name)+".sh; \n"
        script += "\tsbatch " + str(os.getcwd()) + "/"+str(self.job_name)+".sh; \n"

        script += "done"

        with open(os.getcwd()+"/"+str(self.job_name)+"_2.sh", "w") as f:
            f.write(script)

    def generate_cluster_script_cholesky(self):
        script = "#!/bin/bash\n"
        script += "#SBATCH --qos=" + str(self.qos) + "\n"
        script += "#SBATCH --nodes=1\n"
        script += "#SBATCH --time=" + str(self.time_limit_chol) + "\n"
        script += "#SBATCH --cpus-per-task=64\n"
        script += "#SBATCH --mail-type=ALL\n"
        script += "#SBATCH --mail-user=" + str(self.email) + "\n"
        script += "#SBATCH --constraint=" + str(self.constraint) + "\n"

        script += "# run the application: \n"
        script += "module load python \n"
        script += "module load fast-mkl-amd \n"

        script += "conda activate " + self.env + "\n"
        script += "python3 "+os.getcwd()+"/scripts/cholesky.py --config " + str(self.out)+"/"+str(self.config_name) + "\n"
        
        script += "conda deactivate"

        with open(os.getcwd()+"/"+str(self.job_name)+".sh", "w") as f:
            f.write(script)

    def run_cluster_script_covariance(self):
        if self.sh_covariance is not None:
            import subprocess
            cmd = ["sbatch", self.sh_covariance]
            subprocess.call(cmd)
        else:
            if self.nj_cov == 1:
                import subprocess
                cmd = ["sbatch", str(self.job_name)+".sh"]
                subprocess.call(cmd)
            else:
                import subprocess
                cmd = ["sbatch", str(self.job_name)+"_2.sh"]
                subprocess.call(cmd)

    def run_cluster_script_cholesky(self):
        if self.sh_chol is not None:
            import subprocess
            cmd = ["sbatch", self.sh_chol]
            subprocess.call(cmd)
        else:
            import subprocess
            cmd = ["sbatch", str(self.job_name)+".sh"]
            subprocess.call(cmd)

    def run_cluster_script_inpainting(self):
        if self.sh_inp is not None:
            import subprocess
            cmd = ["sbatch", self.sh_inp]
            subprocess.call(cmd)
        else:
            if self.nj_inp == 1:
                import subprocess
                cmd = ["sbatch", str(self.job_name)+".sh"]
                subprocess.call(cmd)
            else:
                import subprocess
                cmd = ["sbatch", str(self.job_name)+"_2.sh"]
                subprocess.call(cmd)

    def calculate_covariance(self, global_path = None):
        print("Calculating covariance matrix...")
        print("Local:", self.local)
        if self.local:
            self.calculate_covariance_matrix(global_path)
        else:
            if self.sh_covariance is not None:
                print("Using provided bash script...")
                self.run_cluster_script_covariance()
            else:
                print("Generating bash script...")
                self.generate_cluster_script_covariance()

                if self.nj_cov != 1:
                    print("Generating multiple bash scripts...")
                    self.generate_multiple_cluster_script_covariance()
                print("Submitting job...")
                self.run_cluster_script_covariance()
    
    def calculate_cholesky(self, global_path = None):
        print("Calculating cholesky decomposition...")
        print("Local:", self.local())
        if self.local:
            self.calculate_chol(global_path)
        else:
            if self.sh_chol is not None:
                print("Using provided bash script...")
                self.run_cluster_script_cholesky()
            else:
                print("Generating bash script...")
                self.generate_cluster_script_cholesky()
                print("Submitting job...")
                self.run_cluster_script_cholesky()

    def calculate_inpainting(self, global_path = None):
        print("Calculating inpainting...")
        print("Local:", self.local)
        if self.local:
            self.calculate_inp(global_path)
        else:
            if self.sh_inp is not None:
                print("Using provided bash script...")
                self.run_cluster_script_inpainting()
            else:
                print("Generating bash script...")
                self.generate_cluster_script_inpainting()

                if self.nj_inp != 1:
                    print("Generating multiple bash scripts...")
                    self.generate_multiple_cluster_script_inpainting()

                print("Submitting job...")
                self.run_cluster_script_inpainting()

if __name__ == "__main__":
    import os
    import time
    import scipy
    import argparse
    import subprocess
    import numpy as np
    import configparser
    import healpy as hp
    from tqdm import tqdm

    def initial():
        """Function to read input arguments from command line and to offer help to user."""
    
        parser = argparse.ArgumentParser(
            description="""Given a fiducial model, this program generates inpainted realizations of an input Cosmic Microwave Background (CMB)
            map(s) based on Gaussian constrained realization methodology.""",
            prog="CMB_PAInT.py",
            epilog="Contact: gimenoc@ifca.unican.es")
    
        # Mandatory parameters

        parser.add_argument("config", type=str, default = str(os.getcwd())+"/docs/config.ini", help='Path to the configuration file.')
        parser.add_argument("level", type=int, default = 0, help="""Select action: \
                            select 0 to compute the pixel covariance matrix, select 1 for cholesky decomposition, or 2 for inpainting step.""")
    
        args = parser.parse_args()
        
        return args

    args = initial()
    config_file = args.config
    level = args.level

    config = configparser.ConfigParser()
    config.read(config_file)

    cls = config['MODEL_PARAMS']['Cls']
    nside = int(config['MODEL_PARAMS']['Nside'])
    lmax = int(config['MODEL_PARAMS']['Lmax'])
    out_path = config['SOFTWARE_PARAMS']['Out_Matrix']
    pol = [False if config['MODEL_PARAMS']['Pol'] == "False" else True][0]
    tpol = [False if config['MODEL_PARAMS']['TPol'] == "False" else True][0]
    eb = [False if config['MODEL_PARAMS']['EB'] == "False" else True][0]
    tb = [False if config['MODEL_PARAMS']['TB'] == "False" else True][0]
    int_mask = [None if config['MODEL_PARAMS']['I_Mask'] == "None" else config['MODEL_PARAMS']['I_Mask']][0]
    pol_mask = [None if config['MODEL_PARAMS']['P_Mask'] == "None" else config['MODEL_PARAMS']['P_Mask']][0]
    external_matrix = [None if config['MODEL_PARAMS']['Ext_Cov'] == "None" else config['MODEL_PARAMS']['Ext_Cov']][0]
    dp = [False if config['SOFTWARE_PARAMS']['DP'] == "False" else True][0]
    noise_level = float(config['SOFTWARE_PARAMS']['Noise_Level'])
    chunks = int(config['SOFTWARE_PARAMS']['Chunks'])
    sh_covariance_path = [None if config['SOFTWARE_PARAMS']['sh_covariance_PATH'] == "None" else config['SOFTWARE_PARAMS']['sh_covariance_PATH']][0]
    sh_chol_path = [None if config['SOFTWARE_PARAMS']['sh_chol_PATH'] == "None" else config['SOFTWARE_PARAMS']['sh_chol_PATH']][0]
    sh_inp_path = [None if config['SOFTWARE_PARAMS']['sh_inp_PATH'] == "None" else config['SOFTWARE_PARAMS']['sh_inp_PATH']][0]
    qos = config['SOFTWARE_PARAMS']['qos']
    nodes_cov = int(config['SOFTWARE_PARAMS']['nodes_cov'])
    nodes_inp = int(config['SOFTWARE_PARAMS']['nodes_inp'])
    time_limit_cov = config['SOFTWARE_PARAMS']['time_limit_cov']
    time_limit_inp = config['SOFTWARE_PARAMS']['time_limit_inp']
    time_limit_chol = config['SOFTWARE_PARAMS']['time_limit_chol']
    env = config['SOFTWARE_PARAMS']['env']
    ntasks_cov = int(config['SOFTWARE_PARAMS']['ntasks_cov'])
    ntasks_inp = int(config['SOFTWARE_PARAMS']['ntasks_inp'])
    cpus_per_task_cov = int(config['SOFTWARE_PARAMS']['cpus_per_task_cov'])
    cpus_per_task_inp = int(config['SOFTWARE_PARAMS']['cpus_per_task_inp'])
    email = config['SOFTWARE_PARAMS']['email']
    constraint = config['SOFTWARE_PARAMS']['constraint']
    nj_cov = int(config['SOFTWARE_PARAMS']['NJ_cov'])
    nj_inp = int(config['SOFTWARE_PARAMS']['NJ_inp'])
    local = [False if config['SOFTWARE_PARAMS']['local'] == "False" else True][0]
    inp_in_path = config['SOFTWARE_PARAMS']['Inp_In_Path']
    inp_out_path = config['SOFTWARE_PARAMS']['Inp_Out_Path']
    name = config['SOFTWARE_PARAMS']['Name']
    num_sims = int(config['SOFTWARE_PARAMS']['Num_Sims'])
    single = [False if config['SOFTWARE_PARAMS']['Single'] == "False" else True][0]
    no_z = [False if config['SOFTWARE_PARAMS']['No_Z'] == "False" else True][0]
    cons_uncons = [False if config['SOFTWARE_PARAMS']['Cons_Uncons'] == "False" else True][0]
    fields = config['SOFTWARE_PARAMS']['Fields']
    zbar_path = [None if config['SOFTWARE_PARAMS']['Zbar_Path'] == "None" else config['SOFTWARE_PARAMS']['Zbar_Path']][0]
    config_name = config['SOFTWARE_PARAMS']['config_name']
    job_name = config['SOFTWARE_PARAMS']['job_name']

    PyGCR = CMB_Paint(Cls = cls, Nside = nside, Lmax = lmax, Out_PATH = out_path, Inp_In_PATH = inp_in_path, Inp_Out_PATH = inp_out_path, In_Name = name, \
                      Num_Sims = num_sims, Single = single, No_Z = no_z, Cons_Uncons = cons_uncons, Fields = fields, Zbar_PATH = zbar_path, I_Mask = int_mask, \
                        P_Mask = pol_mask, Chunks = chunks, Noise_Level = noise_level, DP = dp, local = local, Pol = pol, TPol = tpol, EB = eb, TB = tb, \
                            sh_covariance_PATH = sh_covariance_path, sh_chol_PATH = sh_chol_path, sh_inp_PATH = sh_inp_path, qos = qos, nodes_cov = nodes_cov, \
                                nodes_inp = nodes_inp, time_limit_cov = time_limit_cov, time_limit_chol = time_limit_chol, job_name = job_name, time_limit_inp = time_limit_inp, \
                                     env = env, cpus_per_task_cov = cpus_per_task_cov, cpus_per_task_inp = cpus_per_task_inp, email = email, NJ_cov = nj_cov, NJ_inp = nj_inp, constraint = constraint,
                                        ntasks_cov = ntasks_cov, ntasks_inp = ntasks_inp, config_name = config_name, external_matrix = external_matrix)
    
    if (os.path.exists(out_path) == False):
        os.makedirs(out_path)
    
    if level == 0:
        PyGCR.calculate_covariance()
    elif level == 1:
        PyGCR.calculate_cholesky()
    elif level == 2:
        PyGCR.calculate_inpainting()
