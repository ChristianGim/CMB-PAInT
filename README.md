# **C**osmic **M**icrowave **B**ackground **P**olarization **A**nisotropies **In**painting **T**ool (CMB-PAInT)

A Python package to inpaint Cosmic Microwave Background maps. It works for both intensity and polarization Q and U maps. 
The code is based on Gaussian Constrained Realizations methodology (see [paper](https://arxiv.org/abs/2405.06820)).
CMBPAInT can be used in different ways:

* To compute pixel covariance matrix from an input angular power spectrum.
* To compute Cholesky decomposition from an input covariance matrix.
* To inpaint a CMB map using the precomputed Cholesky decomposition.

The code contains two parallelization levels. The first one can be use in a cluster. It splits the work among different jobs that are submitted at the same time. Then, each of these jobs are parallelized using [mpi4py](https://mpi4py.readthedocs.io/en/stable/), the MPI standard for Python.

If you have any comments or suggestions, please feel free to contact by email (gimenoc@ifca.unican.es) or by opening a discussion thread or issue.

# Input parameters

This code needs an ini file containing the following parameters:



``` 
[MODEL_PARAMS]
cls = /gpfs/projects/astro/gimeno/Inpainting_Test/codes/docs/Cls_PTEP_lmax_192.npy
nside = 64
lmax = 192
pol = True
tpol = True
eb = False
tb = False
i_mask = /gpfs/projects/astro/gimeno/Inpainting_Test/codes/docs/dx12_v3_common_ps_mask_int_160a_0064_v2.fits
p_mask = /gpfs/projects/astro/gimeno/Inpainting_Test/codes/docs/dx12_v3_common_ps_mask_pol_160a_0064_v2.fits
Ext_Cov = None

[SOFTWARE_PARAMS]
sh_covariance_path = /gpfs/projects/astro/gimeno/Inpainting_Test/sh/run_covariance_2.sh
sh_chol_path = /gpfs/projects/astro/gimeno/Inpainting_Test/sh/run_cholesky.sh
sh_inp_path = None
qos = overrun
nodes_cov = 4
nodes_inp = 5
env = cudaaware_chrisenv_2
ntasks_cov = 200
ntasks_inp = 5
cpus_per_task = 2
email = cga119@alumnos.unican.es
constrain = cpu
nj_cov = 16
nj_inp = 64
local = False
time_limit_cov = 00:20:00
time_limit_chol = 00:45:00
time_limit_inp = 00:15:00
dp = True
noise_level = 1e-05
chunks = 12288
inp_in_path = /pscratch/sd/c/chris98/Proyectos/PyGCRecon/Nside_64/Input_Map
inp_out_path = /pscratch/sd/c/chris98/Proyectos/PyGCRecon/Nside_64/Inpainting_Just_CMB
out_matrix = /gpfs/projects/astro/gimeno/Inpainting_Test/Test_ns_64/Cov_Mat/
name = None
num_sims = 1200
single = True
no_z = False
cons_uncons = True
fields = 3
zbar_path = None
config_name = config_test.ini
job_name = test_CMB_PAInT
```



# Installation and dependencies

This package can be installed with:

``` 
pip install CMBPAInT
```

You can also copy the repository, where a setup_environment.sh file will setup conda environment and install all the dependencies.

  1. Starting the terminal
  2. Clone the repository:
  ``` 
     https://github.com/ChristianGim/CMB-PAInT
  ```
  3. Run setup_environment.sh:
  ```  
     chmod +x setup_environment.sh
     ./setup_environment.sh
  ```
  4. Load conda environment and install CMBPAInT:
  ```
     conda activate CMBPAInT
     pip install CMBPAInT
  ```

This code has been tested in Python 3.8 and Python 3.11. 

The dependencies are:

* [numpy](https://numpy.org/): We tested for version 1.21.5 (in local) and 1.25.4 (in a cluster).
* [healpy](https://healpy.readthedocs.io/en/latest/): We tested for version 1.15.2 (in local) and 1.16.6 (in a cluster).
* [tqdm](https://github.com/tqdm/tqdm)
* [dask](https://www.dask.org/)
* [psutil](https://psutil.readthedocs.io/en/latest/)
* [h5py](https://www.h5py.org/): We tested for version 3.6.0 (in local) and 3.11.0 (in a cluster).
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/): We tested for version 3.1.3 (in local) and 3.1.5 (in a cluster).



  
