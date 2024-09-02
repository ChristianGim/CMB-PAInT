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



This is an example:

``` 
[MODEL_PARAMS]
cls = /path/to/cls
nside = 64
lmax = 192
pol = True
tpol = True
eb = False
tb = False
i_mask = /path/to/intensity/mask
p_mask = /path/to/polarization/mask
Ext_Cov = None

[SOFTWARE_PARAMS]
sh_covariance_path = /path/to/external/sh/file/for/covariance_matrix
sh_chol_path = /path/to/external/sh/file/for/cholesky_decomposition
sh_inp_path = /path/to/external/sh/file/for/inpainting
qos = qos
nodes_cov = 4
nodes_inp = 5
env = your_conda_environment
ntasks_cov = 32
ntasks_inp = 5
cpus_per_task = 5gpfs/projects/astro/gimeno/Inpainting_Test/Test_ns_64/Cov_Mat/
email = your_email
constrain = constrain
nj_cov = 32
nj_inp = 64
local = False
time_limit_cov = 00:20:00
time_limit_chol = 00:45:00
time_limit_inp = 00:15:00
dp = True
noise_level = 1e-05
chunks = 12288
inp_in_path = /path/to/input/maps
inp_out_path = /path/where/output/inpainted/simulations/are/stored
out_matrix = /path/where/output/covariance/matrix/is/stored
name = name_input_maps
num_sims = 1000
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

# Usage

# Ouput examples (Jupyter Notebook)

[View the Jupyter Notebook]()


# License

This project is licensed under the MIT License. Feel free to use and modify the code according to the terms specified in the license.

# Citation



# Contact

If you have any questions please contact gimenoc@ifca.unican.es.

We hope this code to be useful.
  
