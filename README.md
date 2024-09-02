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

* cls: Array with the fiducial Cls of the CMB used for the covariance matrix computation. The shape of the array must be (N, $\ell_{\mathrm{max}}$+1), where N is the number of Cls and $\ell_{\mathrm{max}}$ is the maximum multipole. The Cls must be in the following order: TT, EE, BB (optionally TE, TB, EB). If pol = False, then just TT is needed. If pol = True, but tpol = False, then only EE and BB will be used. If tpol = True, TE is also needed.
* $N_{side}$: Resolution.
* $\ell_{\mathrm{max}}$: Maximum multipole.
* pol: If True, the covariance matrix will be computed for Q and U (or T+QU if TPol is True). If False, only temperature will be considered.
* tpol: If True, the covariance matrix will be computed for T+QU. If False, the covariance matrix will be computed for Q and U (or just T if Pol is False).
* eb: If True, the covariance matrix will be computed asumming a non standard model where EB is different from zero. False means EB = 0.
* tb: If True, the covariance matrix will be computed asumming a non standard model where TB is different from zero. False means TB = 0.
* i_mask: Path to the intensity mask.
* p_mask: Path to the polarization mask.
* Ext_Cov: Path to an external covariance matrix which contains noise and systematic properties. It is added tothe signal covariance matrix. By default is None.
* sh_covariance_path: Bash script to run the computation of the covariance matrix in a cluster. If None, the code will generate it taking as reference NERSC system.
* sh_chol_path: Bash script to run the computation of the Cholesky decomposition of the covariance matrix in a cluster. If None, the code will generate it taking as reference NERSC system.
* sh_inp_path: Bash script to run the inpainting in a cluster. If None, the code will generate it taking as reference NERSC system.
* nj_cov: Number of jobs in which the computation of the covariance matrix will be distributed.
* nj_inp: Number of jobs in which the inpainting procedure will be distributed.
* local: If True the code will be launched for a local system (not distributed along different jobs). If False, a cluster will be use.
* dp: If True, precision = DOUBLE. If False, precision = SINGLE.
* noise_level: Regularization noise level.
* chunks: Chunk size, number of elements per row (or column) to be included in the chunk. This is used by dask package to speed up the code. An optimal chunk size is needed,
  not too big (big chunks), not too small (too many chunks). Take into account that the chunk size must be a divisor of the number of pixels.
* inp_in_path: Path to the input maps.
* inp_out_path: Path to the folder where inpainted maps will be saved.
* out_matrix: Path to the folder where covariance matrix (and cholesky decomposition) will be stored.
* name: Name of the input maps. If more than one, the code will concatenate with "_"+str(i)+".fits", where i runs from 0 to num_sims.
* num_sims: If single = False, number of simulations to be inpainted. If single = True, number of inpainted realizations for a single sky.
* single: If True, only one sky will be inpainted. If False, Num_Sims skies will be inpainted.
* no_z: If True, the code assummes that z variables are precomputed. 
* cons_uncons: If True, constrained and unconstrained maps are also provided.
* fields: Number of fields of the input map(s). If TPol = False, it specify if input maps contains also T. If Fields = 3, then the code will read fields 1 and 2 (Q and U)
  If Fields = 2, input map only contains polarization, so the code will read fields 0 and 1.
* zbar_path: Path to the folder where zbar (normal random) variables are stored. This are the seeds of the inpainted maps. By default is None, and the code will generate and save them.
  This is useful as it allows the use of precomputed or presaved seeds.
* config_name: Name of the configuration file.
* job_name: If the code generate by itself the bash script for NERSC system, this is the name of the sh file. 
  
The following parameters are just useful if you are running on NERSC and you are not providing a sh file. The code will generate by itself with the following specifications:

* qos: QOS ("Quality of Service") parameter for the cluster. For instance, "overrun" is the free of charge option with very low priority. 
* nodes_cov: Number of nodes per job for the covariance matrix computation.
* nodes_inp: Number of nodes per job for the inpainting step.
* env: Conda environment to be loaded.
* ntasks_cov: Number of tasks per node for the covariance matrix computation. 
* ntasks_inp: Number of tasks per node for the inpainting step. Be careful with the memory, each task will read large matrices. 
* cpus_per_task_cov: Number of cpus per task to use in the cluster for the covariance matrix computation. Take into account the number of CPUs per node in your cluster.
  cpus_per_task_cov times ntask_cov <= Total number of CPUs per node.
* cpus_per_task_inp: Number of cpus per task to use in the cluster for the inpainting. Take into account the number of CPUs per node in your cluster.
  cpus_per_task_inp times ntask_inp <= Total number of CPUs per node.
* email: Email direction. NERSC will notify by email the status of the job (when it starts and finishs).
* constraint: Constraint to be used. By default this will be "cpu".
* time_limit_cov: Time limit per job for the covariance matrix computation.
* time_limit_chol: Time limit per job for the cholesky decomposition.
* time_limit_inp: Time limit per job for the inpainting step.

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
cpus_per_task_cov = 5
cpus_per_task_inp = 20
email = your_email
constraint = cpu
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

# Output

The inpainted maps are saved to a fits file. Additionally, covariance matrix will be stored, divided along N (the number of jobs) numpy files, and Cholesky decomposition in a hdf5 file. 
Extra numpy files will be also saved containing the z variables (for the constrained part), and zbar variables (for the unconstrained seeds). 

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

This code can be used in two independent ways: 

* Via terminal.

A brief description can be obtained trough:

```
usage: CMB_PAInT.py [-h] config level

Given a fiducial model, this program generates inpainted realizations of an
input Cosmic Microwave Background (CMB) map(s) based on Gaussian constrained
realization methodology.

positional arguments:
  config      Path to the configuration file.
  level       Select action: select 0 to compute the pixel covariance matrix,
              select 1 for cholesky decomposition, or 2 for inpainting step.

optional arguments:
  -h, --help  show this help message and exit

Contact: gimenoc@ifca.unican.es
```

Usage examples:

* Example for covariance matrix computation:

```
python3 CMB_PAInT.py ./test/config_files/config_test.ini 0
```

* Example for cholesky decomposition:

```
python3 CMB_PAInT.py ./test/config_files/config_test.ini 1
```

* Example for inpainting step:

```
python3 CMB_PAInT.py ./test/config_files/config_test.ini 2
```

* As a python class (very low resolution scenarios).

Exceptionally, for very low resolution cases ($N_{\mathrm{side}}$ = 8, 16, 32), any program can import the **CMB_PAInT** class from CMB_PAInT.py.

Once the package is imported, the code can follow the 3 steps: compute covariance matrix, do cholesky, and inpaint the input map (see example jupyter notebook).

# Ouput examples (Jupyter Notebook)

[Exmaple of inpainted low resolution map]()

# License

This project is licensed under the MIT License. Feel free to use and modify the code according to the terms specified in the license.

# Citation

If you use the CMB-PAInT code, please cite its release paper [**CMB-PAInT: An inpainting tool for the cosmic microwave background**](https://arxiv.org/abs/2405.06820) as

```
Gimeno-Amo, C., Martínez-González, E., & Barreiro, R.B., 2024, arXiv eprints, arXiv:2405.06820. https://arxiv.org/abs/2405.06820
```

The corresponding bibtex is:

```
@ARTICLE{2024arXiv240506820G,
       author = {{Gimeno-Amo}, C. and {Mart{\'\i}nez-Gonz{\'a}lez}, E. and {Barreiro}, R.~B.},
        title = "{CMB-PAInT: An inpainting tool for the cosmic microwave background}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = may,
          eid = {arXiv:2405.06820},
        pages = {arXiv:2405.06820},
          doi = {10.48550/arXiv.2405.06820},
archivePrefix = {arXiv},
       eprint = {2405.06820},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240506820G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

# Contact

If you have any questions please contact gimenoc@ifca.unican.es.

We hope this code to be useful.
  
