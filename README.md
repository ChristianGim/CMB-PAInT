# **C**osmic **M**icrowave **B**ackground **P**olarization **A**nisotropies **In**painting **T**ool (CMB-PAInT)

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

  
