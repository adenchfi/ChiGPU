# ChiGPU
A GPU-accelerated set of programs to diagonalize tight-binding Hamiltonians and calculate the real part of the charge susceptibility (Lindhard susceptibility).

# Compilation
Run the make.sh script in /src.

# Requirements
nvcc compiler (CUDA Toolkit). Confirmed to work for CUDA Toolkit 11.0, probably back-compatible to CUDA Toolkit 10.0.


# Typical usage
We expect the user to use Wannier90 to generate the tight-binding moddel, with flag write_hr=.true. or write_tb=.true. Some lines in the files need to be deleted; the util/wannier_filter_hr.py script will do that. 

Refer to example_run.sh for usage for running the code.
