# ChiGPU
A GPU-accelerated set of programs to diagonalize tight-binding Hamiltonians up to 32 orbitals. The tight-binding input file is modeled after the tight-binding output of Wannier90. They calculate the real part of the charge susceptibility (Lindhard susceptibility) perturbatively.

# Requirements
nvcc compiler (CUDA Toolkit). Confirmed to work for CUDA Toolkit 11.0, probably back-compatible to CUDA Toolkit 10.0.

# Compilation
Run the make.sh script in /src after changing the include directory. 

# Typical usage
An example tight-binding model, Nb_tb.dat, is provided for the purposes of testing the code. Any tight-binding model can be used, however, that is up to 32 states/orbitals (due to GPU batch diagonalization limits). 

Refer to example_run.sh for usage for running the code.

We expect the user to use Wannier90 to generate the tight-binding moddel, with flag write_hr=.true. or write_tb=.true. Some lines in the files need to be deleted; the util/wannier_filter_hr.py script will do that for the write_hr output file.

