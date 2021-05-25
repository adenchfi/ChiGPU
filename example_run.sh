### The below is for laptops with both integrated and discrete GPUs. Remove the first two commands otherwise.

############# INPUT FILE FORMAT ########################

# You must have a tight-binding data file on hand, with the following format:
# {Rx} {Ry} {Rz} {orb1} {orb2} {Real part of H} {Imag part of H}

# {Rx, Ry, Rz} are the unit cell numbers in x,y,z directions
# {orb1, orb2} are the tight-binding orbitals used, output by wannier90 or made by hand. They are just numbered 1-{norb}.

############ EXAMPLE USAGE ########################

## an example is in TB_input/Nb_tb.dat which has a d-orbital picture of Nb. This almost but not quite reproduces the bandstructure of Nb; s-orbitals also need to be included when using Wannier90 to account for Nb's s-d hybridization,

# First create a kmesh; either using meshes/write_3Dkmesh.py or your own path. 
   # example IS IN meshes/ folder

# Then run the tight-binding diagonalization calculation.
# ./{program} {TBfile.dat} {GPU_batchsize - 20,000-200,000 typically good values} {kmesh.dat} {outputfile} {num_orbs for TB model}
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./GPU/bandstruct_TB_batch_gpu_complex ./data/TB_hams/Nb_tb.dat 25000 meshes/kmesh50x
50x50.dat TB_output/Nb_TB_solved_50x50x50.dat 10

# Here we compute the susceptibility from the tight-binding calculation. 
# {./program} {TB_output.dat} {susc_output.dat} {frequency response omega} {exponential cutoff parameter - 10.0 good as default} {delta broadening} {beta} {Fermi energy}
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./GPU/susc_calc_GPU  TB_output/SmTe3_TB_solved_100x100x100.dat susc_output/DyTe3_sus
c_beta50_100x100x100.dat 0.00 10.0 0.0005 50.0 18.6928
