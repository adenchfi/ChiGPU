# this routine filters the TB Hamiltonian that Wannier90 outputs, to keep only the elements above an energy cutoff that we can provide, and writes a new file with the filtered values

### we will filter the data, and also count the number of different Delta R (Wannier R) vectors we have, which tells us how many 'sites' we have. But for DMFT purposes we want the individual atoms to be different sites as well as the different unit cells, right?
# or do we want to treat each Delta R as a single site, with all the atomic orbitals that contribute being considered different orbitals of the same 'atom' (unit cell)? 

import numpy as np

tbfilter = open('tb_filtered.dat', 'w')  # this is the file we'll save the TB matrix into after filtering it 

tb = open('Nb_hr.dat', 'r')

while True:
    line = tb.readline()
    # we know the only relevant lines in this file are the ones with 
    words = line.split()
    if len(words) == 7 and float(words[5]) > 0.001:
        tbfilter.write(line)

    if not line:
        break

tbfilter.close()
tb.close()
