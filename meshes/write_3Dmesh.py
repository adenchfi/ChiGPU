# this script writes both an NxMxN kmesh for my TB_batch code to read, and also saves a list of qnums corresponding to a desired path in k-space so that susc_calc.cpp can be run only on a desired 1D curve through k-space
import sys
import numpy as np

nkx = int(sys.argv[1])
nky = int(sys.argv[2])
nkz = int(sys.argv[3])
numk = nkx*nky*nkz

file1 = open(sys.argv[4], "w")
file1.write("KPOINTS crystal\n")
file1.write("{}\n".format(numk))

kxs = np.linspace(0.0, 1.0, nkx)
kys = np.linspace(0.0, 1.0, nky)
kzs = np.linspace(0.0, 1.0, nkz)

for a in range(nkx):
    kx = np.round(kxs[a], 7)
    for b in range(nky):
        ky = np.round(kys[b],7)
        for c in range(nkz):
            kz = np.round(kzs[c], 7)
            file1.write("{}  {}  {}\n".format(kx, ky, kz))

