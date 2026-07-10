#!/usr/bin/env python
# cis-rcis.py
# Restricted Configuration Interaction Singles
# Author: Ziqiu Wang

''' Need PySCF first,
MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 python cis-rcis.py'''

import numpy
import scipy.linalg
from pyscf import gto, scf, ao2mo

############### HF calc in PySCF ##############
# Supposing closed shell configuration
mol = gto.M(atom=
            '''
            O 0. 0. 0.
            H 0.86681633521 0.67723114352 0.
            H -0.86681633521 0.67723114352 0.
            '''
            , basis='sto-3g')
mf = scf.RHF(mol)
mf.kernel()

# Setting variables
nao = len(mol.ao_labels())
noccmo = mol.nelectron // 2
nvirtmo = nao - noccmo

# CIS procedure
# 1. Getting Fock Matrix from linear combination of Gaussian orbitals
#    to Molecular Orbitals
mo_eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(nao, nao, nao, nao)

# 2. Build Hamiltonian Matrix
H_size = noccmo * nvirtmo
H_singlets = numpy.zeros((H_size, H_size))
H_triplets = numpy.zeros((H_size, H_size))

ridx, cidx = 0, 0
# i, j, a, b are spatial molecular orbitals
# i, j range from (0, number of occupied mo)
# a, b range from (0, number of virtual mo)
# so the index of virt mo is (a + noccmo)
for i in range(noccmo):
    for a in range(nvirtmo):
        cidx = 0
        for j in range(noccmo):
            for b in range(nvirtmo):
                if i == j and a == b:
                    e_gap = mf.mo_energy[a + noccmo] - mf.mo_energy[i]
                    H_singlets[ridx][cidx] += e_gap
                    H_triplets[ridx][cidx] += e_gap
                H_singlets[ridx][cidx] \
                    += 2 * mo_eri[j][b + noccmo][i][a + noccmo] \
                    - mo_eri[j][i][a + noccmo][b + noccmo]
                H_triplets[ridx][cidx] -= mo_eri[j][i][a + noccmo][b + noccmo]
                cidx += 1
        ridx += 1

e_cis_singlets, co_cis_singlets = scipy.linalg.eigh(H_singlets)
e_cis_triplets, co_cis_triplets = scipy.linalg.eigh(H_triplets)

print("CIS singlets Energy: ")
print("idx Hartree ev")
for i in range(H_size):
    print("%2i %10.5f %15.5f" % 
          (i, e_cis_singlets[i], e_cis_singlets[i] * 27.211386))

print("CIS triplets Energy: ")
print("idx Hartree ev")
for i in range(H_size):
    print("%2i %10.5f %15.5f" % 
          (i, e_cis_triplets[i], e_cis_triplets[i] * 27.211386))
