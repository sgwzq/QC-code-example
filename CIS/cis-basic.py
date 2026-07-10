#!/usr/bin/env python
# cis-basic.py
# Spin-Orbital Configuration Interaction Singles
# Author: Ziqiu Wang

''' Need PySCF first,
MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 python cis-basic.py'''

import numpy
import scipy.linalg
from pyscf import gto, scf

############### HF calc in PySCF ##############

# Supposing Restricted close-shell configuration
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
nelec = mol.nelectron
nvirt = 2 * nao - nelec
nsingles = nelec * nvirt

# CIS procedure
# 1. Getting Fock Matrix from linear combination of Gaussian orbitals
#    to Molecular Orbitals

# Two-electron Repulsion Integral
eri = mol.intor('int2e').reshape(nao,nao,nao,nao)

# Compute Two-eletron integrals based on Molecular basis

mo_eri = numpy.einsum('abcd,ai,bj,ck,dl->ijkl', eri,
                      mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff)

# 2. Calculating Hamiltonian Matrix
def get_H_single(i, a, j, b):
    # <i->a|H|j->b>
    # i a j b are all spin molecular orbitals
    # supposing closed shell
    # even(0, 2, 4...) idx represents for alpha-spin, odd for beta
    ene = 0.
    if i == j and a == b:
        # one-electron energy for a and i
        ene = mf.mo_energy[(nelec + a) // 2] - mf.mo_energy[i // 2]
    # two-electron integrals -<ja||ib> or -([ji|ab] - [jb|ia])
    if not (i & 1)^(j & 1) and not (a & 1)^(b & 1):
        # i and j are of the same spin, a and b also are
        ene -= mo_eri[j // 2][i // 2][(a + nelec) // 2][(b + nelec) // 2]
    if not (j & 1)^(b & 1) and not (i & 1)^(a & 1):
        ene += mo_eri[j // 2][(b + nelec) // 2][i // 2][(a + nelec) // 2]
    return ene

H = numpy.zeros((nsingles, nsingles))
ridx, cidx = 0, 0
for j in range(nelec):
    for i in range(nvirt):
        cidx = 0
        for l in range(nelec):
            for k in range(nvirt):
                H[ridx][cidx] = get_H_single(j, i, l, k)
                cidx += 1
        ridx += 1

e_cis, co_cis = scipy.linalg.eigh(H)
print("CIS Correlation Energy: ")
print("idx Hartree ev")
for i in range(nsingles):
    print("%2i %10.5f %15.5f" % (i, e_cis[i], e_cis[i] * 27.211386))
print("CIS Total Energy = ", e_cis[0] + mf.e_tot)

