#!/usr/bin/env python
# HF Code Example
# Author: Peng Bao <baopeng@iccas.ac.cn>
# Modified by: Ziqiu Wang <3417847429@qq.com>

''' Need PySCF first,
MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 python rhf.py'''

import numpy
import scipy.linalg
from pyscf import gto, scf

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')
############### HF in PySCF to compare ##############
mf = scf.RHF(mol)

# For debug use, set:
# mf.verbose=4

mf.kernel()
print('Reference HF total energy =', mf.e_tot)
#####################################################

#################### HF Code Example ####################
# RHF. Only need structure information of molecule and electronic integrals 
  
# one-electron integral
hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
# electron-overlap integral
s1e = mol.intor_symmetric('int1e_ovlp')
# number of space orbitals by given basis
nao = hcore.shape[0]
# two-electron integrals
eri = mol.intor('int2e').reshape(nao,nao,nao,nao)
# number of occupied space orbitals
nocc = mol.nelectron // 2

def energy_nuc(mol):
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    rr = numpy.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)
    rr[numpy.diag_indices_from(rr)] = 1e200
    e = numpy.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

# density matrix of initial guess in pyscf.rhf is doubled
# for there is two electrons in one space orbital
# so we divide the initial guess dm by 2
dm = mf.init_guess_by_minao(mol) / 2
vhf = numpy.einsum('ijkl,ji->kl', eri, dm) * 2 - numpy.einsum('ijkl,jk->il', eri, dm) 
# e_tot = 2tr(hcore) + tr(vhf) = tr(2 * hcore + vhf)
e_tot = numpy.einsum('ij, ji->', 2 * hcore + vhf , dm) + energy_nuc(mol)
# for debug, uncomment this:
# print("Initial_e: ", e_tot)

scf_conv = False
cycle = 0
mf_diis = mf.DIIS(mf, mf.diis_file)
while not scf_conv and cycle < 50:
    dm_last = dm
    last_hf_e = e_tot

    fock = hcore + vhf
    # DIIS provided by pyscf, for debug use:
    # fock = mf_diis.update(s1e, dm, fock, mf, hcore)    

    # Diagonalize fock matrix to get mo_energy and mo_coefficients
    mo_energy, mo_coeff = scipy.linalg.eigh(fock, s1e)
    # Take occupied orbitals and form the new density matrix
    dm = mo_coeff[:,:nocc] @ mo_coeff[:,:nocc].T
    # Since new density matrix corresponds to new V_hf,
    # we should update G matrix before calculating total energy
    # otherwise there will be error introduced
    # though it does not affect the final result (dm and dm_last are identical)
    vhf = numpy.einsum('ijkl,ji->kl', eri, dm) * 2 - numpy.einsum('ijkl,jk->il', eri, dm) 
    # Calculate total energy using 'e_tot = tr(2 * hcore + vhf)'
    e_tot = numpy.einsum('ij,ji->', hcore * 2 + vhf, dm)  + energy_nuc(mol)

    # judge if converged
    delta_e = e_tot - last_hf_e
    norm_ddm = numpy.linalg.norm(dm-dm_last)
    if abs(delta_e) < 1.0E-10 and norm_ddm < 1.0E-6:
        scf_conv = True
    cycle += 1

    # Print debug informations:
    # print('Cycle ', cycle, 'Energy: ', e_tot, 'Delta_E: ', delta_e)
    # print('HOMO: ', mo_energy[nocc - 1], 'LUMO: ', mo_energy[nocc])

print('HF Code Example total energy =', e_tot, 'Cycle number=', cycle, 
      'Converged? ', scf_conv)
