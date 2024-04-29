"""This module demonstrates that the area law for entanglement entropy in one dimension.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys
import os
import logging
logging.basicConfig(level=logging.WARNING)

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mps import MPS
from hellomps.algorithms.dmrg import DMRG
from hellomps.models.spin_chains import Heisenberg, TransverseIsing

def AreaLaw(maxiter=2):

    N_list = np.arange(10,60,10, dtype=int)
    g_list = [0.5, 1.0, 1.5]

    fig1, axes = plt.subplots(nrows=len(g_list), ncols=len(N_list), sharex=True, figsize=(16,8)) # history of energies
    fig2, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5)) # entanglement entropy

    for i, g in enumerate(g_list):
        print('g=', g)
        S_list = []
        for j, N in enumerate(N_list):
            print('N=', N)
            model = TransverseIsing(N, g)
            #model = Heisenberg(N, J=[1., 1., 1.5], g=g)
            psi = MPS.gen_polarized_spin_chain(N, '+x')
            lab = DMRG(psi, model.mpo)
            e_list = [model.energy(psi)]
            for k in range(maxiter):
                lab.run_two_sites(1, 1e-7, 50)
                e_list.append(model.energy(psi))
            print('final bond dimensions:', psi.bond_dims)
            axes[i][j].plot(e_list, '-o', label=f'N={N},g={g}')
            S_list.append(psi.entropy(int(N//2)))

        ax1.plot(N_list, S_list, '-o', label=f'g={g}')
        # compute the singular values for the longest chain
        psi.orthonormalize(mode='mixed', center_idx = N//2)
        ml, mr, d = psi[N//2].shape
        svs = np.linalg.svd(psi[N//2].reshape(ml, mr*d), compute_uv=False)
        ax2.plot(svs, 'o', label=f'g={g}')

    ax1.set_xlabel('chain length')
    ax1.set_ylabel('half chain entropy')
    ax1.set_title('Bipartite entanglement entropy')

    ax2.set_ylabel(r'$\sigma_i$')
    ax2.set_yscale('log')
    ax2.set_title('Singular values at the center bond')

    for ax in axes.flat:
        ax.legend()
    for ax in (ax1, ax2):
        ax.legend()

    fig1.tight_layout()
    fig2.tight_layout()

    plt.figure(fig1)
    plt.savefig('dmrg_energy')
    plt.figure(fig2)
    plt.savefig('area_law')

if __name__ == "__main__":
    AreaLaw()