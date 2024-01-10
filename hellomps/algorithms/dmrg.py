#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#----------Density Matrix Renormalization Group (DMRG) algorithm for ground state research----------

__author__='Xianrui Yin'

import numpy as np
from scipy.sparse.linalg import eigsh

import os
import logging
logging.info(f'number of threads in use:{os.environ.get("OMP_NUM_THREADS")}')

from ..networks.mps import MPS
from ..networks.mpo import MPO
from ..networks.operations import *
from ..networks.mpo_projected import *

__all__ = ["DMRG",]

class DMRG(object):
    r"""DMRG algorithm for 1D ground state serach
    
        Parameters
        ----------
        psi : MPS
            the initial MPS to be optimized, optimization is done in place
        H : MPO
            model Hamiltonian in `MPO` format

        Attributes
        ----------
        same as parameters
        Methods
        ----------
        run_one_site()
        run_two_sites()
    """

    def __init__(self, psi:MPS, H: MPO) -> None:
        assert len(psi) == len(H)
        self.psi = psi
        self.H = H

    def run_two_sites(self, Nsweeps: int, tol: float, m_max: int):
        """Two site update

        Parameters
        ----------
        Nsweeps : int
            total sweeps in one run, one sweep goes from left to right and back from right to left.
        tol : float
            the largest singular value to be discarded at each bond
        m_max : int
            largest bond dimension allowed
        """
        N = len(self.psi)
        self.psi.orthonormalize('right')
        Ls = LeftBondTensors(N)
        Rs = RightBondTensors(N)
        Rs.load(self.psi, self.psi, self.H)
        for n in range(Nsweeps):
            # the first and last tensors are only attended once during a back and forth sweep
            for i in range(N-2): # sweep from left to right
                j = i+1
                x = merge(self.psi[i], self.psi[j])
                eff_H = ProjTwoSite(Ls[i], Rs[j], self.H[i], self.H[j])
                _, x = eigsh(eff_H, k=1, which='SA', v0=x)
                x = np.reshape(x, eff_H.dims)
                # split the result tensor
                self.psi[i], self.psi[j] = split(x, 'right', tol, m_max)
                # update the left bond tensor Ls[j]
                Ls.update(j, self.psi[i].conj(), self.psi[i], self.H[i])
            for j in range(N-1,1,-1):
                i = j-1
                x = merge(self.psi[i], self.psi[j])
                eff_H = ProjTwoSite(Ls[i], Rs[j], self.H[i], self.H[j])
                _, x = eigsh(eff_H, k=1, which='SA', v0=x)
                x = np.reshape(x, eff_H.dims)
                # split the result tensor
                self.psi[i], self.psi[j] = split(x, 'left', tol, m_max)
                # update the right bond tensor Rs[i]
                Rs.update(i, self.psi[j].conj(), self.psi[j], self.H[j])

    def run_one_stie(self, Nsweeps: int):
        """One site update

        Parameters
        ----------
        Nsweeps : int
            total sweeps in one run, one sweep goes from left to right and back from right to left.
        """
        N = len(self.psi)
        self.psi.orthonormalize('right')
        Ls = LeftBondTensors(N)
        Rs = RightBondTensors(N)
        Rs.load(self.psi, self.psi, self.H)
        for n in range(Nsweeps):
            # the first and last tensor are only attended once during a back and forth sweep
            for i in range(N-1): # sweep from left to right
                eff_H = ProjOneSite(Ls[i], Rs[i], self.H[i])
                _, self.psi[i] = eigsh(eff_H, k=1, which='SA', v0=self.psi[i])
                self.psi[i] = np.reshape(self.psi[i], eff_H.dims)
                self.psi[i], self.psi[i+1] = qr_step(self.psi[i], self.psi[i+1])
                Ls.update(i+1, self.psi[i].conj(), self.psi[i], self.H[i])
            for i in range(N-1,0,-1):
                eff_H = ProjOneSite(Ls[i], Rs[i], self.H[i])
                _, self.psi[i] = eigsh(eff_H, k=1, which='SA', v0=self.psi[i])
                self.psi[i] = np.reshape(self.psi[i], eff_H.dims)
                self.psi[i-1], self.psi[i] = rq_step(self.psi[i-1], self.psi[i])
                Rs.update(i-1, self.psi[i].conj(), self.psi[i], self.H[i])