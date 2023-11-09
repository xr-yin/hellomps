#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__='Xianrui Yin'

import numpy as np
from scipy import sparse

from ..networks.mpo import MPO
from ..networks.mps import MPS

__all__ = ['BosonChain']

class BosonChain(object):
    """
    1D homogeneous Boson chain
    """
    def __init__(self, N:int, d:int) -> None:
        self._N = N
        self.bt = np.diag([m**0.5 for m in range(1,d)], -1)
        self.bn = np.diag([m**0.5 for m in range(1,d)], +1)
        self.num = np.diag(range(d))
        self.bid = np.eye(d)

    def __len__(self):
        return self._N
    
class BoseHubburd(BosonChain):
    """
    1D Bose-Hubburd model with Hamiltonian
    H = -t \sum ()

    Parameters:
        N: length of the lattice
        d: local Hilbert space dimension
        t: hopping amplitude
        U: onstie interaction strength (U>0 means replusive)
        mu: chemical potential
    """
    def __init__(self, N: int, d: int, t:float, U:float, mu:float) -> None:
        super().__init__(N, d)
        self.U = U
        self.mu = mu

    @property
    def hduo(self):
        bt, bn = self.bt, self.bn
        n, id = self.num, self.bid
        t = self.t
        h_list = []
        for i in range(self._N - 1):
            UL, UR = 0.5 * self.U
            muL, muR = 0.5 * self.mu
            if i == 0: # first bond
                UL = self.U
                muL = self.mu
            if i + 1 == self._N - 1: # last bond
                UR = self.U
                muR = self.mu
            h = - t * (np.kron(bt, bn),np.kron(bn, bt)) \
                - muL * np.kron(n, id) \
                - muR * np.kron(id, n) \
                + UL * np.kron(n@(n-id), id)/2 \
                + UR * np.kron(id, n@(n-id))/2
            # h is a matrix with legs ``(i, j), (i*, j*)``
            # reshape to a tensor with legs ``i, j, i*, j*``
            # reshape is carried out in evolution algorithms after exponetiation
            h_list.append(h)