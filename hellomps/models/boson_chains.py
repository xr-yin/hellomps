#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__='Xianrui Yin'

import numpy as np
from scipy import sparse

from ..networks.mpo import MPO
from ..networks.mps import MPS

__all__ = ['BosonChain', 'BoseHubburd', 'DDBoseHubburd']

class BosonChain(object):
    """1D homogeneous Boson chain
    """
    def __init__(self, N:int, d:int) -> None:
        """we include the local operators as instance attributes instead of 
        class attributes due to the indeterminacy of local dimensions"""
        self._N = N
        self.d = d
        self.bt = np.diag([m**0.5 for m in range(1,d)], -1)
        self.bn = np.diag([m**0.5 for m in range(1,d)], +1)
        self.num = np.diag(range(d))
        self.nu = np.zeros((d,d))
        self.bid = np.eye(d)

    def H_full(self):
        N, d = self._N, self.d
        h_full = sparse.csr_matrix((d**N, d**N))
        for i, hh in enumerate(self.hduo):
            h_full += sparse.kron(sparse.eye(d**i), sparse.kron(hh, np.eye(d**(N-2-i))))
        return h_full
    
    def L_full(self):
        """extend local one-site Lindblad operators into full space"""
        N, d = self._N, self.d
        Ls = []
        for i, L in enumerate(self.Lloc):
            if L is not None:
                Ls.append(sparse.kron(sparse.eye(d**i), sparse.kron(L, np.eye(d**(N-1-i)))))
        return Ls
    
    def Liouvillian(self, H, *Ls):
        """
        calculate the Liouvillian (super)operator

        Paras:
            H: the Hamiltonian in the full Hilbert space
            *L: the Lindblad jump operator(s) in the full Hilbert space

        Return: the Liouvillian operator as a sparse matrix
        """
        Lv = self._Hsup(H)
        for L in Ls:
            Lv += self._Dsup(L)
        return Lv
    
    def _Dsup(self, L):
        """
        calculate the $L\otimes L^\bar - (L^\dagger L\otimes I + I\otimes L^T L^\bar)/2$
        """
        D = self.d**self._N
        return sparse.kron(L,L.conj()) \
            - 0.5*(sparse.kron(L.conj().T@L, sparse.eye(D)) + sparse.kron(sparse.eye(D), L.T@L.conj()))

    def _Hsup(self, H):
        """
        calculate the Hamiltonian superoperator $-iH \otimes I + iI \otimes H^T$
        """
        D = self.d**self._N
        return - 1j*(sparse.kron(H,sparse.eye(D)) - sparse.kron(sparse.eye(D),H.T))

    def __len__(self):
        return self._N
    
class BoseHubburd(BosonChain):
    """
    1D Bose-Hubburd model with Hamiltonian
    H = -t \sum (b)

    Parameters
    ----------
    N : int
        size of the 1D-lattice
    d : int
        local Hilbert space dimension
    t : float or np.complex
        hopping amplitude
    U : float
        onstie interaction strength (U>0 means replusive)
    mu : float
        chemical potential
    F : float or np.complex
        coherent driving strength
    gamma : float
        coupling strength between the system and the environment
    """
    def __init__(self, N: int, d: int, t:float, U:float, mu:float) -> None:
        super().__init__(N, d)
        self.t = t
        self.U = U
        self.mu = mu

    @property
    def hduo(self):
        bt, bn = self.bt, self.bn
        n, id = self.num, self.bid
        t = self.t
        h_list = []
        for i in range(self._N - 1):
            UL = UR = 0.5 * self.U
            muL = muR = 0.5 * self.mu
            if i == 0: # first bond
                UL, muL = self.U, self.mu
            if i + 1 == self._N - 1: # last bond
                UR, muR = self.U, self.mu
            h = - t * (np.kron(bt, bn) + np.kron(bn, bt)) \
                - muL * np.kron(n, id) \
                - muR * np.kron(id, n) \
                + UL * np.kron(n@(n-id), id)/2 \
                + UR * np.kron(id, n@(n-id))/2
            # h is a matrix with legs ``(i, j), (i*, j*)``
            # reshape to a tensor with legs ``i, j, i*, j*``
            # reshape is carried out in evolution algorithms after exponetiation
            h_list.append(h)
        return h_list

    @property
    def mpo(self):
        t, U, mu = self.t, self.U, self.mu
        bt, bn= self.bt, self.bn
        n, nu, id = self.num, self.nu, self.bid
        O = np.array([[id, nu, nu, nu],
                      [bn, nu, nu, nu],
                      [bt, nu, nu, nu],
                      [0.5*U*n@(n-id) - mu*n, -t*bt, -t*bn, id]])
        Os = [O] * self._N
        Os[0] = O[None,-1,:,:,:]
        Os[-1] = O[:,0,None,:,:]
        return MPO(Os)
    
class DDBoseHubburd(BoseHubburd):
    """class for driven-dissipative Bose-Hubburd model"""
    def __init__(self, N: int, d: int, t: float, U: float, mu: float, F=0, gamma=0) -> None:
        super().__init__(N, d, t, U, mu)
        self.F = F
        self.gamma = gamma

    @property
    def hduo(self):
        bt, bn = self.bt, self.bn
        n, id = self.num, self.bid
        t = self.t
        h_list = []
        for i in range(self._N - 1):
            UL = UR = 0.5 * self.U
            muL = muR = 0.5 * self.mu
            FL = FR = 0.5 * self.F
            if i == 0: # first bond
                UL, muL, FL = self.U, self.mu, self.F
            if i + 1 == self._N - 1: # last bond
                UR, muR, FR = self.U, self.mu, self.F
            h = - t * (np.kron(bt, bn) + np.kron(bn, bt)) \
                - muL * np.kron(n, id) \
                - muR * np.kron(id, n) \
                + UL * np.kron(n@(n-id), id)/2 \
                + UR * np.kron(id, n@(n-id))/2 \
                + FL * np.kron(bt, id) \
                + FR * np.kron(id, bt) \
                + FL.conjugate() * np.kron(bn, id) \
                + FR.conjugate() * np.kron(id, bn)
            # h is a matrix with legs ``(i, j), (i*, j*)``
            # reshape to a tensor with legs ``i, j, i*, j*``
            # reshape is carried out in evolution algorithms after exponetiation
            h_list.append(h)
        return h_list
    
    @property
    def mpo(self):
        t, U, mu, F = self.t, self.U, self.mu, self.F
        bt, bn= self.bt, self.bn
        n, nu, id = self.num, self.nu, self.bid
        diag = 0.5*U*n@(n-id) - mu*n + F*bt + F.conjugate()*bn
        O = np.array([[id, nu, nu, nu],
                      [bn, nu, nu, nu],
                      [bt, nu, nu, nu],
                      [diag, -t*bt, -t*bn, id]])
        Os = [O] * self._N
        Os[0] = O[None,-1,:,:,:]
        Os[-1] = O[:,0,None,:,:]
        return MPO(Os)
    
    @property
    def Lloc(self):
        return [np.sqrt(self.gamma) * self.bn] * self._N
    
    @property
    def Liouvillian(self):
        return super().Liouvillian(self.H_full(), *self.L_full())