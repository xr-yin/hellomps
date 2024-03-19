#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__='Xianrui Yin'

import numpy as np
from scipy import sparse

from ..networks.mpo import MPO
from ..networks.mps import MPS

__all__ = ['SpinlessFermion', 'KitaevChain']

class SpinlessFermion(object):
    """class for spinless fermions
    """
    cn = np.array([[0., 1.], [0., 0.]])
    ct = np.array([[0., 0.], [1., 0.]])
    id = np.eye(2)
    num = np.array([[0., 0.], [0., 1.]])

    def __init__(self, N:int) -> None:
        self._N = N

    def Liouvillian(self, H, *Ls):
        """
        calculate the Liouvillian (super)operator

        Paras:
            H: the Hamiltonian in the full Hilbert space
            *L: the Lindblad jump operator(s) in the full Hilbert space

        Return: the Liouvillian operator as a sparse matrix
        """
        D = 2**self._N
        Lv = self._Hsup(H)
        for L in Ls:
            Lv += self._Dsup(L)
        return Lv
    
    def _Dsup(self, L):
        """
        calculate the $L\otimes\L^\bar - (L^\dagger L\otimes I + I\otimes L^T L^\bar)/2$
        """
        D = 2**self._N
        return sparse.kron(L,L.conj()) \
            - 0.5*(sparse.kron(L.conj().T@L, sparse.eye(D)) + sparse.kron(sparse.eye(D), L.T@L.conj()))

    def _Hsup(self, H):
        D = 2**self._N
        return - 1j*(sparse.kron(H,sparse.eye(D)) - sparse.kron(sparse.eye(D),H.T))

    def __len__(self):
        return self._N

class KitaevChain(SpinlessFermion):

    def __init__(self, N: int, t=1., delta=1., mu=0., gamma=0.1) -> None:
        super().__init__(N)
        self.t = t  # hopping 
        self.delta = delta  # creation/annihilate pairs of neighbours 
        self.mu = mu  # on-site energy
        self.gamma = gamma  # damping rate

    @property
    def hduo(self):
        cn, ct, num, id = self.cn, self.ct, self.num, self.id
        t, delta, mu = self.t, self.delta, self.mu
        h_list = []
        for i in range(self._N - 1):
            muL = muR = 0.5 * mu
            if i == 0: # first bond
                muL = mu
            if i + 1 == self._N - 1: # last bond
                muR = mu
            h = - t * (np.kron(ct, cn) + np.kron(cn, ct)) \
                + delta * (np.kron(cn, cn) + np.kron(ct, ct)) \
                - muL * np.kron(num, id) \
                - muR * np.kron(id, num)
            # h is a matrix with legs ``(i, j), (i*, j*)``
            # reshape to a tensor with legs ``i, j, i*, j*``
            # reshape is carried out only after exponetiation
            h_list.append(h)
        return h_list
    
    @property
    def lduo(self):
        cn, ct, num, id = self.cn, self.ct, self.num, self.id
        #0.25*np.sqrt(self.gamma)*(np.kron(ct,id)+np.kron(id,ct))@(np.kron(cn,id)-np.kron(id,cn))
        return [0.25*np.sqrt(self.gamma)*(np.kron(num,id)-np.kron(ct,cn)+np.kron(cn,ct)-np.kron(id,num))]*(self._N-1)
    
    @property
    def H_full(self):
        N = self._N
        h_full = sparse.csr_matrix((2**N, 2**N))
        for i, hh in enumerate(self.hduo):
            h_full += sparse.kron(sparse.eye(2**i), sparse.kron(hh, np.eye(2**(N-2-i))))
        return h_full
    
    @property
    def Liouvillian(self):
        N = self._N
        Ls = []
        for i, ll in enumerate(self.lduo):
            Ls.append(sparse.kron(sparse.eye(2**i), sparse.kron(ll, sparse.eye(2**(N-2-i)))))
        # first transform each full Lindblad operator into superoperator, then sum together, oder is important
        return super().Liouvillian(self.H_full, *Ls)
    
class FermiHubburd():
    pass