#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__='Xianrui Yin'

import numpy as np
from scipy import sparse

from ..networks.mpo import MPO
from ..networks.mps import MPS

__all__ = ['SpinChain', 'TransverseIsing', 'Heisenberg', 'XXZ']

class SpinChain(object):
    """
    Base class for spin one-half chains

    Params:
        N: chain length
    """
    cid = np.eye(2)
    nu = np.zeros([2,2])
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1j], [1j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    splus = np.array([[0., 1.], [0., 0.]])
    sminus = np.array([[0., 0.], [1., 0.]])
    
    def __init__(self, N:int) -> None:
        self._N = N

    @property
    def H_full(self):
        N = self._N
        h_full = sparse.csr_matrix((2**N, 2**N))
        for i, hh in enumerate(self.hduo):
            h_full += sparse.kron(sparse.eye(2**i), sparse.kron(hh, np.eye(2**(N-2-i))))
        return h_full
    
    def energy(self, psi:MPS):
        assert len(psi) == self._N
        return np.sum(psi.bond_expectation_value([h.reshape(2,2,2,2) for h in self.hduo]))
    
    def current(self, psi:MPS):
        assert len(psi) == self._N
        Nbonds = self._N-1
        current_op = 2j*(np.kron(self.splus, self.sminus) - np.kron(self.sminus, self.splus))
        current_op = np.reshape(current_op, (2,2,2,2))
        return psi.bond_expectation_value([current_op]*Nbonds)

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

class TransverseIsing(SpinChain):
    """
    Class for transverse-field Ising model.
    H = sum{ -J*Sz*Sz - g*Sx }
    """

    def __init__(self, N:int, g, J=1.):
        super().__init__(N)
        self.J, self.g = J, g

    @property
    def mpo(self):
        sx, sz, nu, id = self.sx, self.sz, self.nu, self.cid
        J, g = self.J, self.g
        O = np.array([[id, nu, nu],
                      [sz, nu, nu],
                      [-g*sx, -J*sz, id]])
        Os = [O] * self._N
        Os[0] = O[None,-1,:,:,:]
        Os[-1] = O[:,0,None,:,:]
        return MPO(Os)
    
    @property
    def hduo(self):
        sx, sz, id = self.sx, self.sz, self.cid
        J, g = self.J, self.g
        h_list = []
        for i in range(self._N - 1):
            gL = gR = 0.5 * g
            if i == 0: # first bond
                gL = g
            if i + 1 == self._N - 1: # last bond
                gR = g
            h = - J * np.kron(sz, sz) \
                - gL * np.kron(sx, id) \
                - gR * np.kron(id, sx)
            # h is a matrix with legs ``(i, j), (i*, j*)``
            # reshape to a tensor with legs ``i, j, i*, j*``
            # reshape is carried out in evolution algorithms after exponetiation
            h_list.append(h)
        return h_list

    @property
    def Lloc(self):
        return None

class Heisenberg(SpinChain):
    """1D spin 1/2 Heisenberg model
    H = -\sum{Jx*Sx*Sx + Jy*Sy*Sy + Jz*Sz*Sz + g*Sx}
    """
    def __init__(self, N:int, J:list, g:float, gamma=0.):
        super().__init__(N)
        self.J = J
        self.g = g
        self.gamma = gamma

    @property
    def mpo(self):
        sx, sy, sz, nu, id = self.sx, self.sy, self.sz, self.nu, self.cid
        Jx, Jy, Jz = self.J
        g = self.g
        O = np.array([[id, nu, nu, nu, nu],
                      [sx, nu, nu, nu, nu],
                      [sy, nu, nu, nu, nu],
                      [sz, nu, nu, nu, nu],
                      [-g*sx, -Jx*sx, -Jy*sy, -Jz*sz, id]])
        Os = [O] * self._N
        Os[0] = O[None,-1,:,:,:]
        Os[-1] = O[:,0,None,:,:]
        return MPO(Os)
    
    @property
    def hduo(self):
        sx, sy, sz, id = self.sx, self.sy, self.sz, self.cid
        Jx, Jy, Jz = self.J
        g = self.g
        h_list = []
        for i in range(self._N - 1):
            gL = gR = 0.5 * g
            if i == 0: # first bond
                gL = g
            if i + 1 == self._N - 1: # last bond
                gR = g
            h = - Jx * np.kron(sx, sx) \
                - Jy * np.kron(sy, sy) \
                - Jz * np.kron(sz, sz) \
                - gL * np.kron(sx, id) \
                - gR * np.kron(id, sx)
            # h is a matrix with legs ``(i, j), (i*, j*)``
            # reshape to a tensor with legs ``i, j, i*, j*``
            # reshape is carried out in evolution algorithms after exponetiation
            h_list.append(h)
        return h_list

class XXZ(SpinChain):
    def __init__(self, N:int, delta:float, gamma=0,) -> None:
        super().__init__(N)
        self.delta = delta
        self.gamma = gamma

    @property
    def hduo(self):
        sx, sz, sy = self.sx, self.sz, self.sy
        return [np.kron(sx,sx) + np.kron(sy,sy) + self.delta*np.kron(sz,sz)]*(self._N-1)
        
    @property
    def Lloc(self):
        # list of jump operators
        return [np.sqrt(2*self.gamma)*self.splus] \
                + [None]*(self._N-2) \
                + [np.sqrt(2*self.gamma)*self.sminus]
    
    @property
    def Liouvillian(self):
        Ls = []
        
        L0 = sparse.kron(self.Lloc[0], sparse.eye(2**(self._N-1)))
        L1 = sparse.kron(sparse.eye(2**(self._N-1)), self.Lloc[-1])
        return super().Liouvillian(self.H_full, L0, L1)