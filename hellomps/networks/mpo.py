#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__='Xianrui Yin'

import numpy as np

from .operations import orthonormalizer

__all__ = ['MPO']

class MPO(object):
    """class for matrix product operators

    Parameters
    ----------
    As : list 
        a list of rank-4 tensors, each tensor has the following shape

        k |
    i---- A ----j
        k*|

    i (j) is the left (right) bond leg and k (k*) is the ket (bra) physical leg
    the legs are ordered as `i, j, k, k*`

    Attributes
    ----------
    As : list
        as described above

    Methods
    ----------
    orthonormalize()
    conj()
    hc()
    to_matrix()
    """
    def __init__(self, As) -> None:
        self.As = As 
        self._N = len(As)
        self.__bot()

    @classmethod
    def gen_random_mpo(cls, N:int, m_max:int, phy_dims:list, hermitian=False):
        assert len(phy_dims) == N
        rng = np.random.default_rng()
        bond_dims = rng.integers(1, m_max, size=N+1)
        bond_dims[0] = bond_dims[-1] = 1
        As = []
        for i in range(N):
            As.append(rng.random((bond_dims[i],bond_dims[i+1],phy_dims[i], phy_dims[i]))
                      + 1j*rng.random((bond_dims[i],bond_dims[i+1],phy_dims[i], phy_dims[i])))
        if hermitian:
            As = [A + A.swapaxes(2,3).conj() for A in As]
        return cls(As)

    @property
    def bond_dims(self):
        return [A.shape[0] for A in self] + [self[-1].shape[1]]
    
    @property
    def physical_dims(self):
        return [A.shape[2] for A in self]
    
    def orthonormalize(self, mode: str, center_idx=None):
        orthonormalizer(self, mode=mode, center_idx=center_idx)

    def conj(self):
        """
        Return
        ----------
        complex conjugate of the MPO
        """
        return MPO([A.conj() for A in self])
    
    def hc(self):
        """
        Return
        ----------
        Hermitian conjugate of the MPO
        """
        return MPO([A.swapaxes(2,3).conj() for A in self])

    def to_matrix(self):
        """
        convert the MPO into a dense matrix for best compatability. Users
        are free to further convert it into a sparse matrix to explore more
        efficient linear algebra algorithms.
        """
        full = self[0]
        for i in range(1,self._N):
            full = np.tensordot(full, self[i],axes=(1,0))
            full = np.transpose(full, (0,3,1,4,2,5))
            di, dj, dk1, dk2, dk3, dk4 = full.shape
            full = np.reshape(full, (di, dj, dk1*dk2, dk3*dk4))
        return full.squeeze()
        
    def __len__(self):
        return self._N
    
    def __getitem__(self, idx: int):
        return self.As[idx]
    
    def __setitem__(self, idx: int, value):
        self.As[idx] = value
    
    def __iter__(self):
        return iter(self.As)
    
    def __bot(self):
        assert self.As[0].shape[0]==self.As[-1].shape[1]==1
        # check bond dims of neighboring tensors
        for i in range(self._N-1):
             assert self.As[i].shape[1] == self.As[i+1].shape[0]