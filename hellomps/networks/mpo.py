#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__='Xianrui Yin'

import numpy as np
from scipy import sparse
from scipy.linalg import norm, qr, rq

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
        return [A.shape[0] for A in self.As] + [self.As[-1].shape[1]]
    
    @property
    def physical_dims(self):
        return [A.shape[2] for A in self.As]
    
    def orthonormalize(self, mode: str, center_idx=None):
        if mode not in ['left','right','mixed']:
             raise ValueError(
                  'Mode argument should be one of left, right or mixed')
        
        if mode == 'right':
            for i in range(self._N-1, 0,-1):
                self.As[i-1], self.As[i] = self._rq_step(self.As[i-1], self.As[i])
            self.As[0] /= norm(self.As[0].squeeze())
        elif mode == 'left':
            for i in range(self._N - 1):
                self.As[i], self.As[i+1] = self._qr_step(self.As[i], self.As[i+1])
            self.As[-1] /= norm(self.As[-1].squeeze())
        else:
            #assert isinstance(center_idx, int)
            assert center_idx >= 0
            assert center_idx < self._N
            for i in range(center_idx):
                self.As[i], self.As[i+1] = self._qr_step(self.As[i], self.As[i+1])
            for i in range(self._N-1,center_idx,-1):
                self.As[i-1], self.As[i] = self._rq_step(self.As[i-1], self.As[i])
            self.As[center_idx] /= norm(self.As[center_idx].squeeze())
    
    def conj(self):
        """
        Complex conjugate of the MPO
        """
        return MPO([A.conj() for A in self.As])
    
    def hc(self):
        """
        Hermitian conjugate of the MPO
        """
        return MPO([A.swapaxes(2,3).conj() for A in self.As])

    def to_matrix(self):
        """
        convert the MPO into a dense matrix for best compatability. Users
        are free to further convert it into a sparse matrix to explore more
        efficient linear algebra algorithms.
        """
        full = self.As[0]
        for i in range(1,self._N):
            full = np.tensordot(full, self.As[i],axes=(1,0))
            full = np.transpose(full, (0,3,1,4,2,5))
            di, dj, dk1, dk2, dk3, dk4 = full.shape
            full = np.reshape(full, (di, dj, dk1*dk2, dk3*dk4))
        return full.squeeze()
        
    def __len__(self):
        return self._N
    
    @staticmethod
    def _qr_step(ls,rs):
        """
            2,k           2,k
             |             |
         ----ls----    ----rs----  
         0,i |  1,j    0,i | 1,j 
            3,l           3,l
        """
        di, dj, dk, dl = ls.shape
        ls = ls.swapaxes(1,3).reshape(-1,dj) # stick i,k,l together, first need to switch j,k
        # compute QR decomposition of the left matrix
        ls, _r = qr(ls, overwrite_a=True, mode='economic') 
        ls = ls.reshape(di,dl,dk,-1).swapaxes(3,1)
        # multiply matrix R into the right matrix
        rs = np.tensordot(_r, rs, axes=1)
        return ls, rs
    
    @staticmethod
    def _rq_step(ls,rs):
         di, dj, dk, dl = rs.shape
         rs = rs.reshape(di,-1)
         # compute RQ decomposition of the right matrix
         _r, rs = rq(rs, overwrite_a=True, mode='economic')
         rs = rs.reshape(-1,dj,dk,dl)
         # multiply matrix R into the left matrix
         ls = np.tensordot(ls, _r, axes=(1,0)).transpose(0,3,1,2)
         return ls, rs
    
    def __bot(self):
        assert self.As[0].shape[0]==self.As[-1].shape[1]==1
        # check bond dims of neighboring tensors
        for i in range(self._N-1):
             assert self.As[i].shape[1] == self.As[i+1].shape[0]