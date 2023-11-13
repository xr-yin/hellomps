#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__='Xianrui Yin'

import numpy as np
#from numba import jit
from scipy.linalg import qr, rq, norm

from copy import deepcopy
import logging
logging.basicConfig(level=logging.INFO)


from .operations import qr_step, rq_step, merge, split

__all__ = ['MPS', 'as_mps', 'inner', 'compress']

class MPS(object):
    '''
    class for matrix product states

    Parameters:
        As: list of local rank-3 tensors, each tensor has the following shape
                    k
                    |
                i---A---j
            i (j) is the left (right) bond leg and k is the physical leg

    Attributes:
        N: number of sites
        As: list of local tensors

    Methods:
        orthornormalize
        bond_dims
        as_array
    '''

    def __init__(self, As:list) -> None:
        self.As = As
        self.__N = len(As)
        self.__bot()

    @classmethod
    def gen_polarized_spin_chain(cls, N:int, polarization:str):
        if polarization not in ['+z','-z','+x']:
            raise ValueError('Only support polarization +z, -z or +x')
        A = np.zeros([1,1,2])
        if polarization == '+z':
            A[0,0,0] = 1.
        elif polarization == '-z':
            A[0,0,1] = 1.
        else:
            A[0,0,0] = A[0,0,1] = 0.5**0.5
        return cls([A]*N)

    @classmethod
    def gen_neel_state(cls,N:int):
        up = np.zeros([1,1,2])
        down = np.zeros([1,1,2])
        up[0,0,0] = 1.
        down[0,0,1] = 1.
        if N%2 == 0:
            return cls([up,down]*(N//2))
        else:
            return cls([up,down]*(N//2)+[up])

    @classmethod
    def gen_random_state(cls,N:int,m_max:int,phy_dims:list):
        assert len(phy_dims) == N
        rng = np.random.default_rng()
        bond_dims = rng.integers(1, m_max, size=N+1)
        bond_dims[0] = bond_dims[-1] = 1
        As = []
        for i in range(N):
            As.append(rng.random((bond_dims[i],bond_dims[i+1],phy_dims[i])))
        return cls(As)

    def orthonormalize(self, mode:str, center_idx=None):
        """
        Transforming the MPS into canaonical forms by doing successive QR decompositions

        Parameters:
            mode: right, left, mixed. When choosing mixed, the corresponding index of the
            orthogonality center must be given
            center_idx: the index of the orthogonality center

        Return: None

        Notes:
            scipy.linalg.qr, which we use here, only accepts 2-d arrays (matrices) as
            inputs to be diagonalized. Therefore, one must first combine the physical
            and matrix leg by doing a reshape, before calling qr().
            On the other hand, numpy.linalg.qr can take in (N>2)-d arrays, which are 
            regarded as stacks of matrices residing on the last 2 dimensions. Consequently,
            one can call qr() with the original tensors. In this regard, [physical, 
            left bond, right bond] indexing is preferred.
        """

        if mode not in ['left','right','mixed']:
             raise ValueError(
                  'Mode argument should be one of left, right or mixed')

        if mode == 'right':
            for i in range(self.__N-1, 0,-1):
                 self.As[i-1], self.As[i] = rq_step(self.As[i-1], self.As[i])
            #self.As[0], norm = rq_step(np.ones([1,1,1], self.As[0]))
            self.As[0] /= norm(self.As[0].squeeze())
        elif mode == 'left':
            for i in range(self.__N - 1):
                self.As[i], self.As[i+1] = qr_step(self.As[i], self.As[i+1])
            #self.As[-1], norm = qr_step(self.As[-1], np.ones([1,1,1]))
            #norm = norm.ravel()
            self.As[-1] /= norm(self.As[-1].squeeze())
        else:
            #assert isinstance(center_idx, int)
            assert center_idx > 0
            assert center_idx < self.__N
            for i in range(center_idx):
                self.As[i], self.As[i+1] = qr_step(self.As[i], self.As[i+1])
            for i in range(self.__N-1,center_idx,-1):
                self.As[i-1], self.As[i] = rq_step(self.As[i-1], self.As[i])
            self.As[center_idx] /= norm(self.As[center_idx].squeeze())

    @property
    def physical_dims(self):
        return [A.shape[2] for A in self.As]
    
    @property
    def bond_dims(self):
        return [A.shape[0] for A in self.As] + [self.As[-1].shape[1]]
    
    def entropy(self,idx=None):
        if not idx:
            cache = self.As
            S = []
            self.orthonormalize(mode='right')
            for i in range(self.__N - 1):
                self.As[i], self.As[i+1] = qr_step(self.As[i], self.As[i+1])
                s = np.linalg.svd(self.As[i+1],full_matrices=False,compute_uv=False)
                s = s[s>1.e-15]
                ss = s*s
                S.append(-np.sum(ss*np.log(ss)))
            self.As = cache
            return np.array(S)
        else:
            assert isinstance(idx, int)
            cache = self.As
            self.orthonormalize(mode='mixed',center_idx=idx)
            s = np.linalg.svd(self.As[idx],full_matrices=False,compute_uv=False)
            s = s[s>1.e-15]
            ss = s*s
            self.As = cache
            return -np.sum(ss*np.log(ss))

    def site_expectation_value(self, op, idx=None):
        """
        if `idx` is None, `op` must be a list of ordered local operators for every physical site
        if `idx` is an integer, `op` must be a single operator for this specific physical site
        """
        if isinstance(op, list):
            assert len(op) == self.__N
            cache = self.As
            exp = []
            self.orthonormalize(mode='right')
            for i in range(self.__N-1):
                amp = self.As[i]   # amplitude in the Schmidt basis
                opc = np.tensordot(amp, op[i], axes=(2,1)) # apply local operator
                res = np.tensordot(amp.conj(), opc, axes=3)
                exp.append(np.real_if_close(res))
                self.As[i], self.As[i+1] = qr_step(self.As[i], self.As[i+1]) # move the orthogonality center
            amp = self.As[-1]
            opc = np.tensordot(amp, op[-1], axes=(2,1)) # apply local operator
            res = np.tensordot(amp.conj(), opc, axes=3)
            exp.append(np.real_if_close(res))
            self.As =cache
            return exp
        else:
            assert isinstance(idx, int)
            cache = self.As
            self.orthonormalize(mode='mixed', center_idx = idx)
            amp = self.As[idx]   # amplitude in the Schmidt basis
            opc = np.tensordot(amp, op, axes=(2,1)) # apply local operator
            res = np.tensordot(amp.conj(), opc, axes=3)
            self.As = cache
            return np.real_if_close(res)
        
    def bond_expectation_value(self, op, idx=None):
        """
        if `idx` is None, `op` must be a list of ordered two-local operators for every pair of neighbouring site
        if `idx` is an integer, `op` must be a single two-local operator for this specific pair of neighbouring site
        """
        if isinstance(op, list):
            assert len(op) == self.__N-1
            cache = self.As
            exp = []
            self.orthonormalize(mode='right')
            for i in range(self.__N-1):
                j = i+1
                amp = merge(self.As[i], self.As[j])   # amplitude in the Schmidt basis
                opc = np.tensordot(amp, op[i], axes=([2,3],[2,3])) # apply local operator
                res = np.tensordot(amp.conj(), opc, axes=4)
                exp.append(np.real_if_close(res))
                self.As[i], self.As[i+1] = qr_step(self.As[i], self.As[i+1]) # move the orthogonality center
            self.As = cache
            return exp
        else:
            assert isinstance(idx, int)
            cache = self.As
            self.orthonormalize(mode='mixed', center_idx = idx)
            amp = merge(self.As[idx],self.As[idx+1])   # amplitude in the Schmidt basis
            opc = np.tensordot(amp, op, axes=([2,3],[2,3])) # apply local operator
            res = np.tensordot(amp.conj(), opc, axes=4)
            self.As = cache
            return np.real_if_close(res)

    def density_matrix(self):
        pass

    def transfer_matrix(self):
        pass

    def correlation(self, op1, op2, indices):
        pass
    
    def as_array(self):  # can be paralleled?
        """
        convert a MPS into a state vector by iterative contractions
        """
        res = self.As[0]
        for A in self.As[1:]:
            res = np.tensordot(res, A, axes=(1,0))
            res = np.swapaxes(res, 1, 2)
            new_shape = (1,A.shape[1],-1)
            res = np.reshape(res, new_shape)
        return res.ravel()
    
    def conj(self):
        return MPS([A.conj() for A in self.As])
    
    def __len__(self):
        return self.__N

    def __bot(self):
        assert self.As[0].shape[0]==self.As[-1].shape[1]==1
        # check bond dims of neighboring tensors
        for i in range(self.__N-1):
             assert self.As[i].shape[1] == self.As[i+1].shape[0]

def as_mps(psi: np.ndarray, phy_dims:list):
    """
    convert a state vector into a MPS by iterative SVDs
    """
    assert np.prod(phy_dims)==psi.shape[0] & psi.ndim==1
    As = []
    psi_ = psi
    for d in phy_dims:
        psi_ = np.reshape(psi_, (d,-1))
        u, s, vt = np.linalg.svd(psi_, full_matrices=False)
        As.append(u*s)
        phi_ = vt
    return MPS(As)

def inner(amps: MPS, bmps: MPS):
    """
    Evaluating the inner product of two MPSs by bubbling, complexity=O(D^3)

    Parameters:
        amps: the bra MPS
        bmps: the ket MPS

    Return:
        the inner product <amps|bmps>
    """
    assert len(amps) == len(bmps)
    res = np.tensordot(amps.As[0].conj(),bmps.As[0],axes=([0,2],[0,2]))
    for i in range(1,len(amps)):
        res = np.tensordot(amps.As[i].conj(), res, axes=(0,0))
        try:
            res = np.tensordot(res, bmps.As[i], axes=([1,2],[2,0]))
        except ValueError as e:
            logging.exception(e)
            logging.error(f'i={i},shape b:{bmps.As[i].shape},shape a:{amps.As[i].shape}, shape res:{res[i].shape}')
    return res.squeeze()

def compress(psi:MPS, tol:float, max_sweeps:int, m_max:int, overwrite=False):
    """
    Two-site varational compression of a MPS by using a SVD-truncated state as an initial guess.
    """
    N = len(psi)
    phi = deepcopy(psi)  # overwrite set to False, first copy then orthonormalize
    phi.orthonormalize('left')
    # peform a SVD sweep from the right to left
    for i in range(N-1,0,-1):
        di, dj, dk = phi.As[i].shape
        phi.As[i] = np.reshape(phi.As[i], (di, dj*dk))
        u, s, vt = np.linalg.svd(phi.As[i], full_matrices=False)
        mask = s>tol
        phi.As[i] = vt[mask,:].reshape(-1,dj,dk)
        phi.As[i-1] = np.tensordot(phi.As[i-1], u[:,mask]*s[mask], axes=(1,0)).swapaxes(1,2)
    # now we arrive at a right canonical MPS
    RBT = _load_right_bond_tensors(psi,phi)
    LBT = [np.ones((1,1))] * (N+1)
    nsweeps = 0
    for n in range(max_sweeps):
        logging.info('in the loop')
        for i in range(N-1): # sweep from left to right
            j = i+1
            temp = merge(psi.As[i],psi.As[j])
            temp = np.tensordot(LBT[i], temp, axes=(0,0))
            temp = np.tensordot(RBT[j], temp, axes=(0,1)).swapaxes(0,1)
            phi.As[i], phi.As[j] = split(temp, 'right', 0.01*tol, m_max)
            # compute left bond tensor L[j]
            LBT[j] = np.tensordot(psi.As[i], LBT[i], axes=(0,0))
            LBT[j] = np.tensordot(LBT[j], phi.As[i].conj(), axes=([1,2],[2,0]))
        LBT[N] = np.tensordot(psi.As[N-1], LBT[N-1], axes=(0,0))
        LBT[N] = np.tensordot(LBT[N], phi.As[N-1].conj(), axes=([1,2],[2,0]))
        for j in range(N-1,0,-1):  # sweep from right to left
            i = j-1
            temp = merge(psi.As[i],psi.As[j])
            temp = np.tensordot(LBT[i], temp, axes=(0,0))
            temp = np.tensordot(RBT[j], temp, axes=(0,1)).swapaxes(0,1)
            phi.As[i], phi.As[j] = split(temp, 'left', 0.01*tol, m_max)
            # compute right bond tensor R[i]
            RBT[i] = np.tensordot(psi.As[j], RBT[j], axes=(1,0))
            RBT[i] = np.tensordot(RBT[i], phi.As[j].conj(), axes=([1,2],[2,1]))
        # RBT[-1] is essentially the overlap
        RBT[-1] = np.tensordot(psi.As[0], RBT[0], axes=(1,0))
        RBT[-1] = np.tensordot(RBT[N], phi.As[0].conj(), axes=([1,2],[2,1]))
        logging.info(f'The overlap recorded in the #{n} sweep are {LBT[-1].ravel()} and {RBT[-1].ravel()}')
    return phi

def _load_right_bond_tensors(psi:MPS, phi:MPS):
    """
    calculate the right bond tensors while contracting two MPSs.
    RBT[i] is to the right of the MPS.As[i].

    Parameters
    ----------
    psi : MPS 
        used as ket
    phi : MPS 
        used as bra

    Return
    ----------
    RBT: a list of length N+1. The first N elements are the right bond tensors, 
    i.e. RBT[i] is to the right of the i-th local tensor. The last element, i.e.
    RBT[-1] == RBT[N] is the inner product of psi and phi.
    """
    assert len(psi) == len(phi)
    N = len(psi)
    RBT = [np.ones((1,1))] * (N+1)
    for i in range(N-1,-1,-1):
        # optimal contraction scheme, comuptational cost ~ O(m^3 * d)
        RBT[i-1] = np.tensordot(psi.As[i], RBT[i], axes=(1,0))
        RBT[i-1] = np.tensordot(RBT[i-1], phi.As[i].conj(), axes=([1,2],[2,1]))
    return RBT