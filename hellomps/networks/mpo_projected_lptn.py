#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Common interface for applying projected matrix product operators onto local MPO (LPTN) tensor.

    This class acts as an abstract interface between matrix product operators and iterative 
    solvers (for our purposes, mainly eigensolvers), providing methods to perform MPO-MPO/LPTN 
    and adjoint MPO-MPO/LPTN products
"""

__author__='Xianrui Yin'

import numpy as np
from pylops import LinearOperator
from pylops.utils.typing import NDArray

__all__ = ["ProjTwoSite", "LeftBondTensors", "RightBondTensors"]

class ProjTwoSite(LinearOperator):
    r"""Project MPO onto two neighboring sites

                            <bra|
    
    /```\----0        2               2         0----/```\
    |   |           __|__           __|__            |   |
    | L |----1  0---|M 1|---1   0---|M 2|---1   1----| R |
    |   |           ``|``           ``|``            |   |
    \___/----2        3               3         2----\___/

                            |ket>

    Parameters
    ----------
    L : ndarray, ndim==3
        left bond tensor
    R : ndarray, ndim==3
        right bond tensor
    M1 : ndarray, ndim==4
        left local MPO tensor
    M2 : ndarray, ndim==4
        right local MPO tensor
    """

    def __init__(self, L: NDArray, R: NDArray, M1: NDArray, M2: NDArray) -> None:
        self.L = L
        self.R = R
        self.M1 = M1
        self.M2 = M2
        dtype = M2.dtype
        dims = (L.shape[2], R.shape[2], M1.shape[3], M2.shape[3])
        dimsd = (L.shape[0], R.shape[0], M1.shape[2], M2.shape[2])
        super().__init__(dtype=dtype, dims=dims, dimsd=dimsd)
        assert self.shape == (np.prod(dimsd), np.prod(dims))

    def _matvec(self, x: NDArray) -> NDArray:
        y = np.tensordot(self.L, x, axes=(2,0))
        y = np.tensordot(y, self.M1, axes=([1,3],[0,3]))
        y = np.tensordot(y, self.M2, axes=([5,2],[0,3]))
        y = np.tensordot(y, self.R, axes=([1,5],[2,1]))
        return y.transpose(0,5,3,4,1,2)
    
class LeftBondTensors(object):
    r"""Left bond tensor, L

    /```\----0      <bra|
    |   |
    | L |----1
    |   |
    \___/----2      |ket>

    Parameters
    ----------
    N : int
        length of the underlying LPTN/MPO

    Attributes
    ----------
    LBT : list
        a list stores the left bond tensor to each site

    Methods
    ----------
    update()
        calculate the next bond tensor towards right

    Notes
    ----------
    <bra| is NOT complex congugated in the update() method.
    """
    def __init__(self, N) -> None:
        self._N = N
        self.LBT = [np.ones((1,1,1))] * N  # LBT[0] is trivial

    def update(self, idx: int, bra: NDArray, ket: NDArray, op: NDArray):
        self[idx] = np.tensordot(bra, self[idx-1], axes=(0,0))
        self[idx] = np.tensordot(self[idx], op, axes=([1,3],[2,0]))
        self[idx] = np.tensordot(self[idx], ket, axes=([1,2,4],[3,0,2]))
        assert self[idx].shape == (bra.shape[1], op.shape[1], ket.shape[1])

    def __getitem__(self, idx: int):
        return self.LBT[idx]
    
    def __setitem__(self, idx: int, value: NDArray):
        self.LBT[idx] = value
    
    def __iter__(self):
        return iter(self.LBT)

class RightBondTensors(object):
    r"""Right bond tensors

    <bra|  0----/```\
                |   |
           1----| R |
                |   |
    |ket>  2----\___/

    Parameters
    ----------
    N : int
        length of the underlying LPTN/MPO

    Attributes
    ----------
    LBT : list
        a list stores the right bond tensor to each site

    Methods
    ----------
    update()
        calculate the next bond tensor towards left
    load()

    Notes
    ----------
    <bra| is NOT complex congugated in the update() method.
    """
    def __init__(self, N:int) -> None:
        self._N = N
        self.RBT = [np.ones((1,1,1))] * N  # RBT[N-1] is trivial

    def update(self, idx: int, bra: NDArray, ket: NDArray, op: NDArray):
        self[idx] = np.tensordot(bra, self[idx+1], axes=(1,0))
        self[idx] = np.tensordot(self[idx], op, axes=([1,3],[2,1]))
        self[idx] = np.tensordot(self[idx], ket, axes=([2,4,1],[1,2,3]))
        assert self[idx].shape == (bra.shape[0], op.shape[0], ket.shape[0])

    def load(self, phi, psi, O):
        """Iteratively computing the right bond tensors resulted from 
        the following contraction: R_(j-1) = ((R_j & psi_j) & O_j) & phi*_j

        Parameters
        ----------
        phi : LPTN 
            used as bra
        psi : LPTN
            used as ket
        O : MPO
        """
        assert len(psi) == len(O) == len(phi) == self._N
        for i in range(self._N-1,0,-1):
            self.update(i-1, phi[i].conj(), psi.As[i], O.As[i])

    def __getitem__(self, idx: int):
        return self.RBT[idx]
    
    def __setitem__(self, idx: int, value: NDArray):
        self.RBT[idx] = value
    
    def __iter__(self):
        return iter(self.RBT)