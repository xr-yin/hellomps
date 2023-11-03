#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#----------Lindblad evolution for density operators----------

import numpy as np
from scipy.linalg import expm

from .mps_evolution import TEBD2
from ..networks.lptn import LPTN
from ..networks.operations import merge, split

__all__ = ['LindbladOneSite', 'LindbladTwoSite']

class LindbladOneSite(TEBD2):
    """
    Algorithm for evolving a mixed state in locally purified tensor network (LPTN) representation according to the
    Lindblad master equation.

    Parameters:
        psi: the LPTN to be evolved
        model: must include the following two attributes
            hduo: list of ordered two-local Hamiltonian operators
            Lloc: list of ordered local Lindbaldian jump operators
    """
    def __init__(self, psi: LPTN, model) -> None:
        self.Lloc = model.Lloc # list containing all (local) jump operators
        super().__init__(psi, model)
        self.dt = None

    def run_second_order(self, Nsteps, dt: float, tol: float, k_max=30, m_max=60): 
        Nbonds = len(self.psi)-1
        assert Nbonds == len(self.hduo)
        if dt != self.dt:
            self.make_unitaries(1j*dt)
            self.make_krauss(dt)
            self.dt = dt
        # apply a half step at odd-even sites
        for i in range(1, Nbonds, 2):
            self.update_local_two_sites(i, tol, m_max, half=True)
        self.psi.orthonormalize('right')      
        for n in range(Nsteps-1):
            for i in range(0, Nbonds, 2):  # half step at even-odd
                self.update_local_two_sites(i, tol, m_max, half=True)
            self.psi.orthonormalize('right')
            for i in range(Nbonds+1):  # full step
                self.apply_krauss(i, tol, k_max)
            self.psi.orthonormalize('right')
            for i in range(0, Nbonds, 2):  # half step at even-odd
                self.update_local_two_sites(i, tol, m_max, half=True)
            self.psi.orthonormalize('right')
            for i in range(1, Nbonds, 2):  # full step at odd-even
                self.update_local_two_sites(i, tol, m_max)
            self.psi.orthonormalize('right')
            if (n+1)%20 == 0:
                print(f'{n+1} steps have been completed')
                print('current bond dims:', self.psi.bond_dims)
                print('current Krauss dims:', self.psi.krauss_dims)
        for i in range(0, Nbonds, 2):
            self.update_local_two_sites(i, tol, m_max, half=True)
        self.psi.orthonormalize('right')
        for i in  range(Nbonds+1):
            self.apply_krauss(i, tol, k_max)
        self.psi.orthonormalize('right')
        for i in range(0, Nbonds, 2):
            self.update_local_two_sites(i, tol, m_max, half=True)
        self.psi.orthonormalize('right')
        for i in range(1, Nbonds, 2):  # half step trailing at the end
            self.update_local_two_sites(i, tol, m_max, half=True)
        self.psi.orthonormalize('right')

    def run_first_order(self, Nsteps, dt: float, tol: float, k_max=30, m_max=60): 
        Nbonds = len(self.psi)-1
        assert Nbonds == len(self.hduo)
        self.make_unitaries(1j*dt)
        self.make_krauss(dt)      
        for n in range(Nsteps):
            for i in range(0, Nbonds, 2):  # even-odd
                self.update_local_two_sites(i, tol, m_max)
            self.psi.orthonormalize('left')
            for i in range(1, Nbonds, 2):  # odd-even
                self.update_local_two_sites(i, tol, m_max)
            self.psi.orthonormalize('left')
            for i in range(Nbonds+1):
                try:
                    self.apply_krauss(i, tol, k_max)
                except np.linalg.LinAlgError as e:
                    print('except:', e)
                    print(f'Bs[{i}]:{self.B_list[i].shape}')
                    print('Krauss operator:', self.B_list[i])
            self.psi.orthonormalize('left')
            if (n+1)%20 == 0:
                print(f'{n+1} steps have been completed')
                print('current bond dims:', self.psi.bond_dims)
                print('current Krauss dims:', self.psi.krauss_dims)
    
    def make_krauss(self, dt):
        """
        calculate the Kraus operators from the jump operators, the resulting
        Kraus operators have the following shape
                   0|
                :.......:
                :   B   :--:
                :.......:  |
                   1|     2|

        0 : output
        1 : input
        2 : Kraus
        """
        B_list = []
        for i, L in enumerate(self.Lloc):
            if isinstance(L, np.ndarray):
                d = self.dims[i]
                # calculate the disspative part in superoperator form
                D = np.kron(L,L.conj()) \
                - 0.5*(np.kron(L.conj().T@L, np.eye(d)) + np.kron(np.eye(d), L.T@L.conj()))
                eD = expm(D*dt)
                eD = np.reshape(eD, (d,d,d,d))
                eD = np.transpose(eD, (0,2,1,3))
                eD = np.reshape(eD, (d*d,d*d))
                assert np.allclose(eD, eD.conj().T)
                B = _cholesky(eD)
                B = np.reshape(B, (d,d,-1))
                B_list.append(B)
            else:
                B_list.append(None)
        self.B_list = B_list

    def apply_krauss(self, i, tol, k_max):
        # apply bloc
        if self.B_list[i] is not None:
            self.psi.orthonormalize('mixed',i)
            l, r, p, k = self.psi.As[i].shape
            temp = np.tensordot(self.psi.As[i], self.B_list[i], axes=(2,1))
            temp = np.transpose(temp, (0,1,3,2,4))
            temp = np.reshape(temp, (l*r*p,-1))
            u, s, vt = np.linalg.svd(temp, full_matrices=False)
            s1 = s / np.linalg.norm(s)
            pivot = np.sum(s1>tol)
            pivot = min(pivot, k_max)
            temp = u[:,:pivot]*s[:pivot]
            self.psi.As[i] = np.reshape(temp, (l,r,p,-1))

    def update_local_two_sites(self, i, tol, m_max, half=False):
        # construct theta matrix
        j = i+1
        theta = merge(self.psi.As[i], self.psi.As[j])  # i, j, k1, k2
        # apply U
        if half:
            theta = np.tensordot(theta, self.u_duo['half'][i], axes=([2, 3], [2, 3]))
        else:
            theta = np.tensordot(theta, self.u_duo['full'][i], axes=([2, 3], [2, 3]))
        theta = np.transpose(theta, (0,1,4,5,2,3))
        # split and truncate
        self.psi.As[i], self.psi.As[j] = split(theta, mode="sqrt", tol=tol, m_max=m_max)
    
    def _bot(self):
        super()._bot()
        assert len(self.Lloc) == len(self.psi)
    

class LindbladTwoSite(TEBD2):
    """
    Algorithm for evolving a mixed state in locally purified tensor network (LPTN) representation according to the
    Lindblad master equation.

    Parameters:
        psi: the LPTN to be evolved
        model: must include the following two attributes
            hduo: list of ordered two-local Hamiltonian operators
            lduo: list of ordered 2-local Lindbaldian operators

    Be aware there are major architectural differences from both LindbladOneSite and TEBD2.
    """
    def __init__(self, psi: LPTN, model) -> None:
        super().__init__(psi, model)
        self.Lloc = model.lduo # list containing all (local) jump operators
        self.dt = None

    def run_second_order(self, Nsteps, dt: float, tol: float, k_max=20, m_max=100): 
        Nbonds = len(self.psi)-1
        assert Nbonds == len(self.hduo) == len(self.Lloc)
        if dt != self.dt:
            self.make_krauss(dt)
            self.dt = dt
        # apply a half step at odd-even sites
        for i in range(1, Nbonds, 2):
            self.update_local_two_sites(i, tol, m_max, k_max, half=True)
        # ---(even-odd )*(Nsteps-1)---       
        for n in range(Nsteps-1):
            for k in [0, 1]:  # even-odd and odd-even
                for i in range(k, Nbonds, 2):
                    self.update_local_two_sites(i, tol, m_max, k_max)
            self.psi.orthonormalize('right')
        # apply a full step at even bonds
        for i in range(0, Nbonds, 2):
            self.update_local_two_sites(i, tol, m_max, k_max)
        # apply a half step at odd-even sites
        for i in range(1, Nbonds, 2):
            self.update_local_two_sites(i, tol, m_max, k_max, half=True)
        self.psi.orthonormalize('right')

    def update_local_two_sites(self, i, tol, m_max, k_max, half=False):
        # construct theta matrix
        j = i+1
        theta = merge(self.psi.As[i], self.psi.As[j])  # i, j, k1, k2
        # apply U
        if half:
            theta = np.tensordot(theta, self.B_dict['half'][i], axes=([2, 3], [2, 3]))
        else:
            theta = np.tensordot(theta, self.B_dict['full'][i], axes=([2, 3], [2, 3]))
        theta = np.transpose(theta, (0,1,4,5,2,3,6))
        theta = np.reshape(theta, theta.shape[:-2]+(-1,))  # Krauss rank all to the right side
        # split and truncate along the bond dimension
        self.psi.As[i], self.psi.As[j] = split(theta, mode="sqrt", tol=tol, m_max=m_max)
         # split and truncate along the krauss dimension
        l, r, p, k = self.psi.As[j].shape
        temp = np.reshape(self.psi.As[j], (l*r*p, k))
        u, s, vt = np.linalg.svd(temp, full_matrices=False)
        s1 = s / np.linalg.norm(s)
        pivot = np.sum(s1>tol)
        pivot = min(pivot, k_max)
        temp = u[:,:pivot]*s[:pivot]
        self.psi.As[j] = np.reshape(temp, (l,r,p,-1))

    def make_krauss(self, dt):
        """
        calculate the Kraus operators from the Lindbladian operators, which includes 
        both coherent and disspative dynamics (L=H+D), the resulting Kraus operators 
        have the following shape
                    |0
                :.......:
                :   B   :--:
                :.......:  |
                    |1     |2

        0 : output
        1 : input
        2 : Kraus
        """
        B_dict = {'half':[], 'full':[]}
        for i, (H,L) in enumerate(zip(self.hduo, self.Lloc)):
            if isinstance(L, np.ndarray):
                dl, dr = self.dims[i], self.dims[i+1]
                # calculate the Lindbladian in superoperator form
                # L and H are both matrices with size (dl*dr, dl*dr)
                d = dl*dr
                Ls = np.kron(L,L.conj()) \
                - 0.5*(np.kron(L.conj().T@L, np.eye(d)) + np.kron(np.eye(d), L.T@L.conj())) \
                - 1j*np.kron(H,np.eye(d)) + 1j*np.kron(np.eye(d),H.T)
                for tau, key in zip((dt/2, dt),B_dict.keys()):
                    eLt = expm(Ls*tau)
                    eLt = np.reshape(eLt, (d,d,d,d))
                    eLt = np.transpose(eLt, (0,2,1,3))
                    eLt = np.reshape(eLt, (d*d,d*d))
                    assert np.allclose(eLt, eLt.conj().T)
                    B = _cholesky(eLt)
                    B = np.reshape(B, (dl,dr,dl,dr,-1))  # check
                    B_dict[key].append(B)
            else:
                for tau, key in zip((dt/2, dt),B_dict.keys()):
                    u = expm(-1j*H*tau)
                    u = np.reshape(u,(dl,dr,dl,dr))
                    B_dict[key].append(u[:,:,:,:,None])
        self.B_dict = B_dict
        print('Krauss operators prepared')



def _cholesky(a):
    eigvals, eigvecs = np.linalg.eigh(a)
    mask = abs(eigvals) > 1e-15
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:,mask]
    assert min(eigvals) > 0
    B = eigvecs*np.sqrt(eigvals)
    #print('error:', np.linalg.norm(a-B@B.T.conj()))
    return B