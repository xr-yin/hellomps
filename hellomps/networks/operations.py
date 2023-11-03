import numpy as np
from copy import deepcopy
#from numba import jit

import logging
logging.basicConfig(level=logging.ERROR)


__all__ = ['split', 'merge', 'mul', 'apply_mpo', 'zip_up', 'load_right_bond_tensors']

def split(theta, mode:str, tol:float, m_max=None):
    '''
    split a local 4-tensor into two parts by doing a SVD 
    and discard the impertinent singular values
    '''
    if mode not in ["left", "right", "sqrt"]:
        raise ValueError('unknown mode')
    if theta.ndim == 4:
        di, dj, dk1, dk2 = theta.shape
        theta = np.swapaxes(theta, 1, 2)
        theta = np.reshape(theta, (di*dk1, dj*dk2))
        toggle = 'MPS'
    elif theta.ndim == 6:
        di, dj, dk1, dk2, dl1, dl2 = theta.shape
        theta = np.transpose(theta, (0,2,4,1,3,5))
        theta = np.reshape(theta, (di*dk1*dl1, dj*dk2*dl2))
        toggle = 'MPO'
    else:
        raise ValueError('Theta must have rank-4 (for MPS) or rank-6 (for MPO)')
    theta1, s, theta2 = np.linalg.svd(theta, full_matrices=False)
    s = s / np.linalg.norm(s)
    pivot = min(np.sum(s>tol), m_max) if m_max else np.sum(s>tol) # sum over the mask
    theta1, s, theta2 = theta1[:,:pivot], s[:pivot], theta2[:pivot,:]
    s = s / np.linalg.norm(s)
    if mode == 'left':
        theta1 *= s
    elif mode == 'right':
        theta2 = np.diag(s) @ theta2
    else:
        _s = np.sqrt(s)
        theta1 *= _s
        theta2 = np.diag(_s) @ theta2
    if toggle == 'MPS':
        theta1 = np.reshape(theta1, (di,dk1,-1))
        theta1 = np.swapaxes(theta1, 1, 2)
        theta2 = np.reshape(theta2, (-1,dj,dk2))
    else:
        theta1 = np.reshape(theta1, (di,dk1,dl1,-1))
        theta1 = np.transpose(theta1, (0,3,1,2))
        theta2 = np.reshape(theta2, (-1,dj,dk2,dl2))
    return theta1, theta2

def merge(theta1, theta2):
    '''
    merge two local tensors into one, the result has the following shape

            k1          k2             
            |           |
      i---theta1--//--theta2---j  
            |           |      
           (l1)        (l2)
    
                k1    k2
                |     |
          i------theta------j
                |     |
               (l1)  (l2)
    '''
    theta = np.tensordot(theta1, theta2, axes=(1,0))
    if theta1.ndim == 3:
        theta = np.swapaxes(theta, 2, 1)
    elif theta1.ndim == 4:
        theta = np.transpose(theta, (0,3,1,4,2,5)) # i,j,k1,k2,l1,l2
    return theta

def mul(A, B):
    """
    Calculate the product of a MPO and a MPS (or another MPO) by direct 
    contraction. The dimensions of the bonds will simply multiply. This 
    When B is a MPS, you should consider using apply_mpo() or zip_up() 
    instead for achieving optimzed bond dimension.

    Parameters:
        A: a MPO
        B: a MPO or a MPS

    Return:
        A x B: MPO (resp. MPS)
        
                    |k   output
                ----A----
    A x B   =       |k*, 3
                    |k , 2
                ----B----
                (   |k*  input)
    """
    
    from .mpo import MPO
    from .mps import MPS
    Os = []
    for a, b in zip(A.As, B.As): 
        a0, a1, a2, a3 = a.shape
        b0, b1, b2 = b.shape[:3]
        O = np.tensordot(a, b, axes=(3,2))
        O = np.swapaxes(O, 1, 3)
        O = np.reshape(O, (a0*b0, a2, a1*b1, -1))
        O = np.swapaxes(O, 1, 2)
        Os.append(O)
    return MPS([O[:,:,:,0] for O in Os]) if isinstance(B, MPS) else MPO(Os)

def apply_mpo(O, psi, tol:float, m_max:int, max_sweeps:int, overwrite=False):
    """
    Varationally calculate the product of a MPO and a MPS.

    Parameters:
        O: a MPO
        psi: a MPS modified in place

    Return:
        A x B: MPS
        
                    |k   output
                ----O----
    A x B   =       |k*, 3
                    |k , 2
                ---psi----
    """
    N = len(psi)
    phi = deepcopy(psi) # assume psi is only changed slightly, 
    phi.orthonormalize('right') # useful when O is a time evolution operator 
    RBT = load_right_bond_tensors(O, psi, phi) # for a small time step
    LBT = [np.ones((1,1,1))]*(N+1)
    for n in range(max_sweeps):
        for i in range(N-1): # sweep from left to right
            j = i+1
            temp = merge(psi.As[i], psi.As[j])
            # contracting left block LBT[i]
            temp = np.tensordot(LBT[i], temp, axes=(0,0))
            # contracting the MPO tensor
            temp = np.tensordot(temp, O.As[i], axes=([0,3],[0,3]))
            temp = np.tensordot(temp, O.As[j], axes=([3,2],[0,3]))
            # contracting right block RBT[j]
            temp = np.tensordot(temp, RBT[j], axes=[(1,3),(0,1)])
            temp = np.transpose(temp, (0,3,1,2))
            # split the result tensor
            phi.As[i], phi.As[j] = split(temp, 'right', tol, m_max)
            # compute the left bond tensor LBT[j]
            LBT[j] = np.tensordot(psi.As[i], LBT[i], axes=(0,0))
            LBT[j] = np.tensordot(LBT[j], O.As[i], axes=([1,2],[3,0]))
            LBT[j] = np.tensordot(LBT[j], phi.As[i].conj(), axes=([1,3],[0,2]))
        LBT[N] = np.tensordot(psi.As[-1], LBT[N-1], axes=(0,0))
        LBT[N] = np.tensordot(LBT[N], O.As[-1], axes=([1,2],[3,0]))
        LBT[N] = np.tensordot(LBT[N], phi.As[-1].conj(), axes=([1,3],[0,2]))
        for j in range(N-1,0,-1): # sweep from right to left
            i = j-1
            temp = merge(psi.As[i], psi.As[j])
            # contracting left block LBT[i]
            temp = np.tensordot(LBT[i], temp, axes=(0,0))
            # contracting the MPO tensor
            temp = np.tensordot(temp, O.As[i], axes=([0,3],[0,3]))
            temp = np.tensordot(temp, O.As[j], axes=([3,2],[0,3]))
            # contracting right block RBT[j]
            temp = np.tensordot(temp, RBT[j], axes=[(1,3),(0,1)])
            temp = np.transpose(temp, (0,3,1,2))
            # split the result tensor
            phi.As[i], phi.As[j] = split(temp, 'left', tol, m_max)
            # compute the right bond tensor RBT[i]
            RBT[i] = np.tensordot(psi.As[j], RBT[j], axes=(1,0))
            RBT[i] = np.tensordot(RBT[i], O.As[j], axes=([1,2],[3,1]))
            RBT[i] = np.tensordot(RBT[i], phi.As[j].conj(), axes=([1,3],[1,2]))
        RBT[-1] = np.tensordot(psi.As[0], RBT[0], axes=(1,0))
        RBT[-1] = np.tensordot(RBT[-1], O.As[0], axes=([1,2],[3,1]))
        RBT[-1] = np.tensordot(RBT[-1], phi.As[0].conj(), axes=([1,3],[1,2]))
        #logging.info(f'The overlap recorded in the #{n} sweep are {LBT[-1].ravel()} and {RBT[-1].ravel()}')
    if overwrite:
        psi.As = phi.As
    else:
        return phi

def zip_up(O, psi, tol):
    """
    Zip-up method for contracting a MPO with a MPS

    Parameters:
        O: MPO
        psi: MPS to be modified in place
        tol: largest discarded singular value in each truncation step
    """
    assert len(O) == len(psi)
    N = len(psi)
    psi.orthonormalize('right')
    M = np.tensordot(psi.As[0], O.As[0], axes=([0,2],[0,3]))
    M = M[None,:,:,:] # s, m, w, k
    for i in range(N-1):
        M = np.transpose(M, (0,3,1,2)) # s,k,m,w
        s, k, m, w = M.shape
        M = np.reshape(M, (s*k, m*w)) # (s,k), (m,w)
        u, svals, vt = np.linalg.svd(M, full_matrices=False)
        svals1 = svals / np.linalg.norm(svals)
        mask = svals1 > tol
        psi.As[i] = np.reshape(u[:,mask], (s, k, -1)) # s, k, s'
        psi.As[i] = np.swapaxes(psi.As[i], 1, 2)  # s, s', k
        M = np.reshape(np.diag(svals[mask]) @ vt[mask,:], (-1, m, w))
        M = np.tensordot(M, psi.As[i+1], axes=(1,0))
        M = np.tensordot(M, O.As[i+1], axes=([1,3],[0,3]))
    M = M[:,:,0,:]
    psi.As[N-1] = M / np.linalg.norm(M)

def load_right_bond_tensors(O, psi, phi):
    """
    Iteratively computing the right bond tensors resulted from the following contraction:
    R_(j-1) = ((R_j & psi_j) & O_j) & phi*_j

    Parameters:
        O: MPO
        psi: MPS (used as ket)
        phi: MPS (used as bra)

    Note: We adopted a different index convention to someother literatures, R_j (L_j) stands 
    for the bond tensor right (left) to the j-th site. The last element RBT[-1] (LBT[N]) 
    is the inner product <phi|O|psi>.
    """
    assert len(psi) == len(O) == len(phi)
    N = len(psi)
    RBT = [np.ones((1,1,1))] * (N+1)
    for i in range(N-1,-1,-1):
        RBT[i-1] = np.tensordot(psi.As[i], RBT[i], axes=(1,0))
        RBT[i-1] = np.tensordot(RBT[i-1], O.As[i], axes=([1,2],[2,1]))
        RBT[i-1] = np.tensordot(RBT[i-1], phi.As[i].conj(), axes=([1,3],[1,2]))
    return RBT

if __name__ == "__main__":
    pass