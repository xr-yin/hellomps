import numpy as np

class eff_ham_mpo(object):
    '''
    effective model Hamiltonian for architecture I -- Eq.31
    '''


    '''
    --------------device parameters in units of 2*pi---------------

             qubit0----qubit1----qubit2
                    g0        g1
    '''
    def __init__(self) -> None:
        pass

    # qubit frequencies: E_1 - E_0
    omega = [5.100, 8.100, 6.200]
    # anharmonicity: (E_2 - E_1)-(E_1 - E_0)
    alpha = [-0.310, -0.235, -0.285]/2
    # effective coupling strength
    lg = [0.146, 0.164]

    # bosonic creation operator
    bt = np.diag([1., np.sqrt(2)], k=-1)
    # bosonic creation operator
    bn = bt.transpose()
    # number operator
    n = np.kron(bt, bn)
    # charge operator
    q = bt + bn
    # identity
    I = np.eye(3)
    # null matrix
    O = np.zeros((3,3))
    
    onsite = [w*n + a*n@(a-I) for w,a in zip(omega, alpha)]
    inter = [g*q for g in lg]

    W0 = np.array([[onsite[0], inter[0], I]])
    W1 = np.array([
        [I, O, O],
        [q, O, O],
        [onsite[1], inter[q], I]
    ])
    W2 = np.array([
        [I],
        [q],
        [onsite[2]]
    ])