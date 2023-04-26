#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__='Xianrui Yin'

import numpy as np

class MPS:
    '''
    Parameters:
        As: list of local rank-3 tensors, each tensor has the following shape
                    k
                    |
                i---A---j
            i (j) is the left (right) bond leg and k is the physical leg
    Attributes:
        N: number of sites
    Methods:
        orthornormalize
    '''

    def __init__(self, As:list) -> None:
        self.As = As
        self.N = len(As)

    def __init__(self, L:int, ) -> None:
        pass

    def left_orthonormalize():
        pass

    def right_orthonormalize():
        pass

