#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__='Xianrui Yin'

import numpy as np
from scipy import sparse

from ..networks.mpo import MPO
from ..networks.mps import MPS

__all__ = []

class BosonChain(object):
    
    def __init__(self, N:int, phy_dims:list) -> None:
        self._N = N
