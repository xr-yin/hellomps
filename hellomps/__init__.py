#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__='Xianrui Yin'

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"