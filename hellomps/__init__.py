#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__='Xianrui Yin'

import os

cpu_deploy = input('number of threads to use:') or '4'

os.environ["OMP_NUM_THREADS"] = cpu_deploy
os.environ["NUMEXPR_NUM_THREADS"] = cpu_deploy
os.environ["OPENBLAS_NUM_THREADS"] = cpu_deploy