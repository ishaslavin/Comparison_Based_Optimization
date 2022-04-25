#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:32:45 2022

@author: danielmckenzie

Working on doing performance profiles with Pycutest.
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import copy
import pycutest
from matplotlib import pyplot as plt
from benchmarkfunctions import SparseQuadratic, MaxK
from oracle import Oracle, Oracle_pycutest
from Algorithms.stp_optimizer import STPOptimizer
from Algorithms.gld_optimizer import GLDOptimizer
from Algorithms.SignOPT2 import SignOPT
from Algorithms.scobo_optimizer import SCOBOoptimizer
from Algorithms.CMA_2 import CMA