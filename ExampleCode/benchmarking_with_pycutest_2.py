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
from pycutest_utils import run_STP_pycutest, run_GLD_pycutest, run_CMA_pycutest
                           run_SCOBO_pycutest, run_signOPT_pycutest


#==========================
# 
# Identify the relevant problems. Currently, we restrict to unconstrained 
# problems of dimension less than 100
#
#==========================

probs = pycutest.find_problems(constraints='U', userN=True)

probs_under_100 = []

for p in probs:
    prob = pycutest.import_problem(p)
    x0 = prob.x0
    # only want <= 100.
    if len(x0) <= 100:
        probs_under_100.append(p)
        

#==========================
# 
# Initialize arrays to contain results. 
#
#==========================

num_trials = 10
num_problems = len(probs_under_100)
       
f_evals_STP = np.zeros(num_trials, num_problems)
f_evals_GLD = np.zeros(num_trials, num_problems)
f_evals_CMA = np.zeros(num_trials, num_problems)
f_evals_SCOBO = np.zeros(num_trials, num_problems)
f_evals_signOPT = np.zeros(num_trials, num_problems)


#==========================
# 
# Run Experiment
#
#==========================

prob_number = 0
for problem in probs_under_100:
    for i in range(num_trials):
        p_invoke_ = pycutest.import_problem(problem)
        x0 = p_invoke_.x0
        print('dimension of problem: ', len(x0_invoke_))
        function_budget_ = 100  # should make this bigger?
        
        #=========================== STP ====================================#
        print('invoking STP in a loop....')
        stp_f_vals, stp_function_evals = run_STP_pycutest(p_invoke_, 
                                                          copy.copy(x0), 
                                                          function_budget_,
                                                          target_func_value)
        f_evals_STP(i, prob_number) = stp_function_evals
        print('\n')
        
        ## Finish rewriting remaining invocations.
        # GLD.
        print('invoking GLD in a loop....')
        min2 = run_GLD_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        GLD_err_list[i].append(min2)
        print('\n')
        # SignOPT.
        print('invoking SignOPT in a loop....')
        min3 = run_signOPT_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        SignOPT_err_list[i].append(min3)
        print('\n')
        # SCOBO.
        print('invoking SCOBO in a loop....')
        min4 = run_SCOBO_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        SCOBO_err_list[i].append(min4)
        print('\n')
        # CMA.
        print('invoking CMA in a loop....')
        min5 = run_CMA_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        CMA_err_list[i].append(min5)
        print('\n')

