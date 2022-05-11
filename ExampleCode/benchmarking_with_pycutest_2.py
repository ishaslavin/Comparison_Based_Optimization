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
from benchmark_functions import SparseQuadratic, MaxK
from oracle import Oracle, Oracle_pycutest
from pycutest_utils import run_STP_pycutest, run_GLD_pycutest, run_CMA_pycutest, \
    run_SCOBO_pycutest, run_signOPT_pycutest, ConstructProbWithGrad

import scipy.optimize as sciopt
# ==========================
# 
# Identify the relevant problems. Currently, we restrict to unconstrained 
# problems of dimension less than 100
#
# ==========================

probs = pycutest.find_problems(constraints='U', userN=True)
print('probs: ', probs)
probs = sorted(probs)

probs_under_100 = []

for p in probs:
    prob = pycutest.import_problem(p)
    print('prob: ', prob)
    x0 = prob.x0
    # only want <= 100.
    if len(x0) <= 100:
        probs_under_100.append(p)
print('probs under 100: ')
print(probs_under_100)
        

# ==========================
# 
# Initialize arrays to contain results. 
#
# ==========================

num_trials = 10
num_algs = 5
num_problems = len(probs_under_100)

EVALS = np.zeros((num_algs, num_problems, num_trials))
print('EVALS before runs: ')
print(EVALS)

# target_func_value.
"""
Daniel will complete this part.
We need a list, something like alg_target_vals = [stp_targ, gld_targ, signopt_targ, scobo_targ, cma_targ].
Each of the values in the list will have a target value corresponding to that algorithm.
Then, in the code below, after each "alg_num", we say:
    target_func_value = alg_target_vals[alg_num_stp], 
for ex.
"""
#==========================
# 
# Run Experiment
#
#==========================

print('\n')
prob_number = 0
for problem in probs_under_100:
    print('problem: ', problem)
    print('type(problem): ', type(problem))
    # TODO: problem = pycutest.import_problem(pr)
    # TODO: print('NEW problem: ', problem)
    '''
    # TODO: don't output the following line. it breaks (for a good reason) since it's not type 'str'. 
    #  print('NEW type(problem): ', type(problem))
    '''

    #================== Work out true minimum using scipy.optimize
    options = {"maxiter": int(1e3)}
    pr = pycutest.import_problem(problem)
    ProbWithGrad = ConstructProbWithGrad(pr)
    res = sciopt.minimize(ProbWithGrad, pr.x0, method= "BFGS", jac=True, options=options)
    target_fun_val = 1.001*res.fun # give a little leeway
    # TODO: Set max number of iters to 500*len(x0).
    #  sciopt.minimize(problem)
    for i in range(num_trials):
        p_invoke_ = pycutest.import_problem(problem)
        x0 = p_invoke_.x0
        print('dimension of problem: ', len(x0))
        function_budget_ = 100  # should make this bigger?
        
        # =========================== STP ==================================== #
        print('invoking STP in a loop....')
        alg_num_stp = 0
        stp_f_vals, stp_function_evals = run_STP_pycutest(p_invoke_,
                                                          copy.copy(x0),
                                                          function_budget_,
                                                          target_fun_val)
        EVALS[alg_num_stp][prob_number][i] = stp_function_evals
        print('\n')
        # TODO: Finish rewriting remaining invocations.
        # GLD.
        print('invoking GLD in a loop....')
        alg_num_gld = 1
        gld_f_vals, gld_function_evals = run_GLD_pycutest(p_invoke_,
                                                          copy.copy(x0),
                                                          function_budget_,
                                                          target_fun_val)
        EVALS[alg_num_gld][prob_number][i] = gld_function_evals
        '''
        min2 = run_GLD_pycutest(p_invoke_, copy.copy(x0), function_budget_)
        GLD_err_list[i].append(min2)
        '''
        print('\n')
        # SignOPT.
        print('invoking SignOPT in a loop....')
        alg_num_signopt = 2
        signopt_f_vals, signopt_function_evals = run_signOPT_pycutest(p_invoke_,
                                                                      copy.copy(x0),
                                                                      function_budget_,
                                                                      target_fun_val)
        EVALS[alg_num_signopt][prob_number][i] = signopt_function_evals
        '''
        min3 = run_signOPT_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        SignOPT_err_list[i].append(min3)
        '''
        print('\n')
        # SCOBO.
        print('invoking SCOBO in a loop....')
        alg_num_scobo = 3
        scobo_f_vals, scobo_function_evals = run_SCOBO_pycutest(p_invoke_,
                                                                copy.copy(x0),
                                                                function_budget_,
                                                                target_fun_val)
        EVALS[alg_num_scobo][prob_number][i] = signopt_function_evals
        '''
        min4 = run_SCOBO_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        SCOBO_err_list[i].append(min4)
        '''
        print('\n')
        # CMA.
        print('invoking CMA in a loop....')
        alg_num_cma = 4
        cma_f_vals, cma_function_evals = run_CMA_pycutest(p_invoke_,
                                                          copy.copy(x0),
                                                          function_budget_,
                                                          target_fun_val)
        EVALS[alg_num_cma][prob_number][i] = cma_function_evals
        '''
        min5 = run_CMA_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        CMA_err_list[i].append(min5)
        '''
        print('\n')

print('\n')
print('EVALS after runs: ')
print(EVALS)
