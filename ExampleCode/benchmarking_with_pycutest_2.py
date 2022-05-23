#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:32:45 2022

@author: danielmckenzie

Working on doing performance profiles with Pycutest.
"""

from __future__ import print_function
import numpy as np
import copy
import pickle
import pycutest
from ExampleCode.oracle import Oracle_pycutest
from ExampleCode.pycutest_utils import run_STP_pycutest, run_GLD_pycutest
from ExampleCode.pycutest_utils import run_CMA_pycutest, run_signOPT_pycutest
from ExampleCode.pycutest_utils import ConstructProbWithGrad  # ,run_SCOBO_pycutest

import scipy.optimize as sciopt
# ==========================
# 
# Identify the relevant problems. Currently, we restrict to unconstrained 
# problems of dimension less than 100
#
# ==========================

print('/n Finding all problems of size less than 100 /n')

probs = pycutest.find_problems(constraints='U', userN=True)

# old code (checked dimension of each PyCutest problem).
'''
probs_under_100 = []
for p in probs:
    prob = pycutest.import_problem(p)
    print('prob: ', prob)
    x0 = prob.x0
    # only want <= 100.
    if 100 >= len(x0) >= 10:
        probs_under_100.append(p)
print('probs under 100: ')
print(probs_under_100)
'''

# new code (reads in the list of problems to use from pycutest_probs_to_use.txt.
probs_under_100 = []
f = open("pycutest_probs_to_use.txt", "r")
lines = f.readlines()
for line in lines:
    print(line)
    probs_under_100.append(line.rstrip())
print('\n')
print('problems with dimension from 10 to 100: ')
print(probs_under_100)
print('\n')

# ==========================
# 
# Initialize arrays to contain results. 
#
# ==========================

num_trials = 2
num_algs = 4  # Will be tricky to run SCOBO on workstation.
num_problems = len(probs_under_100)

EVALS = np.zeros((num_algs, num_problems, num_trials))

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

prob_number = 0
for problem in probs_under_100:
    #================== Work out true minimum using scipy.optimize
    options = {"maxiter": int(1e4)}
    ProbWithGrad = ConstructProbWithGrad(problem)
    # res = sciopt.minimize(ProbWithGrad, problem.x0, method= "BFGS", jac=True, options=options)
    # target_fun_val = 1.001*res.fun # give a little leeway
    # TODO: Set max number of iters to 500*len(x0).
    #  sciopt.minimize(problem)
    p_invoke_ = pycutest.import_problem(problem)
    oracle = Oracle_pycutest(p_invoke_)
    ## CHECK where oracle should be instantiated and called
    x0 = p_invoke_.x0
    print('dimension of problem: ', len(x0))
    function_budget_ = int(1e4)  # should make this bigger?
    target_fun_val = 0.05*p_invoke_.obj(x0)
    for i in range(num_trials): 
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
        gld_f_vals, gld_function_evals = run_GLD_pycutest(oracle,
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
        signopt_f_vals, signopt_function_evals = run_signOPT_pycutest(oracle,
                                                                      copy.copy(x0),
                                                                      function_budget_,
                                                                      target_fun_val)

        print('\n')
        # CMA.
        print('invoking CMA in a loop....')
        alg_num_cma = 3
        cma_f_vals, cma_function_evals = run_CMA_pycutest(oracle,
                                                          copy.copy(x0),
                                                          function_budget_,
                                                          target_fun_val)
        EVALS[alg_num_cma][prob_number][i] = cma_function_evals
        '''
        min5 = run_CMA_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        CMA_err_list[i].append(min5)
        '''
        print('\n')
        
        #        EVALS[alg_num_signopt][prob_number][i] = signopt_function_evals
#        '''
#        min3 = run_signOPT_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
#        SignOPT_err_list[i].append(min3)
#        '''
#        print('\n')
#        # SCOBO.
#        print('invoking SCOBO in a loop....')
#        alg_num_scobo = 4
#        scobo_f_vals, scobo_function_evals = run_SCOBO_pycutest(oracle,
#                                                                copy.copy(x0),
#                                                                function_budget_,
#                                                                target_fun_val)
#        EVALS[alg_num_scobo][prob_number][i] = scobo_function_evals

        
myFile = open('Results/Comparison_Opt_May_23.p', 'wb')
results = {"Evals": EVALS,
           "target_function_param": 0.05
           }
pickle.dump(results, myFile)
myFile.close()
