# invokes the GLD algorithm class.

from Algorithms.gld_optimizer import GLDOptimizer
from Algorithms.gld_optimizer import GLDOptimizer
from ExampleCode.benchmarkfunctions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ExampleCode.oracle import Oracle

# ---------
print('sample invoke.')
# GLD - FUNCTION sample invocation.
n_def = 200  # problem dimension.
s_exact = 20  # True sparsity.
noise_amp = 0.001  # noise amplitude.
# initialize objective function.
obj_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
obj_func_2 = MaxK(n_def, s_exact, noise_amp)
'''
max_function_evals = 10000
'''
max_function_evals = 500
x_0_ = np.random.rand(n_def)
print('shape of x_0_: ', len(x_0_))
R_ = 10
r_ = 1e-3

# Define the comparison oracle.
oracle = Oracle(obj_func_1)

# GLDOptimizer instance.
# def __init__(self, defined_func, x_0, R, r, function_budget).
gld1 = GLDOptimizer(oracle, obj_func_1, x_0_, R_, r_, max_function_evals)
# stp1 = GLDOptimizer(obj_func_2, x_0_, R_, r_, max_function_evals)
# step.
termination = False
prev_evals = 0
while termination is False:
    # optimization step.
    solution, func_value, termination = gld1.step()
    # print('step')
    print('current value: ', func_value[-1])
# print the solution.
print('\n')
print('solution: ', solution)
# plot the decreasing function.
plt.plot(func_value)
plt.show()
plt.close()
# log x-axis.
plt.semilogy(func_value)
plt.show()
plt.close()
# ---------
print('\n')
print('number of function vals: ', len(func_value))
print('function evaluations: ', func_value)