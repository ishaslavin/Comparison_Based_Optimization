# PROBLEM #1 part c - Week 6 TASKS.
# utils.

"""
util #1.
"""
# we have 3 algorithms: STP, GLD, SignOPT.
# check where they generate random direction vectors.
# STP: one randomly generated direction vector.
# GLD: one randomly generated direction vector (uniform from sphere / n).
# SignOPT: randomly generated MATRIX whose rows are the random sampling directions.


'''
testing.
'''
import numpy as np
import random

"""
x = np.random.randn(4)
y = np.random.randn(4, 1)
z = np.random.randn(1, 4)

print(x)
print(y)
print(z)

a = np.random.randn(4)
b = np.random.randn(4)
c = np.random.randn(4)
d = np.vstack((a, b, c))
list1 = [a, b, c]
print('\n')
print(d)
print('\n')
e = np.vstack((element for element in list1))
print('e:')
print(e)
print('\n')
"""

# print(np.random.randn(2, 4))

# idea: if columns=None: input just 'rows' as a parameter into the call.
# new idea: if there is only 1 ROW, then just input length_of_row as a parameter into the call.
# 4 possible options:
'''
1. original (i.e. vector of length n; all zeroes; one element is 1).
2. gaussian (np.random.randn(n) / sqrt(n)).
3. uniform from sphere (np.random.randn(n)).
4. rademacher.
'''



# _______________________________________________
'''
INPUTS....
'''
# number_of_rows: (type = INT) number of rows in random sampling matrix.
#   if number_of_rows = 1, input only length_of_row into generation,
# length_of_row: (type = INT) length of each row in random sampling matrix (i.e. number of columns).
# type_of_distribution: (type = STRING) 4 possibilities for this input....
'''
1. original (i.e. vector of length n; all zeroes; one element is 1).
2. gaussian np.random.randn(n).
3. uniform from sphere (np.random.randn(n) / norm).
4. rademacher (random n-dim vector of -1, +1).
'''
# OUTPUT: matrix whose rows are random sampling directions.

def random_sampling_directions(number_of_rows, length_of_row, type_of_distribution):
    # original.
    if type_of_distribution == 'original':
        if number_of_rows == 1:
            random_direction = random.randint(0, length_of_row - 1)
            s_k = np.zeros(length_of_row, int)
            s_k[random_direction] = 1
            output = s_k
        else:
            list_of_direction_vectors = []
            for row_index in range(number_of_rows):
                random_direction = random.randint(0, length_of_row - 1)
                s_k = np.zeros(length_of_row, int)
                s_k[random_direction] = 1
                list_of_direction_vectors.append(s_k)
            matrix_of_s_k = np.vstack((element for element in list_of_direction_vectors))
            output = matrix_of_s_k
    # gaussian. (Don't divide by sqrt(n) - do that in the methods since GLD divides by n; STP divides by sqrt(n)).
    elif type_of_distribution == 'gaussian':
        if number_of_rows == 1:
            output = np.random.randn(length_of_row)
        else:
            output = np.random.randn(number_of_rows, length_of_row)
    # uniform from sphere.
    elif type_of_distribution == 'uniform from sphere':
        if number_of_rows == 1:
            s_k = np.random.randn(length_of_row)
            s_k_norm = np.linalg.norm(s_k)
            s_k = s_k / s_k_norm
            output = s_k
        else:
            list_of_direction_vectors = []
            for row_index in range(number_of_rows):
                s_k = np.random.randn(length_of_row)
                s_k_norm = np.linalg.norm(s_k)
                s_k = s_k / s_k_norm
                list_of_direction_vectors.append(s_k)
            matrix_of_s_k = np.vstack((element for element in list_of_direction_vectors))
            output = matrix_of_s_k
    # rademacher.
    elif type_of_distribution == 'rademacher':
        if number_of_rows == 1:
            s_k = 2 * np.round(np.random.rand(length_of_row)) - 1
            # print('ORIGINAL RADEMACHER: ', s_k)
            s_k = s_k / np.sqrt(length_of_row)
            # print('NORMALIZED RADEMACHER: ', s_k)
            output = s_k
        else:
            list_of_direction_vectors = []
            for row_index in range(number_of_rows):
                s_k = 2 * np.round(np.random.rand(length_of_row)) - 1
                s_k = s_k / np.sqrt(length_of_row)
                list_of_direction_vectors.append(s_k)
            matrix_of_s_k = np.vstack((element for element in list_of_direction_vectors))
            output = matrix_of_s_k
    else:
        return ('Incorrect input given for type_of_generation. Possible inputs are: '
                'ORIGINAL, GAUSSIAN, UNIFORM FROM SPHERE, or RADEMACHER.')
    # return MATRIX of randomly sampled ROW direction vectors.
    return output


# now, test this on STP, GLD, and SignOPT.
# first, test each of those functions without this implementation.
# then, test each of these functions with this implementation.
# make sure outcome is the same.
# STP first.
#   checks out!
# Now GLD.
#   checks out!
# Lastly, SignOPT.
#   checks out!
# so this util is implemented in our algorithms.

"""
util #2.
"""
# from ExampleCode.oracle import Oracle, opt_function
from ExampleCode.oracle import Oracle
# # in this util, we will write a function that uses comparison oracle to determine the best point among 3 or more.
# this is used in GLD and STP.
#   STP compares 3, specifically.
#   GLD compares an unknown amount (specified by parameters when called).

'''
INPUT....
'''
# v_list: (type = LIST) list of x_vals we want to compare function values for.
# OUTPUT: first - list containing the argmin (i.e. only one element left); second - function_evals.
def multiple_comparisons_oracle(v_list, objective_function):
    function_evals = 0
    while len(v_list) >= 2:
        new_instance_1 = Oracle(objective_function)
        # print('0:', v_list[0])
        # print('1:', v_list[1])
        # input the first two elements of the list into the oracle.
        first_comparison = new_instance_1(v_list[0], v_list[1])
        # INCREMENT function_evals by 1.
        function_evals += 1
        # possibilities of Oracle output:
        if first_comparison == +1:
            # 0th elem is smaller.
            # remove 1st element.
            v_list.pop(1)
        elif first_comparison == -1:
            # 1st elem is smaller.
            # remove 0th element.
            v_list.pop(0)
        else:
            # function values are equal with elements 0 and 1 of list.
            # choose one at random to drop.
            rand_choice = random.choice([0, 1])
            v_list.pop(rand_choice)
    # return list of values AND function evaluations.
    return v_list, function_evals

# takes in ORACLE instead of Objective Function (the ORACLE should already be initialized with obj. func.).
def multiple_comparisons_oracle_2(v_list, oracle):
    function_evals = 0
    while len(v_list) >= 2:
        oracle_query = oracle
        # new_instance_1 = Oracle(objective_function)
        # print('0:', v_list[0])
        # print('1:', v_list[1])
        # input the first two elements of the list into the oracle.
        first_comparison = oracle_query(v_list[0], v_list[1])
        # INCREMENT function_evals by 1.
        function_evals += 1
        # possibilities of Oracle output:
        if first_comparison == +1:
            # 0th elem is smaller.
            # remove 1st element.
            v_list.pop(1)
        elif first_comparison == -1:
            # 1st elem is smaller.
            # remove 0th element.
            v_list.pop(0)
        else:
            # function values are equal with elements 0 and 1 of list.
            # choose one at random to drop.
            rand_choice = random.choice([0, 1])
            v_list.pop(rand_choice)
    # return list of values AND function evaluations.
    return v_list, function_evals

def BubbleSort(v_arr, oracle):
    '''
    Simple oracle based implementation of bubble sort
    v_arr = (num_items) x (dim) array 
    '''
    v_list = list(v_arr)
    n = v_arr.shape[0]  # number of items
    num_queries = 0  
    for i in range(n-1):
        for j in range(0,n-i-1):
            num_queries += 1 
            temp = oracle(v_list[j+1], v_list[j])
            if temp == 1:
                # Swap these two elements
                v_list[j], v_list[j+1]  = v_list[j+1], v_list[j]
                
                
    return np.array(v_list), num_queries

# ---------
# test.
n_def = 20000  # problem dimension.
s_exact = 200  # True sparsity.
noise_amp = 0.001  # noise amplitude.
# initialize objective function.
#objective_func = SparseQuadratic(n_def, s_exact, noise_amp)
x = np.random.randn(n_def)
y = np.random.randn(n_def)
z = np.random.randn(n_def)
w = np.random.randn(n_def)
#argmin, function_evaluations = multiple_comparisons_oracle([x, y, z, w], objective_func)
# argmin = argmin[0]
#print('\n')
#print('argmin: ', argmin)
#print('func evals: ', function_evaluations)
# seems to work well.
# ---------
# tested on GLD & STP - works for both.
# implemented in both algorithms.
# create the next UTIL after you complete the next part - benchmarking the functions.







