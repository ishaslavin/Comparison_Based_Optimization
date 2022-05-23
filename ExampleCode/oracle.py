import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from numpy import linalg as LA

# class which represents Comparison Oracle.
class Oracle:

    def __init__(self, f_x):
        # f_x is a function.
        # we have to define what that function does... right?
        self.f_x = f_x

    # calls the function f_x the class is initialized with.
    # function takes in inputs x and y.
    #   returns +1 if f(y)-f(x) > 0.
    #   returns -1 if f(y)-f(x) < 0.
    def __call__(self, x, y):
        if self.f_x(y) - self.f_x(x) < 0:
            # print(-1)
            return -1
        elif self.f_x(y) - self.f_x(x) > 0:
            # print(1)
            return 1
        else:
            return 0


# class which represents Comparison Oracle.
class Oracle_pycutest:

    def __init__(self, f_x):
        # f_x is a function.
        # we have to define what that function does... right?
        self.f_x = f_x

    # calls the function f_x the class is initialized with.
    # function takes in inputs x and y.
    #   returns +1 if f(y)-f(x) > 0.
    #   returns -1 if f(y)-f(x) < 0.
    def __call__(self, x, y):
        if self.f_x(y) - self.f_x(x) < 0:
            # print(-1)
            return -1
        elif self.f_x(y) - self.f_x(x) > 0:
            # print(1)
            return 1
        else:
            return 0


# class which represents Comparison Oracle.
# takes into account noise - runs 3 times and uses the majority response.
# also takes into account a margin of error defined by the user.
#   if the difference in function values falls into the margin of error, they are considered equal.
class Oracle_2:

    def __init__(self, f_x):
        # f_x is a function.
        # we have to define what that function does... right?
        self.f_x = f_x
        self.count_plus = 0
        self.count_minus = 0
        self.count_equal = 0
        self.list_of_outputs = []

    # calls the function f_x the class is initialized with.
    # function takes in inputs x and y.
    # input margin_of_error: (type = FLOAT) accounts for noise of Oracle.
    #   user defines what they want the margin of error to be. (ex.: 0.005.)
    #   returns +1 if f(y)-f(x) > + margin_of_error.
    #   returns -1 if f(y)-f(x) < - margin_of_error.
    #   returns 0 if |f(y)-f(x)| < + margin_of_error.
    #   returns NONE if 1, -1, 0 are all found by the Oracle.
    def __call__(self, x, y, margin_of_error):
        for i in range(3):
            if self.f_x(y) - self.f_x(x) < -1 * margin_of_error:
                self.count_minus += 1
                self.list_of_outputs.append(-1)
                # print(-1)
                # return -1
            elif self.f_x(y) - self.f_x(x) > margin_of_error:
                self.count_plus += 1
                self.list_of_outputs.append(1)
                # print(1)
                # return 1
            else:
                self.count_equal += 1
                self.list_of_outputs.append(0)
        print('\n')
        print('+: ', self.count_plus)
        print('-: ', self.count_minus)
        print('*: ', self.count_equal)
        print('\n')
        # returns....
        if self.count_minus > 1:
            return -1
        elif self.count_plus > 1:
            return 1
        elif self.count_equal > 1:
            return 0
        else:
            return None


            # return 0
