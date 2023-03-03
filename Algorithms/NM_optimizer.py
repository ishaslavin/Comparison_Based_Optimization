'''
Daniel McKenzie and Isha Slavin. 
March 2023
Simple, comparison-based implementation of Nelder-Mead Algorithm.
'''

import numpy as np
import copy as copy
from Algorithms.base import BaseOptimizer
from Algorithms.utils import BubbleSort

class NMOptimizer(BaseOptimizer):
    def __init__(self, oracle, query_budget, x0, step_size, function=None):
        super().__init__(oracle, query_budget, x0, function)
        self.step_size = step_size
        self.simplex = np.zeros((self.n+1, self.n))
        self.simplex[0,:] = x0
        # Hard code the constants
        self.alpha=1.0
        self.gamma=2.0
        self.rho=-0.5
        self.sigma=0.5

        ## Initialize simplex
        for i in range(self.n):
            x=copy.copy(self.x)
            x[i] = x[i] + self.step_size
            self.simplex[i+1,:] = x
        ## Sort simplex
        self.simplex, num_query = BubbleSort(self.simplex, self.oracle)
        self.queries += num_query


    def step(self):
        # Compute centroid
        centroid = np.mean(self.simplex[0:self.n,:], axis=0)
        # Reflect
        xr = centroid + self.alpha*(centroid - self.simplex[-1,:])
        test_oracle1 = self.oracle(self.simplex[0,:], xr)
        test_oracle2 = self.oracle(self.simplex[-2,:], xr)
        self.queries += 2
        if test_oracle1 > 0 and test_oracle2 < 0:
            self.simplex[-1,:] = xr
        elif test_oracle1 < 0: # expansion
            xe = centroid + self.gamma*(xr - centroid)
            test_oracle3 = self.oracle(xr, xe)
            self.queries += 1
            if test_oracle3 < 0:
                self.simplex[-1,:] = xe
            else:
                self.simplex[-1,:] = xr
        else: # contraction
            test_oracle4 = self.oracle(self.simplex[-1,:], xr)
            self.queries += 1 
            if test_oracle4 < 0:
                xc = centroid + self.rho*(self.simplex[-1,:] - centroid)
                test_oracle5 = self.oracle(self.simplex[-1,:], xc)
                self.queries += 1
                if test_oracle5 < 0:
                    self.simplex[-1,:] = xc
                else:
                    self.simplex[1:,:] = self.simplex[0,:] + self.sigma*(self.simplex[1:,:] - self.simplex[0,:])
        
        # reorder the points
        self.simplex, num_query = BubbleSort(self.simplex, self.oracle)
        self.queries += num_query

        # Book-keeping
        self.queries_hist.append(self.queries)
        self.x = self.simplex[0,:] # best so far

        if self._function is not None:
            self.f_vals.append(self._function(self.x))
            
        if self.reachedFunctionBudget(self.query_budget, self.queries):
            # if budget is reached return current iterate.
            # solution, list of all function values, termination.
            if self._function is not None:
                return self.x, self.f_vals, 'B', self.queries_hist
            else:
                return None, None, 'B', None
        return self.x, self.f_vals, False, self.queries_hist





        
