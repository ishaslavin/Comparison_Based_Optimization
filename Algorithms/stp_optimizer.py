import numpy as np
from Algorithms.base import BaseOptimizer
from Algorithms.utils import random_sampling_directions, multiple_comparisons_oracle_2


class STPOptimizer(BaseOptimizer):
    def __init__(self, oracle, query_budget, x0, m, step_size, direction_vector_type, function=None):
        super().__init__(oracle, query_budget, x0, function)
        self.direction_vector_type = direction_vector_type
        self.step_size = step_size
        self.list_of_sk = []
        self.x_vals = [x0]
        
        if self._function is not None:
        # In development, we'll have access to both the oracle and the function.
        # In practice this will not be the case.
            self.f_vals = [self._function(x0)]

    def step(self):
        sampling_direction = random_sampling_directions(1, self.n, 'gaussian')
        ## ToDo: Add support for other sampling directions.
        x_plus = self.x + self.step_size*sampling_direction
        x_minus = self.x - self.step_size*sampling_direction
  
        v_list = [self.x, x_plus, x_minus]
        if len(v_list) > 1:
            argmin, function_evaluations = multiple_comparisons_oracle_2(v_list, self.oracle)
            self.queries += function_evaluations
            self.x = argmin[0]
            self.x_vals.append(argmin[0])
            
        # *********
        if self._function is not None:
            self.f_vals.append(self._function(self.x))
            
        if self.reachedFunctionBudget(self.query_budget, self.queries):
            # if budget is reached return current iterate.
            # solution, list of all function values, termination.
            if self._function is not None:
                return self.x, self.f_vals, 'B', self.queries
            else:
                return None, None, 'B', None
        return self.x, self.f_vals, False, self.queries
