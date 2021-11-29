# comparison_based_optimization
This public repository contains code for five comparison - based optimization algorithms, as well as invocations to run each. It also contains code relating to hyper parameter tuning and benchmarking the algorithms against various functions.

This project is licensed under the terms of the MIT license.

We studied zeroth order optimization algorithms and converged them to comparison - based optimization algorithms. This transformation changes the algorithm to assume access to the objective function f(x) is limited to the user. Specifically, it is assumed that the only way to obtain information on the function is to feed two input vectors, x and y, to an Oracle which tells us whether f(x) < f(y) or vice versa.

The algorithms are as follows: Stochastic Three Points (STP), GradientLess Descent (GLD), SignOPT, Sparcity - Aware Comparison Based Optimization (SCOBO), and Covariance Matrix Adaptation (CMA). The code for each algorithm is located in the Algorithms subfolder of this repo, and the code to invoke each algorithm can be found under Invocations. 
