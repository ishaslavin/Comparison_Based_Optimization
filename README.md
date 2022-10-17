# Comparison_based_optimization
This repository contains the code accompanying the paper __Adapting Zeroth Order Algorithms for Comparison-Based Optimization__ by Isha Slavin and Daniel McKenzie. Specifically, it contains our utilities for repurposing ZOO algorithms for CBO, code for the five comparison-based optimization (CBO) algorithms considered, and code relating to hyper parameter tuning and benchmarking the algorithms against various functions. The algorithms are as follows: Stochastic Three Points (STP), GradientLess Descent (GLD), SignOPT, Sparcity - Aware Comparison Based Optimization (SCOBO), and Covariance Matrix Adaptation (CMA). The code for each algorithm is located in the Algorithms subfolder of this repo, and the code to invoke each algorithm can be found under Invocations.

In the CBO paradigm, access to the objective function f(x) is extremely limited. Specifically, it is assumed that the only way to obtain information on the function is to feed two inputs, x and y, to an oracle which tells us whether f(x) < f(y) or vice versa. This paradigm finds application in recommender systems, reinforcement learning, and fine-tuning of robotics---in general any optimization problem in which there is a human in the loop.

This project is licensed under the terms of the MIT license. If you find this code useful, please cite our work:

	@article{slavin2022adapting,
  	title={Adapting Zeroth Order Algorithms for Comparison-Based Optimization},
  	author={Slavin, Isha and McKenzie, Daniel},
  	journal={arXiv preprint arXiv:2210.05824},
  	year={2022}}

