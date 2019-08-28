# example to test optimizer

from __future__ import print_function

from minisam import *
import numpy as np


# ##################################
print('=======================================')
print('optimizer param base')

params = NonlinearOptimizerParams()

print(params.max_iterations)
print(params.min_rel_err_decrease)
print(params.min_abs_err_decrease)
print(params.linear_solver_type)
print(params.verbosity_level)

params.max_iterations = 200

print(params.max_iterations)
print(params.min_rel_err_decrease)
print(params.min_abs_err_decrease)
print(params.linear_solver_type)
print(params.verbosity_level)


# ##################################
print('=======================================')
print('GN param')

params = GaussNewtonOptimizerParams()

print(params.max_iterations)
print(params.min_rel_err_decrease)
print(params.min_abs_err_decrease)
print(params.linear_solver_type)
print(params.verbosity_level)

opt = GaussNewtonOptimizer()
opt = GaussNewtonOptimizer(params)

print(opt.iterations())


# ##################################
print('=======================================')
print('LM param')

params = LevenbergMarquardtOptimizerParams()

print(params.max_iterations)
print(params.min_rel_err_decrease)
print(params.min_abs_err_decrease)
print(params.linear_solver_type)
print(params.verbosity_level)
print(params.lambda_init)
print(params.diagonal_damping)

params.linear_solver_type = LinearSolverType.CG

print(params.max_iterations)
print(params.min_rel_err_decrease)
print(params.min_abs_err_decrease)
print(params.linear_solver_type)
print(params.verbosity_level)
print(params.lambda_init)
print(params.diagonal_damping)

params.diagonal_damping = False

print(params.max_iterations)
print(params.min_rel_err_decrease)
print(params.min_abs_err_decrease)
print(params.linear_solver_type)
print(params.verbosity_level)
print(params.lambda_init)
print(params.diagonal_damping)

opt = LevenbergMarquardtOptimizer()
opt = LevenbergMarquardtOptimizer(params)

print(opt.iterations())
