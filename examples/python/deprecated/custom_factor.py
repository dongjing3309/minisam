# example to custom factor in python

from __future__ import print_function

from minisam import *
import numpy as np

from math import pi


# python implementation of prior factor on SE2
class PyPriorFactorSE2(Factor):

    # constructor
    def __init__(self, key, prior, loss=None):
        Factor.__init__(self, 3, [key], loss)
        self.prior_ = prior

    # make a deep copy
    def copy(self):
        return PyPriorFactorSE2(self.keys()[0], self.prior_, self.lossFunction())

    # error
    def error(self, variables):
        return (self.prior_.inverse() * variables.at(self.keys()[0])).log()

    # jacobians
    def jacobians(self, variables):
        return [np.eye(3)]

    # optional print, if not exist just use default print function minisam.Factor.__repr__
    def __repr__(self):
        return 'py Prior Factor on SE2:\nprior = ' + self.prior_.__repr__() + '\n' + Factor.__repr__(self)


# python implementation of between factor on SE2
class PyBetweenFactorSE2(Factor):

    # constructor
    def __init__(self, key1, key2, diff, loss=None):
        Factor.__init__(self, 3, [key1, key2], loss)
        self.diff_ = diff

    # make a deep copy
    def copy(self):
        return PyBetweenFactorSE2(self.keys()[0], self.keys()[1], self.diff_, self.lossFunction())

    # error
    def error(self, variables):
        p1 = variables.at(self.keys()[0])
        p2 = variables.at(self.keys()[1])
        return (self.diff_.inverse() * (p1.inverse() * p2)).log()

    # jacobians
    def jacobians(self, variables):
        p1 = variables.at(self.keys()[0])
        p2 = variables.at(self.keys()[1])
        return [np.matmul(p2.inverse().Adj(), -p1.Adj()), np.eye(3)]

    # optional print, if not exist just use default print function minisam.Factor.__repr__
    def __repr__(self):
        return 'py Between Factor on SE2:\ndiff = ' + self.diff_.__repr__() + '\n' + Factor.__repr__(self)


# python implementation of numerical between factor on SE2
class PyNumBetweenFactorSE2(NumericalFactor):

    # constructor
    def __init__(self, key1, key2, diff, loss=None, delta=1e-3, numerical_type=NumericalJacobianType.RIDDERS5):
        NumericalFactor.__init__(self, 3, [key1, key2], loss, delta, numerical_type)
        self.diff_ = diff

    # make a deep copy
    def copy(self):
        return PyNumBetweenFactorSE2(self.keys()[0], self.keys()[1], self.diff_, self.lossFunction())

    # error
    def error(self, variables):
        p1 = variables.at(self.keys()[0])
        p2 = variables.at(self.keys()[1])
        return (self.diff_.inverse() * (p1.inverse() * p2)).log()

    # optional print, if not exist just use default print function minisam.NumericalFactor.__repr__
    def __repr__(self):
        return 'py numerical Between Factor on SE2:\ndiff = ' + self.diff_.__repr__() + '\n' + NumericalFactor.__repr__(self)




# ##################################
# graph
graph = FactorGraph()

# prior
lossp = DiagonalLoss.Sigmas(np.array([1.0, 1.0, 0.1]))

graph.add(PyPriorFactorSE2(key('x', 1), sophus.SE2(sophus.SO2(0), np.array([0, 0])), lossp))
# graph.add(PriorFactor(key('x', 1), sophus.SE2(sophus.SO2(0), np.array([0, 0])), lossp))

# between
lossb = DiagonalLoss.Sigmas(np.array([0.5, 0.5, 0.1]))

# graph.add(PyNumBetweenFactorSE2(key('x', 1), key('x', 2), sophus.SE2(sophus.SO2(0), np.array([5, 0])), lossb))
graph.add(PyBetweenFactorSE2(key('x', 1), key('x', 2), sophus.SE2(sophus.SO2(0), np.array([5, 0])), lossb))
# graph.add(BetweenFactor(key('x', 1), key('x', 2), sophus.SE2(sophus.SO2(0), np.array([5, 0])), lossb))
# graph.add(BetweenFactor_SE2_(key('x', 1), key('x', 2), sophus.SE2(sophus.SO2(0), np.array([5, 0])), lossb))

graph.add(PyNumBetweenFactorSE2(key('x', 2), key('x', 3), sophus.SE2(sophus.SO2(-pi/2), np.array([5, 0])), lossb))
graph.add(PyNumBetweenFactorSE2(key('x', 3), key('x', 4), sophus.SE2(sophus.SO2(-pi/2), np.array([5, 0])), lossb, 
    delta=1e-3))
graph.add(PyNumBetweenFactorSE2(key('x', 4), key('x', 5), sophus.SE2(sophus.SO2(-pi/2), np.array([5, 0])), lossb, 
    numerical_type=NumericalJacobianType.CENTRAL, delta=1e-6))

# loop closure
lossl = DiagonalLoss.Sigmas(np.array([0.5, 0.5, 0.1]))

graph.add(PyNumBetweenFactorSE2(key('x', 5), key('x', 2), sophus.SE2(sophus.SO2(-pi/2), np.array([5, 0])), lossl))

print(graph)


# ##################################
# init values
init_values = Variables()

init_values.add(key('x', 1), sophus.SE2(sophus.SO2(0.2), np.array([0.2, -0.3])))
init_values.add(key('x', 2), sophus.SE2(sophus.SO2(-0.1), np.array([5.1, 0.3])))
init_values.add(key('x', 3), sophus.SE2(sophus.SO2(-pi/2 - 0.2), np.array([9.9, -0.1])))
init_values.add(key('x', 4), sophus.SE2(sophus.SO2(-pi + 0.1), np.array([10.2, -5.0])))
init_values.add(key('x', 5), sophus.SE2(sophus.SO2(pi/2 - 0.1), np.array([5.1, -5.1])))

# print(init_values)


# ##################################
# some tests

# test type
print('type ......')
print(type(graph[0]))
print(type(graph[1]))


# test error inside
print('test error ......')

print('dim =', graph.dim())
print('err in graph =', graph.error(init_values))

print('dim0 =', graph[0].dim())
print('dim1 =', graph[1].dim())

print('err0 =', graph[0].error(init_values))
print('err1 =', graph[1].error(init_values))

print('jacobian0 =', graph[0].jacobians(init_values))
print('jacobian1 =', graph[1].jacobians(init_values))

# ##################################
# opt

# opt_param = GaussNewtonOptimizerParams()
# opt_param.linear_solver_type = LinearSolverType.CHOLESKY
# opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel.ITERATION
# opt = GaussNewtonOptimizer(opt_param)

opt_param = LevenbergMarquardtOptimizerParams()
opt_param.linear_solver_type = LinearSolverType.CHOLESKY
opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel.SUBITERATION
opt = LevenbergMarquardtOptimizer(opt_param)

# opt_param = DoglegOptimizerParams()
# opt_param.linear_solver_type = LinearSolverType.CHOLESKY
# opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel.SUBITERATION
# opt = DoglegOptimizer(opt_param)


values = Variables()

status = opt.optimize(graph, init_values, values)

if status != NonlinearOptimizationStatus.SUCCESS:
    print("optimization error")
    print(status)

print("opt values :")
print(values)
print("iter =", opt.iterations())


# ##################################
# marginal cov

mcovinit_timer = global_timer().getTimer('Marginal covariance initialize')
mcov_timer = global_timer().getTimer('Marginal covariance')

mcov_solver = MarginalCovarianceSolver()

mcovinit_timer.tic()
status = mcov_solver.initialize(graph, values)
mcovinit_timer.toc()

if status != MarginalCovarianceSolverStatus.SUCCESS:
    print("maginal covariance error")
    print(status)

mcov_timer.tic()

print("Sigma_x1 =\n", mcov_solver.marginalCovariance(key('x', 1)))
print("Sigma_x3 =\n", mcov_solver.marginalCovariance(key('x', 3)))
print("Sigma_x4 =\n", mcov_solver.marginalCovariance(key('x', 4)))
print("Sigma_x5 =\n", mcov_solver.marginalCovariance(key('x', 5)))

print("Sigma_x3_x4 =\n", mcov_solver.jointMarginalCovariance([key('x', 3), key('x', 4)]))

mcov_timer.toc()

global_timer().print()

