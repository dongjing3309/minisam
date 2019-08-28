# example to test toy SE2 pose graph example 

from __future__ import print_function

from minisam import *
from minisam.sophus import *
import numpy as np

from math import pi


# A simple 2D pose-graph SLAM
# The robot moves from x1 to x5, with odometry information between each pair. 
# the robot moves 5 each step, and makes 90 deg right turns at x3 - x5
# At x5, there is a *loop closure* between x2 is avaible
# The graph strcuture is shown:
# 
#   p-x1 - x2 - x3
#          |    |
#          x5 - x4 


# graph
graph = FactorGraph()

# prior
lossp = DiagonalLoss.Sigmas(np.array([1.0, 1.0, 0.1]))

graph.add(PriorFactor(key('x', 1), SE2(sophus.SO2(0), np.array([0, 0])), lossp))

# between
lossb = DiagonalLoss.Sigmas(np.array([0.5, 0.5, 0.1]))

graph.add(BetweenFactor(key('x', 1), key('x', 2), sophus.SE2(sophus.SO2(0), np.array([5, 0])), lossb))
graph.add(BetweenFactor(key('x', 2), key('x', 3), sophus.SE2(sophus.SO2(-pi/2), np.array([5, 0])), lossb))
graph.add(BetweenFactor(key('x', 3), key('x', 4), sophus.SE2(sophus.SO2(-pi/2), np.array([5, 0])), lossb))
graph.add(BetweenFactor(key('x', 4), key('x', 5), sophus.SE2(sophus.SO2(-pi/2), np.array([5, 0])), lossb))

# loop closure
lossl = DiagonalLoss.Sigmas(np.array([0.5, 0.5, 0.1]))

graph.add(BetweenFactor(key('x', 5), key('x', 2), sophus.SE2(sophus.SO2(-pi/2), np.array([5, 0])), lossl))

print(graph)


# init values
init_values = Variables()

init_values.add(key('x', 1), sophus.SE2(sophus.SO2(0.2), np.array([0.2, -0.3])))
init_values.add(key('x', 2), sophus.SE2(sophus.SO2(-0.1), np.array([5.1, 0.3])))
init_values.add(key('x', 3), sophus.SE2(sophus.SO2(-pi/2 - 0.2), np.array([9.9, -0.1])))
init_values.add(key('x', 4), sophus.SE2(sophus.SO2(-pi + 0.1), np.array([10.2, -5.0])))
init_values.add(key('x', 5), sophus.SE2(sophus.SO2(pi/2 - 0.1), np.array([5.1, -5.1])))

print(init_values)


# opt
opt = GaussNewtonOptimizer()
values = Variables()

status = opt.optimize(graph, init_values, values)

if status != NonlinearOptimizationStatus.SUCCESS:
    print("optimization error")
    print(status)

print("opt values :")
print(values)
print("iter =", opt.iterations())
