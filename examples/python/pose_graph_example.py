"""
A simple 2D pose-graph SLAM
The robot moves from x1 to x5, with odometry information between each pair. 
the robot moves 5 each step, and makes 90 deg right turns at x3 - x5
At x5, there is a *loop closure* between x2 is avaible
The graph strcuture is shown:

 p-x1 - x2 - x3
        |    |
        x5 - x4 
"""

from __future__ import print_function

from minisam import *
from minisam.sophus import *
import numpy as np
import math

import matplotlib.pyplot as plt


# ############################## plot utils ####################################

# plot SE2 with covariance
def plotSE2WithCov(pose, cov, vehicle_size=0.5, line_color='k', vehicle_color='r'):
    # plot vehicle
    p1 = pose.translation() + pose.so2() * np.array([1, 0]) * vehicle_size
    p2 = pose.translation() + pose.so2() * np.array([-0.5, -0.5]) * vehicle_size
    p3 = pose.translation() + pose.so2() * np.array([-0.5, 0.5]) * vehicle_size
    line = plt.Polygon([p1, p2, p3], closed=True, fill=True, edgecolor=line_color, facecolor=vehicle_color)
    plt.gca().add_line(line)
    # plot cov
    ps = []
    circle_count = 50
    for i in range(circle_count):
        t = float(i) / float(circle_count) * math.pi * 2.0
        cp = pose.translation() + np.matmul(cov[0:2, 0:2], np.array([math.cos(t), math.sin(t)]))
        ps.append(cp)
    line = plt.Polygon(ps, closed=True, fill=False, edgecolor=line_color)
    plt.gca().add_line(line)

# ################################# example ####################################

# factor graph container
graph = FactorGraph()

# Add a prior on the first pose, setting it to the origin
# The prior is needed to fix/align the whole trajectory at world frame
# A prior factor consists of a mean value and a loss function (covariance matrix)
priorLoss = DiagonalLoss.Sigmas(np.array([1.0, 1.0, 0.1]))
graph.add(PriorFactor(key('x', 1), SE2(SO2(0), np.array([0, 0])), priorLoss))

# odometry measurement loss function
odomLoss = DiagonalLoss.Sigmas(np.array([0.5, 0.5, 0.1]))

# Add odometry factors
# Create odometry (Between) factors between consecutive poses
# robot makes 90 deg right turns at x3 - x5
graph.add(BetweenFactor(key('x', 1), key('x', 2), SE2(SO2(0), np.array([5, 0])), odomLoss))
graph.add(BetweenFactor(key('x', 2), key('x', 3), SE2(SO2(-1.57), np.array([5, 0])), odomLoss))
graph.add(BetweenFactor(key('x', 3), key('x', 4), SE2(SO2(-1.57), np.array([5, 0])), odomLoss))
graph.add(BetweenFactor(key('x', 4), key('x', 5), SE2(SO2(-1.57), np.array([5, 0])), odomLoss))

# loop closure measurement loss function
loopLoss = DiagonalLoss.Sigmas(np.array([0.5, 0.5, 0.1]))

# Add the loop closure constraint
graph.add(BetweenFactor(key('x', 5), key('x', 2), SE2(SO2(-1.57), np.array([5, 0])), loopLoss))

print(graph, '\n')


# initial varible values for the optimization
# add random noise from ground truth values
initials = Variables()
initials.add(key('x', 1), SE2(SO2(0.2), np.array([0.2, -0.3])))
initials.add(key('x', 2), SE2(SO2(-0.1), np.array([5.1, 0.3])))
initials.add(key('x', 3), SE2(SO2(-1.57 - 0.2), np.array([9.9, -0.1])))
initials.add(key('x', 4), SE2(SO2(-3.14 + 0.1), np.array([10.2, -5.0])))
initials.add(key('x', 5), SE2( SO2(1.57 - 0.1), np.array([5.1, -5.1])))

print(initials, '\n')

# Use LM method optimizes the initial values
opt_param = LevenbergMarquardtOptimizerParams()
opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel.ITERATION
opt = LevenbergMarquardtOptimizer(opt_param)

results = Variables()
status = opt.optimize(graph, initials, results)

if status != NonlinearOptimizationStatus.SUCCESS:
    print("optimization error: ", status)

print(results, '\n')


# Calculate marginal covariances for all poses
mcov_solver = MarginalCovarianceSolver()

status = mcov_solver.initialize(graph, results)
if status != MarginalCovarianceSolverStatus.SUCCESS:
    print("maginal covariance error")
    print(status)

cov1 = mcov_solver.marginalCovariance(key('x', 1))
cov2 = mcov_solver.marginalCovariance(key('x', 2))
cov3 = mcov_solver.marginalCovariance(key('x', 3))
cov4 = mcov_solver.marginalCovariance(key('x', 4))
cov5 = mcov_solver.marginalCovariance(key('x', 5))

print('cov pose 1:', cov1)
print('cov pose 2:', cov2)
print('cov pose 3:', cov3)
print('cov pose 4:', cov4)
print('cov pose 5:', cov5)

# plot
fig, ax = plt.subplots()

plotSE2WithCov(results.at(key('x', 1)), cov1)
plotSE2WithCov(results.at(key('x', 2)), cov2)
plotSE2WithCov(results.at(key('x', 3)), cov3)
plotSE2WithCov(results.at(key('x', 4)), cov4)
plotSE2WithCov(results.at(key('x', 5)), cov5)

plt.axis('equal')
plt.show()
