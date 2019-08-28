"""
A simple 2D pose-graph SLAM with 'GPS' measurement
The robot moves from x1 to x3, with odometry information between each pair. 
each step has an associated 'GPS' measurement by GPSPose2Factor
The graph strcuture is shown:

 g1   g2   g3
 |    |    |
 x1 - x2 - x3

The GPS factor has error function
    e = pose.translation() - measurement 
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


# ################################## factor ####################################

# GPS position factor
class GPSPositionFactor(Factor):
    # ctor
    def __init__(self, key, point, loss):
        Factor.__init__(self, 1, [key], loss)
        self.p_ = point

    # make a deep copy
    def copy(self):
        return GPSPositionFactor(self.keys()[0], self.p_, self.lossFunction())

    # error = y - exp(m * x + c);
    def error(self, variables):
        pose = variables.at(self.keys()[0])
        return pose.translation() - self.p_

    # jacobians
    def jacobians(self, variables):
        return [np.array([[1, 0, 0], [0, 1, 0]])]

    # optional print function
    def __repr__(self):
        return 'GPS Factor on SE(2):\nprior = ' + self.p_.__repr__() + ' on ' + keyString(self.keys()[0]) + '\n'


# ################################# example ####################################

# factor graph container
graph = FactorGraph()

# odometry measurement loss function
odomLoss = ScaleLoss.Scale(1.0)

# Add odometry factors
# Create odometry (Between) factors between consecutive poses
graph.add(BetweenFactor(key('x', 1), key('x', 2), SE2(SO2(0), np.array([5, 0])), odomLoss))
graph.add(BetweenFactor(key('x', 2), key('x', 3), SE2(SO2(0),np.array([5, 0])), odomLoss))

# 2D 'GPS' measurement loss function, 2-dim
gpsLoss = DiagonalLoss.Sigmas(np.array([2.0, 2.0]));

# Add the GPS factors
# note that there is no prior factor needed at first pose, since GPS provides
# the global positions (and rotations given more than 1 GPS measurements)
graph.add(GPSPositionFactor(key('x', 1), np.array([0, 0]), gpsLoss))
graph.add(GPSPositionFactor(key('x', 2), np.array([5, 0]), gpsLoss))
graph.add(GPSPositionFactor(key('x', 3), np.array([10, 0]), gpsLoss))

print(graph, '\n')


# initial varible values for the optimization
# add random noise from ground truth values
initials = Variables()

initials.add(key('x', 1), SE2(SO2(0.2), np.array([0.2, -0.3])))
initials.add(key('x', 2), SE2(SO2(-0.1), np.array([5.1, 0.3])))
initials.add(key('x', 3), SE2(SO2(-0.2), np.array([9.9, -0.1])))

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

print('cov pose 1:', cov1)
print('cov pose 2:', cov2)
print('cov pose 3:', cov3)

# plot
fig, ax = plt.subplots()

plotSE2WithCov(results.at(key('x', 1)), cov1)
plotSE2WithCov(results.at(key('x', 2)), cov2)
plotSE2WithCov(results.at(key('x', 3)), cov3)

plt.axis('equal')
plt.show()
