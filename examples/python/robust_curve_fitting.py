"""
Robust curve fitting example

In this example we fit a curve defined by
    y = exp(m * x + c)
The error function is
    \sum_i f(|y_i - exp(m * x_i + c)|^2)
The loss function f can be either identity or robust Cauchy loss function
"""

from __future__ import print_function

from minisam import *
import numpy as np
import math

import matplotlib.pyplot as plt


# ########################## plot and loading utils ############################

# load x-y value pairs
def loadFromFile(filename):
    data = []
    f = open(filename, "r")
    for line in f:
        arr = [float(num) for num in line.split(' ')]
        data.append(np.array(arr))
    return data

# plot raw x-y points
def plotPoints(data, color, r=10):
    xs, ys = [], []
    for d in data:
        xs.append(d[0])
        ys.append(d[1])
    plt.scatter(xs, ys, s=r, facecolors='none', edgecolors=color, marker='o')

# plot fitted exp curve
# y = exp(m * x + c);
def plotExpCurve(m, c, x_start, x_end, color, linewidth=1):
    x_step = 0.05
    xs, ys = [], []
    for x in np.arange(x_start, x_end, x_step):
        xs.append(x)
        ys.append(math.exp(m * x + c))
    plt.plot(xs, ys, color, linewidth)


# ################################## factor ####################################

# exp curve fitting factor
class ExpCurveFittingFactor(Factor):
    # ctor
    def __init__(self, key, point, loss):
        Factor.__init__(self, 1, [key], loss)
        self.p_ = point

    # make a deep copy
    def copy(self):
        return ExpCurveFittingFactor(self.keys()[0], self.p_, self.lossFunction())

    # error = y - exp(m * x + c);
    def error(self, variables):
        params = variables.at(self.keys()[0])
        return np.array([self.p_[1] - math.exp(params[0] * self.p_[0] + params[1])])

    # jacobians
    def jacobians(self, variables):
        params = variables.at(self.keys()[0])
        J_e_mc = np.array([[-self.p_[0] * math.exp(params[0] * self.p_[0] + params[1]), 
            -math.exp(params[0] * self.p_[0] + params[1])]])
        return [J_e_mc]


# ################################# example ####################################

# load data
data = loadFromFile("../data/exp_curve_fitting_data.txt")

# use robust (Cauchy) loss function or not
useRobustLoss = True
if useRobustLoss:
    loss = CauchyLoss.Cauchy(1.0)
else:
    loss = None

# build graph
graph = FactorGraph()
for d in data:
    graph.add(ExpCurveFittingFactor(key('p', 0), d, loss))

# init estimation of curve parameters
init_values = Variables()
init_values.add(key('p', 0), np.array([0, 0]))

print("initial curve parameters :", init_values.at(key('p', 0)))

# Use LM method optimizes the initial values
opt_param = LevenbergMarquardtOptimizerParams()
opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel.ITERATION
opt = LevenbergMarquardtOptimizer(opt_param)

values = Variables()
status = opt.optimize(graph, init_values, values)

if status != NonlinearOptimizationStatus.SUCCESS:
    print("optimization error: ", status)

print("opitmized curve parameters :", values.at(key('p', 0)))

# plot
fig, ax = plt.subplots()

plotPoints(data, 'b')
plotExpCurve(0.3, 0.1, data[0][0], data[-1][0], 'k')
params_opt = values.at(key('p', 0))
plotExpCurve(params_opt[0], params_opt[1], data[0][0], data[-1][0], 'r')

ax.set_title('red is optimized and black is ground truth')
plt.axis('equal')
plt.show()
