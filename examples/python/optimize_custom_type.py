"""
An example of defining a vector space manifold R^2
In Python this is done by defining manifold-related member functions,
  1. dim() function returns manifold dimensionality, 
  2. local() and retract() functions defines the local coordinate chart.
  3. __repr__() for printing
"""

from __future__ import print_function

from minisam import *
import numpy as np


# ############################## custom type ###################################

class Point2D(object):
    # constructor
    def __init__(self, x, y):
        self.x_ = float(x)
        self.y_ = float(y)

    # print function
    def __repr__(self):
        return 'custom 2D point [' + str(self.x_) + ', ' + str(self.y_) + ']\''

    # local coordinate dimension
    def dim(self):
        return 2

    # map manifold point other to local coordinate
    def local(self, other):
        return np.array([other.x_ - self.x_, other.y_ - self.y_], dtype=np.float)

    # apply changes in local coordinate to manifold, \oplus operator
    def retract(self, vec):
        return Point2D(self.x_ + vec[0], self.y_ + vec[1])


# ################### optimizing custom 2D point type ##########################

# graph container
graph = FactorGraph()

# add a single prior on (0, 0)
graph.add(PriorFactor(key('x', 0), Point2D(0, 0), None))

print(graph, '\n')


# initial variables for optimization
initials = Variables()

# initial point value set to (2, 3)
initials.add(key('x', 0), Point2D(2, 3))

print("initial:\n", initials, '\n')


# optimize
opt_param = GaussNewtonOptimizerParams()
opt = GaussNewtonOptimizer(opt_param)

results = Variables()

status = opt.optimize(graph, initials, results)
if status != NonlinearOptimizationStatus.SUCCESS:
    print("optimization error : ", status)

print("optimized:\n", results, '\n')
