# Optimize pose graph problem by loading data from .g2o file
# Learn more about/download .g2o file at https://lucacarlone.mit.edu/datasets/

from __future__ import print_function

from minisam import *
import numpy as np
import sys

import matplotlib
import matplotlib.pyplot as plt


# ############################## plot utils ####################################

# plot 2D pose graph
def plot2DPoseGraphResult(ax, graph, variables, color, linewidth=1):
    lines = []
    for factor in graph:
        # only plot between factor
        if factor.__class__.__name__ == BetweenFactor_SE2_.__name__:
            keys = factor.keys()
            p1 = variables.at_SE2_(keys[0]).translation()
            p2 = variables.at_SE2_(keys[1]).translation()
            lines.append([p1, p2])
    lc = matplotlib.collections.LineCollection(lines, colors=color, linewidths=linewidth)
    ax.add_collection(lc)
    plt.axis('equal')


# ################################# example ####################################

# pose graph data .g2o file
filename = "../data/input_M3500_g2o.g2o";

if len(sys.argv) == 2:
    filename = sys.argv[1]
elif len(sys.argv) > 2:
    print("usage: pose_graph_g2o.py [g2o_filepath]")
    exit(0)

# factor graph and initiali values are loaded from .g2o file
graph = FactorGraph()
initials = Variables()

# load .g2o file
file3d = loadG2O(filename, graph, initials);

# add a prior factor to first pose to fix the whole system
lossprior = ScaleLoss.Scale(1.0)
graph.add(PriorFactor(key('x', 0), initials.at(key('x', 0)), lossprior));

"""
  Choose an solver from
  CHOLESKY,              // Eigen Direct LDLt factorization
  CHOLMOD,               // SuiteSparse CHOLMOD
  QR,                    // SuiteSparse SPQR
  CG,                    // Eigen Classical Conjugate Gradient Method
  CUDA_CHOLESKY,         // cuSolverSP Cholesky factorization
"""

# optimize by LM
opt_param = LevenbergMarquardtOptimizerParams()
opt_param.linear_solver_type = LinearSolverType.CHOLESKY
opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel.SUBITERATION
opt = LevenbergMarquardtOptimizer(opt_param)

all_timer = global_timer().getTimer("Pose graph all")
all_timer.tic()

results = Variables()
status = opt.optimize(graph, initials, results)

all_timer.toc()

if status != NonlinearOptimizationStatus.SUCCESS:
    print("optimization error: ", status)

global_timer().print()

# plot (only in 2D)
if not file3d:
    fig, ax = plt.subplots()
    plot2DPoseGraphResult(ax, graph, initials, 'r', linewidth=1)
    plot2DPoseGraphResult(ax, graph, results, 'b', linewidth=1)
    ax.set_title('Pose graph, blue is optimized and red is non-optimized')
    plt.show()
