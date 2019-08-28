# Optimize bundle adjustment problem by loading data from .bal file
# Learn more about .bal file at https://grail.cs.washington.edu/projects/bal/

from __future__ import print_function

from minisam import *
import numpy as np
import sys


filename = "../data/problem-49-7776-pre.txt";

if len(sys.argv) == 2:
    filename = sys.argv[1]
elif len(sys.argv) > 2:
    print("usage: bundle_adjustment_bal.py [bal_filepath]")
    exit(0)

# load bundle adjustment data from bal file
ba_data = loadBAL(filename);

# factor graph container
# camera poses use keys start with 'x'
# camera calibrations use keys start with 'c'
# landmarks use keys start with 'l'
graph = FactorGraph()

# add prior on first pose/camera calibration and first land
loss_first = ScaleLoss.Scale(1.0)

graph.add(PriorFactor(key('x', 0), ba_data.init_values.poses[0], loss_first))
graph.add(PriorFactor(key('c', 0), ba_data.init_values.calibrations[0], loss_first))
graph.add(PriorFactor(key('l', 0), ba_data.init_values.lands[0], loss_first))

# add projection factors to graph
for m in ba_data.measurements:
    graph.add(ReprojectionBundlerFactor(key('x', m.pose_idx), key('c', m.pose_idx), 
        key('l', m.land_idx), m.p_measured, None))

# initial variables
initials = Variables()

for i in range(len(ba_data.init_values.poses)):
    initials.add(key('x', i), ba_data.init_values.poses[i])
    initials.add(key('c', i), ba_data.init_values.calibrations[i])

i = 0
for land in ba_data.init_values.lands:
    initials.add(key('l', i), land)
    i = i + 1

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

timer = global_timer().getTimer("Bundle adjustment all")
timer.tic()

results = Variables()
status = opt.optimize(graph, initials, results)

timer.toc()

if status != NonlinearOptimizationStatus.SUCCESS:
    print("optimization error", status)

global_timer().print()
