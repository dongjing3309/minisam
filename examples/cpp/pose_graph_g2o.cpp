// Optimize pose graph problem by loading data from .g2o file
// Learn more about/download .g2o file at https://lucacarlone.mit.edu/datasets/

#include <minisam/core/FactorGraph.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/geometry/Sophus.h>
#include <minisam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <minisam/slam/PriorFactor.h>
#include <minisam/slam/g2oInterface.h>
#include <minisam/utils/Timer.h>

#include <iostream>

using namespace std;
using namespace minisam;

int main(int argc, char* argv[]) {
  // pose graph data .g2o file
  string filename = "../../../examples/data/input_M3500_g2o.g2o";
  if (argc == 2) {
    filename = string(argv[1]);
  } else if (argc > 2) {
    cout << "usage: ./pose_graph_g2o [g2o_filepath]" << endl;
    return 0;
  }

  // factor graph and initiali values are loaded from .g2o file
  FactorGraph graph;
  Variables initials;
  // load .g2o file
  const bool file3d = loadG2O(filename, graph, initials);

  // add a prior factor to first pose to fix the whole system
  std::shared_ptr<LossFunction> lossprior = ScaleLoss::Scale(1);
  if (file3d) {
    graph.add(PriorFactor<Sophus::SE3d>(
        key('x', 0), initials.at<Sophus::SE3d>(key('x', 0)), lossprior));
  } else {
    graph.add(PriorFactor<Sophus::SE2d>(
        key('x', 0), initials.at<Sophus::SE2d>(key('x', 0)), lossprior));
  }

  /**
   * Choose an solver from
   * CHOLESKY,              // Eigen Direct LDLt factorization
   * CHOLMOD,               // SuiteSparse CHOLMOD
   * QR,                    // SuiteSparse SPQR
   * CG,                    // Eigen Classical Conjugate Gradient Method
   * CUDA_CHOLESKY,         // cuSolverSP Cholesky factorization
   */

  // optimize by LM
  LevenbergMarquardtOptimizerParams params;
  params.verbosity_level = NonlinearOptimizerVerbosityLevel::SUBITERATION;
  params.linear_solver_type = LinearSolverType::CHOLESKY;
  LevenbergMarquardtOptimizer opt(params);

  auto all_timer = global_timer().getTimer("Pose graph all");
  all_timer->tic();

  Variables results;
  NonlinearOptimizationStatus status = opt.optimize(graph, initials, results);

  all_timer->toc();

  if (status != NonlinearOptimizationStatus::SUCCESS) {
    cout << "optimization error" << endl;
  }

  global_timer().print();
  return 0;
}
