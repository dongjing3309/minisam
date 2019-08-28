// Optimize bundle adjustment problem by loading data from .bal file
// Learn more about .bal file at https://grail.cs.washington.edu/projects/bal/

#include <minisam/core/Eigen.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/geometry/CalibBundler.h>
#include <minisam/geometry/Sophus.h>
#include <minisam/geometry/projection.h>
#include <minisam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <minisam/slam/BALInterface.h>
#include <minisam/slam/PriorFactor.h>
#include <minisam/slam/ReprojectionFactor.h>
#include <minisam/utils/Timer.h>

#include <iostream>

using namespace std;
using namespace Eigen;
using namespace minisam;

int main(int argc, char* argv[]) {
  // load bundle adjustment data from bal file
  string filename = "../../../examples/data/problem-49-7776-pre.txt";
  if (argc == 2) {
    filename = string(argv[1]);
  } else if (argc > 2) {
    cout << "usage: ./bundle_adjustment_bal [bal_filepath]" << endl;
    return 0;
  }
  BAproblem<CalibBundler> ba_data = loadBAL(filename);

  // factor graph container
  // camera poses use keys start with 'x'
  // camera calibrations use keys start with 'c'
  // landmarks use keys start with 'l'
  FactorGraph graph;

  // add prior on first pose/camera calibration and first land
  std::shared_ptr<LossFunction> lossprior = ScaleLoss::Scale(1);
  graph.add(PriorFactor<Sophus::SE3d>(key('x', 0), ba_data.init_values.poses[0],
                                      lossprior));
  graph.add(PriorFactor<CalibBundler>(
      key('c', 0), ba_data.init_values.calibrations[0], lossprior));
  graph.add(PriorFactor<Vector3d>(key('l', 0), ba_data.init_values.lands[0],
                                  lossprior));

  // add projection factors to graph
  for (size_t i = 0; i < ba_data.measurements.size(); i++) {
    const BAmeasurement& m = ba_data.measurements[i];
    graph.add(ReprojectionBundlerFactor(key('x', m.pose_idx),
                                        key('c', m.pose_idx),
                                        key('l', m.land_idx), m.p_measured));
  }

  // initial variables
  Variables initials;
  for (size_t i = 0; i < ba_data.init_values.poses.size(); i++) {
    initials.add(key('x', i), ba_data.init_values.poses[i]);
    initials.add(key('c', i), ba_data.init_values.calibrations[i]);
  }
  for (size_t i = 0; i < ba_data.init_values.lands.size(); i++) {
    initials.add(key('l', i), ba_data.init_values.lands[i]);
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

  auto all_timer = global_timer().getTimer("Bundle adjustment all");
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
