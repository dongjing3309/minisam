// test BA

#include <minisam/slam/PriorFactor.h>
#include <minisam/slam/ReprojectionFactor.h>
#include <minisam/slam/BALInterface.h>

#include <minisam/nonlinear/GaussNewtonOptimizer.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/Variables.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/SchurComplement.h>

#include <minisam/core/Eigen.h>
#include <minisam/geometry/Sophus.h>
#include <minisam/geometry/CalibKD.h>
#include <minisam/geometry/CalibK.h>
#include <minisam/geometry/projection.h>

#include <minisam/utils/Timer.h>
#include <random>

using namespace std;
using namespace Eigen;
using namespace minisam;



int main() {

  // ===========================================================
  // a simple BA dataset
  CalibK cfixed(100, 100, 300, 200);
  //CalibK cfixed(500, 500, 300, 200);

  BAdataset<CalibK> ba_ground_truth;

  // poses
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*0.0), Vector3d(0, 0, -10)));
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*0.25), Vector3d(-10, 0, -10)));
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*0.5), Vector3d(-10, 0, 0)));
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*0.75), Vector3d(-10, 0, 10)));
  //ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*1.0), Vector3d(0, 0, 10)));
  //ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*1.25), Vector3d(10, 0, 10)));
  //ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*1.5), Vector3d(10, 0, 0)));
  //ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*1.75), Vector3d(10, 0, -10)));
  
  for (size_t i = 0; i < ba_ground_truth.poses.size(); i++)
    ba_ground_truth.calibrations.push_back(cfixed);

  // land at grid
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      ba_ground_truth.lands.push_back(Vector3d(1.0*(i-1), 2.0, 1.0*(j-1)));
      ba_ground_truth.lands.push_back(Vector3d(1.0*(i-1), -2.0, 1.0*(j-1)));
    }
  }

  // ===========================================================
  // synthetic BA

  double init_pose_rot_noise = 0.1, init_pose_trans_noise = 0.5;
  double init_land_noise = 0.5, image_noise = 0.0;

  BAproblem<CalibK> ba_data = syntheticBA(ba_ground_truth, init_pose_rot_noise, 
      init_pose_trans_noise, init_land_noise, image_noise);

  // ===========================================================
  // BA optimization problem

  // values
  Variables init_values;
  FactorGraph graph;

  // init pose values
  for (size_t i = 0; i < ba_data.init_values.poses.size(); i++) {
    init_values.add(key('x', i), ba_data.init_values.poses[i]);
    init_values.add(key('c', i), ba_data.init_values.calibrations[i]);
    graph.add(PriorFactor<CalibK>(key('c', i), ba_ground_truth.calibrations[i], nullptr));
  }
  // init land values
  for (size_t i = 0; i < ba_data.init_values.lands.size(); i++) {
    init_values.add(key('l', i), ba_data.init_values.lands[i]);
  }

  // init_values.print();

  // prior on first two pose
  std::shared_ptr<LossFunction> lossprior6 = DiagonalLoss::Sigmas(
      (Eigen::VectorXd(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
  std::shared_ptr<LossFunction> lossprior3 = DiagonalLoss::Sigmas(
      (Eigen::VectorXd(3) << 0.1, 0.1, 0.1).finished());

  graph.add(PriorFactor<Sophus::SE3d>(key('x', 0), ba_ground_truth.poses[0], lossprior6));
  graph.add(PriorFactor<Sophus::SE3d>(key('x', 1), ba_ground_truth.poses[1], lossprior6));
  //graph.add(PriorFactor<Vector3d>(key('l', 0), ba_ground_truth.lands[0], lossprior3));
  

  // projection factor
  for (size_t i = 0; i < ba_data.measurements.size(); i++) {
    const BAmeasurement& m = ba_data.measurements[i];

    graph.add(ReprojectionFactor<CalibK>(key('x', m.pose_idx), key('c', m.pose_idx), 
        key('l', m.land_idx), m.p_measured));

    // graph.add(ReprojectionPoseFactor<CalibK>(key('x', m.pose_idx), key('l', m.land_idx), 
    //     shared_ptr<CalibK>(new CalibK(ba_data.init_values.calibrations[m.pose_idx])), 
    //     m.p_measured));
  }

  // graph.print();


  // ===========================================================
  // opt
  GaussNewtonOptimizerParams opt_param;
  // Choose an solver from CHOLESKY, QR, CG, LSCG, CUDA_QR, CUDA_CHOLESKY, SCHUR_DENSE_CHOLESKY
  opt_param.linear_solver_type = LinearSolverType::QR;
  opt_param.max_iterations = 1;
  opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel::SUBITERATION;

  // schur complement
  VariablesToEliminate eliminate_variables;
  //eliminate_variables.eliminate('l');

  // optimize!
  GaussNewtonOptimizer opt(opt_param);
  Variables values;

  NonlinearOptimizationStatus status = opt.optimize(graph, init_values, values,
      eliminate_variables);

  if (status != NonlinearOptimizationStatus::SUCCESS) {
    cout << "optimization error" << endl;
  }

  cout << "opt values :" << endl;
  values.print();
  cout << "iter = " << opt.iterations() << endl;

  global_timer().print();

  return 0;
}
