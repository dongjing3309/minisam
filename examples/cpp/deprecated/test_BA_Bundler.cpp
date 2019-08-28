// test BA

#include <minisam/slam/PriorFactor.h>
#include <minisam/slam/ReprojectionFactor.h>
#include <minisam/slam/BALInterface.h>

#include <minisam/nonlinear/GaussNewtonOptimizer.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/Variables.h>
#include <minisam/core/LossFunction.h>

#include <minisam/core/Eigen.h>
#include <minisam/geometry/Sophus.h>
#include <minisam/geometry/CalibBundler.h>
#include <minisam/geometry/projection.h>

#include <random>

using namespace std;
using namespace Eigen;
using namespace minisam;



int main() {

  // ===========================================================
  // a simple BA dataset
  CalibBundler cfixed(100, -0.001, 0.0001);

  BAdataset<CalibBundler> ba_ground_truth;

  // poses
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*0.0), Vector3d(0, 0, -10)).inverse());
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*0.25), Vector3d(-10, 0, -10)).inverse());
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*0.5), Vector3d(-10, 0, 0)).inverse());
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*0.75), Vector3d(-10, 0, 10)).inverse());
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*1.0), Vector3d(0, 0, 10)).inverse());
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*1.25), Vector3d(10, 0, 10)).inverse());
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*1.5), Vector3d(10, 0, 0)).inverse());
  ba_ground_truth.poses.push_back(Sophus::SE3d(Sophus::SO3d::rotY(M_PI*1.75), Vector3d(10, 0, -10)).inverse());
  
  for (size_t i = 0; i < ba_ground_truth.poses.size(); i++)
    ba_ground_truth.calibrations.push_back(cfixed);

  // land at grid
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      ba_ground_truth.lands.push_back(Vector3d(1.0*(i-2), 3.0, 1.0*(j-2)));
      ba_ground_truth.lands.push_back(Vector3d(1.0*(i-2), -3.0, 1.0*(j-2)));
    }
  }

  // ===========================================================
  // synthetic BA

  double init_pose_rot_noise = 0.1, init_pose_trans_noise = 0.1;
  double init_land_noise = 0.5, image_noise = 0.0;

  BAproblem<CalibBundler> ba_data = syntheticBundlerBA(ba_ground_truth, init_pose_rot_noise, 
      init_pose_trans_noise, init_land_noise, image_noise);

  // ===========================================================
  // BA optimization problem

  // values
  Variables init_values;

  // init pose values
  for (size_t i = 0; i < ba_data.init_values.poses.size(); i++) {
    // note that OpenGL model is inverse
    init_values.add(key('x', i), ba_data.init_values.poses[i]);
    init_values.add(key('c', i), ba_data.init_values.calibrations[i]);
  }
  // init land values
  for (size_t i = 0; i < ba_data.init_values.lands.size(); i++) {
    init_values.add(key('l', i), ba_data.init_values.lands[i]);
  }

  init_values.print();

  // factors
  FactorGraph graph;

  // prior on first two pose
  std::shared_ptr<LossFunction> lossprior6 = DiagonalLoss::Sigmas(
      (Eigen::VectorXd(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
  std::shared_ptr<LossFunction> lossprior3 = DiagonalLoss::Sigmas(
      (Eigen::VectorXd(3) << 0.1, 0.1, 0.1).finished());

  graph.add(PriorFactor<Sophus::SE3d>(key('x', 0), ba_ground_truth.poses[0], lossprior6));
  graph.add(PriorFactor<CalibBundler>(key('c', 0), ba_ground_truth.calibrations[0], lossprior3));
  graph.add(PriorFactor<Vector3d>(key('l', 0), ba_ground_truth.lands[0], lossprior3));

  // projection factor
  for (size_t i = 0; i < ba_data.measurements.size(); i++) {
    const BAmeasurement& m = ba_data.measurements[i];
    graph.add(ReprojectionBundlerFactor(key('x', m.pose_idx), key('c', m.pose_idx), 
        key('l', m.land_idx), m.p_measured));
  }

  graph.print();


  // ===========================================================
  // opt
  GaussNewtonOptimizerParams opt_param;
  // Choose an solver from CHOLESKY, QR, CG, LSCG, CUDA_QR, CUDA_CHOLESKY
  opt_param.linear_solver_type = LinearSolverType::CHOLESKY;

  // optimize!
  GaussNewtonOptimizer opt(opt_param);
  Variables values;

  NonlinearOptimizationStatus status = opt.optimize(graph, init_values, values);

  if (status != NonlinearOptimizationStatus::SUCCESS) {
    cout << "optimization error" << endl;
    return 1;
  }

  cout << "opt values :" << endl;
  values.print();
  cout << "iter = " << opt.iterations() << endl;

  return 0;
}
