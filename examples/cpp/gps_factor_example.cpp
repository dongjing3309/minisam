/**
 * A simple 2D pose-graph SLAM with 'GPS' measurement
 * The robot moves from x1 to x3, with odometry information between each pair.
 * each step has an associated 'GPS' measurement by GPSPose2Factor
 * The graph strcuture is shown:
 *
 *  g1   g2   g3
 *  |    |    |
 *  x1 - x2 - x3
 *
 * The GPS factor has error function
 *     e = pose.translation() - measurement
 */

#include <minisam/core/Factor.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/geometry/Sophus.h>  // include when use Sophus types in optimization
#include <minisam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <minisam/nonlinear/MarginalCovariance.h>
#include <minisam/slam/BetweenFactor.h>

#include <iostream>

using namespace std;
using namespace minisam;

/* ******************************** factor ********************************** */

// GPS position factor
class GPSPositionFactor : public Factor {
 private:
  Eigen::Vector2d p_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GPSPositionFactor(Key key, const Eigen::Vector2d& translation,
                    const std::shared_ptr<LossFunction>& loss)
      : Factor(1, std::vector<Key>{key}, loss), p_(translation) {}
  virtual ~GPSPositionFactor() = default;

  // make a deep copy
  std::shared_ptr<Factor> copy() const override {
    return std::shared_ptr<Factor>(new GPSPositionFactor(*this));
  }

  // error = y - exp(m * x + c);
  Eigen::VectorXd error(const Variables& variables) const override {
    const Sophus::SE2d& pose = variables.at<Sophus::SE2d>(keys()[0]);
    return pose.translation() - p_;
  }

  // jacobians
  std::vector<Eigen::MatrixXd> jacobians(
      const Variables& /*values*/) const override {
    Eigen::MatrixXd J(2, 3);
    // clang-format off
        J << 1, 0, 0,
             0, 1, 0;
    // clang-format on
    return std::vector<Eigen::MatrixXd>{J};
  }

  // optional print function
  void print(std::ostream& out = std::cout) const override {
    out << "GPS Factor on SE(2) on " << keyString(keys()[0]) << std::endl;
  }
};

/* ******************************* example ********************************** */

int main() {
  // factor graph container
  FactorGraph graph;

  // odometry measurement loss function
  const std::shared_ptr<LossFunction> odomLoss = ScaleLoss::Scale(1.0);

  // Add odometry factors
  // Create odometry (Between) factors between consecutive poses
  graph.add(BetweenFactor<Sophus::SE2d>(
      key('x', 1), key('x', 2), Sophus::SE2d(0.0, Eigen::Vector2d(5, 0)),
      odomLoss));
  graph.add(BetweenFactor<Sophus::SE2d>(
      key('x', 2), key('x', 3), Sophus::SE2d(0.0, Eigen::Vector2d(5, 0)),
      odomLoss));

  // 2D 'GPS' measurement loss function, 2-dim
  const std::shared_ptr<LossFunction> gpsLoss =
      DiagonalLoss::Sigmas(Eigen::Vector2d(2.0, 2.0));

  // Add the GPS factors
  // note that there is no prior factor needed at first pose, since GPS provides
  // the global positions (and rotations given more than 1 GPS measurements)
  graph.add(GPSPositionFactor(key('x', 1), Eigen::Vector2d(0, 0), gpsLoss));
  graph.add(GPSPositionFactor(key('x', 2), Eigen::Vector2d(5, 0), gpsLoss));
  graph.add(GPSPositionFactor(key('x', 3), Eigen::Vector2d(10, 0), gpsLoss));

  graph.print();
  cout << endl;

  // initial varible values for the optimization
  // add random noise from ground truth values
  Variables initials;

  initials.add(key('x', 1), Sophus::SE2d(0.2, Eigen::Vector2d(0.2, -0.3)));
  initials.add(key('x', 2), Sophus::SE2d(-0.1, Eigen::Vector2d(5.1, 0.3)));
  initials.add(key('x', 3), Sophus::SE2d(-0.2, Eigen::Vector2d(9.9, -0.1)));

  initials.print();
  cout << endl;

  // Use LM method optimizes the initial values
  LevenbergMarquardtOptimizerParams opt_param;
  opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel::ITERATION;
  LevenbergMarquardtOptimizer opt(opt_param);

  Variables results;

  auto status = opt.optimize(graph, initials, results);
  if (status != NonlinearOptimizationStatus::SUCCESS) {
    cout << "optimization error" << endl;
  }

  results.print();
  cout << endl;

  // Calculate marginal covariances for all poses
  MarginalCovarianceSolver mcov_solver;

  auto cstatus = mcov_solver.initialize(graph, results);
  if (cstatus != MarginalCovarianceSolverStatus::SUCCESS) {
    cout << "maginal covariance error" << endl;
  }

  Eigen::Matrix3d cov1 = mcov_solver.marginalCovariance(key('x', 1));
  Eigen::Matrix3d cov2 = mcov_solver.marginalCovariance(key('x', 2));
  Eigen::Matrix3d cov3 = mcov_solver.marginalCovariance(key('x', 3));

  cout << "cov pose 1:" << cov1 << endl;
  cout << "cov pose 2:" << cov2 << endl;
  cout << "cov pose 3:" << cov3 << endl;

  return 0;
}
