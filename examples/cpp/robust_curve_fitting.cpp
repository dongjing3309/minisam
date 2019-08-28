/**
 * Robust curve fitting example
 *
 * In this example we fit a curve defined by
 *     y = exp(m * x + c)
 * The error function is
 *     \sum_i f(|y_i - exp(m * x_i + c)|^2)
 * The loss function f can be either identity or robust Cauchy loss function
 */

#include <minisam/core/Eigen.h>  // include when use Eigen vector in optimization
#include <minisam/core/Factor.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <fstream>
#include <iostream>

using namespace std;
using namespace minisam;

/* ************************** file loading utils **************************** */

// load x-y value pairs from file
std::vector<Eigen::Vector2d> loadFromFile(const std::string& filename) {
  ifstream file(filename.c_str(), ifstream::in);
  if (!file) {
    throw invalid_argument("ERROR: cannot load file " + filename);
  }
  std::vector<Eigen::Vector2d> points;
  string line;
  while (getline(file, line)) {
    stringstream ss(line);
    double x, y;
    ss >> x >> y;
    points.push_back(Eigen::Vector2d(x, y));
  }
  return points;
}

/* ******************************** factor ********************************** */

// exp curve fitting factor
class ExpCurveFittingFactor : public Factor {
 private:
  Eigen::Vector2d p_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ExpCurveFittingFactor(Key key, const Eigen::Vector2d& point,
                        const std::shared_ptr<LossFunction>& lossfunc)
      : Factor(1, std::vector<Key>{key}, lossfunc), p_(point) {}

  virtual ~ExpCurveFittingFactor() = default;

  // deep copy function
  std::shared_ptr<Factor> copy() const override {
    return std::shared_ptr<Factor>(new ExpCurveFittingFactor(*this));
  }

  // error function
  // clang-format off
  Eigen::VectorXd error(const Variables& values) const override {
    const Eigen::Vector2d& params = values.at<Eigen::Vector2d>(keys()[0]);
    return (Eigen::VectorXd(1) << p_(1) - std::exp(params(0) * p_(0) + params(1)))
        .finished();
  }

  // jacobians function
  std::vector<Eigen::MatrixXd> jacobians(
      const Variables& values) const override {
    const Eigen::Vector2d& params = values.at<Eigen::Vector2d>(keys()[0]);
    return std::vector<Eigen::MatrixXd>{
        (Eigen::MatrixXd(1, 2) <<
            -p_(0) * std::exp(params(0) * p_(0) + params(1)),
            -std::exp(params(0) * p_(0) + params(1)))
        .finished()};
  }
  // clang-format on
};

/* ******************************* example ********************************** */

int main(int argc, char* argv[]) {
  // load data
  string filename = "../../../examples/data/exp_curve_fitting_data.txt";
  if (argc == 2) {
    filename = string(argv[1]);
  } else if (argc > 2) {
    cout << "usage: ./robust_curve_fitting [data_filename.txt]" << endl;
    return 0;
  }
  const auto data = loadFromFile(filename);

  // loss function
  const bool useRobustLossFuction = true;
  std::shared_ptr<LossFunction> loss;
  if (useRobustLossFuction) {
    loss = CauchyLoss::Cauchy(1.0);
  } else {
    loss = nullptr;
  }

  // build graph
  FactorGraph graph;
  for (const auto& d : data) {
    graph.add(ExpCurveFittingFactor(key('p', 0), d, loss));
  }

  // init estimation of curve parameters
  Variables init_values;
  init_values.add(key('p', 0), Eigen::Vector2d(0, 0));
  cout << "initial curve parameters :" << endl
       << init_values.at<Eigen::Vector2d>(key('p', 0)) << endl;

  // optimize!
  LevenbergMarquardtOptimizerParams opt_param;
  opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel::ITERATION;
  LevenbergMarquardtOptimizer opt(opt_param);

  Variables values;
  opt.optimize(graph, init_values, values);
  cout << "opitmized curve parameters :" << endl
       << values.at<Eigen::Vector2d>(key('p', 0)) << endl;

  return 0;
}
