// test Levenberg-Marquardt
// use the same factor graph in testlinearization

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/nonlinear/DoglegOptimizer.h>

#include <minisam/nonlinear/linearization.h>
#include <minisam/nonlinear/SparsityPattern.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Factor.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/core/Key.h>
#include <minisam/core/Scalar.h>

#include <string>
#include <sstream>

using namespace std;
using namespace minisam;


// example factor for test
class PFactor: public Factor {
private:
  double prior_;

public:
  PFactor(Key key, double prior, const std::shared_ptr<LossFunction>& lossfunc): 
    Factor(1, std::vector<Key>{key}, lossfunc), prior_(prior) {}
  virtual ~PFactor() {}

  std::shared_ptr<Factor> copy() const { return std::shared_ptr<Factor>(new PFactor(*this)); }

  Eigen::VectorXd error(const Variables& values) const {
    return (Eigen::VectorXd(1) << values.at<double>(keys()[0]) - prior_).finished();
  }
  std::vector<Eigen::MatrixXd> jacobians(const Variables& /*values*/) const {
    return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Identity(1, 1)};
  }
};

// prepared graph and values
Variables values_prep;
FactorGraph graph_prep;

Eigen::SparseMatrix<double> A, AtA, AtA_low;
Eigen::VectorXd b, Atb;
internal::JacobianSparsityPattern jcache;
internal::LowerHessianSparsityPattern hcache;


/* ************************************************************************** */
TEST_CASE("LevenbergMarquardt_prep_static_values", "[nonlinear]") {
  
  values_prep.add<double>(0, 0.5);
  values_prep.add<double>(1, 1.2);

  graph_prep.add(PFactor(0, 0.0, nullptr));
  graph_prep.add(PFactor(1, 1.0, nullptr));

  VariableOrdering vordering({0, 1});

  // jaocbian linearization
  jcache = internal::constructJacobianSparsity(graph_prep, values_prep, vordering);
  internal::linearzationJacobian(graph_prep, values_prep, jcache, A, b);

  // hessian linearization
  hcache = internal::constructLowerHessianSparsity(graph_prep, values_prep, vordering);
  internal::linearzationLowerHessian(graph_prep, values_prep, hcache, AtA_low, Atb);
  internal::linearzationFullHessian(graph_prep, values_prep, hcache, AtA, Atb);
}

/* ************************************************************************** */
TEST_CASE("DoglegSteepestDescent", "[nonlinear]") {

  Eigen::VectorXd dx_sd_exp(2), dx_sd_act(2);
  double alpha, gnorm2;
  dx_sd_exp << -0.5, -0.2;

  dx_sd_act = internal::steepestDescentJacobian(A, b, gnorm2, alpha);
  CHECK(assert_equal(dx_sd_exp, dx_sd_act));
  CHECK(assert_equal(Atb.squaredNorm(), gnorm2));
  CHECK(assert_equal(dx_sd_act.norm() / Atb.norm(), alpha));

  dx_sd_act = internal::steepestDescentHessian(AtA, Atb, gnorm2, alpha);
  CHECK(assert_equal(dx_sd_exp, dx_sd_act));
  CHECK(assert_equal(Atb.squaredNorm(), gnorm2));
  CHECK(assert_equal(dx_sd_act.norm() / Atb.norm(), alpha));

  dx_sd_act = internal::steepestDescentHessian(AtA_low, Atb, gnorm2, alpha);
  CHECK(assert_equal(dx_sd_exp, dx_sd_act));
  CHECK(assert_equal(Atb.squaredNorm(), gnorm2));
  CHECK(assert_equal(dx_sd_act.norm() / Atb.norm(), alpha));
}

/* ************************************************************************** */
TEST_CASE("DoglegStep", "[nonlinear]") {

  Eigen::VectorXd dx_gn(2), dx_sd(2), dx_dl_exp(2), dx_dl_act(2);
  double r = 1.0, beta;

  // use GN
  dx_gn << 0., 0.;
  dx_sd << 0., 0.;
  dx_dl_exp = dx_gn;
  beta = internal::doglegStep(dx_gn, dx_sd, r, dx_dl_act);
  CHECK(assert_equal(dx_dl_exp, dx_dl_act));
  CHECK(beta > 1.0);

  dx_gn << 0.4, 0.3;
  dx_sd << 0.2, 1.2;
  dx_dl_exp = dx_gn;
  beta = internal::doglegStep(dx_gn, dx_sd, r, dx_dl_act);
  CHECK(assert_equal(dx_dl_exp, dx_dl_act));
  CHECK(beta > 1.0);

  // use SD
  dx_gn << 1.4, 1.3;
  dx_sd << 2.0, 0.0;
  dx_dl_exp << 1.0, 0.0;
  beta = internal::doglegStep(dx_gn, dx_sd, r, dx_dl_act);
  CHECK(assert_equal(dx_dl_exp, dx_dl_act));
  CHECK(beta < 0.0);

  dx_gn << 1.4, 1.3;
  dx_sd << 1.6, 1.2;
  dx_dl_exp << 0.8, 0.6;
  beta = internal::doglegStep(dx_gn, dx_sd, r, dx_dl_act);
  CHECK(assert_equal(dx_dl_exp, dx_dl_act));
  CHECK(beta < 0.0);

  // use blended
  dx_gn << 2.0, 0.0;
  dx_sd << 0.0, 0.0;    // zero SD
  dx_dl_exp << 1.0, 0.0;
  beta = internal::doglegStep(dx_gn, dx_sd, r, dx_dl_act);
  CHECK(assert_equal(dx_dl_exp, dx_dl_act));
  CHECK(assert_equal(0.5, beta));

  dx_gn << 0.0, 1.732050807568877;
  dx_sd << 1.0, 0.0;    // one SD
  dx_dl_exp << 0.5, 0.866025403784439;
  beta = internal::doglegStep(dx_gn, dx_sd, r, dx_dl_act);
  CHECK(assert_equal(dx_dl_exp, dx_dl_act));
  CHECK(assert_equal(0.5, beta));

  // use blended: just test norm
  dx_gn << 2.3, 4.5;
  dx_sd << 0.2, 0.5;
  r = 0.98345;
  beta = internal::doglegStep(dx_gn, dx_sd, r, dx_dl_act);
  CHECK(assert_equal(r, dx_dl_act.norm()));
}
