// test Levenberg-Marquardt
// use the same factor graph in testlinearization

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/nonlinear/LevenbergMarquardtOptimizer.h>

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


class BFactor: public Factor {
private:
  double diff_;

public:
  BFactor(Key key1, Key key2, double diff, const std::shared_ptr<LossFunction>& lossfunc): 
    Factor(1, std::vector<Key>{key1, key2}, lossfunc), diff_(diff) {}
  virtual ~BFactor() {}

  std::shared_ptr<Factor> copy() const { return std::shared_ptr<Factor>(new BFactor(*this)); }

  Eigen::VectorXd error(const Variables& values) const {
    double v1 = values.at<double>(keys()[0]);
    double v2 = values.at<double>(keys()[1]);
    return (Eigen::VectorXd(1) << v2 - v1 - diff_).finished();
  }
  std::vector<Eigen::MatrixXd> jacobians(const Variables& /*values*/) const {
    return std::vector<Eigen::MatrixXd>{-Eigen::MatrixXd::Identity(1, 1), 
        Eigen::MatrixXd::Identity(1, 1)};
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
  values_prep.add<double>(2, 2.8);
  values_prep.add<double>(3, 3.3);
  values_prep.add<double>(4, 4.4);

  graph_prep.add(PFactor(0, 0.0, nullptr));
  graph_prep.add(PFactor(1, 1.0, nullptr));
  graph_prep.add(PFactor(2, 2.0, nullptr));
  graph_prep.add(PFactor(3, 3.0, nullptr));
  graph_prep.add(PFactor(4, 4.0, nullptr));

  graph_prep.add(BFactor(0, 1, 1.0, nullptr));
  graph_prep.add(BFactor(1, 2, 1.0, nullptr));
  graph_prep.add(BFactor(2, 3, 1.0, nullptr));
  graph_prep.add(BFactor(3, 4, 1.0, nullptr));

  graph_prep.add(BFactor(1, 0, -1.0, nullptr));
  graph_prep.add(BFactor(2, 1, -1.0, nullptr));

  VariableOrdering vordering({0, 1, 2, 3, 4});

  // jaocbian linearization
  jcache = internal::constructJacobianSparsity(graph_prep, values_prep, vordering);
  internal::linearzationJacobian(graph_prep, values_prep, jcache, A, b);

  // hessian linearization
  hcache = internal::constructLowerHessianSparsity(graph_prep, values_prep, vordering);
  internal::linearzationLowerHessian(graph_prep, values_prep, hcache, AtA_low, Atb);
  internal::linearzationFullHessian(graph_prep, values_prep, hcache, AtA, Atb);
}

/* ************************************************************************** */
TEST_CASE("LevenbergMarquardtUpdateDumpingHessian", "[nonlinear]") {

  Eigen::SparseMatrix<double> H_dump_exp, H_dump_act;
  H_dump_exp = AtA;
  H_dump_act = AtA;
  double diag = 12.3, diag_last = 3.4;

  for (int i = 0; i < 5; i++)
    H_dump_exp.coeffRef(i,i) += diag;
  for (int i = 0; i < 5; i++)
    H_dump_act.coeffRef(i,i) += diag_last;

  internal::updateDumpingHessian(H_dump_act, diag, diag_last);
  CHECK(assert_equal(H_dump_exp, H_dump_act));
}

/* ************************************************************************** */
TEST_CASE("LevenbergMarquardtUpdateDumpingHessianDiag", "[nonlinear]") {

  Eigen::SparseMatrix<double> H_dump_exp, H_dump_act;
  H_dump_exp = AtA;
  H_dump_act = AtA;
  double lambda = 12.3, lambda_last = 3.4;
  Eigen::VectorXd diags = AtA.diagonal();

  for (int i = 0; i < 5; i++)
    H_dump_exp.coeffRef(i,i) += lambda * diags(i);
  for (int i = 0; i < 5; i++)
    H_dump_act.coeffRef(i,i) += lambda_last * diags(i);

  internal::updateDumpingHessianDiag(H_dump_act, diags, lambda, lambda_last);
  CHECK(assert_equal(H_dump_exp, H_dump_act));

  // test consistency
  lambda = 1e-2;
  for (int i = 0; i < 100; i++) {
    internal::updateDumpingHessianDiag(H_dump_act, diags, lambda, -lambda);
    lambda = -lambda;
  }
  CHECK(assert_equal(H_dump_exp, H_dump_act));
}

/* ************************************************************************** */
TEST_CASE("LevenbergMarquardtUpdateDumpingLowerHessian", "[nonlinear]") {

  Eigen::SparseMatrix<double> H_dump_exp, H_dump_act;
  H_dump_exp = AtA_low;
  H_dump_act = AtA_low;
  double diag = 12.3, diag_last = 3.4;

  for (int i = 0; i < 5; i++)
    H_dump_exp.coeffRef(i,i) += diag;
  for (int i = 0; i < 5; i++)
    H_dump_act.coeffRef(i,i) += diag_last;

  internal::updateDumpingHessian(H_dump_act, diag, diag_last);
  CHECK(assert_equal(H_dump_exp, H_dump_act));
}

/* ************************************************************************** */
TEST_CASE("LevenbergMarquardtUpdateDumpingLowerHessianDiag", "[nonlinear]") {

  Eigen::SparseMatrix<double> H_dump_exp, H_dump_act;
  H_dump_exp = AtA_low;
  H_dump_act = AtA_low;
  double lambda = 12.3, lambda_last = 3.4;
  Eigen::VectorXd diags = AtA_low.diagonal();

  for (int i = 0; i < 5; i++)
    H_dump_exp.coeffRef(i,i) += lambda * diags(i);
  for (int i = 0; i < 5; i++)
    H_dump_act.coeffRef(i,i) += lambda_last * diags(i);

  internal::updateDumpingHessianDiag(H_dump_act, diags, lambda, lambda_last);
  CHECK(assert_equal(H_dump_exp, H_dump_act));

  // test consistency
  lambda = 1e-2;
  for (int i = 0; i < 100; i++) {
    internal::updateDumpingHessianDiag(H_dump_act, diags, lambda, -lambda);
    lambda = -lambda;
  }
  CHECK(assert_equal(H_dump_exp, H_dump_act));
}

/* ************************************************************************** */
TEST_CASE("LevenbergMarquardtDumpingJacobianAlloc", "[nonlinear]") {

  Eigen::VectorXd b_dump_exp(16), b_dump_act = b;
  Eigen::SparseMatrix<double> A_dump_exp, A_dump_act;
  A_dump_exp = A;
  A_dump_act = A;
  double diag = 12.3;

  b_dump_exp << b, Eigen::VectorXd::Zero(5);

  A_dump_exp.conservativeResize(jcache.A_rows + jcache.A_cols, jcache.A_cols);

  // empty dumping elements
  for (int i = 0; i < 5; i++)
    A_dump_exp.insert(11 + i,i) = diag;
  A_dump_exp.makeCompressed();

  // allocate space and assign (zero) element for dumping a Jacobian
  internal::allocateDumpingJacobian(A_dump_act, b_dump_act, jcache, diag);
  CHECK(assert_equal(A_dump_exp, A_dump_act));
  CHECK(assert_equal(b_dump_exp, b_dump_act));
}

/* ************************************************************************** */
TEST_CASE("LevenbergMarquardtDumpingJacobianAllocDiag", "[nonlinear]") {

  Eigen::VectorXd b_dump_exp(16), b_dump_act = b;
  Eigen::SparseMatrix<double> A_dump_exp, A_dump_act;
  A_dump_exp = A;
  A_dump_act = A;
  Eigen::VectorXd diag_vec(5);
  diag_vec << 0.1, 2.3, 3.6, 6.7, 89.9;
  double lambda = 4.5;

  b_dump_exp << b, Eigen::VectorXd::Zero(5);

  A_dump_exp.conservativeResize(jcache.A_rows + jcache.A_cols, jcache.A_cols);

  // empty dumping elements
  for (int i = 0; i < 5; i++)
    A_dump_exp.insert(11 + i,i) = lambda * diag_vec(i);
  A_dump_exp.makeCompressed();

  // allocate space and assign (zero) element for dumping a Jacobian
  internal::allocateDumpingJacobianDiag(A_dump_act, b_dump_act, jcache, lambda, diag_vec);
  CHECK(assert_equal(A_dump_exp, A_dump_act));
  CHECK(assert_equal(b_dump_exp, b_dump_act));
}

/* ************************************************************************** */
TEST_CASE("LevenbergMarquardtUpdateDumpingJacobian", "[nonlinear]") {

  Eigen::VectorXd b_dummy1 = b, b_dummy2 = b;
  Eigen::SparseMatrix<double> A_dump_exp, A_dump_act;
  A_dump_exp = A;
  A_dump_act = A;
  double diag = 12.3, diag_last = 3.4;

  internal::allocateDumpingJacobian(A_dump_exp, b_dummy1, jcache, diag);
  internal::allocateDumpingJacobian(A_dump_act, b_dummy2, jcache, diag_last);

  internal::updateDumpingJacobian(A_dump_act, diag, diag_last);
  CHECK(assert_equal(A_dump_exp, A_dump_act));
}

/* ************************************************************************** */
TEST_CASE("LevenbergMarquardtUpdateDumpingJacobianDiag", "[nonlinear]") {

  Eigen::VectorXd b_dummy1 = b, b_dummy2 = b;
  Eigen::SparseMatrix<double> A_dump_exp, A_dump_act;
  A_dump_exp = A;
  A_dump_act = A;
  double lambda = 12.3, lambda_last = 3.4;
  Eigen::VectorXd diags = AtA.diagonal();

  internal::allocateDumpingJacobianDiag(A_dump_exp, b_dummy1, jcache, lambda, diags);
  internal::allocateDumpingJacobianDiag(A_dump_act, b_dummy2, jcache, lambda_last, diags);

  internal::updateDumpingJacobianDiag(A_dump_act, diags, lambda, lambda_last);
  CHECK(assert_equal(A_dump_exp, A_dump_act));
}

/* ************************************************************************** */
TEST_CASE("LevenbergMarquardtHessianDiag", "[nonlinear]") {

  // get hessian diagonal (for dumping) from a jacobian
  Eigen::VectorXd hdiag_act = internal::hessianDiagonal(A);
  Eigen::VectorXd hdiag_exp = Eigen::SparseMatrix<double>(A.transpose() * A).diagonal();
  CHECK(assert_equal(hdiag_exp, hdiag_act));
}
