// test CPU direct R solver

#include "test_common_factors.h"

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/nonlinear/MarginalCovariance.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/Factor.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/core/Eigen.h>  // optimizing Eigen

#include <Eigen/Eigenvalues> // eigen value

using namespace minisam;


using test::PFactor;
using test::BFactor;

// random symmetric matrix positive-defined
Eigen::MatrixXd randomCovMat(int rows) {
  Eigen::MatrixXd symmat;
  double min_eig;
  do {
    symmat = Eigen::MatrixXd::Random(rows, rows);
    symmat = 0.5 * (symmat + symmat.transpose()).eval();  // make symmetric
    Eigen::VectorXcd eigs_complex = Eigen::EigenSolver<Eigen::MatrixXd>(symmat, false).eigenvalues(); 
    min_eig = 1e9;
    for (int i = 0; i < rows; i++)
      if (min_eig > eigs_complex[i].real()) 
        min_eig = eigs_complex[i].real();
  } while (min_eig < 1e-3);
  return symmat;
}

/* ************************************************************************** */
TEST_CASE("MarginalCovarianceVariableIndices1", "[nonlinear]") {
  
  // problem setup: all natural ordering
  // x1(3),  x2(4),    x3(5)
  // 0,1,2,  3,4,5,6,  7,8,9,10,11
  internal::LowerHessianSparsityPattern sparisty;
  sparisty.var_col = {0, 3, 7};
  sparisty.var_dim = {3, 4, 5};
  sparisty.var_ordering.push_back(key('x', 1));
  sparisty.var_ordering.push_back(key('x', 2));
  sparisty.var_ordering.push_back(key('x', 3));

  NaturalOrdering ordering(Eigen::SparseMatrix<double>(12, 12));
  std::vector<int> var_idx_exp, var_idx_act;

  var_idx_exp = {0, 1, 2};
  var_idx_act = internal::getVariableIndices(key('x', 1), ordering, sparisty);
  CHECK(assert_equal_vector(var_idx_exp, var_idx_act));

  var_idx_exp = {3, 4, 5, 6};
  var_idx_act = internal::getVariableIndices(key('x', 2), ordering, sparisty);
  CHECK(assert_equal_vector(var_idx_exp, var_idx_act));

  var_idx_exp = {7, 8, 9, 10, 11};
  var_idx_act = internal::getVariableIndices(key('x', 3), ordering, sparisty);
  CHECK(assert_equal_vector(var_idx_exp, var_idx_act));

  CHECK_THROWS_WITH(internal::getVariableIndices(key('x', 4), ordering, sparisty), 
      "[VariableOrdering::searchKey] cannot find key x4 in variable ordering");
}

/* ************************************************************************** */
TEST_CASE("MarginalCovarianceVariableIndices2", "[nonlinear]") {
  
  // problem setup: all natural ordering
  // x3(5),       x2(4),     x1(3),
  // 4,8,10,5,2,  7,11,0,1,  3,9,6
  internal::LowerHessianSparsityPattern sparisty;
  sparisty.var_col = {0, 5, 9};
  sparisty.var_dim = {5, 4, 3};
  sparisty.var_ordering.push_back(key('x', 3));
  sparisty.var_ordering.push_back(key('x', 2));
  sparisty.var_ordering.push_back(key('x', 1));

  Eigen::VectorXi ordering_idx(12);
  ordering_idx << 4, 8, 10, 5, 2, 7, 11, 0, 1, 3, 9, 6;
  Ordering ordering(ordering_idx);
  std::vector<int> var_idx_exp, var_idx_act;

  var_idx_exp = {3, 9, 6};
  var_idx_act = internal::getVariableIndices(key('x', 1), ordering, sparisty);
  CHECK(assert_equal_vector(var_idx_exp, var_idx_act));

  var_idx_exp = {7, 11, 0, 1};
  var_idx_act = internal::getVariableIndices(key('x', 2), ordering, sparisty);
  CHECK(assert_equal_vector(var_idx_exp, var_idx_act));

  var_idx_exp = {4, 8, 10, 5, 2};
  var_idx_act = internal::getVariableIndices(key('x', 3), ordering, sparisty);
  CHECK(assert_equal_vector(var_idx_exp, var_idx_act));
}

/* ************************************************************************** */
TEST_CASE("MarginalCovariance1", "[nonlinear]") {
  MarginalCovarianceSolver mcov;

  Eigen::MatrixXd mcov_exp;
  Variables vars1, vars2;
  vars1.add(0, (Eigen::Vector3d() << 0, 0, 0).finished());
  vars2.add(0, (Eigen::Vector3d() << 3.4, 6.2, -9.1).finished());
  vars2.add(1, (Eigen::Vector3d() << 0.4, -2.2, 7.8).finished());
  
  // not well conditioned
  // TODO: not sure why not throw
  FactorGraph graph0;
  CHECK(MarginalCovarianceSolverStatus::SUCCESS != mcov.initialize(graph0, vars1));

  // single prior 1: unit
  FactorGraph graph1;
  graph1.add(PFactor<Eigen::Vector3d>(0, Eigen::Vector3d::Zero(), nullptr));
  CHECK(MarginalCovarianceSolverStatus::SUCCESS == mcov.initialize(graph1, vars1));
  CHECK(assert_equal(Eigen::MatrixXd(Eigen::MatrixXd::Identity(3,3)), 
      mcov.marginalCovariance(0)));

  // single prior 2: random
  FactorGraph graph2;
  mcov_exp = randomCovMat(3);
  graph2.add(PFactor<Eigen::Vector3d>(0, Eigen::Vector3d::Zero(), 
      GaussianLoss::Covariance(mcov_exp)));
  CHECK(MarginalCovarianceSolverStatus::SUCCESS == mcov.initialize(graph2, vars1));
  CHECK(assert_equal(mcov_exp, mcov.marginalCovariance(0)));

  // single prior 3: doubled
  FactorGraph graph3;
  mcov_exp = randomCovMat(3);
  graph3.add(PFactor<Eigen::Vector3d>(0, Eigen::Vector3d::Zero(), 
      GaussianLoss::Covariance(mcov_exp)));
  graph3.add(PFactor<Eigen::Vector3d>(0, Eigen::Vector3d::Zero(), 
      GaussianLoss::Covariance(mcov_exp)));
  CHECK(MarginalCovarianceSolverStatus::SUCCESS == mcov.initialize(graph3, vars1));
  mcov_exp = mcov_exp * 0.5;
  CHECK(assert_equal(mcov_exp, mcov.marginalCovariance(0)));

  // prior + between
  FactorGraph graph4;
  Eigen::MatrixXd pcov = randomCovMat(3);
  Eigen::MatrixXd bcov = randomCovMat(3);
  graph4.add(PFactor<Eigen::Vector3d>(0, Eigen::Vector3d::Zero(), 
      GaussianLoss::Covariance(pcov)));
  graph4.add(BFactor<Eigen::Vector3d>(0, 1, Eigen::Vector3d::Zero(), 
      GaussianLoss::Covariance(bcov)));
  CHECK(MarginalCovarianceSolverStatus::SUCCESS == mcov.initialize(graph4, vars2));
  CHECK(assert_equal(pcov, mcov.marginalCovariance(0)));
  mcov_exp = pcov + bcov;
  CHECK(assert_equal(mcov_exp, mcov.marginalCovariance(1)));
}

/* ************************************************************************** */
TEST_CASE("MarginalCovarianceJoint1", "[nonlinear]") {
  MarginalCovarianceSolver mcov;

  Eigen::MatrixXd mcov_exp;
  Variables vars1, vars2;
  vars2.add(0, (Eigen::Vector3d() << 3.4, 6.2, -9.1).finished());
  vars2.add(1, (Eigen::Vector3d() << 0.4, -2.2, 7.8).finished());

  // not well conditioned
  FactorGraph graph1;
  graph1.add(PFactor<Eigen::Vector3d>(0, Eigen::Vector3d::Zero(), nullptr));
  CHECK(MarginalCovarianceSolverStatus::SUCCESS != mcov.initialize(graph1, vars2));
  
  // double prior 1: unit
  FactorGraph graph2;
  graph2.add(PFactor<Eigen::Vector3d>(0, Eigen::Vector3d::Zero(), nullptr));
  graph2.add(PFactor<Eigen::Vector3d>(1, Eigen::Vector3d::Zero(), nullptr));
  CHECK(MarginalCovarianceSolverStatus::SUCCESS == mcov.initialize(graph2, vars2));
  CHECK(assert_equal(Eigen::MatrixXd(Eigen::MatrixXd::Identity(6,6)), 
      mcov.jointMarginalCovariance({0, 1})));
  
  // double prior 2: random
  Eigen::MatrixXd rcov1 = randomCovMat(3);
  Eigen::MatrixXd rcov2 = randomCovMat(3);
  FactorGraph graph3;
  graph3.add(PFactor<Eigen::Vector3d>(0, Eigen::Vector3d::Zero(), 
      GaussianLoss::Covariance(rcov1)));
  graph3.add(PFactor<Eigen::Vector3d>(1, Eigen::Vector3d::Zero(), 
      GaussianLoss::Covariance(rcov2)));
  CHECK(MarginalCovarianceSolverStatus::SUCCESS == mcov.initialize(graph3, vars2));

  mcov_exp = Eigen::MatrixXd::Zero(6,6);
  mcov_exp.block<3,3>(0, 0) = rcov1;
  mcov_exp.block<3,3>(3, 3) = rcov2;
  CHECK(assert_equal(mcov_exp, mcov.jointMarginalCovariance({0, 1})));
  mcov_exp.block<3,3>(3, 3) = rcov1;
  mcov_exp.block<3,3>(0, 0) = rcov2;
  CHECK(assert_equal(mcov_exp, mcov.jointMarginalCovariance({1, 0})));
}
