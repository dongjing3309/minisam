/**
 * @file    MarginalCovariance.h
 * @brief   Class of marginal covariance given graph and linearize point
 * @author  Jing Dong
 * @date    Mar 18, 2019
 */

#pragma once

#include <minisam/linear/Covariance.h>
#include <minisam/linear/SquareRoot.h>
#include <minisam/nonlinear/SparsityPattern.h>

#include <memory>

namespace minisam {

// forward declearation
class FactorGraph;
class Variables;

/** enum of square root solver types */
enum class SquareRootSolverType {
  CHOLESKY,  // Direct LDLt factorization Eigen Implementation
};

/** return status of nonlinear optimization */
enum class MarginalCovarianceSolverStatus {
  SUCCESS = 0,      // nonlinear optimization meets converge requirement
  RANK_DEFICIENCY,  // linear system has rank deficiency
  INVALID,          // something else is wrong
};

/** marginal covariance settings */
struct MarginalCovarianceSolverParams {
  SquareRootSolverType sqr_solver_type = SquareRootSolverType::CHOLESKY;
  OrderingMethod ordering_method = OrderingMethod::AMD;
};

/** class for marginal covariance */
class MarginalCovarianceSolver {
 private:
  // settings
  MarginalCovarianceSolverParams params_;
  // initialized covariance
  std::unique_ptr<SquareRootSolver> ptr_sqr_;
  std::unique_ptr<Covariance> ptr_cov_;
  internal::LowerHessianSparsityPattern sparsity_;

 public:
  explicit MarginalCovarianceSolver(
      const MarginalCovarianceSolverParams& params =
          MarginalCovarianceSolverParams());

  ~MarginalCovarianceSolver() = default;

  // initialize with a factor graph and llinearization point
  MarginalCovarianceSolverStatus initialize(const FactorGraph& graph,
                                            const Variables& values);

  // get marginal covariance of a key
  Eigen::MatrixXd marginalCovariance(Key key) const;

  // get joint marginal covariance of a list of key
  // order of joint marginal covariance row/col is same as keys
  Eigen::MatrixXd jointMarginalCovariance(const std::vector<Key>& keys) const;
};

namespace internal {

// get variable index in ordering
// make this outside MarginalCovarianceSolver for unit test purpose
std::vector<int> getVariableIndices(
    Key key, const Ordering& ordering,
    const internal::LowerHessianSparsityPattern& sparsity);

}  // namespace internal
}  // namespace minisam
