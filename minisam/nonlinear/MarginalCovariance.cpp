/**
 * @file    MarginalCovariance.cpp
 * @brief   Class of marginal covariance given graph and linearize point
 * @author  Jing Dong
 * @date    Mar 18, 2019
 */

#include <minisam/nonlinear/MarginalCovariance.h>

#include <minisam/core/FactorGraph.h>
#include <minisam/core/Variables.h>
#include <minisam/linear/SquareRootCholesky.h>
#include <minisam/nonlinear/linearization.h>

namespace minisam {

/* ************************************************************************** */
MarginalCovarianceSolver::MarginalCovarianceSolver(
    const MarginalCovarianceSolverParams& params)
    : params_(params) {
  // sqr solver
  switch (params_.sqr_solver_type) {
    case SquareRootSolverType::CHOLESKY: {
      ptr_sqr_ =
          std::unique_ptr<SquareRootSolver>(new SquareRootSolverCholesky());
    } break;
    default:
      throw std::invalid_argument(
          "[MarginalCovarianceSolver] sqr solver type is wrong");
  }
}

/* ************************************************************************** */
MarginalCovarianceSolverStatus MarginalCovarianceSolver::initialize(
    const FactorGraph& graph, const Variables& values) {
  // linearize
  Eigen::SparseMatrix<double> H_low, L;
  Eigen::VectorXd Atb;  // not used
  // use default var ordering
  sparsity_ = internal::constructLowerHessianSparsity(
      graph, values, values.defaultVariableOrdering());
  internal::linearzationLowerHessian(graph, values, sparsity_, H_low, Atb);

  // solve square root information matrix
  SquareRootSolverStatus sqr_status = ptr_sqr_->initialize(H_low);
  if (sqr_status != SquareRootSolverStatus::SUCCESS) {
    if (sqr_status == SquareRootSolverStatus::RANK_DEFICIENCY) {
      return MarginalCovarianceSolverStatus::INVALID;
    }
    if (sqr_status == SquareRootSolverStatus::RANK_DEFICIENCY) {
      return MarginalCovarianceSolverStatus::INVALID;
    }
  }

  sqr_status = ptr_sqr_->solveL(H_low, L);
  if (sqr_status != SquareRootSolverStatus::SUCCESS) {
    if (sqr_status == SquareRootSolverStatus::RANK_DEFICIENCY) {
      return MarginalCovarianceSolverStatus::INVALID;
    }
    if (sqr_status == SquareRootSolverStatus::RANK_DEFICIENCY) {
      return MarginalCovarianceSolverStatus::INVALID;
    }
  }

  // init covariance
  ptr_cov_ = std::unique_ptr<Covariance>(new Covariance(L));
  return MarginalCovarianceSolverStatus::SUCCESS;
}

/* ************************************************************************** */
Eigen::MatrixXd MarginalCovarianceSolver::marginalCovariance(Key key) const {
  const std::vector<Key> key_list = {key};
  return jointMarginalCovariance(key_list);
}

/* ************************************************************************** */
Eigen::MatrixXd MarginalCovarianceSolver::jointMarginalCovariance(
    const std::vector<Key>& keys) const {
  std::vector<int> cov_idx;
  for (Key key : keys) {
    std::vector<int> cov_idx_key =
        internal::getVariableIndices(key, *(ptr_sqr_->ordering()), sparsity_);
    cov_idx.insert(cov_idx.end(), cov_idx_key.begin(), cov_idx_key.end());
  }
  return ptr_cov_->marginalCovariance(cov_idx);
}

namespace internal {

/* ************************************************************************** */
std::vector<int> getVariableIndices(
    Key key, const Ordering& ordering,
    const internal::LowerHessianSparsityPattern& sparsity) {
  size_t var_idx = sparsity.var_ordering.searchKey(key);
  int var_dim = sparsity.var_dim[var_idx];
  int var_col = sparsity.var_col[var_idx];

  std::vector<int> matidx;
  matidx.reserve(var_dim);
  for (int i = 0; i < var_dim; i++) {
    matidx.push_back(ordering.indices()(var_col + i));
  }
  return matidx;
}

}  // namespace internal
}  // namespace minisam
