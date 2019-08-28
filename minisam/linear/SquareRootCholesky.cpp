/**
 * @file    SquareRootCholesky.cpp
 * @brief   Calculate square root information matrix by Cholesky factorization
 * @author  Jing Dong
 * @date    Mar 15, 2019
 */

#include <minisam/linear/SquareRootCholesky.h>

namespace minisam {

/* ************************************************************************** */
SquareRootSolverStatus SquareRootSolverCholesky::initialize(
    const Eigen::SparseMatrix<double>& H) {
  // get ordering from base class
  SquareRootSolver::initialize(H);
  // init cholesky solver
  Eigen::SparseMatrix<double> H_reorderd(H.rows(), H.cols());
  ordering_->permuteSystemSelfAdjoint<Eigen::Lower>(H, H_reorderd);
  chol_.analyzePattern(H_reorderd);

  return SquareRootSolverStatus::SUCCESS;
}

/* ************************************************************************** */
SquareRootSolverStatus SquareRootSolverCholesky::factorize_(
    const Eigen::SparseMatrix<double>& H) {
  // apply ordering
  Eigen::SparseMatrix<double> H_reorderd(H.rows(), H.cols());
  ordering_->permuteSystemSelfAdjoint<Eigen::Lower>(H, H_reorderd);

  chol_.factorize(H_reorderd);
  if (chol_.info() != Eigen::Success) {
    if (chol_.info() == Eigen::InvalidInput) {
      return SquareRootSolverStatus::INVALID;
    }
    if (chol_.info() == Eigen::NumericalIssue) {
      return SquareRootSolverStatus::RANK_DEFICIENCY;
    }
  }
  return SquareRootSolverStatus::SUCCESS;
}

/* ************************************************************************** */
SquareRootSolverStatus SquareRootSolverCholesky::solveR(
    const Eigen::SparseMatrix<double>& H, Eigen::SparseMatrix<double>& R) {
  // factorize
  SquareRootSolverStatus factorize_status = factorize_(H);
  if (factorize_status != SquareRootSolverStatus::SUCCESS) {
    return factorize_status;
  }
  const Eigen::VectorXd diag_sqrtD = chol_.vectorD().cwiseSqrt();
  R = chol_.matrixU();
  R = diag_sqrtD.asDiagonal() * R;
  return SquareRootSolverStatus::SUCCESS;
}

/* ************************************************************************** */
SquareRootSolverStatus SquareRootSolverCholesky::solveL(
    const Eigen::SparseMatrix<double>& H, Eigen::SparseMatrix<double>& L) {
  // factorize
  SquareRootSolverStatus factorize_status = factorize_(H);
  if (factorize_status != SquareRootSolverStatus::SUCCESS) {
    return factorize_status;
  }
  const Eigen::VectorXd diag_sqrtD = chol_.vectorD().cwiseSqrt();
  L = chol_.matrixL();
  L = L * diag_sqrtD.asDiagonal();
  return SquareRootSolverStatus::SUCCESS;
}

}  // namespace minisam
