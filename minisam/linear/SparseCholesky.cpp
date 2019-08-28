/**
 * @file    SparseCholesky.cpp
 * @brief   Direct Cholesky linear solver
 * @author  Zhaoyang Lv, Jing Dong
 * @date    April 11, 2018
 */

#include <minisam/linear/SparseCholesky.h>

namespace minisam {

/* ************************************************************************** */
LinearSolverStatus SparseCholeskySolver::initialize(
    const Eigen::SparseMatrix<double>& A) {
  chol_.analyzePattern(A);
  return LinearSolverStatus::SUCCESS;
}

/* ************************************************************************** */
LinearSolverStatus SparseCholeskySolver::solve(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    Eigen::VectorXd& x) {
  chol_.factorize(A);

  if (chol_.info() != Eigen::Success) {
    if (chol_.info() == Eigen::InvalidInput) {
      return LinearSolverStatus::INVALID;
    }
    if (chol_.info() == Eigen::NumericalIssue) {
      return LinearSolverStatus::RANK_DEFICIENCY;
    }
  }

  x = chol_.solve(b);
  return LinearSolverStatus::SUCCESS;
}

}  // namespace minisam
