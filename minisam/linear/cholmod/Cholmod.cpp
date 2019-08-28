/**
 * @file    Cholmod.cpp
 * @brief   Wrapped SuitSparse Cholmod solver
 * @author  Zhaoyang Lv, Jing Dong
 * @date    Oct 30, 2018
 */

#include <minisam/linear/cholmod/Cholmod.h>

namespace minisam {

/* ************************************************************************** */
LinearSolverStatus CholmodSolver::initialize(
    const Eigen::SparseMatrix<double>& A) {
  chol_.analyzePattern(A);
  return LinearSolverStatus::SUCCESS;
}

/* ************************************************************************** */
LinearSolverStatus CholmodSolver::solve(const Eigen::SparseMatrix<double>& A,
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
