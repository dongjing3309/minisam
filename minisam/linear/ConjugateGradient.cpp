/**
 * @file    ConjugateGradient.cpp
 * @brief   Conjugate gradient (CG) linear solvers of Ax = b
 * @author  Zhaoyang Lv, Jing Dong
 * @date    April 11, 2018
 */

#include <minisam/linear/ConjugateGradient.h>

namespace minisam {

/* ************************************************************************** */
LinearSolverStatus ConjugateGradientSolver::initialize(
    const Eigen::SparseMatrix<double>& A) {
  cg_.analyzePattern(A);
  return LinearSolverStatus::SUCCESS;
}

/* ************************************************************************** */
LinearSolverStatus ConjugateGradientSolver::solve(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
    Eigen::VectorXd& x) {
  cg_.factorize(A);

  if (cg_.info() != Eigen::Success) {
    return LinearSolverStatus::RANK_DEFICIENCY;
  }

  x = cg_.solve(b);
  return LinearSolverStatus::SUCCESS;
}

/* ************************************************************************** */
LinearSolverStatus ConjugateGradientLeastSquareSolver::initialize(
    const Eigen::SparseMatrix<double>& A) {
  cg_.analyzePattern(A);
  return LinearSolverStatus::SUCCESS;
}

/* ************************************************************************** */
LinearSolverStatus ConjugateGradientLeastSquareSolver::solve(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
    Eigen::VectorXd& x) {
  cg_.factorize(A);

  if (cg_.info() != Eigen::Success) {
    return LinearSolverStatus::RANK_DEFICIENCY;
  }

  x = cg_.solve(b);
  return LinearSolverStatus::SUCCESS;
}

}  // namespace minisam
