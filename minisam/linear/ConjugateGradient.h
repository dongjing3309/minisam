/**
 * @file    ConjugateGradient.h
 * @brief   Conjugate gradient (CG) linear solvers of Ax = b
 * @author  Zhaoyang Lv, Jing Dong
 * @date    April 11, 2018
 */

#pragma once

#include <minisam/linear/LinearSolver.h>

#include <Eigen/IterativeLinearSolvers>

namespace minisam {

// wrapper of Eigen::ConjugateGradient
class ConjugateGradientSolver : public SparseLinearSolver {
 private:
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower,
                           Eigen::DiagonalPreconditioner<double>>
      cg_;

 public:
  ConjugateGradientSolver() : cg_() {}
  virtual ~ConjugateGradientSolver() = default;

  LinearSolverStatus initialize(const Eigen::SparseMatrix<double>& A) override;

  LinearSolverStatus solve(const Eigen::SparseMatrix<double>& A,
                           const Eigen::VectorXd& b,
                           Eigen::VectorXd& x) override;

  // solver properties
  bool is_normal() const override { return true; }
  bool is_normal_lower() const override { return true; }
};

// wrapper of Eigen::LeastSquaresConjugateGradient
class ConjugateGradientLeastSquareSolver : public SparseLinearSolver {
 private:
  Eigen::LeastSquaresConjugateGradient<
      Eigen::SparseMatrix<double>,
      Eigen::LeastSquareDiagonalPreconditioner<double>>
      cg_;

 public:
  ConjugateGradientLeastSquareSolver() : cg_() {}
  virtual ~ConjugateGradientLeastSquareSolver() = default;

  LinearSolverStatus initialize(const Eigen::SparseMatrix<double>& A) override;

  LinearSolverStatus solve(const Eigen::SparseMatrix<double>& A,
                           const Eigen::VectorXd& b,
                           Eigen::VectorXd& x) override;

  // solver properties
  bool is_normal() const override { return false; }
  bool is_normal_lower() const override { return false; }
};

}  // namespace minisam
