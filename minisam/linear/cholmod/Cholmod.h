/**
 * @file    Cholmod.h
 * @brief   Wrapped SuitSparse Cholmod solver
 * @author  Zhaoyang Lv, Jing Dong
 * @date    Oct 30, 2018
 */

#pragma once

#include <minisam/linear/LinearSolver.h>

#include <Eigen/CholmodSupport>

namespace minisam {

// wrapper of SuitSparse cholmod, using built-in COLAMD ordering
class CholmodSolver : public SparseLinearSolver {
 private:
  Eigen::CholmodDecomposition<Eigen::SparseMatrix<double>, Eigen::Lower> chol_;

 public:
  CholmodSolver() : chol_() {}
  virtual ~CholmodSolver() {}

  LinearSolverStatus initialize(const Eigen::SparseMatrix<double>& A) override;

  LinearSolverStatus solve(const Eigen::SparseMatrix<double>& A,
                           const Eigen::VectorXd& b,
                           Eigen::VectorXd& x) override;

  // solver properties
  bool is_normal() const override { return true; }
  bool is_normal_lower() const override { return true; }
};

}  // namespace minisam
