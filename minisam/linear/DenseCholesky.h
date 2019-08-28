/**
 * @file    DenseCholesky.h
 * @brief   Direct Cholesky linear solver
 * @author  Jing Dong
 * @date    June 25, 2019
 */

#pragma once

#include <minisam/linear/LinearSolver.h>

#include <Eigen/Cholesky>

namespace minisam {

// wrapper of Eigen SparseLDLT, using patched Eigen AMD ordering
class DenseCholeskySolver : public DenseLinearSolver {
 public:
  DenseCholeskySolver() = default;
  virtual ~DenseCholeskySolver() = default;

  LinearSolverStatus solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                           Eigen::VectorXd& x) override;

  // solver properties
  bool is_normal() const override { return true; }
  bool is_normal_lower() const override { return true; }
};

}  // namespace minisam
