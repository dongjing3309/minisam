/**
 * @file    SparseCholesky.h
 * @brief   Direct Cholesky linear solver
 * @author  Zhaoyang Lv, Jing Dong
 * @date    April 11, 2018
 */

#pragma once

#include <minisam/linear/LinearSolver.h>

#include <Eigen/SparseCholesky>

// CholeskySolver uses a patched (bugfix) AMD ordering method
#include <minisam/3rdparty/eigen3/OrderingMethods>

namespace minisam {

// wrapper of Eigen SparseLDLT, using patched Eigen AMD ordering
class SparseCholeskySolver : public SparseLinearSolver {
 private:
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                        Eigen::AMDOrderingPatched<int>>
      chol_;

 public:
  SparseCholeskySolver() : chol_() {}
  virtual ~SparseCholeskySolver() = default;

  LinearSolverStatus initialize(const Eigen::SparseMatrix<double>& A) override;

  LinearSolverStatus solve(const Eigen::SparseMatrix<double>& A,
                           const Eigen::VectorXd& b,
                           Eigen::VectorXd& x) override;

  // solver properties
  bool is_normal() const override { return true; }
  bool is_normal_lower() const override { return true; }
};

}  // namespace minisam
