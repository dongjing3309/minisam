/**
 * @file    SPQR.h
 * @brief   Wrapped SuitSparse SPQR solver
 * @author  Zhaoyang Lv, Jing Dong
 * @date    Oct 30, 2018
 */

#pragma once

#include <minisam/linear/LinearSolver.h>

#include <Eigen/SPQRSupport>

namespace minisam {

// wrapper of SuitSparse SPQR, using built-in COLAMD ordering
class QRSolver : public SparseLinearSolver {
 private:
  Eigen::SPQR<Eigen::SparseMatrix<double>> qr_;

 public:
  QRSolver() : qr_() {
    // qr_.setPivotThreshold(0.0f);
  }
  virtual ~QRSolver() {}

  LinearSolverStatus solve(const Eigen::SparseMatrix<double>& A,
                           const Eigen::VectorXd& b,
                           Eigen::VectorXd& x) override;

  // solver properties
  bool is_normal() const override { return false; }
  bool is_normal_lower() const override { return false; }
};

}  // namespace minisam
