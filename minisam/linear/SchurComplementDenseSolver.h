/**
 * @file    SchurComplementDenseSolver.h
 * @brief   Abstract linear solver class of Ax = b
 * @author  Jing Dong
 * @date    Jun 24, 2019
 */

#pragma once

#include <minisam/core/SchurComplement.h>
#include <minisam/linear/LinearSolver.h>

#include <memory>
#include <vector>

namespace minisam {

// abstract class of linear solvers to solve Ax = b
class SchurComplementDenseSolver : public SparseLinearSolver {
 private:
  // actual solver of reduced system
  std::unique_ptr<DenseLinearSolver> reduced_solver_;
  // eliminated/reduced systems ordering information
  // saved outside after constructed
  std::unique_ptr<SchurComplementOrdering> sc_ordering_;

 public:
  SchurComplementDenseSolver(
      std::unique_ptr<DenseLinearSolver>&& reduced_solver)
      : reduced_solver_(std::move(reduced_solver)) {}

  virtual ~SchurComplementDenseSolver() = default;

  // solve Ax = b
  LinearSolverStatus solve(const Eigen::SparseMatrix<double>& A,
                           const Eigen::VectorXd& b,
                           Eigen::VectorXd& x) override;

  // is it a normal equation solver
  bool is_normal() const override { return true; }
  bool is_normal_lower() const override { return true; }

  // set internal schur ordering information
  void setSchurComplementOrdering(
      std::unique_ptr<SchurComplementOrdering>&& sc_info) {
    sc_ordering_ = std::move(sc_info);
  }
  const std::unique_ptr<SchurComplementOrdering>& schur_complement_ordering()
      const {
    return sc_ordering_;
  }

 private:
  // build reduced system
  LinearSolverStatus buildReducedSystem_(
      const Eigen::SparseMatrix<double>& AtA_lower, const Eigen::VectorXd& Atb,
      Eigen::MatrixXd& H_reduced_lower, Eigen::SparseMatrix<double>& Her,
      Eigen::SparseMatrix<double>& He_inv_lower, Eigen::VectorXd& g_reduced);
};

}  // namespace minisam
