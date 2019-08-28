/**
 * @file    LinearSolver.h
 * @brief   Abstract linear solver class of Ax = b
 * @author  Jing Dong
 * @date    Sep 19, 2018
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace minisam {

/** return status of solving linear system */
enum class LinearSolverStatus {
  SUCCESS = 0,      // problem solve successfully (iterative methods converge)
  RANK_DEFICIENCY,  // linear system has rank deficiency
  INVALID,  // something wrong with the system, e.g. matrix size incompatible
};

// abstract class of linear solvers to solve sparse Ax = b
class SparseLinearSolver {
 public:
  virtual ~SparseLinearSolver() = default;

  // initialize the solver with sparsity pattern of system Ax = b
  // call once before solving Ax = b share the same sparsity structure
  // needs an actual implementation, if the empty one if not used
  virtual LinearSolverStatus initialize(
      const Eigen::SparseMatrix<double>& /*A*/) {
    return LinearSolverStatus::SUCCESS;
  }

  // solve Ax = b, return solving status
  // request A's sparsity pattern is setup by initialize();
  // needs an actual implementation
  virtual LinearSolverStatus solve(const Eigen::SparseMatrix<double>& A,
                                   const Eigen::VectorXd& b,
                                   Eigen::VectorXd& x) = 0;

  // is it a normal equation solver
  // if ture, the solver solves A'Ax = A'b, request input A is SPD
  // if false, the solver solves Ax = b
  virtual bool is_normal() const = 0;

  // does the normal equation solver only request lower part of the SPD matrix
  // if ture, only lower part of A (which is assume SPD) is needed, and self
  // adjoint view is considered
  // if false, the full SPD A must be provided
  virtual bool is_normal_lower() const = 0;

 protected:
  SparseLinearSolver() = default;
};

// abstract class of linear solvers to solve dense Ax = b
class DenseLinearSolver {
 public:
  virtual ~DenseLinearSolver() = default;

  // solve Ax = b, return solving status
  // needs an actual implementation
  virtual LinearSolverStatus solve(const Eigen::MatrixXd& A,
                                   const Eigen::VectorXd& b,
                                   Eigen::VectorXd& x) = 0;

  // is it a normal equation solver
  // if ture, the solver solves A'Ax = A'b, request input A is SPD
  // if false, the solver solves Ax = b
  virtual bool is_normal() const = 0;

  // does the normal equation solver only request lower part of the SPD matrix
  // if ture, only lower part of A (which is assume SPD) is needed, and self
  // adjoint view is considered
  // if false, the full SPD A must be provided
  virtual bool is_normal_lower() const = 0;

 protected:
  DenseLinearSolver() = default;
};

}  // namespace minisam
