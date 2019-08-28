/**
 * @file    CUDASolver.h
 * @brief   CUDA Direct linear solvers of Ax = b
 * @author  Zhaoyang Lv, Jing Dong
 * @date    Sep 20, 2018
 */

#pragma once

#include <minisam/linear/AMDOrdering.h>
#include <minisam/linear/LinearSolver.h>

#include <cusolverSp.h>
#include <cusparse_v2.h>

#include <memory>

namespace minisam {

// wrapper of cuSolver::Cholesky
class CUDACholeskySolver : public SparseLinearSolver {
 private:
  // ordering
  std::shared_ptr<Ordering> amd_;

  // solver handlers
  cusolverSpHandle_t cusolverSpH_;  // solver handler
  cusparseMatDescr_t descrA_;       // A is a base-0 general matrix
  // cudaStream_t stream_;

  // device sparse matrix in CSR(A)
  int *d_csrRowPtrA_;  // <int> n+1
  int *d_csrColIndA_;  // <int> nnzA
  double *d_csrValA_;  // <double> nnzA
  double *d_x_;        // <double> n, x = A \ b
  double *d_b_;        // <double> n, a copy of h_b

  // host
  double *h_x_;  // <double> n,  x = A \ b

 public:
  CUDACholeskySolver();
  virtual ~CUDACholeskySolver();

  LinearSolverStatus initialize(
      const Eigen::SparseMatrix<double> &AtA) override;

  LinearSolverStatus solve(const Eigen::SparseMatrix<double> &AtA,
                           const Eigen::VectorXd &Atb,
                           Eigen::VectorXd &x) override;

  // solver properties
  bool is_normal() const override { return true; }
  bool is_normal_lower() const override { return true; }

 private:
  // free used memory
  void free_();
};
}
