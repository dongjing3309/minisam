/**
 * @file    SquareRootCholesky.h
 * @brief   Calculate square root information matrix by Cholesky factorization
 * @author  Jing Dong
 * @date    Mar 14, 2019
 */

#pragma once

#include <minisam/linear/SquareRoot.h>

#include <Eigen/SparseCholesky>

namespace minisam {

/** Square root information matrix by Cholesky factorization, using AMD ordering
 */
class SquareRootSolverCholesky : public SquareRootSolver {
 private:
  // cholesky solver
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                        Eigen::NaturalOrdering<int>>
      chol_;

 public:
  // by default using AMD ordering, can override with no ordering
  explicit SquareRootSolverCholesky(
      OrderingMethod ordering_method = OrderingMethod::AMD)
      : SquareRootSolver(ordering_method), chol_() {}

  virtual ~SquareRootSolverCholesky() = default;

  SquareRootSolverStatus initialize(
      const Eigen::SparseMatrix<double>& H) override;

  SquareRootSolverStatus solveR(const Eigen::SparseMatrix<double>& H,
                                Eigen::SparseMatrix<double>& R) override;

  SquareRootSolverStatus solveL(const Eigen::SparseMatrix<double>& H,
                                Eigen::SparseMatrix<double>& L) override;

 private:
  // factorization shared by solveR and solveL
  SquareRootSolverStatus factorize_(const Eigen::SparseMatrix<double>& H);
};

}  // namespace minisam
