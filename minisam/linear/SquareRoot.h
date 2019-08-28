/**
 * @file    SquareRoot.h
 * @brief   Abstract square root information matrix R from information/Hessian H
 * @author  Jing Dong
 * @date    Mar 14, 2019
 */

#pragma once

#include <minisam/linear/AMDOrdering.h>
#include <minisam/linear/Ordering.h>

#include <memory>

namespace minisam {

/** return status of solving square root information */
enum class SquareRootSolverStatus {
  SUCCESS = 0,      // problem solve successfully (iterative methods converge)
  RANK_DEFICIENCY,  // linear system has rank deficiency
  INVALID,  // something wrong with the system, e.g. matrix size incompatible
};

// abstract class of solving upper square root information matrix R from
// information/Hessian H
// R'R = ordering.permute(H)
class SquareRootSolver {
 protected:
  // ordering
  OrderingMethod ordering_method_;
  std::shared_ptr<Ordering> ordering_;

 public:
  virtual ~SquareRootSolver() = default;

  // initialize the solver with sparsity pattern of information matrix H
  // call once before solving H the same sparsity structure
  // needs an actual implementation, if the empty one if not used
  virtual SquareRootSolverStatus initialize(
      const Eigen::SparseMatrix<double>& H) {
    switch (ordering_method_) {
      case OrderingMethod::NONE: {
        ordering_ = std::shared_ptr<Ordering>(new NaturalOrdering(H));
      } break;
      case OrderingMethod::AMD: {
        ordering_ = std::shared_ptr<Ordering>(new AMDOrdering(H));
      } break;
      default: {
        throw std::runtime_error(
            "[SquareRootSolver::initialize] ERROR: ordering method unknown");
      }
    }
    return SquareRootSolverStatus::SUCCESS;
  }

  // solve square root information matrix R from information/Hessian H
  // R'R = ordering.permute(H)
  // only lower part of H will be used
  // request H's sparsity pattern is setup by initialize();
  // needs an actual implementation
  virtual SquareRootSolverStatus solveR(const Eigen::SparseMatrix<double>& H,
                                        Eigen::SparseMatrix<double>& R) = 0;

  // solve transposed (lower-tri) square root information matrix L
  // L = R', LL' = ordering.permute(H)
  virtual SquareRootSolverStatus solveL(const Eigen::SparseMatrix<double>& H,
                                        Eigen::SparseMatrix<double>& L) = 0;

  // access ordering
  const std::shared_ptr<Ordering>& ordering() const { return ordering_; }

 protected:
  explicit SquareRootSolver(OrderingMethod ordering_method)
      : ordering_method_(ordering_method), ordering_(nullptr) {}
};

}  // minisam
