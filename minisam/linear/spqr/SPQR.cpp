/**
 * @file    SPQR.cpp
 * @brief   Wrapped SuitSparse SPQR solver
 * @author  Zhaoyang Lv, Jing Dong
 * @date    Oct 30, 2018
 */

#include <minisam/linear/spqr/SPQR.h>

namespace minisam {

/* ************************************************************************** */
LinearSolverStatus QRSolver::solve(const Eigen::SparseMatrix<double>& A,
                                   const Eigen::VectorXd& b,
                                   Eigen::VectorXd& x) {
  qr_.compute(A);

  if (qr_.info() != Eigen::Success) {
    if (qr_.info() == Eigen::InvalidInput) {
      return LinearSolverStatus::INVALID;
    }
    if (qr_.info() == Eigen::NumericalIssue) {
      return LinearSolverStatus::RANK_DEFICIENCY;
    }
  }

  x = qr_.solve(b);
  return LinearSolverStatus::SUCCESS;
}

}  // namespace minisam
