/**
 * @file    DenseCholesky.cpp
 * @brief   Direct Cholesky linear solver
 * @author  Jing Dong
 * @date    June 28, 2019
 */

#include <minisam/linear/DenseCholesky.h>

namespace minisam {

/* ************************************************************************** */
LinearSolverStatus DenseCholeskySolver::solve(const Eigen::MatrixXd& A,
                                              const Eigen::VectorXd& b,
                                              Eigen::VectorXd& x) {
  const Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower> ldlt(A);
  if (ldlt.info() != Eigen::Success) {
    return LinearSolverStatus::RANK_DEFICIENCY;
  }
  x = ldlt.solve(b);
  return LinearSolverStatus::SUCCESS;
}

}  // namespace minisam
