/**
 * @file    SchurComplementDenseSolver.cpp
 * @brief   Abstract linear solver class of Ax = b
 * @author  Jing Dong
 * @date    Jun 24, 2019
 */

#include <minisam/linear/SchurComplementDenseSolver.h>
#include <minisam/utils/Timer.h>

#include <Eigen/Cholesky>

namespace minisam {

/* ************************************************************************** */
LinearSolverStatus SchurComplementDenseSolver::solve(
    const Eigen::SparseMatrix<double>& AtA, const Eigen::VectorXd& Atb,
    Eigen::VectorXd& x) {
  // static auto sr_timer = global_timer().getTimer("* Schur solve reduced");
  // static auto se_timer = global_timer().getTimer("* Schur solve elimiated");

  const int rs_until_idx = sc_ordering_->reducedSysUntilIndex();

  Eigen::MatrixXd H_reduced_lower;
  Eigen::SparseMatrix<double> Her;
  Eigen::SparseMatrix<double> He_inv_lower;
  Eigen::VectorXd g_reduced;

  // build reduced system  H_reduced * x_reduced = g_reduced
  LinearSolverStatus build_status = buildReducedSystem_(
      AtA, Atb, H_reduced_lower, Her, He_inv_lower, g_reduced);

  if (build_status != LinearSolverStatus::SUCCESS) {
    return build_status;
  }

  // sr_timer->tic();

  // reduced solution
  Eigen::VectorXd x_reduced;
  LinearSolverStatus reduced_status =
      reduced_solver_->solve(H_reduced_lower, g_reduced, x_reduced);

  if (reduced_status != LinearSolverStatus::SUCCESS) {
    return reduced_status;
  }

  // sr_timer->toc();
  // se_timer->tic();

  Eigen::VectorXd x_elimiated =
      He_inv_lower.selfadjointView<Eigen::Lower>() *
      (Atb.tail(AtA.rows() - rs_until_idx) - Her * x_reduced);

  // se_timer->toc();

  x = Eigen::VectorXd(x_reduced.size() + x_elimiated.size());
  x << x_reduced, x_elimiated;

  return LinearSolverStatus::SUCCESS;
}

/* ************************************************************************** */
LinearSolverStatus SchurComplementDenseSolver::buildReducedSystem_(
    const Eigen::SparseMatrix<double>& AtA_lower, const Eigen::VectorXd& Atb,
    Eigen::MatrixXd& H_reduced_lower, Eigen::SparseMatrix<double>& Her,
    Eigen::SparseMatrix<double>& He_inv_lower, Eigen::VectorXd& g_reduced) {
  // static auto br1_timer = global_timer().getTimer("* Schur build reduced");

  const int rs_until_idx = sc_ordering_->reducedSysUntilIndex();

  He_inv_lower = AtA_lower.block(rs_until_idx, rs_until_idx,
                                 AtA_lower.rows() - rs_until_idx,
                                 AtA_lower.rows() - rs_until_idx);
  Her = AtA_lower.block(rs_until_idx, 0, AtA_lower.rows() - rs_until_idx,
                        rs_until_idx);
  He_inv_lower.makeCompressed();
  Her.makeCompressed();

  H_reduced_lower =
      Eigen::MatrixXd(AtA_lower.block(0, 0, rs_until_idx, rs_until_idx));

  // br1_timer->tic();

  // block inversed based method
  for (size_t i = 0; i < sc_ordering_->eliminatedVariableSize(); i++) {
    const int var_pos = sc_ordering_->eliminatedVariablePosition(i);
    const int var_dim = sc_ordering_->eliminatedVariableDim(i);

    // br11_timer->tic();
    const Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower> ldlt(
        He_inv_lower.block(var_pos, var_pos, var_dim, var_dim));
    if (ldlt.info() != Eigen::Success) {
      return LinearSolverStatus::RANK_DEFICIENCY;
    }
    const Eigen::MatrixXd Hs_block_inv =
        ldlt.solve(Eigen::MatrixXd::Identity(var_dim, var_dim));
    // br11_timer->toc();
    // replace with inverse
    // br12_timer->tic();
    for (int ii = 0; ii < var_dim; ii++) {
      for (int ij = 0; ij < var_dim - ii; ij++) {
        double* value_ptr = He_inv_lower.valuePtr() +
                            He_inv_lower.outerIndexPtr()[ii + var_pos] + ij;
        *value_ptr = Hs_block_inv(ii, ii + ij);
      }
    }
    // br12_timer->toc();
  }

  H_reduced_lower -=
      Her.transpose() * (He_inv_lower.selfadjointView<Eigen::Lower>() * Her);

  // br1_timer->toc();

  g_reduced.noalias() =
      Atb.head(rs_until_idx) -
      Her.transpose() * (He_inv_lower.selfadjointView<Eigen::Lower>() *
                         Atb.tail(AtA_lower.rows() - rs_until_idx));

  return LinearSolverStatus::SUCCESS;
}

}  // namspace minisam
