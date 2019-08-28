/**
 * @file    GaussNewtonOptimizer.cpp
 * @brief   Gauss-Newton nonlinear optimizer
 * @author  Jing Dong, Zhaoyang Lv
 * @date    Oct 17, 2017
 */

#include <minisam/nonlinear/GaussNewtonOptimizer.h>

#include <minisam/core/FactorGraph.h>
#include <minisam/core/Variables.h>
#include <minisam/linear/DenseCholesky.h>
#include <minisam/linear/SchurComplementDenseSolver.h>
#include <minisam/nonlinear/SparsityPattern.h>
#include <minisam/nonlinear/linearization.h>
#include <minisam/utils/Timer.h>

#include <iostream>

using namespace std;

namespace minisam {

/* ************************************************************************** */
GaussNewtonOptimizer::GaussNewtonOptimizer(
    const GaussNewtonOptimizerParams& params)
    : NonlinearOptimizer(params) {}

/* ************************************************************************** */
void GaussNewtonOptimizerParams::print(std::ostream& out) const {
  out << "GaussNewtonOptimizerParams:" << endl;
  NonlinearOptimizerParams::print(out);
}

/* ************************************************************************** */
void GaussNewtonOptimizer::print(std::ostream& out) const {
  out << "GaussNewtonOptimizer : ";
  params_.print(out);
}

/* ************************************************************************** */
NonlinearOptimizationStatus GaussNewtonOptimizer::iterate(
    const FactorGraph& graph, Variables& values) {
  // profiling
  static auto init_timer =
      global_timer().getTimer("* Ordering/Linear solver init");
  static auto linsolve_timer = global_timer().getTimer("* Linear system solve");
  static auto lin_timer = global_timer().getTimer("* Graph linearization");
  static auto retract_timer = global_timer().getTimer("* Solution update");

  // linearize
  Eigen::SparseMatrix<double> A;
  Eigen::VectorXd dx, b;

  lin_timer->tic_();

  if (linear_solver_->is_normal()) {
    if (linear_solver_->is_normal_lower()) {
      // lower hessian linearization
      internal::linearzationLowerHessian(graph, values, h_sparsity_cache_, A,
                                         b);

    } else {
      // full hessian linearization
      internal::linearzationFullHessian(graph, values, h_sparsity_cache_, A, b);
    }

  } else {
    // jacobian linearization
    internal::linearzationJacobian(graph, values, j_sparsity_cache_, A, b);
  }

  lin_timer->toc_();

  // initiailize the linear solver if needed at first iteration
  if (iterations() == 0) {
    init_timer->tic_();

    linear_solver_->initialize(A);

    init_timer->toc_();
  }

  // solve linear system

  linsolve_timer->tic_();

  LinearSolverStatus linear_solver_status = linear_solver_->solve(A, b, dx);

  linsolve_timer->toc_();

  // retract
  // cout << "retracting ... " << endl;
  if (linear_solver_status == LinearSolverStatus::SUCCESS) {
    retract_timer->tic_();

    if (linear_solver_->is_normal())
      values = values.retract(dx, h_sparsity_cache_.var_ordering);
    else
      values = values.retract(dx, j_sparsity_cache_.var_ordering);

    retract_timer->toc_();

    return NonlinearOptimizationStatus::SUCCESS;

  } else if (linear_solver_status == LinearSolverStatus::RANK_DEFICIENCY) {
    cerr << "Warning: linear system has rank deficiency" << endl;
    return NonlinearOptimizationStatus::RANK_DEFICIENCY;

  } else {
    cerr << "Warning: linear solver returns invalid state" << endl;
    return NonlinearOptimizationStatus::INVALID;
  }
}
}
