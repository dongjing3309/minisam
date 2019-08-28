/**
 * @file    DoglegOptimizer.cpp
 * @brief   Dogleg trust region nonlinear optimizer
 * @author  Jing Dong
 * @date    Nov 4, 2018
 */

#include <minisam/nonlinear/DoglegOptimizer.h>

#include <minisam/core/FactorGraph.h>
#include <minisam/core/Variables.h>
#include <minisam/nonlinear/SparsityPattern.h>
#include <minisam/nonlinear/linearization.h>
#include <minisam/utils/Timer.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

namespace minisam {

/* ************************************************************************** */
DoglegOptimizer::DoglegOptimizer(const DoglegOptimizerParams& params)
    : NonlinearOptimizer(params), params_(params) {
  reset();
}

/* ************************************************************************** */
void DoglegOptimizer::reset() { radius_ = params_.radius_init; }

/* ************************************************************************** */
NonlinearOptimizationStatus DoglegOptimizer::optimize(
    const FactorGraph& graph, const Variables& init_values,
    Variables& opt_values, const VariablesToEliminate& var_elimiated) {
  reset();
  return NonlinearOptimizer::optimize(graph, init_values, opt_values,
                                      var_elimiated);
}

/* ************************************************************************** */
NonlinearOptimizationStatus DoglegOptimizer::iterate(const FactorGraph& graph,
                                                     Variables& values) {
  // profiling
  static auto lin_timer = global_timer().getTimer("* Graph linearization");
  static auto init_timer =
      global_timer().getTimer("* Ordering/Linear solver init");
  static auto linsolve_timer = global_timer().getTimer("* Linear system solve");

  // linearize
  Eigen::SparseMatrix<double> A;
  Eigen::VectorXd dx_gn, b;

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

  LinearSolverStatus linear_solver_status = linear_solver_->solve(A, b, dx_gn);

  linsolve_timer->toc_();

  // if linear solver works, use linear solution to get dogleg solution
  if (linear_solver_status == LinearSolverStatus::SUCCESS) {
    // get SD direction
    Eigen::VectorXd dx_sd;
    double g_squared_norm, alpha;
    if (linear_solver_->is_normal())
      dx_sd = internal::steepestDescentHessian(A, b, g_squared_norm, alpha);
    else
      dx_sd = internal::steepestDescentJacobian(A, b, g_squared_norm, alpha);

    return tryRadius_(graph, values, dx_gn, dx_sd, g_squared_norm, alpha);

  } else if (linear_solver_status == LinearSolverStatus::RANK_DEFICIENCY) {
    cerr << "Warning: linear system has rank deficiency" << endl;
    return NonlinearOptimizationStatus::RANK_DEFICIENCY;

  } else {
    cerr << "Warning: linear solver returns invalid state" << endl;
    return NonlinearOptimizationStatus::INVALID;
  }
}

/* ************************************************************************** */
NonlinearOptimizationStatus DoglegOptimizer::tryRadius_(
    const FactorGraph& graph, Variables& values, const Eigen::VectorXd& dx_gn,
    const Eigen::VectorXd& dx_sd, double g_squared_norm, double alpha) {
  // profiling
  static auto error_timer = global_timer().getTimer("* Graph error");
  static auto retract_timer = global_timer().getTimer("* Solution update");

  // get gain ratio (quality factor)

  // double values_origin_err = 0.5 * graph.errorSquaredNorm(values);
  const double values_origin_err = last_err_squared_norm_;

  while (radius_ > params_.radius_min) {
    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::SUBITERATION) {
      cout << "trust region radius = " << radius_ << ",  ";
    }

    // get dogleg direction with current radius
    Eigen::VectorXd dx_dl;
    double beta = internal::doglegStep(dx_gn, dx_sd, radius_, dx_dl);

    retract_timer->tic_();

    Variables values_update;

    if (linear_solver_->is_normal()) {
      values_update = values.retract(dx_dl, h_sparsity_cache_.var_ordering);
    } else {
      values_update = values.retract(dx_dl, j_sparsity_cache_.var_ordering);
    }

    retract_timer->toc_();

    error_timer->tic_();

    const double values_update_err =
        0.5 * graph.errorSquaredNorm(values_update);

    error_timer->toc_();

    // linear error update
    double linear_err_update;
    if (beta > 1.0) {
      // use GN
      if (params_.verbosity_level >=
          NonlinearOptimizerVerbosityLevel::SUBITERATION) {
        cout << "use GN step, ";
      }

      // g = dx_sd / alpha, which is -g in imm3215.pdf p.31
      linear_err_update = 0.5 * dx_gn.dot(dx_sd) / alpha;

    } else if (beta < 0.0) {
      // use SD
      if (params_.verbosity_level >=
          NonlinearOptimizerVerbosityLevel::SUBITERATION) {
        cout << "use SD step, ";
      }
      // clang-format off
      linear_err_update = 0.5 * radius_ * (2.0 * alpha * std::sqrt(g_squared_norm) - radius_) / alpha;
      // clang-format on

    } else {
      // blended
      if (params_.verbosity_level >=
          NonlinearOptimizerVerbosityLevel::SUBITERATION) {
        cout << "use blended step (beta = " << beta << "), ";
      }

      linear_err_update =
          0.5 * alpha * std::pow(1.0 - beta, 2.0) * g_squared_norm +
          beta * (2.0 - beta) * values_origin_err;
    }

    const double gain_ratio =
        (values_origin_err - values_update_err) / linear_err_update;

    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::SUBITERATION) {
      cout << "gain_ratio = " << gain_ratio << endl;
    }

    if (gain_ratio >= 0.0) {
      if (gain_ratio >= 0.75) {
        // good appox: accept step and increase radius
        double dx_dl_norm = dx_dl.norm();
        radius_ = std::max(radius_, 3.0 * dx_dl_norm);
      } else if (gain_ratio < 0.75 && gain_ratio >= 0.25) {
        // so-so appox: accept step and radius not change
      } else {
        // not good appox: accept step and reduce radius
        radius_ /= 2.0;
      }

      values = values_update;

      err_squared_norm_ = values_update_err;
      err_uptodate_ = true;

      return NonlinearOptimizationStatus::SUCCESS;

    } else {
      // gain_ratio < 0, error increase, reduce radius and try again
      radius_ /= 2.0;
    }
  }

  // cannot decrease error with minimal radius
  return NonlinearOptimizationStatus::ERROR_INCREASE;
}

/* ************************************************************************** */
void DoglegOptimizerParams::print(std::ostream& out) const {
  out << "DoglegOptimizerParams:" << endl;
  out << "  radius_init = " << radius_init << endl;
  out << "  radius_min = " << radius_min << endl;
  NonlinearOptimizerParams::print(out);
}

/* ************************************************************************** */
void DoglegOptimizer::print(std::ostream& out) const {
  out << "DoglegOptimizer : ";
  params_.print(out);
}

namespace internal {

/* ************************************************************************** */
Eigen::VectorXd steepestDescentJacobian(const Eigen::SparseMatrix<double>& A,
                                        const Eigen::VectorXd& b,
                                        double& g_squared_norm, double& alpha) {
  // g here is -g in imm3215.pdf
  // TODO: improve efficiency?
  const Eigen::VectorXd g = A.transpose() * b;
  const Eigen::VectorXd Jg = A * g;

  g_squared_norm = g.squaredNorm();
  alpha = g_squared_norm / Jg.squaredNorm();
  return alpha * g;
}

/* ************************************************************************** */
Eigen::VectorXd steepestDescentHessian(const Eigen::SparseMatrix<double>& AtA,
                                       const Eigen::VectorXd& Atb,
                                       double& g_squared_norm, double& alpha) {
  // g = Atb here is -g in imm3215.pdf
  // TODO: improve efficiency?
  const Eigen::VectorXd Hg = AtA.selfadjointView<Eigen::Lower>() * Atb;

  g_squared_norm = Atb.squaredNorm();
  alpha = g_squared_norm / Hg.dot(Atb);
  return alpha * Atb;
}

/* ************************************************************************** */
double doglegStep(const Eigen::VectorXd& dx_gn, const Eigen::VectorXd& dx_sd,
                  double radius, Eigen::VectorXd& dx_dl) {
  const double norm_gn = dx_gn.norm();
  const double norm_sd = dx_sd.norm();

  if (norm_gn < radius) {
    // use GN
    dx_dl = dx_gn;
    return 2.0;

  } else if (norm_sd > radius) {
    // use SD
    dx_dl = (radius / norm_sd) * dx_sd;
    return -1.0;

  } else {
    // blend GN and SD
    // dx_dl = dx_sd + beta * (dx_gn - dx_sd)
    // see imm3215.pdf p.31
    Eigen::VectorXd diff = dx_gn - dx_sd;
    double c = dx_sd.dot(diff);
    double norm2_diff = diff.squaredNorm();

    double beta;
    if (c < 0) {
      // clang-format off
      beta = (std::sqrt(c * c + norm2_diff * (radius * radius - norm_sd * norm_sd)) - c) / norm2_diff;
      // clang-format onn
    } else {
      const double t = radius * radius - norm_sd * norm_sd;
      beta = t / (c + std::sqrt(c * c + norm2_diff * t));
    }
    dx_dl = dx_sd + beta * diff;
    return beta;
  }
}

}  // namespace internal
}  // namespace minisam
