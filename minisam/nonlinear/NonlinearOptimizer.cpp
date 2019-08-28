/**
 * @file    NonlinearOptimizer.cpp
 * @brief   Base class of nonlinear optimizer
 * @author  Jing Dong
 * @date    Oct 17, 2017
 */

#include <minisam/config.h>

#include <minisam/nonlinear/NonlinearOptimizer.h>

#include <minisam/core/FactorGraph.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Variables.h>
#include <minisam/linear/ConjugateGradient.h>
#include <minisam/linear/DenseCholesky.h>
#include <minisam/linear/SchurComplementDenseSolver.h>
#include <minisam/linear/SparseCholesky.h>
#include <minisam/utils/Timer.h>

// non-default solvers
#ifdef MINISAM_USE_CUSOLVER
#include <minisam/linear/cuda/CUDASolver.h>
#endif

#ifdef MINISAM_USE_CHOLMOD
#include <minisam/linear/cholmod/Cholmod.h>
#endif

#ifdef MINISAM_USE_SPQR
#include <minisam/linear/spqr/SPQR.h>
#endif

#include <algorithm>
#include <limits>

using namespace std;

namespace minisam {

/* ************************************************************************** */
NonlinearOptimizer::NonlinearOptimizer(const NonlinearOptimizerParams& params)
    : params_(params),
      iterations_(0),
      last_err_squared_norm_(numeric_limits<double>::max()),
      err_squared_norm_(numeric_limits<double>::max()),
      err_uptodate_(false) {
  // linear solver
  switch (params_.linear_solver_type) {
    case LinearSolverType::CHOLESKY: {
      linear_solver_ =
          unique_ptr<SparseLinearSolver>(new SparseCholeskySolver());
    } break;

    case LinearSolverType::CHOLMOD: {
#ifdef MINISAM_USE_CHOLMOD
      linear_solver_ = unique_ptr<SparseLinearSolver>(new CholmodSolver());
#else
      throw invalid_argument(
          "[NonlinearOptimizer] Cannot use Cholmod Solver, since miniSAM is "
          "not compiled with SuiteSparse");
#endif
    } break;

    case LinearSolverType::QR: {
#ifdef MINISAM_USE_SPQR
      linear_solver_ = unique_ptr<SparseLinearSolver>(new QRSolver());
#else
      throw invalid_argument(
          "[NonlinearOptimizer] Cannot use SPQR Solver, since miniSAM is not "
          "compiled with SuiteSparse");
#endif
    } break;

    case LinearSolverType::CG: {
      linear_solver_ =
          unique_ptr<SparseLinearSolver>(new ConjugateGradientSolver());
    } break;

    case LinearSolverType::LSCG: {
      linear_solver_ = unique_ptr<SparseLinearSolver>(
          new ConjugateGradientLeastSquareSolver());
    } break;

    case LinearSolverType::CUDA_CHOLESKY: {
#ifdef MINISAM_USE_CUSOLVER
      linear_solver_ = unique_ptr<SparseLinearSolver>(new CUDACholeskySolver());
#else
      throw invalid_argument(
          "[NonlinearOptimizer] Cannot use CUDA Cholesky Solver, since miniSAM "
          "is not compiled with CUDA");
#endif
    } break;

    case LinearSolverType::SCHUR_DENSE_CHOLESKY: {
      linear_solver_ =
          unique_ptr<SparseLinearSolver>(new SchurComplementDenseSolver(
              unique_ptr<DenseCholeskySolver>(new DenseCholeskySolver())));
    } break;

    default: {
      throw invalid_argument(
          "[NonlinearOptimizer] linear solver type is unknown");
    }
  }
}

/* ************************************************************************** */
NonlinearOptimizationStatus NonlinearOptimizer::optimize(
    const FactorGraph& graph, const Variables& init_values,
    Variables& opt_values, const VariablesToEliminate& var_elimiated) {
  // profiling
  static auto pattern_timer =
      global_timer().getTimer("* Linearization pattern");
  static auto err_timer = global_timer().getTimer("* Graph error");

  // linearization sparsity pattern
  VariableOrdering vordering = init_values.defaultVariableOrdering();

  // check whether calculate schur complement ordering
  if (params_.linear_solver_type == LinearSolverType::SCHUR_DENSE_CHOLESKY) {
    // schur complement solver should have at least one variable to eliminate
    if (!var_elimiated.isEliminatedAny()) {
      throw invalid_argument(
          "[NonlinearOptimizer::optimize] Schur complement solver must have at "
          "least one variable to eliminate");
    }

    static auto scordering_timer = global_timer().getTimer("* Schur ordering");
    scordering_timer->tic_();

    // schur complement ordering
    std::unique_ptr<SchurComplementOrdering> sc_ordering(
        new SchurComplementOrdering(vordering, var_elimiated, init_values));

    // update ordering for linearization
    vordering = sc_ordering->ordering();

    // prepare schur complement solver
    SchurComplementDenseSolver* sc_linear_solver =
        dynamic_cast<SchurComplementDenseSolver*>(linear_solver_.get());
    sc_linear_solver->setSchurComplementOrdering(std::move(sc_ordering));

    scordering_timer->toc_();

  } else {
    // regular sparse solvers should not have any variable to eliminate
    if (var_elimiated.isEliminatedAny()) {
      throw invalid_argument(
          "[NonlinearOptimizer::optimize] Non-Schur complement solver cannot "
          "eliminate any variable");
    }
  }

  pattern_timer->tic_();

  if (linear_solver_->is_normal()) {
    // hessian linearization
    h_sparsity_cache_ =
        internal::constructLowerHessianSparsity(graph, init_values, vordering);

  } else {
    // jacobian linearization
    j_sparsity_cache_ =
        internal::constructJacobianSparsity(graph, init_values, vordering);
  }

  pattern_timer->toc_();

  // init vars and errors
  iterations_ = 0;

  err_timer->tic_();

  last_err_squared_norm_ = 0.5 * graph.errorSquaredNorm(init_values);

  err_timer->toc_();

  opt_values = init_values;

  if (params_.verbosity_level >= NonlinearOptimizerVerbosityLevel::ITERATION) {
    cout << "initial error = " << last_err_squared_norm_ << endl;
  }

  while (iterations_ < params_.max_iterations) {
    // iterate through
    NonlinearOptimizationStatus iterate_status = iterate(graph, opt_values);
    iterations_++;

    // check linear solver status and return if not success
    if (iterate_status != NonlinearOptimizationStatus::SUCCESS) {
      return iterate_status;
    }

    // check error for stop condition
    double curr_err;
    if (err_uptodate_) {
      // err has be updated by iterate()
      curr_err = err_squared_norm_;
      err_uptodate_ = false;
    } else {
      err_timer->tic_();

      curr_err = 0.5 * graph.errorSquaredNorm(opt_values);

      err_timer->toc_();
    }

    if (params_.verbosity_level >=
        NonlinearOptimizerVerbosityLevel::ITERATION) {
      cout << "iteration " << iterations_ << ", error = " << curr_err << endl;
    }

    if (curr_err - last_err_squared_norm_ > 1e-20) {
      cerr << "Warning: optimizer cannot decrease error" << endl;
      return NonlinearOptimizationStatus::ERROR_INCREASE;
    }

    if (errorStopCondition_(last_err_squared_norm_, curr_err)) {
      if (params_.verbosity_level >=
          NonlinearOptimizerVerbosityLevel::ITERATION) {
        cout << "reach stop condition, optimization success" << endl;
      }
      return NonlinearOptimizationStatus::SUCCESS;
    }

    last_err_squared_norm_ = curr_err;
  }

  cerr << "Warning: reach max iteration without reaching stop condition"
       << endl;

  return NonlinearOptimizationStatus::MAX_ITERATION;
}

/* ************************************************************************** */
bool NonlinearOptimizer::errorStopCondition_(double last_err, double curr_err) {
  return ((last_err - curr_err) < params_.min_abs_err_decrease) ||
         ((last_err - curr_err) / last_err < params_.min_rel_err_decrease);
}

/* ************************************************************************** */
void NonlinearOptimizerParams::print(std::ostream& out) const {
  out << "NonlinearOptimizerParams:" << endl;
  out << "  max_iterations = " << max_iterations << endl;
  out << "  min_rel_err_decrease = " << min_rel_err_decrease << endl;
  out << "  min_abs_err_decrease = " << min_abs_err_decrease << endl;
  out << "  linear_solver_type = ";
  switch (linear_solver_type) {
    case LinearSolverType::CHOLESKY: {
      out << "CHOLESKY";
    } break;
    case LinearSolverType::CHOLMOD: {
      out << "CHOLMOD";
    } break;
    case LinearSolverType::QR: {
      out << "QR";
    } break;
    case LinearSolverType::CG: {
      out << "CG";
    } break;
    case LinearSolverType::LSCG: {
      out << "LSCG";
    } break;
    case LinearSolverType::CUDA_CHOLESKY: {
      out << "CUDA_CHOLESKY";
    } break;
    case LinearSolverType::SCHUR_DENSE_CHOLESKY: {
      out << "SCHUR_DENSE_CHOLESKY";
    } break;
    default: {
      throw invalid_argument(
          "[NonlinearOptimizer] linear solver type is unknown");
    }
  }
  out << endl;
  out << "  verbosity_level = ";
  switch (verbosity_level) {
    case NonlinearOptimizerVerbosityLevel::WARNING: {
      out << "WARNING";
    } break;
    case NonlinearOptimizerVerbosityLevel::ITERATION: {
      out << "ITERATION";
    } break;
    case NonlinearOptimizerVerbosityLevel::SUBITERATION: {
      out << "SUBITERATION";
    } break;
    default: {
      throw invalid_argument("[NonlinearOptimizer] verbosity_level is unknown");
    }
  }
  out << endl;
}

/* ************************************************************************** */
void NonlinearOptimizer::print(std::ostream& out) const {
  out << "NonlinearOptimizer : ";
  params_.print(out);
}

}  // namespace minisam
