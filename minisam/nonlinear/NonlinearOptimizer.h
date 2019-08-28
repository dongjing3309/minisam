/**
 * @file    NonlinearOptimizer.h
 * @brief   Base class of nonlinear optimizer
 * @author  Jing Dong, Zhaoyang Lv
 * @date    Oct 17, 2017
 */

#pragma once

#include <minisam/core/SchurComplement.h>
#include <minisam/linear/LinearSolver.h>
#include <minisam/nonlinear/SparsityPattern.h>

#include <iostream>
#include <memory>

namespace minisam {

// forward declearation
class FactorGraph;
class Variables;

// enum of linear solver types
enum class LinearSolverType {
  CHOLESKY,              // Eigen Direct LDLt factorization
  CHOLMOD,               // SuiteSparse CHOLMOD
  QR,                    // SuiteSparse SPQR
  CG,                    // Eigen Classical Conjugate Gradient Method
  LSCG,                  // Eigen Conjugate Gradient Method without forming A'A
  CUDA_CHOLESKY,         // cuSolverSP Cholesky factorization
  SCHUR_DENSE_CHOLESKY,  // Schur complement with reduced system solved by Eigen
                         // dense Cholesky
};

// enum of nonlinear optimization verbosity level
enum class NonlinearOptimizerVerbosityLevel {
  // only print warning message to std::cerr when optimization does not success
  // and terminated abnormally. Default verbosity level
  WARNING,
  // print per-iteration least square error sum to std::cout, also print
  // profiling defails to std::cout after optimization is done, if miniSAM is
  // compiled with internal profiling enabled
  ITERATION,
  // print more per-iteration detailed information to std::cout, e.g. trust
  // regoin searching information
  SUBITERATION,
};

// base class for nonlinear optimization settings
struct NonlinearOptimizerParams {
  // max number of iterations
  size_t max_iterations = 100;
  // relative error decrease threshold to stop
  double min_rel_err_decrease = 1e-5;
  // absolute error decrease threshold to stop
  double min_abs_err_decrease = 1e-5;
  // linear solver
  LinearSolverType linear_solver_type = LinearSolverType::CHOLESKY;
  // warning verbosity
  NonlinearOptimizerVerbosityLevel verbosity_level =
      NonlinearOptimizerVerbosityLevel::WARNING;

  void print(std::ostream& out = std::cout) const;
};

// return status of nonlinear optimization
enum class NonlinearOptimizationStatus {
  SUCCESS = 0,      // nonlinear optimization meets converge requirement
  MAX_ITERATION,    // reach max iterations but not reach converge requirement
  ERROR_INCREASE,   // optimizer cannot decrease error and give up
  RANK_DEFICIENCY,  // linear system has rank deficiency
  INVALID,          // something else is wrong with the optimization
};

/** base class for nonlinear optimizer */
class NonlinearOptimizer {
 protected:
  // settings
  NonlinearOptimizerParams params_;

  // linear system
  std::unique_ptr<SparseLinearSolver> linear_solver_;  // linear solver
  // linearization sparsity pattern
  internal::JacobianSparsityPattern j_sparsity_cache_;
  internal::LowerHessianSparsityPattern h_sparsity_cache_;

  // cached internal optimization status, used by iterate() method
  size_t iterations_;
  // error norm of values pass in iterate(), can be used by iterate
  // (should be read-only)
  double last_err_squared_norm_;
  // error norm of values pass out iterate()
  // writable by iterate(), if iterate update this value
  // then set err_squared_norm_ to true
  double err_squared_norm_;
  // flag err_squared_norm_ is up to date by iterate()
  bool err_uptodate_;

 public:
  virtual ~NonlinearOptimizer() = default;

  // default optimization method with default error termination condition
  // can be override in derived classes
  // by default VariablesToEliminate is empty, do not eliminate any variable
  // - if the optimization is successful return SUCCESS
  // - if something else is returned, the value of opt_values may be undefined
  // (depends on solver implementaion)

  virtual NonlinearOptimizationStatus optimize(
      const FactorGraph& graph, const Variables& init_values,
      Variables& opt_values,
      const VariablesToEliminate& var_elimiated = VariablesToEliminate());

  // method to run a single iteration to update variables
  // use to implement your own optimization iterate procedure
  // need a implementation
  // - if the iteration is successful return SUCCESS

  virtual NonlinearOptimizationStatus iterate(const FactorGraph& graph,
                                              Variables& update_values) = 0;

  // print
  virtual void print(std::ostream& out = std::cout) const;

  // read internal cached optimization status
  size_t iterations() const { return iterations_; }

 protected:
  explicit NonlinearOptimizer(
      const NonlinearOptimizerParams& params = NonlinearOptimizerParams());

  // default stop condition using error threshold
  // return true if stop condition meets
  bool errorStopCondition_(double last_err, double curr_err);
};

}  // namespace minisam
