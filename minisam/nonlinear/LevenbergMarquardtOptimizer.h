/**
 * @file    LevenbergMarquardtOptimizer.h
 * @brief   Levenberg-Marquardt nonlinear optimizer
 * @author  Jing Dong
 * @date    Nov 2, 2018
 */

#pragma once

#include <minisam/nonlinear/NonlinearOptimizer.h>
#include <minisam/nonlinear/SparsityPattern.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace minisam {

// params of LM
struct LevenbergMarquardtOptimizerParams : NonlinearOptimizerParams {
  // initial lambda
  double lambda_init = 1e-5;
  // initial multiply factor to increase lambda
  double lambda_increase_factor_init = 2.0;
  // multiply factor to increase lambda multiply factor
  double lambda_increase_factor_update = 2.0;
  // minimal lambda decrease factor
  double lambda_decrease_factor_min = 1.0 / 3.0;
  // minimal lambda
  double lambda_min = 1e-20;
  // max lambda
  double lambda_max = 1e10;
  // minimal gain ratio (quality factor) to accept a step
  double gain_ratio_thresh = 1e-3;
  // if true use lambda * diag(A'A) for dumping,
  // if false use lambda * max(diag(A'A)) * I
  bool diagonal_damping = true;

  void print(std::ostream& out = std::cout) const;
};

// Levenberg-Marquardt nonlinear optimizer
class LevenbergMarquardtOptimizer : public NonlinearOptimizer {
 private:
  LevenbergMarquardtOptimizerParams params_;

  // internal states used for optimization across different call of tryLambda_
  double lambda_, last_lambda_, last_lambda_sqrt_;
  double lambda_increase_factor_;
  double gain_ratio_;
  bool linear_solver_inited_;
  bool try_lambda_inited_;

 public:
  explicit LevenbergMarquardtOptimizer(
      const LevenbergMarquardtOptimizerParams& params =
          LevenbergMarquardtOptimizerParams());

  virtual ~LevenbergMarquardtOptimizer() = default;

  /**
   *  Levenberg-Marquardt solver uses default optimize() implementation
   *
   *  - If the optimization it is successful return SUCCESS.
   *
   *  - If a LM iteration cannot decrease error with even max lambda,
   *    ERROR_INCREASE is returned.
   *    In this case opt_values will be the values BEFORE increased error.
   */
  NonlinearOptimizationStatus optimize(
      const FactorGraph& graph, const Variables& init_values,
      Variables& opt_values, const VariablesToEliminate& var_elimiated =
                                 VariablesToEliminate()) override;

  // iteration implementation
  // - if a LM step successfully decrease error, it returns SUCCESS.
  // - if a LM step cannot decrease error even with max lambda,
  //   it returns ERROR_INCREASE, and update_values will be the values BEFORE
  //   increased error.
  NonlinearOptimizationStatus iterate(const FactorGraph& graph,
                                      Variables& update_values) override;

  // reset interal state with differnt runs of optimization
  // only call by users when start a new optimization,
  // and when implement your optimization by calling iterate(), not optimize();
  void reset();

  void print(std::ostream& out = std::cout) const override;

 private:
  // try current lambda with solving dumped system and check error by gain ratio
  // return SUCCESS if current lambda can decrease error
  // return ERROR_INCREASE if current lambda cannot decrease error
  NonlinearOptimizationStatus tryLambda_(
      Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
      const Eigen::VectorXd& g, const Eigen::VectorXd& hessian_diag,
      double hessian_diag_max, const Eigen::VectorXd& hessian_diag_sqrt,
      double hessian_diag_max_sqrt, const FactorGraph& graph, Variables& values,
      double values_curr_err);

  // dump linear system using current lambda
  void dumpLinearSystem_(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
                         const Eigen::VectorXd& hessian_diag,
                         double hessian_diag_max,
                         const Eigen::VectorXd& hessian_diag_sqrt,
                         double hessian_diag_max_sqrt);

  // increase lambda
  void increaseLambda_();

  // decrease lambda
  void decreaseLambda_();
};

// internal functions needed for Levenberg-Marquardt implementation
namespace internal {

/**
 * Hessian dumping utils
 */

// update hessian dumping, use a single diag element
// assume diagonal elements already exists
void updateDumpingHessian(Eigen::SparseMatrix<double>& H, double diag,
                          double diag_last);

// update hessian dumping, use a hessian diagonal vector
// assume diagonal elements already exists
void updateDumpingHessianDiag(Eigen::SparseMatrix<double>& H,
                              const Eigen::VectorXd& diags, double lambda,
                              double lambda_last);

/**
 * Jacobian dumping utils
 */

// resize dumping jacobian
void resizeDumpingJacobian(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
                           const internal::JacobianSparsityPattern& sparsity);

// allocate space and assign a single element value for dumping a Jacobian
void allocateDumpingJacobian(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
                             const internal::JacobianSparsityPattern& sparsity,
                             double diag);

// allocate space and assign hessian diagonal vector for dumping a Jacobian
// use lambda * diags
void allocateDumpingJacobianDiag(
    Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
    const internal::JacobianSparsityPattern& sparsity, double lambda,
    const Eigen::VectorXd& diags);

// update jaocbian dumping, use a single diag element
void updateDumpingJacobian(Eigen::SparseMatrix<double>& A_dump, double diag,
                           double diag_last);

// update jaocbian dumping, use a hessian diagonal vector
void updateDumpingJacobianDiag(Eigen::SparseMatrix<double>& A_dump,
                               const Eigen::VectorXd& diags, double lambda,
                               double lambda_last);

// get hessian diagonal from a jacobian for dumping
Eigen::VectorXd hessianDiagonal(const Eigen::SparseMatrix<double>& A);

}  // namespace internal
}  // namespace minisam
