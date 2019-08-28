/**
 * @file    DoglegOptimizer.h
 * @brief   Dogleg trust region nonlinear optimizer
 * @author  Jing Dong
 * @date    Nov 4, 2018
 */

#pragma once

#include <minisam/nonlinear/NonlinearOptimizer.h>
#include <minisam/nonlinear/SparsityPattern.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace minisam {

/** params of Dogleg */
struct DoglegOptimizerParams : NonlinearOptimizerParams {
  double radius_init = 1.0;  // initial radius
  double radius_min = 1e-5;  // minimal radius

  void print(std::ostream& out = std::cout) const;
};

/** Dogleg nonlinear optimizer, see imm3215.pdf p.29 */
class DoglegOptimizer : public NonlinearOptimizer {
 private:
  DoglegOptimizerParams params_;

  // internal states used for optimization across different call of tryRadius_
  double radius_;

 public:
  explicit DoglegOptimizer(
      const DoglegOptimizerParams& params = DoglegOptimizerParams());

  virtual ~DoglegOptimizer() = default;

  /**
   *  Dogleg solver uses default optimize() implementation
   *
   *  - If the optimization it is successful return SUCCESS
   *  - If max iteration is reached returns MAX_ITERATION
   *  - If a radius cannot found to decrease error with even min radius,
   *    ERROR_INCREASE is returned.
   *  - If linear system has rank deficiency returns RANK_DEFICIENCY
   */
  NonlinearOptimizationStatus optimize(
      const FactorGraph& graph, const Variables& init_values,
      Variables& opt_values, const VariablesToEliminate& var_elimiated =
                                 VariablesToEliminate()) override;

  // iteration implementation
  // return SUCCESS / RANK_DEFICIENCY if happens
  NonlinearOptimizationStatus iterate(const FactorGraph& graph,
                                      Variables& update_values) override;

  // print
  void print(std::ostream& out = std::cout) const override;

  // reset interal state with differnt runs of optimization
  // only call by users when start a new optimization,
  // and when implement your optimization by calling iterate(), not optimize();
  void reset();

 private:
  // try current radius and update radius according to gain ratio
  // - if a radius found to decrease error, return SUCCESS, and update values.
  // - if a radius cannot found to decrease error with even min radius,
  //   ERROR_INCREASE is returned, and values won't be updated.
  NonlinearOptimizationStatus tryRadius_(const FactorGraph& graph,
                                         Variables& values,
                                         const Eigen::VectorXd& dx_gn,
                                         const Eigen::VectorXd& dx_sd,
                                         double g_squared_norm, double alpha);
};

namespace internal {

// given GN and steepest decent step and radius, calculate dogleg step
// return beta (for calculating gain ratio)
// if beta < 0, dx_dl = normed dx_sd
// if beta > 1, dx_dl = dx_gn
// if 0 <= beta <= 1, dx_dl = blended
double doglegStep(const Eigen::VectorXd& dx_gn, const Eigen::VectorXd& dx_sd,
                  double radius, Eigen::VectorXd& dx_dl);

// calculate steepest descent direction using linear system
// |g|^2 and alpha output by reference
Eigen::VectorXd steepestDescentJacobian(const Eigen::SparseMatrix<double>& A,
                                        const Eigen::VectorXd& b,
                                        double& g_squared_norm, double& alpha);

// only use lower part of Hessian
Eigen::VectorXd steepestDescentHessian(const Eigen::SparseMatrix<double>& AtA,
                                       const Eigen::VectorXd& Atb,
                                       double& g_squared_norm, double& alpha);

}  // namespace internal
}  // namespace minisam
