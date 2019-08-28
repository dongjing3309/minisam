/**
 * @file    GaussNewtonOptimizer.h
 * @brief   Gauss-Newton nonlinear optimizer
 * @author  Jing Dong
 * @date    Oct 17, 2017
 */

#pragma once

#include <minisam/nonlinear/NonlinearOptimizer.h>

namespace minisam {

/** params of GN, nothing more than general nonlinear optimization */
struct GaussNewtonOptimizerParams : NonlinearOptimizerParams {
  void print(std::ostream& out = std::cout) const;
};

/** Gauss-Newton nonlinear optimizer */
class GaussNewtonOptimizer : public NonlinearOptimizer {
 public:
  explicit GaussNewtonOptimizer(
      const GaussNewtonOptimizerParams& params = GaussNewtonOptimizerParams());

  virtual ~GaussNewtonOptimizer() = default;

  /**
   *  Gauss-Newton solver uses default optimize() implementation
   *
   *  - If the optimization it is successful return SUCCESS.
   *
   *  - If the linear system has rank deficiency (determinant \approx 0),
   *    RANK_DEFICIENCY is returned.
   *    In this case opt_values will be init_values at first iteration,
   *    or optimized values of last iteration which has solvable Ax = b.
   *
   *  - If a GN iteration increases error, ERROR_INCREASE is returned.
   *    In this case opt_values will be the values WITH increased error.
   */

  // iteration implementation
  // - if linear solver success it returns SUCCESS.
  // - if linear solver has singularity it returns RANK_DEFICIENCY, and
  // update_values untouched
  NonlinearOptimizationStatus iterate(const FactorGraph& graph,
                                      Variables& update_values) override;

  void print(std::ostream& out = std::cout) const override;
};

}  // namespace minisam
