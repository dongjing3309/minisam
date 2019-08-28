/**
 * @file    numericalJacobianImpl.h
 * @brief   Utils for calculating numerical Jacobians
 * @author  Jing Dong
 * @date    Apr 1, 2019
 */

#pragma once

#include <minisam/core/Traits.h>
#include <minisam/core/Variable.h>

#include <Eigen/Core>

#include <functional>
#include <memory>
#include <vector>

namespace minisam {
namespace internal {

// helper for dim and retract
template <class T>
inline T retraceHelper(const T& x, const Eigen::VectorXd& dx) {
  static_assert(is_manifold<T>::value, "type must be a manifold");
  return traits<T>::Retract(x, dx);
}
template <class T>
inline size_t dimHelper(const T& x) {
  static_assert(is_manifold<T>::value, "type must be a manifold");
  return traits<T>::Dim(x);
}

// specialization for Variable
template <>
inline std::shared_ptr<Variable> retraceHelper<std::shared_ptr<Variable>>(
    const std::shared_ptr<Variable>& x, const Eigen::VectorXd& dx) {
  return x->retract(dx);
}
template <>
inline size_t dimHelper<std::shared_ptr<Variable>>(
    const std::shared_ptr<Variable>& x) {
  return x->dim();
}

/**
 * central numerical jacobian of a 1-argument function, with argument type X and
 * return value type Y, Y must by manifold types and have traits, X should be
 * either Varible or manifold type
 */
template <class Y, class X>
Eigen::MatrixXd numericalJacobianCentral(std::function<Y(const X&)> f,
                                         const X& x, double delta) {
  // check manifold types
  static_assert(is_manifold<Y>::value,
                "numericalJacobian request output Y type must be a manifold");
  // center
  const Y fx = f(x);

  // jacobian matrix
  Eigen::MatrixXd J(traits<Y>::Dim(fx), dimHelper(x));

  for (size_t j = 0; j < dimHelper(x); j++) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(dimHelper(x));
    // right
    dx(j) = delta;
    Eigen::VectorXd dy1 = traits<Y>::Local(fx, f(retraceHelper(x, dx)));
    // left
    dx(j) = -delta;
    Eigen::VectorXd dy2 = traits<Y>::Local(fx, f(retraceHelper(x, dx)));
    // get jacobian at center
    J.col(j) << (dy1 - dy2) / (2.0 * delta);
  }

  return J;
}

/**
 * numerical jacobian of a 1-argument function, with argument type X and return
 * value type Y, Y must by manifold types and have traits, X should be either
 * Varible or manifold type Central difference is equivalent to N = 1
 *
 * using Ridders' Method, see http://ceres-solver.org/numerical_derivatives.html
 * C. Ridders, Accurate computation of F'(x) and F'(x) F''(x), Advances in
 * Engineering Software 4(2), 75-76, 1978.
 */
template <class Y, class X>
Eigen::MatrixXd numericalJacobianRidders(std::function<Y(const X&)> f,
                                         const X& x, int N, double delta) {
  // check manifold types
  static_assert(is_manifold<Y>::value,
                "numericalJacobian request output Y type must be a manifold");
  // cache
  // TODO: how to avoid function call here just for dim?
  std::vector<Eigen::MatrixXd> cacheJ(
      N, Eigen::MatrixXd(traits<Y>::Dim(f(x)), dimHelper(x)));

  double weight = 1.0;
  for (int n = 1; n <= N; n++) {
    if (n == 1) {
      // central
      double delta_curr = delta;
      for (int m = 1; m <= N; m++) {
        cacheJ[m - 1] = numericalJacobianCentral(f, x, delta_curr);
        delta_curr *= 0.5;
      }
    } else {
      // cache in-place calculation
      for (int m = 1; m <= N - n + 1; m++) {
        cacheJ[m - 1].noalias() =
            (weight * cacheJ[m] - cacheJ[m - 1]) / (weight - 1.0);
      }
    }
    weight *= 4.0;
  }
  return cacheJ[0];
}

}  // namespace internal
}  // namespace minisam
