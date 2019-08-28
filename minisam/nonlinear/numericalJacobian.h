/**
 * @file    numericalJacobian.h
 * @brief   Utils for calculating numerical Jacobians
 * @author  Jing Dong
 * @date    Sep 14, 2018
 */

#pragma once

#include <minisam/core/Traits.h>
#include <minisam/nonlinear/numericalJacobianImpl.h>

#include <Eigen/Core>

#include <functional>

namespace minisam {

/// enum of numerical jacobian types
/// see http://ceres-solver.org/numerical_derivatives.html
enum class NumericalJacobianType {
  CENTRAL,   // central difference
  RIDDERS3,  // 3th order Ridders' Method
  RIDDERS5,  // 5th order Ridders' Method
};

/**
 * numerical jacobian of a 1-argument function, with argument type X and return
 * value type Y
 * Y must by manifold types and have traits, X should be either Varible or
 * manifold type
 */
template <class Y, class X>
Eigen::MatrixXd numericalJacobian(
    std::function<Y(const X&)> f, const X& x, double delta = 1e-3,
    NumericalJacobianType numerical_type = NumericalJacobianType::RIDDERS5) {
  // check manifold types
  static_assert(is_manifold<Y>::value,
                "numericalJacobian request output Y type must be a manifold");

  switch (numerical_type) {
    case NumericalJacobianType::CENTRAL: {
      return internal::numericalJacobianCentral(f, x, delta);
    }
    case NumericalJacobianType::RIDDERS3: {
      return internal::numericalJacobianRidders(f, x, 3, delta);
    }
    case NumericalJacobianType::RIDDERS5: {
      return internal::numericalJacobianRidders(f, x, 5, delta);
    }
    default:
      throw std::runtime_error(
          "[numericalJacobian] numerical jacobian type wrong");
  }
}

// clang-format off

/// Convenient wrappers for multi-arguments functions
/// naming convention is numericalJacobianNM
/// where N is total number of arguments, M is the argument perturbed
/// if you have more than four arguments, you need to use numericalJacobian and
/// std::bind

/// 1-argument

template<class Y, class X1>
Eigen::MatrixXd numericalJacobian11(Y (*f)(const X1&), const X1& x1,
    double delta = 1e-3) {
  return numericalJacobian<Y, X1>(f, x1, delta);
}


/// 2-arguments

template<class Y, class X1, class X2>
Eigen::MatrixXd numericalJacobian21(Y (*f)(const X1&, const X2&), const X1& x1,
    const X2& x2, double delta = 1e-3) {
  using namespace std::placeholders;
  return numericalJacobian<Y, X1>(std::bind(f, _1, x2), x1, delta);
}

template<class Y, class X1, class X2>
Eigen::MatrixXd numericalJacobian22(Y (*f)(const X1&, const X2&), const X1& x1,
    const X2& x2, double delta = 1e-3) {
  using namespace std::placeholders;
  return numericalJacobian<Y, X2>(std::bind(f, x1, _1), x2, delta);
}


/// 3-arguments

template<class Y, class X1, class X2, class X3>
Eigen::MatrixXd numericalJacobian31(Y (*f)(const X1&, const X2&, const X3&),
    const X1& x1, const X2& x2, const X3& x3, double delta = 1e-3) {
  using namespace std::placeholders;
  return numericalJacobian<Y, X1>(std::bind(f, _1, x2, x3), x1, delta);
}

template<class Y, class X1, class X2, class X3>
Eigen::MatrixXd numericalJacobian32(Y (*f)(const X1&, const X2&, const X3&),
    const X1& x1, const X2& x2, const X3& x3, double delta = 1e-3) {
  using namespace std::placeholders;
  return numericalJacobian<Y, X2>(std::bind(f, x1, _1, x3), x2, delta);
}

template<class Y, class X1, class X2, class X3>
Eigen::MatrixXd numericalJacobian33(Y (*f)(const X1&, const X2&, const X3&),
    const X1& x1, const X2& x2, const X3& x3, double delta = 1e-3) {
  using namespace std::placeholders;
  return numericalJacobian<Y, X3>(std::bind(f, x1, x2, _1), x3, delta);
}


/// 4-arguments

template<class Y, class X1, class X2, class X3, class X4>
Eigen::MatrixXd numericalJacobian41(Y (*f)(const X1&, const X2&, const X3&, const X4&),
    const X1& x1, const X2& x2, const X3& x3, const X4& x4, double delta = 1e-3) {
  using namespace std::placeholders;
  return numericalJacobian<Y, X1>(std::bind(f, _1, x2, x3, x4), x1, delta);
}

template<class Y, class X1, class X2, class X3, class X4>
Eigen::MatrixXd numericalJacobian42(Y (*f)(const X1&, const X2&, const X3&, const X4&),
    const X1& x1, const X2& x2, const X3& x3, const X4& x4, double delta = 1e-3) {
  using namespace std::placeholders;
  return numericalJacobian<Y, X1>(std::bind(f, x1, _1, x3, x4), x2, delta);
}

template<class Y, class X1, class X2, class X3, class X4>
Eigen::MatrixXd numericalJacobian43(Y (*f)(const X1&, const X2&, const X3&, const X4&),
    const X1& x1, const X2& x2, const X3& x3, const X4& x4, double delta = 1e-3) {
  using namespace std::placeholders;
  return numericalJacobian<Y, X1>(std::bind(f, x1, x2, _1, x4), x3, delta);
}

template<class Y, class X1, class X2, class X3, class X4>
Eigen::MatrixXd numericalJacobian44(Y (*f)(const X1&, const X2&, const X3&, const X4&),
    const X1& x1, const X2& x2, const X3& x3, const X4& x4, double delta = 1e-3) {
  using namespace std::placeholders;
  return numericalJacobian<Y, X1>(std::bind(f, x1, x2, x3, _1), x4, delta);
}
// clang-format on
}  // namespace minisam
