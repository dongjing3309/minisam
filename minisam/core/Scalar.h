/**
 * @file    Scalar.h
 * @brief   Manifold/Lie group traits for scalar double types
 * @author  Jing Dong
 * @date    Oct 13, 2017
 */

#pragma once

#include <minisam/core/Traits.h>

#include <Eigen/Core>

#include <iostream>
#include <vector>

namespace minisam {
namespace internal {

// A helper class implement trais for any scalar types
// requires Scalar type has:
// 1. operator +
// 2. operator -
// 3. constructor from float/double
template <typename Scalar>
struct ScalarTraitsImpl {
  // type tag
  typedef lie_group_tag type_category;

  // print
  static void Print(Scalar m, std::ostream& out = std::cout) { out << m; }

  /** manifold */

  // tangent vector type defs
  typedef Eigen::Matrix<double, 1, 1> TangentVector;

  // dimension
  static constexpr size_t Dim(Scalar) { return 1; }

  // local coordinate
  static TangentVector Local(Scalar origin, Scalar s) {
    return (TangentVector() << s - origin).finished();
  }

  // retract
  static Scalar Retract(Scalar origin, const TangentVector& v) {
    return origin + v[0];
  }

  /** Lie group */

  // identity
  static constexpr Scalar Identity(Scalar) { return Scalar(0); }

  // inverse, with jacobians
  static Scalar Inverse(Scalar s) { return -s; }
  static void InverseJacobian(Scalar /*s*/, Eigen::MatrixXd& H) {
    H = -Eigen::MatrixXd::Identity(1, 1);
  }

  // compose, with jacobians
  static Scalar Compose(Scalar s1, Scalar s2) { return s1 + s2; }
  static void ComposeJacobians(Scalar /*s1*/, Scalar /*s2*/,
                               Eigen::MatrixXd& H1, Eigen::MatrixXd& H2) {
    H1 = Eigen::MatrixXd::Identity(1, 1);
    H2 = Eigen::MatrixXd::Identity(1, 1);
  }

  // logmap
  static TangentVector Logmap(Scalar s) {
    return (TangentVector() << s).finished();
  }

  // expmap
  static Scalar Expmap(Scalar /*s*/, const TangentVector& v) { return v[0]; }
};

}  // namespace internal

/// double traits
template <>
struct traits<double> : internal::ScalarTraitsImpl<double> {};

}  // namespace minisam
