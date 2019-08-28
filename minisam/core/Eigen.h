/**
 * @file    Eigen.h
 * @brief   Manifold/Lie group traits for Eigen::Vector types
 * @author  Jing Dong
 * @date    Oct 15, 2017
 */

#pragma once

#include <minisam/core/Traits.h>

#include <Eigen/Core>

#include <iostream>
#include <vector>

namespace minisam {
namespace internal {

/// A helper class implement trais for any fixed dim Eigen double vector type
template <typename Vector, int N>
struct VectorSpaceTraitsImpl {
  // type tag
  typedef lie_group_tag type_category;

  // print
  static void Print(const Vector& m, std::ostream& out = std::cout) {
    out << "[" << m.transpose() << "]'";
  }

  /** manifold */

  // tangent vector type defs
  typedef Eigen::Matrix<double, N, 1> TangentVector;

  // dimension
  // fixed size implementation
  static constexpr size_t Dim(const Vector&) { return N; }

  // local coordinate
  static TangentVector Local(const Vector& origin, const Vector& s) {
    return s - origin;
  }

  // retract
  static Vector Retract(const Vector& origin, const TangentVector& v) {
    return origin + v;
  }

  /** Lie group */

  // identity
  static constexpr Vector Identity(const Vector&) { return Vector::Zero(); }

  // inverse, with jacobians
  static Vector Inverse(const Vector& s) { return -s; }
  static void InverseJacobian(const Vector& /*s*/, Eigen::MatrixXd& H) {
    H = -Eigen::MatrixXd::Identity(N, N);
  }

  // compose, with jacobians
  static Vector Compose(const Vector& s1, const Vector& s2) { return s1 + s2; }
  static void ComposeJacobians(const Vector& /*s1*/, const Vector& /*s2*/,
                               Eigen::MatrixXd& H1, Eigen::MatrixXd& H2) {
    H1 = Eigen::MatrixXd::Identity(N, N), H2 = Eigen::MatrixXd::Identity(N, N);
  }

  // logmap
  static TangentVector Logmap(const Vector& s) { return s; }

  // expmap
  static Vector Expmap(const Vector& /*s*/, const TangentVector& v) {
    return v;
  }
};

// partial specialization trais for dynamic dim Eigen double vector
template <typename Vector>
struct VectorSpaceTraitsImpl<Vector, Eigen::Dynamic> {
  // type tag
  typedef lie_group_tag type_category;

  // print
  static void Print(const Vector& m, std::ostream& out = std::cout) {
    out << "[" << m.transpose() << "]'";
  }

  /** manifold */

  // tangent vector type defs
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> TangentVector;

  // dimension
  // fixed size implementation
  static size_t Dim(const Vector& s) { return s.size(); }

  // local coordinate
  static TangentVector Local(const Vector& origin, const Vector& s) {
    assert(origin.size() == s.size() &&
           "[VectorSpaceTraitsImpl::Local] vector size incompatible");
    return s - origin;
  }

  // retract
  static Vector Retract(const Vector& origin, const TangentVector& v) {
    assert(origin.size() == v.size() &&
           "[VectorSpaceTraitsImpl::Retract] vector size incompatible");
    return origin + v;
  }

  /** Lie group */

  // identity
  static Vector Identity(const Vector& s) { return Vector::Zero(s.size()); }

  // inverse, with jacobians
  static Vector Inverse(const Vector& s) { return -s; }
  static void InverseJacobian(const Vector& s, Eigen::MatrixXd& H) {
    H = -Eigen::MatrixXd::Identity(Dim(s), Dim(s));
  }

  // compose, with jacobians
  static Vector Compose(const Vector& s1, const Vector& s2) {
    assert(s1.size() == s2.size() &&
           "[VectorSpaceTraitsImpl::Compose] vector size incompatible");
    return s1 + s2;
  }
  static void ComposeJacobians(const Vector& s1, const Vector& s2,
                               Eigen::MatrixXd& H1, Eigen::MatrixXd& H2) {
    assert(
        s1.size() == s2.size() &&
        "[VectorSpaceTraitsImpl::ComposeJacobians] vector size incompatible");
    H1 = Eigen::MatrixXd::Identity(Dim(s1), Dim(s2)),
    H2 = Eigen::MatrixXd::Identity(Dim(s1), Dim(s2));
  }

  // logmap
  static TangentVector Logmap(const Vector& s) { return s; }

  // expmap
  static Vector Expmap(const Vector& /*s*/, const TangentVector& v) {
    return v;
  }
};
}  // namespace internal

/// traits for fixed size Eigen vector types
/// cannot auto deduct Options, MaxRows and MaxCols on VC++
template <int N, int Options, int MaxRows, int MaxCols>
struct traits<Eigen::Matrix<double, N, 1, Options, MaxRows, MaxCols> >
    : internal::VectorSpaceTraitsImpl<
          Eigen::Matrix<double, N, 1, Options, MaxRows, MaxCols>, N> {};

}  // namespace minisam
