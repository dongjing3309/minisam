/**
 * @file    Sophus.h
 * @brief   Traits wrapping Sophus classes, to optimize Sophus types in miniSAM
 * @author  Jing Dong
 * @date    Oct 21, 2017
 */

#pragma once

#include <minisam/core/Traits.h>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <sophus/so2.hpp>
#include <sophus/so3.hpp>

#include <Eigen/Core>

#include <iostream>
#include <vector>

namespace minisam {
namespace internal {

/**
 * A helper class implement trais for any Sophus types
 */
template <typename T>
struct SophusTraitsImpl {
  // type tag
  typedef lie_group_tag type_category;

  // print
  static void Print(const T& m, std::ostream& out = std::cout) { out << m; }

  /** manifold */

  // tangent vector type defs
  typedef Eigen::Matrix<double, T::DoF, 1> TangentVector;

  // dimension
  static constexpr size_t Dim(const T&) { return T::DoF; }

  // local coordinate
  static TangentVector Local(const T& origin, const T& t) {
    // use logmap, right muliplication
    // TODO: may exist non-exact and faster version
    return (origin.inverse() * t).log();
  }

  // retract
  static T Retract(const T& origin, const TangentVector& v) {
    // use expmap
    // TODO: may exist non-exact and faster version
    return origin * T::exp(v);
  }

  /** Lie group */

  // identity
  // Sophus default constructor always give identity
  static constexpr T Identity(const T&) { return T(); }

  // inverse, with jacobians
  static T Inverse(const T& s) { return s.inverse(); }
  static void InverseJacobian(const T& s, Eigen::MatrixXd& H) { H = -s.Adj(); }

  // compose, with jacobians
  static T Compose(const T& s1, const T& s2) { return s1 * s2; }
  static void ComposeJacobians(const T& /*s1*/, const T& s2,
                               Eigen::MatrixXd& H1, Eigen::MatrixXd& H2) {
    H1 = s2.inverse().Adj();
    H2 = Eigen::MatrixXd::Identity(T::DoF, T::DoF);
  }

  // logmap
  static TangentVector Logmap(const T& s) { return s.log(); }

  // expmap
  static T Expmap(const T& /*s*/, const TangentVector& v) { return T::exp(v); }
};

/**
 * specialization of SO2d, since in Sophus Tangent is double, but we need
 * Eigen1d
 */

// local coordinate
template <>
inline SophusTraitsImpl<Sophus::SO2d>::TangentVector
SophusTraitsImpl<Sophus::SO2d>::Local(const Sophus::SO2d& origin,
                                      const Sophus::SO2d& t) {
  // use logmap, right muliplication
  // TODO: may exist non-exact and faster version
  return Eigen::Matrix<double, 1, 1>((origin.inverse() * t).log());
}

// retract
template <>
inline Sophus::SO2d SophusTraitsImpl<Sophus::SO2d>::Retract(
    const Sophus::SO2d& origin, const TangentVector& v) {
  // use expmap
  // TODO: may exist non-exact and faster version
  return origin * Sophus::SO2d::exp(v(0));
}

// inverse jacobians
template <>
inline void SophusTraitsImpl<Sophus::SO2d>::InverseJacobian(
    const Sophus::SO2d& s, Eigen::MatrixXd& H) {
  H = (Eigen::MatrixXd(1, 1) << -s.Adj()).finished();
}

// compose jacobians
template <>
inline void SophusTraitsImpl<Sophus::SO2d>::ComposeJacobians(
    const Sophus::SO2d& /*s1*/, const Sophus::SO2d& s2, Eigen::MatrixXd& H1,
    Eigen::MatrixXd& H2) {
  H1 = (Eigen::MatrixXd(1, 1) << s2.inverse().Adj()).finished();
  H2 = Eigen::MatrixXd::Identity(1, 1);
}

// logmap
template <>
inline SophusTraitsImpl<Sophus::SO2d>::TangentVector
SophusTraitsImpl<Sophus::SO2d>::Logmap(const Sophus::SO2d& s) {
  return (Eigen::MatrixXd(1, 1) << s.log()).finished();
}

// expmap
template <>
inline Sophus::SO2d SophusTraitsImpl<Sophus::SO2d>::Expmap(
    const Sophus::SO2d& /*s*/, const TangentVector& v) {
  return Sophus::SO2d::exp(v(0));
}

}  // namespace internal

/// Sophus trait, only for double types for now, since Sophus float types using
/// Eigen::Matrix<float>,
/// not compatible with current all Eigen::Matrix<double> code
template <>
struct traits<Sophus::SO2d> : internal::SophusTraitsImpl<Sophus::SO2d> {};
template <>
struct traits<Sophus::SE2d> : internal::SophusTraitsImpl<Sophus::SE2d> {};
template <>
struct traits<Sophus::SO3d> : internal::SophusTraitsImpl<Sophus::SO3d> {};
template <>
struct traits<Sophus::SE3d> : internal::SophusTraitsImpl<Sophus::SE3d> {};
template <>
struct traits<Sophus::Sim3d> : internal::SophusTraitsImpl<Sophus::Sim3d> {};

}  // namespace minisam

// print
namespace Sophus {

/// for double Sophus types
inline std::ostream& operator<<(std::ostream& os, const Sophus::SO2d& s) {
  os << "Sophus::SO2d(" << std::atan2(s.data()[1], s.data()[0]) << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Sophus::SE2d& s) {
  os << "Sophus::SE2d(" << s.translation()[0] << ", " << s.translation()[1]
     << ", " << std::atan2(s.data()[1], s.data()[0]) << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Sophus::SO3d& s) {
  os << "Sophus::SO3d" << std::endl
     << "R = [" << std::endl
     << s.matrix() << "]";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Sophus::SE3d& s) {
  os << "Sophus::SE3d" << std::endl
     << "t = [" << s.translation()[0] << ", " << s.translation()[1] << ", "
     << s.translation()[2] << "]'" << std::endl
     << "R = " << std::endl
     << s.rotationMatrix();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Sophus::Sim3d& s) {
  os << "Sophus::Sim3d" << std::endl
     << "s = " << s.rxso3().scale() << std::endl
     << "t = [" << s.translation()[0] << ", " << s.translation()[1] << ", "
     << s.translation()[2] << "]'" << std::endl
     << "R = [" << std::endl
     << s.rotationMatrix() << "]";
  return os;
}

}  // namespace Sophus
