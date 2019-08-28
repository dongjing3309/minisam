/**
 * @file    Traits.h
 * @brief   MiniSAM calib traits defines manifold and calibration related
 * behavior
 * @author  Jing Dong
 * @date    Mar 30, 2019
 */

#pragma once

#include <minisam/core/Traits.h>

#include <Eigen/Core>
#include <iostream>

namespace minisam {
namespace internal {

/// A helper class implement camera calibration trais for minisam types
template <typename CALIBRATION>
struct CameraIntrinsicsTraitsImpl {
  // type camera intrinsics tag
  typedef camera_intrinsics_tag type_category;

  // print
  static void Print(const CALIBRATION& m, std::ostream& out = std::cout) {
    m.print(out);
  }

  /** manifold */

  // tangent vector type defs
  typedef Eigen::Matrix<double, CALIBRATION::dim(), 1> TangentVector;

  // dimension
  static constexpr size_t Dim(const CALIBRATION&) { return CALIBRATION::dim(); }

  // local coordinate
  static TangentVector Local(const CALIBRATION& origin, const CALIBRATION& s) {
    return s.vector() - origin.vector();
  }

  // retract
  static CALIBRATION Retract(const CALIBRATION& origin,
                             const TangentVector& v) {
    return CALIBRATION(origin.vector() + v);
  }

  /** projection */

  // convert camera coordinates xy to image coordinates uv (pixel unit)
  static Eigen::Vector2d project(const CALIBRATION& calib,
                                 const Eigen::Vector2d& pc) {
    return calib.project(pc);
  }

  // Jacobians of project
  static void projectJacobians(
      const CALIBRATION& calib, const Eigen::Vector2d& pc,
      Eigen::Matrix<double, 2, CALIBRATION::dim()>& J_K,
      Eigen::Matrix<double, 2, 2>& J_p) {
    calib.projectJacobians(pc, J_K, J_p);
  }

  // convert image coordinates uv (pixel unit) to camera coordinates xy
  static Eigen::Vector2d unproject(const CALIBRATION& calib,
                                   const Eigen::Vector2d& pi) {
    return calib.unproject(pi);
  }
};
}  // namespace internal
}  // namespace minisam
