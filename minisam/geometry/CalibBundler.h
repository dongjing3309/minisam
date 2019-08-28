/**
 * @file    CalibBundler.h
 * @brief   intrinsic camera calibration with distortion, used by Bundler
 * @author  Jing Dong
 * @date    Oct 18, 2018
 */

#pragma once

#include <minisam/geometry/CalibTraits.h>

#include <Eigen/Core>
#include <iostream>

namespace minisam {

/**
 * An intrinsic camera calibration class with distortion,
 *
 * p' =  f * r(p) * p
 * r(p) = 1.0 + k1 * |p|^2 + k2 * |p|^4.
 */
class CalibBundler {
 private:
  Eigen::Vector3d params_;  // [f, k1, k2]

 public:
  // constructor from parameters f, k1, k2
  CalibBundler(double f, double k1, double k2)
      : params_((Eigen::Vector3d() << f, k1, k2).finished()) {
    assert(f > 0 && "[CalibBundler::CalibBundler] focal length smaller than 0");
  }

  // constructor from parameter vector [f, k1, k2]
  CalibBundler(const Eigen::Vector3d& params) : params_(params) {
    assert(params_(0) > 0 &&
           "[CalibBundler::CalibBundler] focal length smaller than 0");
  }

  ~CalibBundler() = default;

  // stream print operator
  friend std::ostream& operator<<(std::ostream& os, const CalibBundler& cal) {
    os << "CalibBundler { f: " << cal.f() << ", k1: " << cal.k1()
       << ", k2: " << cal.k2() << " }";
    return os;
  }

  // print
  void print(std::ostream& out = std::cout) const { out << *this; }

  /** manifold related */

  static constexpr size_t dim() { return 3; }

  /** data access */

  // intrinsic matrix
  double f() const { return params_(0); }
  double k1() const { return params_(1); }
  double k2() const { return params_(2); }

  // vector access
  const Eigen::Vector3d& vector() const { return params_; }

  // 3x3 camera calibration matrix K
  Eigen::Matrix3d matrix() const {
    Eigen::Matrix3d K;
    // clang-format off
    K <<  f(),    0.0,    0.0,
          0.0,    f(),    0.0,
          0.0,    0.0,    1.0;
    // clang-formato ob
    return K;
  }

  // inverse camera calibration matrix K^{-1}
  Eigen::Matrix3d inverse_matrix() const {
    Eigen::Matrix3d invK;
    // clang-format off
    invK << 1.0/f(),  0.0,      0.0,
            0.0,      1.0/f(),  0.0,
            0.0,      0.0,      1.0;
    // clang-format on
    return invK;
  }

  /** multiview geometry */

  // convert camera coordinates xy to image coordinates uv (pixel unit)
  Eigen::Vector2d project(const Eigen::Vector2d& pc) const {
    const double r2 = pc.squaredNorm();
    const double fd = f() * (1.0 + (k1() + k2() * r2) * r2);
    return fd * pc;
  }

  // Jacobians of project
  void projectJacobians(const Eigen::Vector2d& pc,
                        Eigen::Matrix<double, 2, 3>& J_K,
                        Eigen::Matrix<double, 2, 2>& J_p) const;

  // convert image coordinates uv (pixel unit) to camera coordinates xy
  Eigen::Vector2d unproject(const Eigen::Vector2d& pi) const;
};

// traits of CalibBundler
template <>
struct traits<CalibBundler>
    : internal::CameraIntrinsicsTraitsImpl<CalibBundler> {};

}  // namespace minisam
