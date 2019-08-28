/**
 * @file    CalibKD.h
 * @brief   calibration with radial distortion, representing a 3x3 K matrix, and
 * radial and tangential distortion
 * @author  Jing Dong
 * @date    Oct 11, 2018
 */

#pragma once

#include <minisam/geometry/CalibTraits.h>

#include <Eigen/Core>
#include <iostream>

namespace minisam {

/**
 * An intrinsic camera calibration class without distortion,
 * uses same distortion model as OpenCV, with
 * http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
 * but only consider radial distortion coefficients k1, k2,
 * and tangential distortion coefficients p1, p2
 *
 *     | fx  0   cx |
 * K = | 0   fy  cy |,
 *     | 0   0   1  |
 *
 * x' = x*(1 + k1*r^2 + k2*r^4) + 2*p1*x*y + p2*(r^2 + 2*x^2)
 * y' = y*(1 + k1*r^2 + k2*r^4) + p1*(r^2 + 2*y^2) + 2*p2*x*y
 *
 * with r^2 = x^2 + y^2
 */
class CalibKD {
 private:
  Eigen::Matrix<double, 8, 1> params_;  // [fx, fy, cx, cy, k1, k2, p1, p2]

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // constructor from parameters fx, fy, cx, cy, k1, k2, p1, p2
  CalibKD(double fx, double fy, double cx, double cy, double k1, double k2,
          double p1, double p2)
      : params_((Eigen::Matrix<double, 8, 1>() << fx, fy, cx, cy, k1, k2, p1,
                 p2).finished()) {
    assert((fx > 0 && fy > 0) &&
           "[CalibKD::CalibKD] focal length smaller than 0");
  }

  // constructor from parameter vector [fx, fy, cx, cy]
  CalibKD(const Eigen::Matrix<double, 8, 1>& params) : params_(params) {
    assert((params_(0) > 0 && params_(1) > 0) &&
           "[CalibKD::CalibKD] focal length smaller than 0");
  }

  ~CalibKD() = default;

  // stream print operator
  friend std::ostream& operator<<(std::ostream& os, const CalibKD& cal) {
    os << "CalibKD { fx: " << cal.fx() << ", fy: " << cal.fy()
       << ", cx: " << cal.cx() << ", cy: " << cal.cy() << ", k1: " << cal.k1()
       << ", k2: " << cal.k2() << ", p1: " << cal.p1() << ", p2: " << cal.p2()
       << " }";
    return os;
  }

  // print
  void print(std::ostream& out = std::cout) const { out << *this; }

  /** manifold related */
  static constexpr size_t dim() { return 8; }

  /** data access */

  // intrinsic matrix
  double fx() const { return params_(0); }
  double fy() const { return params_(1); }
  double cx() const { return params_(2); }
  double cy() const { return params_(3); }

  // radial distortion
  double k1() const { return params_(4); }
  double k2() const { return params_(5); }

  // tangential distortion
  double p1() const { return params_(6); }
  double p2() const { return params_(7); }

  // vector access
  const Eigen::Matrix<double, 8, 1>& vector() const { return params_; }

  // 3x3 camera calibration matrix K
  Eigen::Matrix3d matrix() const {
    Eigen::Matrix3d K;
    // clang-format off
    K <<  fx(),   0.0,    cx(),
          0.0,    fy(),   cy(),
          0.0,    0.0,    1.0;
    // clang-format on
    return K;
  }

  // inverse camera calibration matrix K^{-1}
  Eigen::Matrix3d inverse_matrix() const {
    Eigen::Matrix3d invK;
    // clang-format off
    invK << 1.0/fx(), 0.0,      -cx()/fx(),
            0.0,      1.0/fy(), -cy()/fy(),
            0.0,      0.0,      1.0;
    // clang-format on
    return invK;
  }

  /** multiview geometry */

  // convert camera coordinates xy to image coordinates uv (pixel unit)
  Eigen::Vector2d project(const Eigen::Vector2d& pc) const;

  // Jacobians of project
  void projectJacobians(const Eigen::Vector2d& pc,
                        Eigen::Matrix<double, 2, 8>& J_K,
                        Eigen::Matrix<double, 2, 2>& J_p) const;

  // convert image coordinates uv (pixel unit) to camera coordinates xy
  Eigen::Vector2d unproject(const Eigen::Vector2d& pi) const;
};

// traits of CalibKD
template <>
struct traits<CalibKD> : internal::CameraIntrinsicsTraitsImpl<CalibKD> {};

}  // namespace minisam
