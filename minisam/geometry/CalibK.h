/**
 * @file    CalibK.h
 * @brief   intrinsic camera calibration without distortion, representing a 3x3
 * K matrix
 * @author  Jing Dong
 * @date    Oct 9, 2018
 */

#pragma once

#include <minisam/geometry/CalibTraits.h>

#include <Eigen/Core>
#include <iostream>

namespace minisam {

/**
 * An intrinsic camera calibration class without distortion,
 * representing a 3x3 K matrix
 *
 *     | fx  0   cx |
 * K = | 0   fy  cy |
 *     | 0   0   1  |
 */
class CalibK {
 private:
  Eigen::Vector4d params_;  // [fx, fy, cx, cy]

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // constructor from parameters fx, fy, cx and cy
  CalibK(double fx, double fy, double cx, double cy)
      : params_((Eigen::Vector4d() << fx, fy, cx, cy).finished()) {
    assert((fx > 0 && fy > 0) &&
           "[CalibK::CalibK] focal length smaller than 0");
  }

  // constructor from parameter vector [fx, fy, cx, cy]
  CalibK(const Eigen::Vector4d& params) : params_(params) {
    assert((params_(0) > 0 && params_(1) > 0) &&
           "[CalibK::CalibK] focal length smaller than 0");
  }

  ~CalibK() = default;

  // stream print operator
  friend std::ostream& operator<<(std::ostream& os, const CalibK& cal) {
    os << "CalibK { fx: " << cal.fx() << ", fy: " << cal.fy()
       << ", cx: " << cal.cx() << ", cy: " << cal.cy() << " }";
    return os;
  }

  // print
  void print(std::ostream& out = std::cout) const { out << *this; }

  /** manifold related */
  static constexpr size_t dim() { return 4; }

  /** data access */

  // intrinsic matrix
  double fx() const { return params_(0); }
  double fy() const { return params_(1); }
  double cx() const { return params_(2); }
  double cy() const { return params_(3); }

  // vector access
  const Eigen::Vector4d& vector() const { return params_; }

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
  // pi = K * pc
  Eigen::Vector2d project(const Eigen::Vector2d& pc) const {
    return (Eigen::Vector2d() << fx() * pc(0) + cx(), fy() * pc(1) + cy())
        .finished();
  }

  // Jacobians of project
  void projectJacobians(const Eigen::Vector2d& pc,
                        Eigen::Matrix<double, 2, 4>& J_K,
                        Eigen::Matrix<double, 2, 2>& J_p) const;

  // convert image coordinates uv (pixel unit) to camera coordinates xy
  // pc = invK * pi
  Eigen::Vector2d unproject(const Eigen::Vector2d& pi) const {
    return (Eigen::Vector2d() << (pi(0) - cx()) / fx(), (pi(1) - cy()) / fy())
        .finished();
  }

  // Jacobians of unproject. Note this is not required by traits
  void unprojectJacobians(const Eigen::Vector2d& pi,
                          Eigen::Matrix<double, 2, 4>& J_K,
                          Eigen::Matrix<double, 2, 2>& J_p) const;
};

// traits of CalibK
template <>
struct traits<CalibK> : internal::CameraIntrinsicsTraitsImpl<CalibK> {};

}  // namespace minisam
