/**
 * @file    CalibK.cpp
 * @brief   intrinsic camera calibration without distortion, representing a 3x3
 * K matrix
 * @author  Jing Dong
 * @date    Oct 9, 2018
 */

#include <minisam/geometry/CalibK.h>

namespace minisam {

/* ************************************************************************** */
void CalibK::projectJacobians(const Eigen::Vector2d& pc,
                              Eigen::Matrix<double, 2, 4>& J_K,
                              Eigen::Matrix<double, 2, 2>& J_p) const {
  // clang-format off
  J_K <<  pc(0),  0.0,    1.0,  0.0,
          0.0,    pc(1),  0.0,  1.0;
  J_p <<  fx(), 0.0,
          0.0,  fy();
  // clang-format on
}

/* ************************************************************************** */
void CalibK::unprojectJacobians(const Eigen::Vector2d& pi,
                                Eigen::Matrix<double, 2, 4>& J_K,
                                Eigen::Matrix<double, 2, 2>& J_p) const {
  // clang-format off
  J_K << -(pi(0)-cx())/(fx()*fx()),  0.0,   -1.0/fx(),  0.0,
          0.0,   -(pi(1)-cy())/(fy()*fy()),  0.0,      -1.0/fy();
  J_p <<  1.0/fx(),   0.0,
          0.0,        1.0/fy();
  // clang-format on
}

}  // namespace minisam
