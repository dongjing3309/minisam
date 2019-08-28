/**
 * @file    Projection.h
 * @brief   3D single-view projection functions
 * @author  Jing Dong
 * @date    Oct 11, 2018
 */

#pragma once

#include <minisam/core/Traits.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace minisam {

// forward declearation
class CalibBundler;

// transfrom a point in world coordinates to sensor coordinates
inline Eigen::Vector3d transform2sensor(const Sophus::SE3d& pose,
                                        const Eigen::Vector3d& pw) {
  return pose.so3().inverse() * (pw - pose.translation());
}

// jacobians of transform2camera of pose and pw
void transform2sensorJacobians(const Sophus::SE3d& pose,
                               const Eigen::Vector3d& pw,
                               Eigen::Matrix<double, 3, 6>& J_pose,
                               Eigen::Matrix<double, 3, 3>& J_pw);

// transfrom a point in sensor coordinates to world coordinates
inline Eigen::Vector3d transform2world(const Sophus::SE3d& pose,
                                       const Eigen::Vector3d& ps) {
  return pose * ps;
}

// jacobians of transform2world of pose and ps
void transform2worldJacobians(const Sophus::SE3d& pose,
                              const Eigen::Vector3d& ps,
                              Eigen::Matrix<double, 3, 6>& J_pose,
                              Eigen::Matrix<double, 3, 3>& J_ps);

// project a point in world coordinates on image plane, but in mertic unit
// (image)
inline Eigen::Vector2d transform2image(const Sophus::SE3d& pose,
                                       const Eigen::Vector3d& pw) {
  const Eigen::Vector3d ps = transform2sensor(pose, pw);
  return Eigen::Vector2d(ps(0) / ps(2), ps(1) / ps(2));
}

// jacobians of transform2world of pose and pc
void transform2imageJacobians(const Sophus::SE3d& pose,
                              const Eigen::Vector3d& pw,
                              Eigen::Matrix<double, 2, 6>& J_pose,
                              Eigen::Matrix<double, 2, 3>& J_pw);

// project 3D point in world frame to 2D point in image frame
template <class CALIBRATION>
Eigen::Vector2d project(const Sophus::SE3d& pose, const CALIBRATION& calib,
                        const Eigen::Vector3d& pw) {
  // check T is camera intrinsics type
  static_assert(
      is_camera_intrinsics<CALIBRATION>::value,
      "Variable type T in project<T> must be a camera intrinsics type");

  return traits<CALIBRATION>::project(calib, transform2image(pose, pw));
}

// jacobians of project of pose, calibration, and pw
template <class CALIBRATION>
void projectJacobians(const Sophus::SE3d& pose, const CALIBRATION& calib,
                      const Eigen::Vector3d& pw,
                      Eigen::Matrix<double, 2, 6>& J_pose,
                      Eigen::Matrix<double, 2, CALIBRATION::dim()>& J_calib,
                      Eigen::Matrix<double, 2, 3>& J_pw) {
  // check T is camera intrinsics type
  static_assert(is_camera_intrinsics<CALIBRATION>::value,
                "Variable type T in projectJacobians<T> must be a camera "
                "intrinsics type");

  Eigen::Matrix<double, 2, 6> J_pc_pose;
  Eigen::Matrix<double, 2, 3> J_pc_pw;
  Eigen::Vector2d pc = transform2image(pose, pw);
  transform2imageJacobians(pose, pw, J_pc_pose, J_pc_pw);

  Eigen::Matrix<double, 2, 2> J_pc;
  traits<CALIBRATION>::projectJacobians(calib, pc, J_calib, J_pc);

  J_pose = J_pc * J_pc_pose;
  J_pw = J_pc * J_pc_pw;
}

/**
 * Project 3D point in world frame to 2D point in image frame
 * using Bundler/OpenGL's pose convension
 *
 *   P = R * X + t       (conversion from world to camera coordinates)
 *   p = -P / P.z        (perspective division)
 *   p' = f * r(p) * p   (conversion to pixel coordinates)
 *
 * Note that the pose convension is Bundler is different from what we used
 * since in Bundler/OpenGL the camera center is -R' * t, and camera look at
 * negative z direction
 *
 * see http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html
 * see http://grail.cs.washington.edu/projects/bal/
 */
Eigen::Vector2d projectBundler(const Sophus::SE3d& pose,
                               const CalibBundler& calib,
                               const Eigen::Vector3d& pw);

void projectBundlerJacobians(const Sophus::SE3d& pose,
                             const CalibBundler& calib,
                             const Eigen::Vector3d& pw,
                             Eigen::Matrix<double, 2, 6>& J_pose,
                             Eigen::Matrix<double, 2, 3>& J_calib,
                             Eigen::Matrix<double, 2, 3>& J_pw);

}  // namespace minisam
