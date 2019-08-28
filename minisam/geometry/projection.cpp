/**
 * @file    Projection.h
 * @brief   3D single-view projection functions
 * @author  Jing Dong
 * @date    Oct 11, 2018
 */

#include <minisam/geometry/CalibBundler.h>
#include <minisam/geometry/projection.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace minisam {

namespace {
// 3x3 skew symmetric matrix from a Vector3d
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  // clang-format off
  m <<  0.0,  -v(2),  v(1), 
        v(2),  0.0,  -v(0),  
       -v(1),  v(0),  0.0;
  // clang-format on
  return m;
}
}  // namespace

/* ************************************************************************** */
void transform2sensorJacobians(const Sophus::SE3d& pose,
                               const Eigen::Vector3d& pw,
                               Eigen::Matrix<double, 3, 6>& J_pose,
                               Eigen::Matrix<double, 3, 3>& J_pw) {
  Eigen::Vector3d pc = transform2sensor(pose, pw);
  J_pose << -Eigen::Matrix3d::Identity(), skewSymmetric(pc);
  J_pw = pose.so3().inverse().matrix();
}

/* ************************************************************************** */
void transform2worldJacobians(const Sophus::SE3d& pose,
                              const Eigen::Vector3d& ps,
                              Eigen::Matrix<double, 3, 6>& J_pose,
                              Eigen::Matrix<double, 3, 3>& J_ps) {
  Eigen::Matrix3d R = pose.so3().matrix();
  J_pose << R, R * skewSymmetric(-ps);
  J_ps = R;
}

/* ************************************************************************** */
void transform2imageJacobians(const Sophus::SE3d& pose,
                              const Eigen::Vector3d& pw,
                              Eigen::Matrix<double, 2, 6>& J_pose,
                              Eigen::Matrix<double, 2, 3>& J_pw) {
  // see gtsam/geometry/CalibratedCamera.cpp
  Eigen::Vector3d ps = transform2sensor(pose, pw);
  Eigen::Matrix3d Rt = pose.so3().inverse().matrix();
  const double u = ps(0) / ps(2);
  const double v = ps(1) / ps(2);
  const double d = 1.0 / ps(2);
  const double uv = u * v;
  const double uu = u * u;
  const double vv = v * v;
  // clang-format off
  J_pose << -d,   0,   d*u,   uv,   -1-uu,  v,
            0,   -d,   d*v,   1+vv, -uv,   -u;
  J_pw << Rt(0, 0)-u*Rt(2, 0), Rt(0, 1)-u*Rt(2, 1), Rt(0, 2)-u*Rt(2, 2),
          Rt(1, 0)-v*Rt(2, 0), Rt(1, 1)-v*Rt(2, 1), Rt(1, 2)-v*Rt(2, 2);
  // clang-format on
  J_pw *= d;
}

/* ************************************************************************** */
Eigen::Vector2d projectBundler(const Sophus::SE3d& pose,
                               const CalibBundler& calib,
                               const Eigen::Vector3d& pw) {
  Eigen::Vector3d pc = pose * pw;
  double invz = 1.0 / pc(2);
  Eigen::Vector2d pi(-pc(0) * invz, -pc(1) * invz);
  return calib.project(pi);
}

/* ************************************************************************** */
void projectBundlerJacobians(const Sophus::SE3d& pose,
                             const CalibBundler& calib,
                             const Eigen::Vector3d& pw,
                             Eigen::Matrix<double, 2, 6>& J_pose,
                             Eigen::Matrix<double, 2, 3>& J_calib,
                             Eigen::Matrix<double, 2, 3>& J_pw) {
  Eigen::Vector3d pc = pose * pw;
  double invz = 1.0 / pc(2);
  double invz2 = invz * invz;
  Eigen::Vector2d pi(-pc(0) * invz, -pc(1) * invz);

  Eigen::Matrix<double, 2, 2> J_pi;
  calib.projectJacobians(pi, J_calib, J_pi);

  Eigen::Matrix<double, 2, 3> J_pc;
  // clang-format off
  J_pc << -invz*J_pi(0,0), -invz*J_pi(0,1), (pc(0)*J_pi(0,0) + pc(1)*J_pi(0,1)) * invz2,
          -invz*J_pi(1,0), -invz*J_pi(1,1), (pc(0)*J_pi(1,0) + pc(1)*J_pi(1,1)) * invz2;
  // clang-format on

  Eigen::Matrix3d R = pose.so3().matrix();
  Eigen::Matrix<double, 2, 3> J_pc_R = J_pc * R;
  J_pose << J_pc_R,
      J_pc_R * skewSymmetric(-pw);  // J_pc_pose << R, R * skew_symmetric(-pw);
  J_pw = J_pc_R;                    // J_pc_pw = R
}

}  // namespace minisam
