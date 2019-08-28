/**
 * @file    ReprojectionFactor.cpp
 * @brief   Factor for reprojection error
 * @author  Jing Dong
 * @date    Oct 15, 2018
 */

#include <minisam/core/Eigen.h>
#include <minisam/geometry/CalibBundler.h>
#include <minisam/geometry/Sophus.h>
#include <minisam/slam/ReprojectionFactor.h>

namespace minisam {

/* ************************************************************************** */
void ReprojectionBundlerFactor::print(std::ostream& out) const {
  out << "Reprojection (Bundler) Factor, measured = ["
      << p_measured_.transpose() << "]'" << std::endl;
  Factor::print(out);
}

/* ************************************************************************** */
std::shared_ptr<Factor> ReprojectionBundlerFactor::copy() const {
  return std::shared_ptr<Factor>(new ReprojectionBundlerFactor(*this));
}

/* ************************************************************************** */
Eigen::VectorXd ReprojectionBundlerFactor::error(
    const Variables& values) const {
  const Sophus::SE3d& pose = values.at<Sophus::SE3d>(keys()[0]);
  const CalibBundler& calib = values.at<CalibBundler>(keys()[1]);
  const Eigen::Vector3d& land = values.at<Eigen::Vector3d>(keys()[2]);
  return projectBundler(pose, calib, land) - p_measured_;
}

/* ************************************************************************** */
std::vector<Eigen::MatrixXd> ReprojectionBundlerFactor::jacobians(
    const Variables& values) const {
  const Sophus::SE3d& pose = values.at<Sophus::SE3d>(keys()[0]);
  const CalibBundler& calib = values.at<CalibBundler>(keys()[1]);
  const Eigen::Vector3d& land = values.at<Eigen::Vector3d>(keys()[2]);
  Eigen::Matrix<double, 2, 6> J_pose;
  Eigen::Matrix<double, 2, 3> J_calib;
  Eigen::Matrix<double, 2, 3> J_land;
  projectBundlerJacobians(pose, calib, land, J_pose, J_calib, J_land);
  return std::vector<Eigen::MatrixXd>{J_pose, J_calib, J_land};
}

}  // namespace minisam
