/**
 * @file    ReprojectionFactor.h
 * @brief   Factor for reprojection error
 * @author  Jing Dong
 * @date    Oct 15, 2018
 */

#pragma once

#include <minisam/core/Factor.h>
#include <minisam/core/Traits.h>
#include <minisam/core/Variables.h>
#include <minisam/geometry/projection.h>

namespace minisam {

template <class CALIBRATION>
class ReprojectionFactor : public Factor {
  // check T is camera intrinsics type
  static_assert(
      is_camera_intrinsics<CALIBRATION>::value,
      "Variable type T in ReprojectionFactor must be a camera intrinsics type");

 private:
  Eigen::Vector2d p_measured_;  // image point measured

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionFactor(Key key_pose, Key key_calib, Key key_land,
                     const Eigen::Vector2d& p,
                     const std::shared_ptr<LossFunction>& lossfunc = nullptr)
      : Factor(2, std::vector<Key>{key_pose, key_calib, key_land}, lossfunc),
        p_measured_(p) {}

  virtual ~ReprojectionFactor() = default;

  /** factor implementation */

  // print
  void print(std::ostream& out = std::cout) const override {
    out << "Reprojection Factor, measured = [" << p_measured_.transpose()
        << "]'" << std::endl;
    Factor::print(out);
  }

  // deep copy function
  std::shared_ptr<Factor> copy() const override {
    return std::shared_ptr<Factor>(new ReprojectionFactor(*this));
  }

  // error function
  Eigen::VectorXd error(const Variables& values) const override {
    const Sophus::SE3d& pose = values.at<Sophus::SE3d>(keys()[0]);
    const CALIBRATION& calib = values.at<CALIBRATION>(keys()[1]);
    const Eigen::Vector3d& land = values.at<Eigen::Vector3d>(keys()[2]);

    return project(pose, calib, land) - p_measured_;
  }

  // jacobians function
  std::vector<Eigen::MatrixXd> jacobians(
      const Variables& values) const override {
    const Sophus::SE3d& pose = values.at<Sophus::SE3d>(keys()[0]);
    const CALIBRATION& calib = values.at<CALIBRATION>(keys()[1]);
    const Eigen::Vector3d& land = values.at<Eigen::Vector3d>(keys()[2]);
    Eigen::Matrix<double, 2, 6> J_pose;
    Eigen::Matrix<double, 2, CALIBRATION::dim()> J_calib;
    Eigen::Matrix<double, 2, 3> J_land;

    projectJacobians(pose, calib, land, J_pose, J_calib, J_land);
    return std::vector<Eigen::MatrixXd>{J_pose, J_calib, J_land};
  }
};

template <class CALIBRATION>
class ReprojectionPoseFactor : public Factor {
  // check T is camera intrinsics type
  static_assert(is_camera_intrinsics<CALIBRATION>::value,
                "Variable type T in ReprojectionPoseFactor must be a camera "
                "intrinsics type");

 private:
  std::shared_ptr<CALIBRATION> ptr_calib_;
  Eigen::Vector2d p_measured_;  // image point measured

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionPoseFactor(
      Key key_pose, Key key_land, const std::shared_ptr<CALIBRATION>& ptr_calib,
      const Eigen::Vector2d& p,
      const std::shared_ptr<LossFunction>& lossfunc = nullptr)
      : Factor(2, std::vector<Key>{key_pose, key_land}, lossfunc),
        ptr_calib_(ptr_calib),
        p_measured_(p) {}

  virtual ~ReprojectionPoseFactor() = default;

  /** factor implementation */

  // print
  void print(std::ostream& out = std::cout) const override {
    out << "Reprojection Pose Factor, measured = [" << p_measured_.transpose()
        << "]'" << std::endl;
    out << "Calibration : ";
    ptr_calib_->print(out);
    out << std::endl;
    Factor::print(out);
  }

  // deep copy function
  std::shared_ptr<Factor> copy() const override {
    return std::shared_ptr<Factor>(new ReprojectionPoseFactor(*this));
  }

  // error function
  Eigen::VectorXd error(const Variables& values) const override {
    const Sophus::SE3d& pose = values.at<Sophus::SE3d>(keys()[0]);
    const Eigen::Vector3d& land = values.at<Eigen::Vector3d>(keys()[1]);

    return project(pose, *ptr_calib_, land) - p_measured_;
  }

  // jacobians function
  std::vector<Eigen::MatrixXd> jacobians(
      const Variables& values) const override {
    const Sophus::SE3d& pose = values.at<Sophus::SE3d>(keys()[0]);
    const Eigen::Vector3d& land = values.at<Eigen::Vector3d>(keys()[1]);
    Eigen::Matrix<double, 2, 6> J_pose;
    Eigen::Matrix<double, 2, CALIBRATION::dim()> J_calib;
    Eigen::Matrix<double, 2, 3> J_land;

    projectJacobians(pose, *ptr_calib_, land, J_pose, J_calib, J_land);
    return std::vector<Eigen::MatrixXd>{J_pose, J_land};
  }
};

class ReprojectionBundlerFactor : public Factor {
 private:
  Eigen::Vector2d p_measured_;  // image point measured

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionBundlerFactor(
      Key key_pose, Key key_calib, Key key_land, const Eigen::Vector2d& p,
      const std::shared_ptr<LossFunction>& lossfunc = nullptr)
      : Factor(2, std::vector<Key>{key_pose, key_calib, key_land}, lossfunc),
        p_measured_(p) {}

  virtual ~ReprojectionBundlerFactor() = default;

  /** factor implementation */

  // print
  void print(std::ostream& out = std::cout) const override;

  // deep copy function
  std::shared_ptr<Factor> copy() const override;

  // error function
  Eigen::VectorXd error(const Variables& values) const override;

  // jacobians function
  std::vector<Eigen::MatrixXd> jacobians(
      const Variables& values) const override;
};

}  // namespace minisam
