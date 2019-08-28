/**
 * @file    BALInterface.h
 * @brief   File interface of Bundler BAL format
 * @author  Jing Dong
 * @date    Oct 18, 2018
 */

#pragma once

#include <minisam/geometry/CalibBundler.h>
#include <minisam/geometry/projection.h>

#include <Eigen/Core>
#include <iostream>
#include <sophus/se3.hpp>
#include <vector>

namespace minisam {

/**
 * data structure for bundle adjustment
 */

// data structure for BA results/groud truth storage
template <class CALIBRATION>
struct BAdataset {
  std::vector<Sophus::SE3d> poses;
  std::vector<CALIBRATION> calibrations;
  std::vector<Eigen::Vector3d> lands;
};

// a projection measurement
struct BAmeasurement {
  size_t pose_idx;
  size_t land_idx;
  Eigen::Vector2d p_measured;

  BAmeasurement(size_t pi, size_t li, const Eigen::Vector2d& pm)
      : pose_idx(pi), land_idx(li), p_measured(pm) {}
};

// data structure for BA problem storage
template <class CALIBRATION>
struct BAproblem {
  BAdataset<CALIBRATION> init_values;
  std::vector<BAmeasurement> measurements;
};

/**
 * function to read BAL files
 */
BAproblem<CalibBundler> loadBAL(const std::string& filename);

/**
 * synthetic dataset for testing bundle adjustment
 */

// add noise to pose
Sophus::SE3d addPoseNoise(const Sophus::SE3d& p, double rot_noise,
                          double trans_noise);

// add noise to point
Eigen::Vector3d addPointNoise(const Eigen::Vector3d& p, double p_noise);
Eigen::Vector2d addPointNoise(const Eigen::Vector2d& p, double p_noise);

// prepare a fully-connected synthetic BA problem
template <class CALIBRATION>
BAproblem<CALIBRATION> syntheticBA(const BAdataset<CALIBRATION>& ground_truth,
                                   double init_pose_rot_noise,
                                   double init_pose_trans_noise,
                                   double init_land_noise, double image_noise) {
  BAproblem<CALIBRATION> ba_data;
  ba_data.init_values.calibrations = ground_truth.calibrations;
  // init pose values
  for (size_t i = 0; i < ground_truth.poses.size(); i++) {
    ba_data.init_values.poses.push_back(addPoseNoise(
        ground_truth.poses[i], init_pose_rot_noise, init_pose_trans_noise));
  }
  // init land values
  for (size_t i = 0; i < ground_truth.lands.size(); i++) {
    ba_data.init_values.lands.push_back(
        addPointNoise(ground_truth.lands[i], init_land_noise));
  }
  // projection measurements
  for (size_t i = 0; i < ground_truth.poses.size(); i++) {
    for (size_t j = 0; j < ground_truth.lands.size(); j++) {
      Eigen::Vector2d pp =
          project(ground_truth.poses[i], ground_truth.calibrations[i],
                  ground_truth.lands[j]);
      // cout << i << ", " << j << " : " << pp.transpose() << endl;
      ba_data.measurements.push_back(
          BAmeasurement(i, j, addPointNoise(pp, image_noise)));
    }
  }
  return ba_data;
}

// prepare a fully-connected synthetic BA problem, Bundler/OpenGL format pose
BAproblem<CalibBundler> syntheticBundlerBA(
    const BAdataset<CalibBundler>& ground_truth, double init_pose_rot_noise,
    double init_pose_trans_noise, double init_land_noise, double image_noise);

}  // namespace minisam
