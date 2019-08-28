/**
 * @file    BALInterface.cpp
 * @brief   File interface of Bundler BAL format
 * @author  Jing Dong
 * @date    Oct 18, 2018
 */

#include <minisam/slam/BALInterface.h>

#include <fstream>

using namespace std;
using namespace Eigen;

namespace minisam {

// random generator
static default_random_engine rand_gen;
static normal_distribution<double> dist;

/* ************************************************************************** */
BAproblem<CalibBundler> loadBAL(const std::string& filename) {
  BAproblem<CalibBundler> ba_data;

  // open file
  ifstream fs(filename.c_str(), ifstream::in);
  if (!fs) {
    throw runtime_error("[loadBAL] Cannot load BAL file : " + filename);
  }

  // size
  size_t nr_poses, nr_point, nr_measurements;
  fs >> nr_poses >> nr_point >> nr_measurements;
  ba_data.measurements.reserve(nr_measurements);
  ba_data.init_values.calibrations.reserve(nr_poses);
  ba_data.init_values.poses.reserve(nr_poses);
  ba_data.init_values.lands.reserve(nr_point);

  // measurements
  size_t pose_idx, land_idx;
  double u, v;
  for (size_t i = 0; i < nr_measurements; i++) {
    fs >> pose_idx >> land_idx >> u >> v;
    ba_data.measurements.push_back(
        BAmeasurement(pose_idx, land_idx, Vector2d(u, v)));
  }

  // camera
  double rx, ry, rz, tx, ty, tz, f, k1, k2;
  for (size_t i = 0; i < nr_poses; i++) {
    fs >> rx >> ry >> rz;
    fs >> tx >> ty >> tz;
    ba_data.init_values.poses.push_back(Sophus::SE3d(
        Sophus::SO3d::exp(Vector3d(rx, ry, rz)), Vector3d(tx, ty, tz)));
    fs >> f >> k1 >> k2;
    ba_data.init_values.calibrations.push_back(CalibBundler(f, k1, k2));
  }

  // landmark
  double lx, ly, lz;
  for (size_t i = 0; i < nr_point; i++) {
    fs >> lx >> ly >> lz;
    ba_data.init_values.lands.push_back(Vector3d(lx, ly, lz));
  }

  return ba_data;
}

/* ************************************************************************** */
Sophus::SE3d addPoseNoise(const Sophus::SE3d& p, double rot_noise,
                          double trans_noise) {
  // clang-format off
  return p * Sophus::SE3d::exp((Eigen::Matrix<double, 6, 1>() <<
      dist(rand_gen) * trans_noise, dist(rand_gen) * trans_noise,
      dist(rand_gen) * trans_noise, dist(rand_gen) * rot_noise,
      dist(rand_gen) * rot_noise, dist(rand_gen) * rot_noise).finished());
  // clang-format on
}

/* ************************************************************************** */
Vector3d addPointNoise(const Vector3d& p, double p_noise) {
  return p + Vector3d(dist(rand_gen) * p_noise, dist(rand_gen) * p_noise,
                      dist(rand_gen) * p_noise);
}

/* ************************************************************************** */
Vector2d addPointNoise(const Vector2d& p, double p_noise) {
  return p + Vector2d(dist(rand_gen) * p_noise, dist(rand_gen) * p_noise);
}

/* ************************************************************************** */
BAproblem<CalibBundler> syntheticBundlerBA(
    const BAdataset<CalibBundler>& ground_truth, double init_pose_rot_noise,
    double init_pose_trans_noise, double init_land_noise, double image_noise) {
  BAproblem<CalibBundler> ba_data;
  // init calibration: no distortion
  for (size_t i = 0; i < ground_truth.poses.size(); i++) {
    ba_data.init_values.calibrations.push_back(
        CalibBundler(ground_truth.calibrations[i].f(), 0.0, 0.0));
  }
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
      Vector2d pp =
          projectBundler(ground_truth.poses[i], ground_truth.calibrations[i],
                         ground_truth.lands[j]);
      // cout << i << ", " << j << " : " << pp.transpose() << endl;
      ba_data.measurements.push_back(
          BAmeasurement(i, j, addPointNoise(pp, image_noise)));
    }
  }
  return ba_data;
}

}  // namespace minisam
