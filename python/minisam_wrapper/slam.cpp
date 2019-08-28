/**
 * @file    slam.cpp
 * @author  Jing Dong
 * @date    Nov 18, 2017
 */

#include "print.h"
#include "pyobject_traits.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <minisam/core/Eigen.h>
#include <minisam/core/Scalar.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/geometry/Sophus.h>
#include <minisam/geometry/CalibK.h>
#include <minisam/geometry/CalibKD.h>
#include <minisam/geometry/CalibBundler.h>

#include <minisam/slam/PriorFactor.h>
#include <minisam/slam/BetweenFactor.h>
#include <minisam/slam/ReprojectionFactor.h>
#include <minisam/slam/g2oInterface.h>
#include <minisam/slam/BALInterface.h>


using namespace minisam;
namespace py = pybind11;


// wrap prior factor, static and dynamic typed
#define WRAP_TYPE_PRIOR_FACTOR(T, TYPENAME) \
  py::class_<PriorFactor<T>, Factor, std::shared_ptr<PriorFactor<T>>>(m, \
      std::string(std::string("PriorFactor_") + TYPENAME + "_").c_str()) \
    .def(py::init<Key, const T&, const std::shared_ptr<LossFunction>&>()); \
  m.def("PriorFactor", [](Key k, const T& p, const std::shared_ptr<LossFunction>& l) { \
        return PriorFactor<T>(k, p, l); \
      });

// wrap between factor, static and dynamic typed
#define WRAP_TYPE_BETWEEN_FACTOR(T, TYPENAME) \
  py::class_<BetweenFactor<T>, Factor, std::shared_ptr<BetweenFactor<T>>>(m, \
      std::string(std::string("BetweenFactor_") + TYPENAME + "_").c_str()) \
    .def(py::init<Key, Key, const T&, const std::shared_ptr<LossFunction>&>()); \
  m.def("BetweenFactor", [](Key k1, Key k2, const T& d, const std::shared_ptr<LossFunction>& l) { \
        return BetweenFactor<T>(k1, k2, d, l); \
      });

// wrap reprojection factor, only static typed
#define WRAP_TYPE_REPROJECTION_FACTOR(T, TYPENAME) \
  py::class_<ReprojectionFactor<T>, Factor, std::shared_ptr<ReprojectionFactor<T>>>(m, \
      std::string(std::string("ReprojectionFactor_") + TYPENAME + "_").c_str()) \
    .def(py::init<Key, Key, Key, const Eigen::Vector2d&, const std::shared_ptr<LossFunction>&>(), \
        py::arg("key_pose"), py::arg("key_calib"), py::arg("key_land"), py::arg("p"), \
        py::arg("lossfunc") = nullptr);

// wrap reprojection pose factor, static and dynamic typed
#define WRAP_TYPE_REPROJECTION_POSE_FACTOR(T, TYPENAME) \
  py::class_<ReprojectionPoseFactor<T>, Factor, std::shared_ptr<ReprojectionPoseFactor<T>>>(m, \
      std::string(std::string("ReprojectionPoseFactor_") + TYPENAME + "_").c_str()) \
    .def(py::init<Key, Key, const std::shared_ptr<T>&, const Eigen::Vector2d&, \
        const std::shared_ptr<LossFunction>&>(), py::arg("key_pose"), py::arg("key_land"), \
        py::arg("calib"), py::arg("p"), py::arg("lossfunc") = nullptr); \
  m.def("ReprojectionPoseFactor", [](Key k1, Key k2, const std::shared_ptr<T>& c, \
          const Eigen::Vector2d& p, const std::shared_ptr<LossFunction>& l) { \
        return ReprojectionPoseFactor<T>(k1, k2, c, p, l); \
      }, py::arg("key_pose"), py::arg("key_land"), py::arg("calib"), py::arg("p"), \
      py::arg("lossfunc") = nullptr);

// wrap bundle adjustment classes
#define WRAP_BUNDLE_ADJUSTMENT_CLASSES(T, TYPENAME) \
  py::class_<BAdataset<T>>(m, std::string(std::string("BAdataset_") + TYPENAME + "_").c_str()) \
    .def_readwrite("poses", &BAdataset<T>::poses) \
    .def_readwrite("calibrations", &BAdataset<T>::calibrations) \
    .def_readwrite("lands", &BAdataset<T>::lands); \
  py::class_<BAproblem<T>>(m, std::string(std::string("BAproblem_") + TYPENAME + "_").c_str()) \
    .def_readwrite("init_values", &BAproblem<T>::init_values) \
    .def_readwrite("measurements", &BAproblem<T>::measurements);


void wrap_slam(py::module& m) {

  // prior factor
  WRAP_TYPE_PRIOR_FACTOR(Sophus::SE2d, "SE2")
  WRAP_TYPE_PRIOR_FACTOR(Sophus::SE3d, "SE3")
  WRAP_TYPE_PRIOR_FACTOR(Sophus::SO2d, "SO2")
  WRAP_TYPE_PRIOR_FACTOR(Sophus::SO3d, "SO3")

  WRAP_TYPE_PRIOR_FACTOR(CalibK, "CalibK")
  WRAP_TYPE_PRIOR_FACTOR(CalibKD, "CalibKD")
  WRAP_TYPE_PRIOR_FACTOR(CalibBundler, "CalibBundler")

  WRAP_TYPE_PRIOR_FACTOR(double, "double")
  WRAP_TYPE_PRIOR_FACTOR(Eigen::Vector2d, "Vector2")
  WRAP_TYPE_PRIOR_FACTOR(Eigen::Vector3d, "Vector3")
  WRAP_TYPE_PRIOR_FACTOR(Eigen::Vector4d, "Vector4")
  WRAP_TYPE_PRIOR_FACTOR(Eigen::VectorXd, "Vector")

  WRAP_TYPE_PRIOR_FACTOR(py::object, "pyobject") // py::obj should be last

  // between factor
  WRAP_TYPE_BETWEEN_FACTOR(Sophus::SE2d, "SE2")
  WRAP_TYPE_BETWEEN_FACTOR(Sophus::SE3d, "SE3")
  WRAP_TYPE_BETWEEN_FACTOR(Sophus::SO2d, "SO2")
  WRAP_TYPE_BETWEEN_FACTOR(Sophus::SO3d, "SO3")
  
  WRAP_TYPE_BETWEEN_FACTOR(double, "double")
  WRAP_TYPE_BETWEEN_FACTOR(Eigen::Vector2d, "Vector2")
  WRAP_TYPE_BETWEEN_FACTOR(Eigen::Vector3d, "Vector3")
  WRAP_TYPE_BETWEEN_FACTOR(Eigen::Vector4d, "Vector4")
  WRAP_TYPE_BETWEEN_FACTOR(Eigen::VectorXd, "Vector")

  WRAP_TYPE_BETWEEN_FACTOR(py::object, "pyobject") // py::obj should be last

  // projection factors
  WRAP_TYPE_REPROJECTION_FACTOR(CalibK, "CalibK")
  WRAP_TYPE_REPROJECTION_FACTOR(CalibKD, "CalibKD")
  WRAP_TYPE_REPROJECTION_FACTOR(CalibBundler, "CalibBundler")

  WRAP_TYPE_REPROJECTION_POSE_FACTOR(CalibK, "CalibK")
  WRAP_TYPE_REPROJECTION_POSE_FACTOR(CalibKD, "CalibKD")
  WRAP_TYPE_REPROJECTION_POSE_FACTOR(CalibBundler, "CalibBundler")

  py::class_<ReprojectionBundlerFactor, Factor, std::shared_ptr<ReprojectionBundlerFactor>>(m, 
      "ReprojectionBundlerFactor")
    .def(py::init<Key, Key, Key, const Eigen::Vector2d&, const std::shared_ptr<LossFunction>&>());

  py::class_<BAmeasurement>(m, "BAmeasurement")
    .def(py::init<size_t, size_t, const Eigen::Vector2d&>())
    .def_readwrite("pose_idx", &BAmeasurement::pose_idx)
    .def_readwrite("land_idx", &BAmeasurement::land_idx)
    .def_readwrite("p_measured", &BAmeasurement::p_measured);

  WRAP_BUNDLE_ADJUSTMENT_CLASSES(CalibK, "CalibK")
  WRAP_BUNDLE_ADJUSTMENT_CLASSES(CalibKD, "CalibKD")
  WRAP_BUNDLE_ADJUSTMENT_CLASSES(CalibBundler, "CalibBundler")

  // slam util
  m.def("loadG2O", &loadG2O);
  m.def("loadBAL", &loadBAL);
}