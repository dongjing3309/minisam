/**
 * @file    geometry.cpp
 * @author  Jing Dong
 * @date    Nov 18, 2017
 */

#include "print.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <minisam/geometry/CalibK.h>
#include <minisam/geometry/CalibKD.h>
#include <minisam/geometry/CalibBundler.h>
#include <minisam/geometry/projection.h>

#include <tuple>


using namespace minisam;
namespace py = pybind11;


// wrap calibration type basics
#define WRAP_TYPE_CALIBRATION(T) \
  .def_static("dim", &T::dim) \
  .def("vector", &T::vector) \
  .def("matrix", &T::matrix) \
  .def("inverse_matrix", &T::inverse_matrix) \
  .def("project", &T::project) \
  .def("projectJacobians", &T::projectJacobians) \
  .def("unproject", &T::unproject)


// wrap projection and its jacobians Eigen by-reference interface
#define WRAP_TYPE_PROJECTION(T, TYPENAME) \
  m.def(std::string(std::string("project_") + TYPENAME + "_").c_str(), &project<T>); \
  m.def("project", &project<T>); \
  m.def(std::string(std::string("projectJacobians_") + TYPENAME + "_").c_str(), \
    [](const Sophus::SE3d& pose, const T& calib, const Eigen::Vector3d& pw) { \
      Eigen::Matrix<double, 2, 6> J_pose_in; \
      Eigen::Matrix<double, 2, T::dim()> J_calib_in; \
      Eigen::Matrix<double, 2, 3> J_pw_in; \
      projectJacobians<T>(pose, calib, pw, J_pose_in, J_calib_in, J_pw_in); \
      return std::make_tuple(J_pose_in, J_calib_in, J_pw_in); \
    }); \
  m.def("projectJacobians", [](const Sophus::SE3d& pose, const T& calib, const Eigen::Vector3d& pw) { \
      Eigen::Matrix<double, 2, 6> J_pose_in; \
      Eigen::Matrix<double, 2, T::dim()> J_calib_in; \
      Eigen::Matrix<double, 2, 3> J_pw_in; \
      projectJacobians<T>(pose, calib, pw, J_pose_in, J_calib_in, J_pw_in); \
      return std::make_tuple(J_pose_in, J_calib_in, J_pw_in); \
    });

void wrap_geometry(py::module& m) {

  // CalibK
  py::class_<CalibK, std::shared_ptr<CalibK>>(m, "CalibK")
    // type particular ctor
    .def(py::init<double, double, double, double>())
    .def(py::init<const Eigen::Matrix<double, 4, 1>&>())
    // type particular
    .def("fx", &CalibK::fx)
    .def("fy", &CalibK::fy)
    .def("cx", &CalibK::cx)
    .def("cy", &CalibK::cy)
    .def("unprojectJacobians", &CalibK::unprojectJacobians)
    // lie group
    WRAP_TYPE_CALIBRATION(CalibK)
    WRAP_TYPE_PYTHON_PRINT_NOT_REMOVE_LAST(CalibK)
    ;

  // CalibKD
  py::class_<CalibKD, std::shared_ptr<CalibKD>>(m, "CalibKD")
    // type particular ctor
    .def(py::init<double, double, double, double, double, double, double, double>())
    .def(py::init<const Eigen::Matrix<double, 8, 1>&>())
    // type particular
    .def("fx", &CalibKD::fx)
    .def("fy", &CalibKD::fy)
    .def("cx", &CalibKD::cx)
    .def("cy", &CalibKD::cy)
    .def("k1", &CalibKD::k1)
    .def("k2", &CalibKD::k2)
    .def("p1", &CalibKD::p1)
    .def("p2", &CalibKD::p2)
    // lie group
    WRAP_TYPE_CALIBRATION(CalibKD)
    WRAP_TYPE_PYTHON_PRINT_NOT_REMOVE_LAST(CalibKD)
    ;

  // CalibBundler
  py::class_<CalibBundler, std::shared_ptr<CalibBundler>>(m, "CalibBundler")
    // type particular ctor
    .def(py::init<double, double, double>())
    .def(py::init<const Eigen::Matrix<double, 3, 1>&>())
    // type particular
    .def("f", &CalibBundler::f)
    .def("k1", &CalibBundler::k1)
    .def("k2", &CalibBundler::k2)
    // lie group
    WRAP_TYPE_CALIBRATION(CalibBundler)
    WRAP_TYPE_PYTHON_PRINT_NOT_REMOVE_LAST(CalibBundler)
    ;


  // coordinate frame transformation
  m.def("transform2sensor", &transform2sensor);
  m.def("transform2sensorJacobians", [](const Sophus::SE3d& pose, const Eigen::Vector3d& pw) {
        Eigen::Matrix<double, 3, 6> J_pose_in;
        Eigen::Matrix<double, 3, 3> J_pw_in;
        transform2sensorJacobians(pose, pw, J_pose_in, J_pw_in);
        return std::make_tuple(J_pose_in, J_pw_in);
      });

  m.def("transform2world", &transform2world);
  m.def("transform2worldJacobians", [](const Sophus::SE3d& pose, const Eigen::Vector3d& ps) {
        Eigen::Matrix<double, 3, 6> J_pose_in;
        Eigen::Matrix<double, 3, 3> J_ps_in;
        transform2worldJacobians(pose, ps, J_pose_in, J_ps_in);
        return std::make_tuple(J_pose_in, J_ps_in);
      });

  m.def("transform2image", &transform2image);
  m.def("transform2imageJacobians", [](const Sophus::SE3d& pose, const Eigen::Vector3d& pw) {
        Eigen::Matrix<double, 2, 6> J_pose_in;
        Eigen::Matrix<double, 2, 3> J_pw_in;
        transform2imageJacobians(pose, pw, J_pose_in, J_pw_in);
        return std::make_tuple(J_pose_in, J_pw_in);
      });


  // projection
  WRAP_TYPE_PROJECTION(CalibK, "CalibK")
  WRAP_TYPE_PROJECTION(CalibKD, "CalibKD")
  WRAP_TYPE_PROJECTION(CalibBundler, "CalibBundler")

  m.def("projectBundler", &projectBundler);
  m.def("projectBundlerJacobians", [](const Sophus::SE3d& pose, const CalibBundler& calib, 
          const Eigen::Vector3d& pw) {
        Eigen::Matrix<double, 2, 6> J_pose_in;
        Eigen::Matrix<double, 2, 3> J_calib_in, J_pw_in;
        projectBundlerJacobians(pose, calib, pw, J_pose_in, J_calib_in, J_pw_in);
        return std::make_tuple(J_pose_in, J_calib_in, J_pw_in);
      });
}
