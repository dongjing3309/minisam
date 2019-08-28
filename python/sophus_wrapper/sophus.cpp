/**
 * @file    sophus.cpp
 * @brief   sophus wrapper module
 * @author  Jing Dong
 * @date    Nov 14, 2017
 */

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>

#include <minisam/geometry/Sophus.h>

#include <complex>
#include <sstream>

using namespace Sophus;
namespace py = pybind11;


// convert Eigen::Vector2d between std::complex<double>
inline Eigen::Vector2d complex_to_vector2(const std::complex<double>& c) {
  return Eigen::Vector2d(std::real(c), std::imag(c));
}
inline std::complex<double> vector2_to_complex(const Eigen::Vector2d v) {
  return std::complex<double>(v[0], v[1]);
}


// wrap base lie group properties and basic ctors
// - fix log() ambiguity in old sophus version
// - multiply assignment operator needs special care since pybind11 
//   cannot correctly convert Base<Derived> to Derived
#define WRAP_SOPHUS_TYPE_LIE_GROUP(T) \
  .def(py::init<>()) \
  .def(py::init<const T::Transformation&>()) \
  .def("log", (T::Tangent (T::*)() const) &T::log) \
  .def_static("exp", &T::exp) \
  .def_static("vee", &T::vee) \
  .def_static("hat", &T::hat) \
  .def("inverse", &T::inverse) \
  .def("params", &T::params) \
  .def("matrix", &T::matrix) \
  .def("Adj", &T::Adj) \
  .def(py::self * py::self) \
  .def(py::self * T::Point()) \
  .def("__imul__", [](const T& a, const T& b) { \
        return T(a * b); \
      }, py::is_operator())


// wrap print of sophus type
#define WRAP_SOPHUS_TYPE_PRINT(T) \
  .def("__repr__", [](const T &obj) { \
        std::stringstream ss; \
        ss << obj; \
        return ss.str(); \
      })


// sophus module
PYBIND11_MODULE(_minisam_sophus_py_wrapper, m) {

  // SO2
  py::class_<SO2d>(m, "SO2")
    // type particular ctor
    .def(py::init<double>())
    .def(py::init([](const std::complex<double>& c) {
          return SO2d(complex_to_vector2(c));
        }))
    // type particular
    .def("theta", (double (SO2d::*)() const) &SO2d::log)
    .def("unit_complex", [](const SO2d& obj) {
          return vector2_to_complex(obj.unit_complex());
        })
    // lie group
    WRAP_SOPHUS_TYPE_LIE_GROUP(SO2d)
    WRAP_SOPHUS_TYPE_PRINT(SO2d)
    ;

  // SE2
  py::class_<SE2d>(m, "SE2")
    // ctor
    .def(py::init<const SO2d&, const Eigen::Vector2d&>()) \
    .def_static("trans", (SE2d (*)(const Eigen::Vector2d&)) &SE2d::trans) \
    .def_static("transX", &SE2d::transX) \
    .def_static("transY", &SE2d::transY) \
    .def_static("rot", &SE2d::rot) \
    // type particular
    .def("so2", (const SO2d& (SE2d::*)() const) &SE2d::so2, 
        py::return_value_policy::copy) \
    .def("translation", (const Eigen::Vector2d& (SE2d::*)() const) &SE2d::translation, 
        py::return_value_policy::copy) \
    // lie group
    WRAP_SOPHUS_TYPE_LIE_GROUP(SE2d)
    WRAP_SOPHUS_TYPE_PRINT(SE2d)
    ;

  // SO3
  py::class_<SO3d>(m, "SO3")
    // type particular ctor
    .def(py::init([](double x, double y, double z, double w) {
          return SO3d(SO3d::QuaternionType(w, x, y, z));
        }))
    .def_static("rotX", &SO3d::rotX) \
    .def_static("rotY", &SO3d::rotY) \
    .def_static("rotZ", &SO3d::rotZ) \
    // type particular
    .def("unit_quaternion", [](const SO3d& obj) {
          SO3d::QuaternionType q = obj.unit_quaternion();
          Eigen::Vector4d qv = q.coeffs();
          return qv;
        })
    // lie group
    WRAP_SOPHUS_TYPE_LIE_GROUP(SO3d)
    WRAP_SOPHUS_TYPE_PRINT(SO3d)
    ;

  // SE3
  py::class_<SE3d>(m, "SE3")
    // ctor
    .def(py::init<const SO3d&, const Eigen::Vector3d&>()) \
    .def_static("trans", (SE3d (*)(const Eigen::Vector3d&)) &SE3d::trans) \
    .def_static("transX", &SE3d::transX) \
    .def_static("transY", &SE3d::transY) \
    .def_static("transZ", &SE3d::transZ) \
    .def_static("rotX", &SE3d::rotX) \
    .def_static("rotY", &SE3d::rotY) \
    .def_static("rotZ", &SE3d::rotZ) \
    // type particular
    .def("so3", (const SO3d& (SE3d::*)() const) &SE3d::so3, 
        py::return_value_policy::copy) \
    .def("translation", (const Eigen::Vector3d& (SE3d::*)() const) &SE3d::translation, 
        py::return_value_policy::copy) \
    // lie group
    WRAP_SOPHUS_TYPE_LIE_GROUP(SE3d)
    WRAP_SOPHUS_TYPE_PRINT(SE3d)
    ;
}
