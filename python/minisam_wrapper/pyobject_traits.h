/**
 * @file    pyobject_traits.h
 * @brief   utils to support optimizing Python type
 * @author  Jing Dong
 * @date    Apr 9, 2019
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <minisam/core/Traits.h>

#include <Eigen/Core>
#include <iostream>
#include <vector>

namespace py = pybind11;


namespace minisam {
namespace internal {

struct PyObjTraitsImpl {

  // type tag, TODO: has a general traits for python obj?
  typedef lie_group_tag type_category;

  // print
  static void Print(const py::object& m, std::ostream& out = std::cout) {
    out << std::string(py::repr(m));
  }


  /** manifold */

  // tangent vector type defs
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> TangentVector;

  // dimension
  static size_t Dim(const py::object& m) { 
    return m.attr("dim")().cast<size_t>();
  }

  // local coordinate
  static TangentVector Local(const py::object& origin, const py::object& s) {
    return origin.attr("local")(s).cast<TangentVector>();
  }

  // retract
  static py::object Retract(const py::object& origin, const TangentVector& v) {
    return origin.attr("retract")(py::cast(v));
  }


  /** Lie group */

  // identity
  static py::object Identity(const py::object& m) { return m.attr("identity")(); }

  // inverse, with jacobians
  static py::object Inverse(const py::object& m) { return m.attr("inverse")(); }

  static void InverseJacobian(const py::object& m, Eigen::MatrixXd& H) {
    H = m.attr("inverse_jacobian")().cast<Eigen::MatrixXd>();
  }

  // compose, with jacobians
  static py::object Compose(const py::object& m1, const py::object& m2) {
    return m1.attr("compose")(m2); 
  }

  static void ComposeJacobians(const py::object& m1, const py::object& m2, 
      Eigen::MatrixXd& H1, Eigen::MatrixXd& H2) {
    std::vector<Eigen::MatrixXd> Hs = m1.attr("compose_jacobians")(m2).cast<std::vector<Eigen::MatrixXd>>();
    H1 = Hs.at(0);
    H2 = Hs.at(1); 
  }

  // logmap
  static TangentVector Logmap(const py::object& m) {
    return m.attr("logmap")().cast<TangentVector>();
  }

  // expmap
  static py::object Expmap(const py::object& m, const TangentVector& v) {
    return m.attr("expmap")(py::cast(v));;
  }
};
} // namespace internal

template<> struct traits<py::object>: internal::PyObjTraitsImpl {};

} // namespace minisam
