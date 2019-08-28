/**
 * @file    factor.cpp
 * @author  Jing Dong
 * @date    Nov 26, 2017
 */

#include "print.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <minisam/core/Key.h>
#include <minisam/core/Factor.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/nonlinear/NumericalFactor.h>


namespace py = pybind11;


namespace minisam {


// trampoline class for overloading Factor's method in python
class PyFactor_ : public Factor {

public:

  // static create constructor
  static PyFactor_ create(size_t dim, const std::vector<Key>& keylist, 
      const std::shared_ptr<LossFunction>& lossfunc) {
    return PyFactor_(dim, keylist, lossfunc);
  }

  // trampoline error function
  Eigen::VectorXd error(const Variables& variables) const override {
    PYBIND11_OVERLOAD_PURE(Eigen::VectorXd, Factor, error, variables);
  }

  // trampoline jacobian function
  std::vector<Eigen::MatrixXd> jacobians(const Variables& variables) const override {
    PYBIND11_OVERLOAD_PURE(std::vector<Eigen::MatrixXd>, Factor, jacobians, variables);
  }

  // trampoline copy function
  std::shared_ptr<Factor> copy() const override {
    // see https://github.com/pybind/pybind11/issues/1049
    auto self = py::cast(this);
    auto cloned = self.attr("copy")();
    auto keep_python_state_alive = std::make_shared<py::object>(cloned);
    auto ptr = cloned.cast<PyFactor_*>();
    // aliasing shared_ptr: points to `PyFactor_* ptr` but refcounts the Python object
    return std::shared_ptr<Factor>(keep_python_state_alive, ptr);
  }

  // trampoline print function __repr__
  void print(std::ostream& out) const override {
    // see PYBIND11_OVERLOAD_PURE
    py::gil_scoped_acquire gil;
    py::function overload = py::get_overload(static_cast<const Factor*>(this), "__repr__");
    if (overload) {
      auto o = overload();
      out << py::detail::cast_safe<std::string>(std::move(o));
    } else {
      // default print
      out << "Python implemented Factor, ";
      Factor::print(out);
    }
  }

private:

  // constructor
  PyFactor_(size_t dim, const std::vector<Key>& keylist, 
      const std::shared_ptr<LossFunction>& lossfunc): Factor(dim, keylist, lossfunc) {}
};



// trampoline class for overloading NumericalFactor's method in python
class PyNumericalFactor_ : public NumericalFactor {

public:

  // static create constructor
  static PyNumericalFactor_ create(size_t dim, const std::vector<Key>& keylist, 
      const std::shared_ptr<LossFunction>& lossfunc, double delta, NumericalJacobianType numerical_type) {
    return PyNumericalFactor_(dim, keylist, lossfunc, delta, numerical_type);
  }

  // trampoline error function
  Eigen::VectorXd error(const Variables& variables) const override {
    PYBIND11_OVERLOAD_PURE(Eigen::VectorXd, NumericalFactor, error, variables);
  }

  // trampoline jacobian function
  std::vector<Eigen::MatrixXd> jacobians(const Variables& variables) const override {
    PYBIND11_OVERLOAD(std::vector<Eigen::MatrixXd>, NumericalFactor, jacobians, variables);
  }

  // trampoline copy function
  std::shared_ptr<Factor> copy() const override {
    // see https://github.com/pybind/pybind11/issues/1049
    auto self = py::cast(this);
    auto cloned = self.attr("copy")();
    auto keep_python_state_alive = std::make_shared<py::object>(cloned);
    auto ptr = cloned.cast<PyNumericalFactor_*>();
    // aliasing shared_ptr: points to `PyNumericalFactor_* ptr` but refcounts the Python object
    return std::shared_ptr<Factor>(keep_python_state_alive, ptr);
  }

  // trampoline print function __repr__
  void print(std::ostream& out) const override {
    // see PYBIND11_OVERLOAD_PURE
    py::gil_scoped_acquire gil;
    py::function overload = py::get_overload(static_cast<const NumericalFactor*>(this), "__repr__");
    if (overload) {
      auto o = overload();
      out << py::detail::cast_safe<std::string>(std::move(o));
    } else {
      // default print
      out << "Python implemented numerical Factor, ";
      Factor::print(out);
    }
  }

private:

  // constructor
  PyNumericalFactor_(size_t dim, const std::vector<Key>& keylist, 
      const std::shared_ptr<LossFunction>& lossfunc, double delta, NumericalJacobianType type): 
          NumericalFactor(dim, keylist, lossfunc, delta, type) {}
};

} // namespace minisam


using namespace minisam;


void wrap_factor(py::module& m) {

  // factor base
  py::class_<Factor, PyFactor_, std::shared_ptr<Factor>>(m, "Factor")

    // constructor: expose protected in C++ to python
    .def(py::init(&PyFactor_::create), py::arg("dim"), py::arg("keylist"), 
        py::arg("lossfunc") = nullptr)

    // keys
    .def("size", &Factor::size)
    .def("keys", &Factor::keys, py::return_value_policy::copy)

    // error
    .def("dim", &Factor::dim)
    .def("lossFunction", &Factor::lossFunction)
    .def("weightedError", &Factor::weightedError)
    .def("weightedJacobiansError", &Factor::weightedJacobiansError)

    // override-able
    .def("copy", &Factor::copy)
    .def("error", &Factor::error)
    .def("jacobians", &Factor::jacobians)

    // print should not remove last char
    // TODO: this is related to how print is overrided, still not sure why...
    WRAP_TYPE_PYTHON_PRINT_NOT_REMOVE_LAST(Factor)
    ;

  // numerical types
  py::enum_<NumericalJacobianType>(m, "NumericalJacobianType", py::arithmetic())
    .value("CENTRAL", NumericalJacobianType::CENTRAL)
    .value("RIDDERS3", NumericalJacobianType::RIDDERS3)
    .value("RIDDERS5", NumericalJacobianType::RIDDERS5)
    ;

  // numerical factor base
  py::class_<NumericalFactor, PyNumericalFactor_, std::shared_ptr<NumericalFactor>>(m, "NumericalFactor")

    // constructor: expose protected in C++ to python
    .def(py::init(&PyNumericalFactor_::create), py::arg("dim"), py::arg("keylist"), 
        py::arg("lossfunc") = nullptr, py::arg("delta") = 1e-3, 
        py::arg("numerical_type") = NumericalJacobianType::RIDDERS5)

    // keys
    .def("size", &NumericalFactor::size)
    .def("keys", &NumericalFactor::keys, py::return_value_policy::copy)

    // error
    .def("dim", &NumericalFactor::dim)
    .def("lossFunction", &NumericalFactor::lossFunction)
    .def("weightedError", &NumericalFactor::weightedError)
    .def("weightedJacobiansError", &NumericalFactor::weightedJacobiansError)

    // override-able
    .def("copy", &NumericalFactor::copy)
    .def("error", &NumericalFactor::error)
    .def("jacobians", &NumericalFactor::jacobians)

    // print should not remove last char
    // TODO: this is related to how print is overrided, still not sure why...
    WRAP_TYPE_PYTHON_PRINT_NOT_REMOVE_LAST(NumericalFactor)
    ;
}
