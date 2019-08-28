/**
 * @file    module.cpp
 * @brief   root of the minisam wrapper module
 * @author  Jing Dong
 * @date    Nov 14, 2017
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

// parts

// core, seperate variables and loss function for later flexibility
void wrap_core(py::module& m);
void wrap_variables(py::module& m);
void wrap_factor(py::module& m);
void wrap_loss_function(py::module& m);

// multi-view geometry
void wrap_geometry(py::module& m);

// optimizer implementation
void wrap_optimizer(py::module& m);

// factors and slam utils
void wrap_slam(py::module& m);

// utils
void wrap_utils(py::module& m);


PYBIND11_MODULE(_minisam_py_wrapper, module) {
  wrap_core(module);
  wrap_variables(module);
  wrap_factor(module);
  wrap_loss_function(module);
  wrap_geometry(module);
  wrap_optimizer(module);
  wrap_slam(module);
  wrap_utils(module);
}

