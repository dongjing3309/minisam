/**
 * @file    utils.cpp
 * @author  Jing Dong
 * @date    Apr 13, 2019
 */

#include "print.h"

#include <pybind11/pybind11.h>

#include <minisam/utils/Timer.h>

using namespace minisam;
namespace py = pybind11;

void wrap_utils(py::module& m) {

  // single timer class
  py::class_<Timer>(m, "Timer")
    // no ctor
    .def("tic", &Timer::tic)
    .def("toc", &Timer::toc)
    // statistics
    .def("size", &Timer::size)
    .def("sum", &Timer::sum)
    .def("last", &Timer::last)
    .def("max", &Timer::max)
    .def("min", &Timer::min)
    ;

  // global timer class
  py::class_<GlobalTimer>(m, "GlobalTimer")
    // no ctor
    .def("getTimer", &GlobalTimer::getTimer, py::return_value_policy::reference)
    .def("reset", &GlobalTimer::reset)
    // print to std::cout
    .def("print", [](GlobalTimer& timer) { timer.print(); })
    // support python print to string
    WRAP_TYPE_PYTHON_PRINT(GlobalTimer)
    ;

  // wrap function reference to global timer
  m.def("global_timer", &global_timer, py::return_value_policy::reference);
}
