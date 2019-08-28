/**
 * @file    variables.cpp
 * @author  Jing Dong
 * @date    Nov 14, 2018
 */

#include "print.h"
#include "type_cast.h"
#include "pyobject_traits.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <minisam/config.h>
#include <minisam/core/Variables.h>
#include <minisam/core/VariableOrdering.h>

#include <minisam/core/Eigen.h>
#include <minisam/core/Scalar.h>
#ifdef MINISAM_USE_SOPHUS
#  include <minisam/geometry/Sophus.h>
#  include <minisam/geometry/CalibK.h>
#  include <minisam/geometry/CalibKD.h>
#  include <minisam/geometry/CalibBundler.h>
#endif


namespace py = pybind11;


namespace minisam {
namespace internal {


// caller for dynamic variables.at
class VariablesAtCaller : public ClassMemberCallerBase<const Variables&, py::object, Key> {
  typedef ClassMemberCallerBase<const Variables&, py::object, Key> Base;

public:
  // ctor passes casters to base
  VariablesAtCaller(const type_casters_type& casters) : Base(casters) {}

  // template type caster for each type
  // which implement a call() to process actual member function call
  template <class T>
  class type_caster : public Base::type_caster_base {
  public:
    py::object call(const Variables& values, Key key) const override {
      return py::cast(static_cast<const T*>(&(values.template at<T>(key))));
    }
  };
};

VariablesAtCaller::type_casters_type variables_at_type_casters = {
    // multi-view geometry and SLAM types
#ifdef MINISAM_USE_SOPHUS
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, Sophus::SE3d)
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, Sophus::SE2d)
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, Sophus::SO3d)
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, Sophus::SO2d)
    //ADD_TYPE_TO_CASTERS(VariablesAtCaller, Sophus::Sim3d)
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, CalibK)
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, CalibKD)
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, CalibBundler)
#endif
    // built-in types
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, double)
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, Eigen::Vector2d)
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, Eigen::Vector3d)
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, Eigen::Vector4d)
    ADD_TYPE_TO_CASTERS(VariablesAtCaller, Eigen::VectorXd)
};

// Variables with unified at in python
static const VariablesAtCaller variables_at_caller(variables_at_type_casters);


// caller for dynamic variables.add
class VariablesAddCaller : public ClassMemberCallerBase<Variables&, void, Key, py::object> {
  typedef ClassMemberCallerBase<Variables&, void, Key, py::object> Base;

public:
  // ctor passes casters to base
  VariablesAddCaller(const type_casters_type& casters) : Base(casters) {}

  // template type caster for each type
  // which implement a call() to process actual member function call
  template <class T>
  class type_caster : public Base::type_caster_base {
  public:
    void call(Variables& values, Key key, py::object value) const override {
      values.template add<T>(key, value.cast<T>());
    }
  };
};

VariablesAddCaller::type_casters_type variables_add_type_casters = {
    // multi-view geometry and SLAM types
#ifdef MINISAM_USE_SOPHUS
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, Sophus::SE3d)
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, Sophus::SE2d)
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, Sophus::SO3d)
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, Sophus::SO2d)
    //ADD_TYPE_TO_CASTERS(VariablesAddCaller, Sophus::Sim3d)
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, CalibK)
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, CalibKD)
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, CalibBundler)
#endif
    // built-in types
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, double)
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, Eigen::Vector2d)
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, Eigen::Vector3d)
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, Eigen::Vector4d)
    ADD_TYPE_TO_CASTERS(VariablesAddCaller, Eigen::VectorXd)
};

// Variables with unified at in python
static const VariablesAddCaller variables_add_caller(variables_add_type_casters);



// caller for dynamic variables.update
class VariablesUpdateCaller : public ClassMemberCallerBase<Variables&, void, Key, py::object> {
  typedef ClassMemberCallerBase<Variables&, void, Key, py::object> Base;

public:
  // ctor passes casters to base
  VariablesUpdateCaller(const type_casters_type& casters) : Base(casters) {}

  // template type caster for each type
  // which implement a call() to process actual member function call
  template <class T>
  class type_caster : public Base::type_caster_base {
  public:
    void call(Variables& values, Key key, py::object value) const override {
      values.template update<T>(key, value.cast<T>());
    }
  };
};

VariablesUpdateCaller::type_casters_type variables_update_type_casters = {
    // multi-view geometry and SLAM types
#ifdef MINISAM_USE_SOPHUS
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, Sophus::SE3d)
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, Sophus::SE2d)
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, Sophus::SO3d)
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, Sophus::SO2d)
    //ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, Sophus::Sim3d)
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, CalibK)
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, CalibKD)
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, CalibBundler)
#endif
    // built-in types
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, double)
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, Eigen::Vector2d)
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, Eigen::Vector3d)
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, Eigen::Vector4d)
    ADD_TYPE_TO_CASTERS(VariablesUpdateCaller, Eigen::VectorXd)
};

// Variables with unified at in python
static const VariablesUpdateCaller variables_update_caller(variables_update_type_casters);


} // namespace internal
} // namespace minisam


using namespace minisam;


// wrap a C++ type with Variables in Python
// make sure appropriate traits are defined
// support static typed at, update, add function in Python
//
//.def("add", (void (Variables::*)(Key, const T&)) &Variables::add<T>)
//.def("update", (void (Variables::*)(Key, const T&)) &Variables::update<T>)
#define WRAP_TYPE_TO_VARIABLES(T, TYPENAME) \
    .def(std::string(std::string("at_") + TYPENAME + "_").c_str(), \
        (const T& (Variables::*)(Key) const) &Variables::at<T>, py::return_value_policy::copy) \
    .def("__setitem__", [](Variables &obj, Key k, const T& v) { \
          if (!obj.exists(k)) obj.add<T>(k, v); \
          else obj.update<T>(k, v); \
        })


void wrap_variables(py::module& m) {

  py::class_<Variables>(m, "Variables")

    // ctor
    .def(py::init<>())

    // C++ container 
    .def("size", &Variables::size)
    .def("exists", &Variables::exists)
    .def("erase", &Variables::erase)
    
    // dynamic type at
    .def("at", [](Variables &obj, Key k) {
          try {
            return internal::variables_at_caller.call(obj, k);
          } catch (std::runtime_error&) {
            return obj.at<py::object>(k);
          }
        }, py::return_value_policy::copy)
    // for py::object
    .def("at_pyobject_", (const py::object& (Variables::*)(Key) const) 
        &Variables::at<py::object>, py::return_value_policy::copy)

    // dynamic type add/update
    .def("add", [](Variables &obj, Key k, py::object value) {
          try {
            return internal::variables_add_caller.call_void(obj, k, value);
          } catch (std::runtime_error&) {
            return obj.add<py::object>(k, value);
          }
        })
    .def("update", [](Variables &obj, Key k, py::object value) {
          try {
            return internal::variables_update_caller.call_void(obj, k, value);
          } catch (std::runtime_error&) {
            return obj.update<py::object>(k, value);
          }
        })

    // python emulating container type
    .def("__len__", &Variables::size)
    .def("__getitem__", [](Variables &obj, Key k) {
          if (!obj.exists(k)) 
            throw py::key_error("[python::Variables] key does not exist");
          else 
            return internal::variables_at_caller.call(obj, k);
        }, py::return_value_policy::copy)
    .def("__delitem__", [](Variables &obj, Key k) {
          obj.erase(k);
        })

    .def("defaultVariableOrdering", &Variables::defaultVariableOrdering)

    // manifold
    .def("dim", &Variables::dim)
    .def("retract", &Variables::retract)
    .def("local", &Variables::local)

    // support python print
    WRAP_TYPE_PYTHON_PRINT(Variables)

    // multi-view geometry and SLAM types
#ifdef MINISAM_USE_SOPHUS
    WRAP_TYPE_TO_VARIABLES(Sophus::SE3d, "SE3")
    WRAP_TYPE_TO_VARIABLES(Sophus::SE2d, "SE2")
    WRAP_TYPE_TO_VARIABLES(Sophus::SO3d, "SO3")
    WRAP_TYPE_TO_VARIABLES(Sophus::SO2d, "SO2")
    //WRAP_TYPE_TO_VARIABLES(Sophus::Sim3d, "Sim3")

    WRAP_TYPE_TO_VARIABLES(CalibK, "CalibK")
    WRAP_TYPE_TO_VARIABLES(CalibKD, "CalibKD")
    WRAP_TYPE_TO_VARIABLES(CalibBundler, "CalibBundler")
#endif

    // built-in types
    WRAP_TYPE_TO_VARIABLES(double, "double")
    WRAP_TYPE_TO_VARIABLES(Eigen::Vector2d, "Vector2")
    WRAP_TYPE_TO_VARIABLES(Eigen::Vector3d, "Vector3")
    WRAP_TYPE_TO_VARIABLES(Eigen::Vector4d, "Vector4")
    WRAP_TYPE_TO_VARIABLES(Eigen::VectorXd, "Vector")

    WRAP_TYPE_TO_VARIABLES(py::object, "pyobject")
    ;
}
