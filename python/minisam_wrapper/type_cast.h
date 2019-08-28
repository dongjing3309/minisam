/**
 * @file    type_cast.h
 * @brief   utils to support cast between C++ and Python object 
 * @author  Jing Dong
 * @date    Nov 27, 2017
 */

#pragma once

#include <vector>
#include <memory>

//
// Concepts: caller
//
// If you have a template C++ function func<T>(...) you want to call in Python 
// without template parameter by func(...), you can setup a caller,
// and wrap caller.call(...) by pybind11. 
//
// The caller caller.call(...) handles all dynamic python type objects with py::object
// 
//
// Concepts: caster
//
// a caster is a templated class, which is templated on the type you would like to 
// register with the caller, and has an actual T typed call() implementation.
// Inside call() implementation py::object and T will cast back and forth,
// and throw a std::runtime_error if cast is failed.
//
// To Let the caller know which types should be registered with caller, 
// you need to construct to caller with a caster list.  
//
//
// Define your caller
// 
// 1. Select and derive the base class type based on the function you call is member function or not:
//
//      FunctionCallerBase<ReturnType, ArgsTypes...> for func<T>(...)
//      ClassMemberCallerBase<ClassSignature, ReturnType, ArgsTypes...> for obj.func<T>(...)
//
//    If the object is expected to be a dynamic python type, use py::object as template type argument
// 
// 2. Implement your caller by
//    1.  A ctor passes casters args to base ctor.
//    2.  A templated type_caster class based on Base::type_caster_base, 
//        which has a call(...) contains your actual function implementation.
// 
// 3. Define a static const caller instance by passing a Caller::type_casters_type,
//    which is statically constructed by macro ADD_TYPE_TO_CASTERS


namespace minisam {
namespace internal {


// non-member function caller
template <class Trtn, class... Targs>
class FunctionCallerBase {

public:
  // type caster base
  class type_caster_base {
  public:
    virtual ~type_caster_base() = default;
    virtual Trtn call(Targs... args) const = 0;
  };
  
  // caster list
  typedef std::vector<std::shared_ptr<type_caster_base>> type_casters_type;
  const type_casters_type type_casters_;

  // ctor
  FunctionCallerBase(const type_casters_type& casters) : type_casters_(casters) {}

  // cast
  Trtn call(Targs... args) const {
    bool call_success = false;
    Trtn value;
    for (const auto& caster : type_casters_) {
      try {
        value = caster->call(args...);
        call_success = true;
        break;
      } 
      catch (std::runtime_error&) {}  // both call<T> error and pybind11::cast_error will be caught here
    }
    if (call_success)
      return value;
    else
      throw std::runtime_error("[python::FunctionCaller] cannot cast to registered minisam type");
  }
};


// class member function caller
template <class C, class Trtn, class... Targs>
class ClassMemberCallerBase {

public:
  // type caster base
  class type_caster_base {
  public:
    virtual ~type_caster_base() = default;
    virtual Trtn call(C obj, Targs... args) const = 0;
  };
  
  // caster list
  typedef std::vector<std::shared_ptr<type_caster_base>> type_casters_type;
  const type_casters_type type_casters_;

  // ctor
  ClassMemberCallerBase(const type_casters_type& casters) : type_casters_(casters) {}

  // cast
  Trtn call(C obj, Targs... args) const {
    bool call_success = false;
    Trtn value;
    for (const auto& caster : type_casters_) {
      try {
        value = caster->call(obj, args...);
        call_success = true;
        break;
      } 
      catch (std::runtime_error&) {}  // both call<T> error and pybind11::cast_error will be caught here
    }
    if (call_success)
      return value;
    else
      throw std::runtime_error("[python::ClassMemberCaller] cannot cast to registered minisam type");
  }

  void call_void(C obj, Targs... args) const {
    bool call_success = false;
    for (const auto& caster : type_casters_) {
      try {
        caster->call(obj, args...);
        call_success = true;
        break;
      } 
      catch (std::runtime_error&) {}  // both call<T> error and pybind11::cast_error will be caught here
    }
    if (call_success)
      return;
    else
      throw std::runtime_error("[python::ClassMemberCaller] cannot cast to registered minisam type");
  }

};

} // namespace internal
} // namespace minisam



// macro to add type to caster list
#define ADD_TYPE_TO_CASTERS(CASTER, T) \
  std::shared_ptr<CASTER::type_caster_base>(new CASTER::type_caster<T>()),
