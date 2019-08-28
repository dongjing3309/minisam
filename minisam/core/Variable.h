/**
 * @file    Variable.h
 * @brief   Abstract value base class for wrapping templated VariableType<T>
 * @author  Jing Dong
 * @date    Oct 13, 2017
 */

#pragma once

#include <minisam/core/Traits.h>

#include <Eigen/Core>

#include <iostream>
#include <memory>
#include <sstream>

namespace minisam {

/**
 * Variable class is only used for wrapping VariableType<T> in Variables,
 * which does not allow (different) template types as elements in STL container
 */
class Variable {
 public:
  virtual ~Variable() = default;

  // print
  virtual void print(std::ostream& out = std::cout) const = 0;

  /** data related */

  // make a *deep* copy of shared pointer, implemented by VariableType<T>
  virtual std::shared_ptr<Variable> copy() const = 0;

  // cast to stored data type T, implemented by VariableType<T>
  template <typename T>
  const T& cast() const;

  template <typename T>
  T& cast();

  /** manifold related */

  // dimension
  virtual size_t dim() const = 0;

  // retract
  virtual std::shared_ptr<Variable> retract(
      const Eigen::VectorXd& delta) const = 0;

  // local coordinate
  virtual Eigen::VectorXd local(const Variable& value) const = 0;

 protected:
  // only derived class should call
  // default constructor
  Variable() = default;
};

/**
 * VariableType class is a container class holds any type T for optimization
 * which requires T has appropriate minisam::traits<T>
 */
template <typename T>
class VariableType : public Variable {
  // check T is manifold type
  static_assert(is_manifold<T>::value, "Variable type must be a manifold type");

 private:
  T value_;  // data wrapped and stored

 public:
  explicit VariableType(const T& value) : value_(value) {}

  virtual ~VariableType() = default;

  typedef T type;

  // make a deep copy (of content) of shared pointer
  std::shared_ptr<Variable> copy() const override {
    return std::shared_ptr<Variable>(new VariableType<T>(value_));
  }

  // print, use traits
  void print(std::ostream& out = std::cout) const override {
    out << /*"Type = " << typeid(T).name() << */ "value = ";
    traits<T>::Print(value_, out);
  }

  /** data related */

  // return value
  T& value() { return value_; }
  const T& value() const { return value_; }

  /** manifold related */

  // dimension
  size_t dim() const override { return traits<T>::Dim(value_); }

  // retract
  std::shared_ptr<Variable> retract(
      const Eigen::VectorXd& delta) const override {
    const T retract_result = traits<T>::Retract(value_, delta);
    return std::shared_ptr<Variable>(new VariableType<T>(retract_result));
  }

  // local coordinate
  Eigen::VectorXd local(const Variable& value2) const override {
    const VariableType<T>& value2_type =
        static_cast<const VariableType<T>&>(value2);
    return traits<T>::Local(value_, value2_type.value());
  }
};

// define Variable::cast here since now VariableType has been declared
template <typename T>
const T& Variable::cast() const {
  // will throw exception
  try {
    return dynamic_cast<const VariableType<T>&>(*this).value();
  } catch (std::bad_cast&) {
    std::stringstream ss;
    ss << "[Variable::cast] cannot find cast Variable to " << typeid(T).name();
    throw std::runtime_error(ss.str());
  }
}

template <typename T>
T& Variable::cast() {
  // will throw exception
  try {
    return dynamic_cast<VariableType<T>&>(*this).value();
  } catch (std::bad_cast&) {
    std::stringstream ss;
    ss << "[Variable::cast] cannot find cast Variable to " << typeid(T).name();
    throw std::runtime_error(ss.str());
  }
}

}  // namespace minisam
