/**
 * @file    Variables.h
 * @brief   Multiple value container, for a solution
 * @author  Jing Dong
 * @date    Oct 13, 2017
 */

#pragma once

#include <minisam/core/Key.h>
#include <minisam/core/Variable.h>

#include <Eigen/Core>

#include <memory>
#include <unordered_map>

namespace minisam {

// forward declearation
class VariableOrdering;

// container class for variables
class Variables {
 private:
  std::unordered_map<Key, std::shared_ptr<Variable>> keyvalues_;

 public:
  // default constructor
  Variables() = default;

  // deep copy constructor
  Variables(const Variables& variables);

  ~Variables() = default;

  // print
  void print(std::ostream& out = std::cout) const;

  /** data operators */

  // insert data by variable type
  template <typename T>
  void add(Key i, const T& v) {
    add(i, std::shared_ptr<Variable>(new VariableType<T>(v)));
  }

  // insert value by shared pointer
  void add(Key i, const std::shared_ptr<Variable>& ptr_v);

  // access stored data by variable type
  template <typename T>
  const T& at(Key i) const {
    return at(i)->cast<T>();
  }

  // access value by shared pointer
  const std::shared_ptr<Variable>& at(Key i) const;

  // update stored data by variable type
  template <typename T>
  void update(Key i, const T& v) {
    update(i, std::shared_ptr<Variable>(new VariableType<T>(v)));
  }

  // update value by shared pointer
  void update(Key i, const std::shared_ptr<Variable>& ptr_v);

  // exist or not
  bool exists(Key i) const { return keyvalues_.find(i) != keyvalues_.end(); }

  // erase value by key, OK if key does not exist
  void erase(Key i) { keyvalues_.erase(i); }

  // size
  size_t size() const { return keyvalues_.size(); }

  // iterators
  std::unordered_map<Key, std::shared_ptr<Variable>>::iterator begin() {
    return keyvalues_.begin();
  }
  std::unordered_map<Key, std::shared_ptr<Variable>>::const_iterator begin()
      const {
    return keyvalues_.begin();
  }
  std::unordered_map<Key, std::shared_ptr<Variable>>::iterator end() {
    return keyvalues_.end();
  }
  std::unordered_map<Key, std::shared_ptr<Variable>>::const_iterator end()
      const {
    return keyvalues_.end();
  }

  // get variable ordering
  VariableOrdering defaultVariableOrdering() const;

  /** manifold related */

  // dim (= A.cols)
  size_t dim() const;

  // retract, input size should be dim
  Variables retract(const Eigen::VectorXd& delta,
                    const VariableOrdering& variable_ordering) const;

  // local coordinates
  Eigen::VectorXd local(const Variables& v,
                        const VariableOrdering& variable_ordering) const;
};

}  // namespace minisam
