/**
 * @file    Variables.cpp
 * @brief   Multiple value container, for a solution
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#include <minisam/core/Variables.h>

#include <minisam/core/VariableOrdering.h>

#include <algorithm>
#include <sstream>

namespace minisam {

/* ************************************************************************** */
Variables::Variables(const Variables& variables)
    : keyvalues_(variables.keyvalues_) {
  if (keyvalues_.size() > 0) {
    for (auto it1 = keyvalues_.begin(); it1 != keyvalues_.end(); it1++) {
      it1->second = it1->second->copy();
    }
  }
}

/* ************************************************************************** */
void Variables::print(std::ostream& out) const {
  if (size() == 0) {
    out << "Empty Variables" << std::endl;
  } else {
    // sort key list and print
    std::vector<Key> var_list;
    var_list.reserve(size());
    for (auto it = keyvalues_.begin(); it != keyvalues_.end(); it++) {
      var_list.push_back(it->first);
    }
    std::sort(var_list.begin(), var_list.end());
    for (Key key : var_list) {
      out << "Key = " << keyString(key) << ", ";
      at(key)->print(out);
      out << std::endl;
    }
  }
}

/* ************************************************************************** */
void Variables::add(Key i, const std::shared_ptr<Variable>& ptr_v) {
  if (keyvalues_.find(i) != keyvalues_.end()) {
    std::stringstream ss;
    ss << "[Variables::add] key " << keyString(i) << " is already in Variables";
    throw std::runtime_error(ss.str());
  }
  keyvalues_[i] = ptr_v;
}

/* ************************************************************************** */
const std::shared_ptr<Variable>& Variables::at(Key i) const {
  auto it = keyvalues_.find(i);
  if (it == keyvalues_.end()) {
    std::stringstream ss;
    ss << "[Variables::at] cannot find key " << keyString(i) << " in Variables";
    throw std::runtime_error(ss.str());
  }
  return it->second;
}

/* ************************************************************************** */
void Variables::update(Key i, const std::shared_ptr<Variable>& ptr_v) {
  auto it = keyvalues_.find(i);
  if (it == keyvalues_.end()) {
    std::stringstream ss;
    ss << "[Variables::update] cannot find key " << keyString(i)
       << " in Variables";
    throw std::runtime_error(ss.str());
  }
  it->second = ptr_v;
}

/* ************************************************************************** */
VariableOrdering Variables::defaultVariableOrdering() const {
  std::vector<Key> ordering_list;
  ordering_list.reserve(size());
  for (auto it = keyvalues_.begin(); it != keyvalues_.end(); it++) {
    ordering_list.push_back(it->first);
  }
  return VariableOrdering(ordering_list);
}

/* ************************************************************************** */
size_t Variables::dim() const {
  size_t d = 0;
  for (auto it = keyvalues_.begin(); it != keyvalues_.end(); it++) {
    d += it->second->dim();
  }
  return d;
}

/* ************************************************************************** */
Variables Variables::retract(const Eigen::VectorXd& delta,
                             const VariableOrdering& variable_ordering) const {
  Variables new_values;
  new_values.keyvalues_ = this->keyvalues_;  // avoid expensive deep copy

  size_t d = 0;  // dim accumulator
  for (size_t i = 0; i < variable_ordering.size(); i++) {
    Key k = variable_ordering[i];
    auto it = new_values.keyvalues_.find(k);
    if (it == new_values.keyvalues_.end()) {
      std::stringstream ss;
      ss << "[Variables::retract] cannot find key " << keyString(k)
         << " in Variables";
      throw std::runtime_error(ss.str());
    }
    std::shared_ptr<Variable>& var = it->second;
    size_t vd = var->dim();
    var = var->retract(delta.segment(d, vd));
    d += vd;
  }
  return new_values;
}

/* ************************************************************************** */
Eigen::VectorXd Variables::local(
    const Variables& v, const VariableOrdering& variable_ordering) const {
  Eigen::VectorXd delta(dim());
  size_t d = 0;
  for (size_t i = 0; i < variable_ordering.size(); i++) {
    const std::shared_ptr<Variable>& var = v.at(variable_ordering[i]);
    const std::shared_ptr<Variable>& var_this = at(variable_ordering[i]);
    size_t vd = var->dim();
    delta.segment(d, vd) = var_this->local(*var);
    d += vd;
  }
  return delta;
}
}
