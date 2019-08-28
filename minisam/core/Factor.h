/**
 * @file    Factor.h
 * @brief   Factor base class
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#pragma once

#include <minisam/core/Key.h>

#include <Eigen/Core>

#include <iostream>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace minisam {

// forward declearation
class LossFunction;
class Variables;

/** base class for factors, need actual error and jacobians implementation */
class Factor {
 protected:
  size_t dim_;                // dimension of error vector
  std::vector<Key> keylist_;  // list of keys' of connected variables
  // optional loss function, if not use set as nullptr
  std::shared_ptr<LossFunction> lossfunc_;

 public:
  virtual ~Factor() = default;

  /** factor properties */

  virtual void print(std::ostream& out = std::cout) const;

  // size (number of variables connected)
  size_t size() const { return keylist_.size(); }

  // non-const and const access of keys
  const std::vector<Key>& keys() const { return keylist_; }

  // const access of noisemodel
  const std::shared_ptr<LossFunction>& lossFunction() const {
    return lossfunc_;
  }

  // error dimension is dim of noisemodel
  size_t dim() const { return dim_; }

  /** implementation needed for actual factors */

  // deep copy of a shared pointer
  virtual std::shared_ptr<Factor> copy() const = 0;

  // error function
  // error vector dimension should meet dim()
  virtual Eigen::VectorXd error(const Variables& variables) const = 0;

  // jacobians function
  // jacobians vector sequence meets key list, size error.dim x var.dim
  virtual std::vector<Eigen::MatrixXd> jacobians(
      const Variables& variables) const = 0;

  /** optimization related */

  // whiten error
  virtual Eigen::VectorXd weightedError(const Variables& variables) const;

  // whiten jacobian matrix
  virtual std::pair<std::vector<Eigen::MatrixXd>, Eigen::VectorXd>
  weightedJacobiansError(const Variables& variables) const;

 protected:
  // only derived class should call
  Factor(size_t dim, const std::vector<Key>& keylist,
         const std::shared_ptr<LossFunction>& lossfunc = nullptr)
      : dim_(dim), keylist_(keylist), lossfunc_(lossfunc) {
    // no duplicated keys
    assert(std::set<Key>(keylist.begin(), keylist.end()).size() ==
               keylist.size() &&
           "[Factor::Factor] duplicated keys");
  }
};

}  // namespace minisam
