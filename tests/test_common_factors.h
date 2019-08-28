/**
 * @file    test_common_factors.h
 * @brief   some common utils used in unit tests
 * @author  Jing Dong
 * @date    Mar 30, 2019
 */

#pragma once

#include <minisam/core/Factor.h>
#include <minisam/core/Variables.h>


namespace minisam {
namespace test {

// example prior factor for test
// define since minisam/slam may not include
template <class T>
class PFactor: public Factor {
private:
  T prior_;

public:
  PFactor(Key key, const T& prior, const std::shared_ptr<LossFunction>& lossfunc): 
    Factor(traits<T>::Dim(prior), std::vector<Key>{key}, lossfunc), prior_(prior) {}
  virtual ~PFactor() {}

  std::shared_ptr<Factor> copy() const { return std::shared_ptr<Factor>(new PFactor(*this)); }

  Eigen::VectorXd error(const Variables& values) const {
    return (Eigen::VectorXd(traits<T>::Dim(prior_)) << values.at<T>(keys()[0]) - prior_).finished();
  }
  std::vector<Eigen::MatrixXd> jacobians(const Variables& /*values*/) const {
    return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Identity(traits<T>::Dim(prior_), 
        traits<T>::Dim(prior_))};
  }

  // keylist accessor
  std::vector<Key>& keylist_nonconst() { return keylist_; }

};

// example between factor for test
// define since minisam/slam may not include
template <class T>
class BFactor: public Factor {
private:
  T diff_;

public:
  BFactor(Key key1, Key key2, const T& diff, const std::shared_ptr<LossFunction>& lossfunc): 
    Factor(traits<T>::Dim(diff), std::vector<Key>{key1, key2}, lossfunc), diff_(diff) {}
  virtual ~BFactor() {}

  std::shared_ptr<Factor> copy() const { return std::shared_ptr<Factor>(new BFactor(*this)); }

  Eigen::VectorXd error(const Variables& values) const {
    const T& v1 = values.at<T>(keys()[0]);
    const T& v2 = values.at<T>(keys()[1]);
    return (Eigen::VectorXd(traits<T>::Dim(diff_)) << v2 - v1 - diff_).finished();
  }
  std::vector<Eigen::MatrixXd> jacobians(const Variables& /*values*/) const {
    return std::vector<Eigen::MatrixXd>{
        -Eigen::MatrixXd::Identity(traits<T>::Dim(diff_), traits<T>::Dim(diff_)), 
        Eigen::MatrixXd::Identity(traits<T>::Dim(diff_), traits<T>::Dim(diff_))};
  }

  // keylist accessor
  std::vector<Key>& keylist_nonconst() { return keylist_; }

};

} // namespace test
} // namespace minisam
