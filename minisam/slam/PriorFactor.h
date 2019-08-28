/**
 * @file    PriorFactor.h
 * @brief   Soft prior factor for any manifold types
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#pragma once

#include <minisam/core/Factor.h>
#include <minisam/core/Traits.h>
#include <minisam/core/Variables.h>

namespace minisam {

template <typename T>
class PriorFactor : public Factor {
  // check T is manifold type
  static_assert(is_manifold<T>::value,
                "Variable type T in PriorFactor<T> must be a manifold type");

 private:
  T prior_;

 public:
  PriorFactor(Key key, const T& prior,
              const std::shared_ptr<LossFunction>& lossfunc)
      : Factor(traits<T>::Dim(prior), std::vector<Key>{key}, lossfunc),
        prior_(prior) {}

  virtual ~PriorFactor() = default;

  /** factor implementation */

  // print
  void print(std::ostream& out = std::cout) const override {
    out << "Prior Factor, ";
    Factor::print(out);
    out << "measured = ";
    traits<T>::Print(prior_, out);
    out << std::endl;
  }

  // deep copy function
  std::shared_ptr<Factor> copy() const override {
    return std::shared_ptr<Factor>(new PriorFactor(*this));
  }

  // error function
  Eigen::VectorXd error(const Variables& values) const override {
    // manifold equivalent of x-z -> Local(z,x)
    return traits<T>::Local(prior_, values.at<T>(keys()[0]));
  }

  // jacobians function
  std::vector<Eigen::MatrixXd> jacobians(
      const Variables& /*values*/) const override {
    // indentity jacobians
    return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Identity(
        traits<T>::Dim(prior_), traits<T>::Dim(prior_))};
  }
};

}  // namespace minisam
