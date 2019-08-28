/**
 * @file    BetweenFactor.h
 * @brief   Soft between factor for any Lie group types
 * @author  Jing Dong
 * @date    Oct 15, 2017
 */

#pragma once

#include <minisam/core/Factor.h>
#include <minisam/core/Traits.h>
#include <minisam/core/Variables.h>

namespace minisam {

template <typename T>
class BetweenFactor : public Factor {
  // check T is manifold type
  static_assert(is_lie_group<T>::value,
                "Variable type T in BetweenFactor<T> must be a Lie group type");

 private:
  T diff_; // difference

 public:
  BetweenFactor(Key key1, Key key2, const T& diff,
                const std::shared_ptr<LossFunction>& lossfunc)
      : Factor(traits<T>::Dim(diff), std::vector<Key>{key1, key2}, lossfunc),
        diff_(diff) {}

  virtual ~BetweenFactor() = default;

  /** factor implementation */

  // print
  void print(std::ostream& out = std::cout) const override {
    out << "Between Factor, ";
    Factor::print(out);
    out << "measured = ";
    traits<T>::Print(diff_, out);
    out << std::endl;
  }

  // deep copy function
  std::shared_ptr<Factor> copy() const override {
    return std::shared_ptr<Factor>(new BetweenFactor(*this));
  }

  // error function
  Eigen::VectorXd error(const Variables& values) const override {
    const T& v1 = values.at<T>(keys()[0]);
    const T& v2 = values.at<T>(keys()[1]);
    const T diff = traits<T>::Compose(traits<T>::Inverse(v1), v2);
    // manifold equivalent of x-z -> Local(z,x)
    return traits<T>::Local(diff_, diff);
  }

  // jacobians function
  std::vector<Eigen::MatrixXd> jacobians(
      const Variables& values) const override {
    const T& v1 = values.at<T>(keys()[0]);
    const T& v2 = values.at<T>(keys()[1]);
    Eigen::MatrixXd Hinv, Hcmp1, Hcmp2;
    traits<T>::InverseJacobian(v1, Hinv);
    traits<T>::ComposeJacobians(traits<T>::Inverse(v1), v2, Hcmp1, Hcmp2);
    return std::vector<Eigen::MatrixXd>{Hcmp1 * Hinv, Hcmp2};
  }
};

}  // namespace minisam
