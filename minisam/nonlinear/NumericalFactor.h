/**
 * @file    NumericalFactor.h
 * @brief   Numerical jacobians Factor
 * @author  Jing Dong
 * @date    Nov 6, 2018
 */

#pragma once

#include <minisam/core/Factor.h>
#include <minisam/core/Key.h>
#include <minisam/nonlinear/numericalJacobian.h>

#include <Eigen/Core>

#include <iostream>
#include <memory>
#include <vector>

namespace minisam {

/**
 * Abstract class for factor with numerical jacobians, need actual error/copy
 * implementation
 * when analytic Jacobians are not possible, use this class to provide numerical
 * Jacobians
 * Not recommended if analytic Jacobians are possible
 */
class NumericalFactor : public Factor {
 private:
  double delta_;
  NumericalJacobianType numerical_type_;

 public:
  virtual ~NumericalFactor() = default;

  void print(std::ostream& out = std::cout) const override;

  // numerical jacobians function
  // jacobians vector sequence meets key list, size error.dim x var.dim
  std::vector<Eigen::MatrixXd> jacobians(
      const Variables& variables) const override {
    return numericalJacobiansImpl_(variables);
  }

 protected:
  // only derived class should call
  NumericalFactor(size_t dim, const std::vector<Key>& keylist,
                  const std::shared_ptr<LossFunction>& lossfunc = nullptr,
                  double delta = 1e-3, NumericalJacobianType numerical_type =
                                           NumericalJacobianType::RIDDERS5)
      : Factor(dim, keylist, lossfunc),
        delta_(delta),
        numerical_type_(numerical_type) {}

 private:
  // numerical Jacobians implementation
  std::vector<Eigen::MatrixXd> numericalJacobiansImpl_(
      const Variables& variables) const;
};

}  // namespace minisam
