/**
 * @file    NumericalFactor.cpp
 * @brief   Numerical jacobians Factor
 * @author  Jing Dong
 * @date    Nov 6, 2018
 */

#include <minisam/nonlinear/NumericalFactor.h>

#include <minisam/core/Eigen.h>  // traits for Eigen
#include <minisam/core/Variables.h>

namespace minisam {

/* ************************************************************************** */
void NumericalFactor::print(std::ostream& out) const {
  out << "Numerical jacobians Factor ";
  switch (numerical_type_) {
    case NumericalJacobianType::CENTRAL: {
      out << "Central";
      break;
    }
    case NumericalJacobianType::RIDDERS3: {
      out << "Ridders(3, 1)";
      break;
    }
    case NumericalJacobianType::RIDDERS5: {
      out << "Ridders(5, 1)";
      break;
    }
    default:
      break;
  }
  out << ", delta = " << delta_ << std::endl;
  Factor::print(out);
}

/* ************************************************************************** */
std::vector<Eigen::MatrixXd> NumericalFactor::numericalJacobiansImpl_(
    const Variables& values) const {
  // csst values to non-const for fast error evaluation by error() without copy
  // TODO: breaking constness is nasty, a better way to avoid copying?
  Variables& values_nonconst = const_cast<Variables&>(values);

  // calculate error from values by modifying values
  auto fv = [&values_nonconst, this](const std::shared_ptr<Variable>& v,
                                     Key k) -> Eigen::VectorXd {
    // keep a copy of original value
    std::shared_ptr<Variable> v_copy = values_nonconst.at(k);
    values_nonconst.update(k, v);  // assign new value
    // get error of updated values
    Eigen::VectorXd err_updated = this->error(values_nonconst);
    values_nonconst.update(k, v_copy);  // copy back orginal value
    return err_updated;
  };

  std::vector<Eigen::MatrixXd> jacobians;
  for (Key k : keylist_) {
    jacobians.push_back(
        numericalJacobian<Eigen::VectorXd, std::shared_ptr<Variable>>(
            std::bind(fv, std::placeholders::_1, k), values.at(k), delta_,
            numerical_type_));
  }
  return jacobians;
}

}  // namespace minisam
