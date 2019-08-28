/**
 * @file    Factor.cpp
 * @brief   Factor base class
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#include <minisam/core/Factor.h>

#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>

namespace minisam {

/* ************************************************************************** */
void Factor::print(std::ostream& out) const {
  out << "Factor dim = " << dim_ << std::endl;
  out << "Factor keys : ";
  if (size() > 0) {
    out << keyString(keys()[0]);
  }
  for (size_t i = 1; i < size(); i++) {
    out << ", " << keyString(keys()[i]);
  }
  out << std::endl;
  if (lossfunc_) {
    lossfunc_->print(out);
  }
}

/* ************************************************************************** */
Eigen::VectorXd Factor::weightedError(const Variables& variables) const {
  Eigen::VectorXd err = error(variables);
  if (lossfunc_) {
    lossfunc_->weightInPlace(err);
  }
  return err;
}

/* ************************************************************************** */
std::pair<std::vector<Eigen::MatrixXd>, Eigen::VectorXd>
Factor::weightedJacobiansError(const Variables& variables) const {
  auto pair = make_pair(jacobians(variables), error(variables));
  if (lossfunc_) {
    lossfunc_->weightInPlace(pair.first, pair.second);
  }
  return pair;
}

}  // namespace minisam
