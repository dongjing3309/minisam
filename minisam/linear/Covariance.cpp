/**
 * @file    Covariance.cpp
 * @brief   Calculate (marginal) covariance from information/Hessian matrix
 * @author  Jing Dong
 * @date    Mar 14, 2019
 */

#include <minisam/linear/Covariance.h>

#include <algorithm>

namespace minisam {

/* ************************************************************************** */
Eigen::MatrixXd Covariance::marginalCovariance(
    const std::vector<int>& indices) const {
  // calculate upper marginal part
  Eigen::MatrixXd mcov_upper(indices.size(), indices.size());
  for (int i = 0; i < static_cast<int>(indices.size()); i++) {
    for (int j = i; j < static_cast<int>(indices.size()); j++) {
      mcov_upper(i, j) = value_(indices[i], indices[j]);
    }
  }
  return mcov_upper.selfadjointView<Eigen::Upper>();
}

/* ************************************************************************** */
double Covariance::value_(int row, int col) const {
  int64_t value_key = HinvMapIndex_(row, col);
  const auto iter_value = map_Hinv_.find(value_key);
  if (iter_value != map_Hinv_.end()) {
    return iter_value->second;

  } else {
    double value = 0;
    if (row == col) {
      value = L_diag_inv_(col) * (L_diag_inv_(col) - sumCol_(col, col));
    } else {
      value = -sumCol_(row, col) * L_diag_inv_(row);
    }
    map_Hinv_[value_key] = value;
    return value;
  }
}

/* ************************************************************************** */
double Covariance::sumCol_(int col, int l) const {
  // get sum over a col over j
  double sumj = 0.0;
  // loop non-zero element of column col of L
  // j_idx is index of InnerIndices/Values of j
  for (int j_idx = L_.outerIndexPtr()[col]; j_idx < L_.outerIndexPtr()[col + 1];
       j_idx++) {
    int j = L_.innerIndexPtr()[j_idx];
    if (col != j) {  // skip diag
      double lj = j > l ? value_(l, j) : value_(j, l);
      double L_j_col = L_.valuePtr()[j_idx];  // R(col,j)
      sumj += L_j_col * lj;
    }
  }
  return sumj;
}

}  // namespace minisam
