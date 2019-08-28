/**
 * @file    Covariance.h
 * @brief   Calculate (marginal) covariance from square root information matrix
 * @author  Jing Dong
 * @date    Mar 14, 2019
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <unordered_map>

namespace minisam {

/**
 * Kaess et.al. Covariance recovery from a square root information matrix
 * for data association, in RAS 2009
 */
class Covariance {
 private:
  // Hinv upper triangle value cache
  // indexed by int64_t, calculated by two int32_t
  mutable std::unordered_map<int64_t, double> map_Hinv_;

  // copy to upper square root information matrix
  Eigen::SparseMatrix<double> L_;
  Eigen::VectorXd L_diag_inv_;  // pre-calculated 1 ./ diag(R)

 public:
  // L is lower square root information matrix
  // L must be compressed
  explicit Covariance(const Eigen::SparseMatrix<double>& L)
      : L_(L), L_diag_inv_(L.diagonal().cwiseInverse()) {}

  ~Covariance() = default;

  // get marginal covariance of partial indices of H, a.k.a. part of Hinv
  // indices must be sorted
  Eigen::MatrixXd marginalCovariance(const std::vector<int>& indices) const;

 private:
  // get a single entry value of Hinv
  double value_(int row, int col) const;

  // sum over sparse entries of col
  double sumCol_(int col, int l) const;

  // row/col to int64_t index
  inline static int64_t HinvMapIndex_(int row, int col) {
    return static_cast<int64_t>(static_cast<int32_t>(row)) << 32 |
           static_cast<int64_t>(static_cast<int32_t>(col));
  }
};

}  // namespace minisam
