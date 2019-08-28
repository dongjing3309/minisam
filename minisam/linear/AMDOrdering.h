/**
 * @file    AMDOrdering.h
 * @brief   Wrapped (patched) Eigen AMD ordering
 * @author  Jing Dong
 * @date    Oct 28, 2018
 */

#pragma once

#include <minisam/linear/Ordering.h>

// use a patched (bugfix) AMD ordering method
#include <minisam/3rdparty/eigen3/OrderingMethods>

namespace minisam {

/**
 * Apply (patched) AMD ordering of Hx = b
 * H should be SPD matrix (only lower part will be used)
 * reordered system is (PHP^-1) (Px) = (Pb)
 */
class AMDOrdering : public Ordering {
 public:
  explicit AMDOrdering(const Eigen::SparseMatrix<double>& H) {
    assert(H.rows() == H.cols());
    Eigen::AMDOrderingPatched<int> ordering;
    ordering(H.selfadjointView<Eigen::Lower>(), Pinv_);
    P_ = Pinv_.inverse();
  }
  virtual ~AMDOrdering() = default;
};

}  // namespace minisam
