/**
 * @file    Ordering.h
 * @brief   Abstract ordering method, used by square root information etc., but
 * not linear solver
 * @author  Jing Dong
 * @date    Oct 28, 2018
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace minisam {

/** ordering methods available, natural ordering (none) or AMD ordering */
enum class OrderingMethod {
  NONE = 0,
  AMD,
};

// ordering base method for SPD system, by supply perm matrix
class Ordering {
 public:
  typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>
      PermMatrix;

 protected:
  PermMatrix P_, Pinv_;

 public:
  explicit Ordering(const PermMatrix& perm) : P_(perm), Pinv_(perm.inverse()) {}
  explicit Ordering(const Eigen::VectorXi& indices) : P_(indices) {
    Pinv_ = P_.inverse();
  }

  virtual ~Ordering() = default;

  // permute (full) linear system H
  // only use lower part of H
  virtual void permuteSystemFull(const Eigen::SparseMatrix<double>& H,
                                 Eigen::SparseMatrix<double>& PHPt) const {
    PHPt = H.selfadjointView<Eigen::Lower>().twistedBy(P_);
  }

  // permute adjoint view of linear system H
  // only use lower part of H
  template <int UpLo>
  void permuteSystemSelfAdjoint(const Eigen::SparseMatrix<double>& H,
                                Eigen::SparseMatrix<double>& PHPt) const {
    PHPt.selfadjointView<UpLo>() =
        H.selfadjointView<Eigen::Lower>().twistedBy(P_);
  }

  // permute right hand side vector b by P
  virtual void permuteRhs(const Eigen::VectorXd& b, Eigen::VectorXd& Pb) const {
    Pb = P_ * b;
  }

  // permute back solution by P^-1
  virtual void permuteBackSolution(const Eigen::VectorXd& Px,
                                   Eigen::VectorXd& x) const {
    x = Pinv_ * Px;
  }

  // access permutation matrix
  const PermMatrix& P() const { return P_; }
  const PermMatrix& Pinv() const { return Pinv_; }

  // permutation indices
  const Eigen::VectorXi& indices() const { return P_.indices(); }
  const Eigen::VectorXi& reversedIndices() const { return Pinv_.indices(); }

 protected:
  // default ctor should be only called by child classes since it's unsafe
  Ordering() = default;
};

// natural ordering, no pernumation will be apply
class NaturalOrdering : public Ordering {
 public:
  explicit NaturalOrdering(const Eigen::SparseMatrix<double>& H) {
    assert(H.rows() == H.cols());
    Eigen::VectorXi idx(H.cols());
    for (int i = 0; i < H.cols(); i++) {
      idx[i] = i;
    }
    P_ = PermMatrix(idx);
    Pinv_ = P_.inverse();
  }

  virtual ~NaturalOrdering() = default;

  // override permutation methods with more efficient implementations
  void permuteSystemFull(const Eigen::SparseMatrix<double>& H,
                         Eigen::SparseMatrix<double>& PHPt) const override {
    PHPt = H.selfadjointView<Eigen::Lower>();
  }

  // only use lower part of H
  template <int UpLo>
  void permuteSystemSelfAdjoint(const Eigen::SparseMatrix<double>& H,
                                Eigen::SparseMatrix<double>& PHPt) const {
    PHPt.selfadjointView<UpLo>() = H.selfadjointView<Eigen::Lower>();
  }

  void permuteRhs(const Eigen::VectorXd& b,
                  Eigen::VectorXd& Pb) const override {
    Pb = b;
  }

  void permuteBackSolution(const Eigen::VectorXd& Px,
                           Eigen::VectorXd& x) const override {
    x = Px;
  }
};

}  // namespace minisam
