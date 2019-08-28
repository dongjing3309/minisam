/**
 * @file    LossFunction.h
 * @brief   Gaussian loss function with mahalanobis distance, and robust loss
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#pragma once

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace minisam {

/** abstract loss function class, without implementation */
class LossFunction {
 public:
  virtual ~LossFunction() = default;

  virtual void print(std::ostream& out = std::cout) const {
    out << "Abstract loss function";
  }

  // non-in-place whitening in case needed
  Eigen::VectorXd weightError(const Eigen::VectorXd& b) const {
    Eigen::VectorXd wht_b = b;
    weightInPlace(wht_b);
    return wht_b;
  }

  // non-in-place whitening in case needed
  std::vector<Eigen::MatrixXd> weightJacobians(
      const std::vector<Eigen::MatrixXd>& As, const Eigen::VectorXd& b) const {
    std::vector<Eigen::MatrixXd> wht_As = As;
    Eigen::VectorXd wht_b = b;
    weightInPlace(wht_As, wht_b);
    return wht_As;
  }

  /** implementation needed for actual  */

  // weight error: apply loss function
  // in place operation to avoid excessive memory operation
  virtual void weightInPlace(Eigen::VectorXd& b) const = 0;

  // weight jacobian matrices and error: apply loss function
  // in place operation to avoid excessive memory operation
  virtual void weightInPlace(std::vector<Eigen::MatrixXd>& As,
                             Eigen::VectorXd& b) const = 0;

 protected:
  LossFunction() = default;
};

// implementation of loss functions

/** general Gaussian loss function with mahalanobis distance */
class GaussianLoss : public LossFunction {
 private:
  // squared root information matrix
  // upper triangular by Cholesky: sqrt_info_' * sqrt_info_ = I = \Sigma^{-1}
  Eigen::MatrixXd sqrt_info_;

  // private constructor from squared root information matrix
  explicit GaussianLoss(const Eigen::MatrixXd& R)
      : LossFunction(), sqrt_info_(R) {}

 public:
  // static shared pointer constructors
  static std::shared_ptr<LossFunction> SqrtInformation(
      const Eigen::MatrixXd& R);

  // for information and covariance matrix
  // only use UPPER triangular part as self-adjoint
  static std::shared_ptr<LossFunction> Information(const Eigen::MatrixXd& I);
  static std::shared_ptr<LossFunction> Covariance(const Eigen::MatrixXd& Sigma);

  virtual ~GaussianLoss() = default;

  void print(std::ostream& out = std::cout) const override;

  // weight error
  void weightInPlace(Eigen::VectorXd& b) const override;

  // weight jacobian matrix and error
  void weightInPlace(std::vector<Eigen::MatrixXd>& As,
                     Eigen::VectorXd& b) const override;

  // access
  const Eigen::MatrixXd R() const { return sqrt_info_; }
};

/** Gaussian loss function with diagonal covariance matrix */
class DiagonalLoss : public LossFunction {
 private:
  // squared root information matrix diagonals
  Eigen::VectorXd sqrt_info_diag_;

  // private constructor
  explicit DiagonalLoss(const Eigen::VectorXd& R_diag)
      : LossFunction(), sqrt_info_diag_(R_diag) {}

 public:
  // static shared pointer constructors
  static std::shared_ptr<LossFunction> Precisions(
      const Eigen::VectorXd& I_diag);
  static std::shared_ptr<LossFunction> Sigmas(const Eigen::VectorXd& S_diag);
  static std::shared_ptr<LossFunction> Variances(const Eigen::VectorXd& V_diag);
  static std::shared_ptr<LossFunction> Scales(const Eigen::VectorXd& s);

  virtual ~DiagonalLoss() = default;

  void print(std::ostream& out = std::cout) const override;

  // weight error
  void weightInPlace(Eigen::VectorXd& b) const override;

  // weight jacobian matrix and error
  void weightInPlace(std::vector<Eigen::MatrixXd>& As,
                     Eigen::VectorXd& b) const override;
};

/** loss function with a single scale */
class ScaleLoss : public LossFunction {
 private:
  // squared root information matrix precision
  double inv_sigma_;

  // private constructor
  explicit ScaleLoss(double inv_sigma)
      : LossFunction(), inv_sigma_(inv_sigma) {}

 public:
  // static shared pointer constructors
  static std::shared_ptr<LossFunction> Precision(double prec);
  static std::shared_ptr<LossFunction> Sigma(double sigma);
  static std::shared_ptr<LossFunction> Variance(double var);
  static std::shared_ptr<LossFunction> Scale(double s);

  virtual ~ScaleLoss() = default;

  void print(std::ostream& out = std::cout) const override;

  // weight error
  void weightInPlace(Eigen::VectorXd& b) const override;

  // weight jacobian matrix and error
  void weightInPlace(std::vector<Eigen::MatrixXd>& As,
                     Eigen::VectorXd& b) const override;
};

/** Cauchy robust loss function */
class CauchyLoss : public LossFunction {
 private:
  double k_, k2_;

  // private constructor
  explicit CauchyLoss(double k) : LossFunction(), k_(k), k2_(k * k) {}

 public:
  // static shared pointer constructors
  static std::shared_ptr<LossFunction> Cauchy(double k);

  virtual ~CauchyLoss() = default;

  void print(std::ostream& out = std::cout) const override;

  // robust error weight
  double weight(double err) const { return k2_ / (k2_ + err * err); }

  // weight error
  void weightInPlace(Eigen::VectorXd& b) const override;

  // weight jacobian matrix and error
  void weightInPlace(std::vector<Eigen::MatrixXd>& As,
                     Eigen::VectorXd& b) const override;
};

/** Huber robust loss function */
class HuberLoss : public LossFunction {
 private:
  double k_;

  // private constructor
  explicit HuberLoss(double k) : LossFunction(), k_(k) {}

 public:
  // static shared pointer constructors
  static std::shared_ptr<LossFunction> Huber(double k);

  virtual ~HuberLoss() = default;

  void print(std::ostream& out = std::cout) const override;

  // robust error weight
  double weight(double err) const {
    return (err < k_) ? 1.0 : (k_ / std::fabs(err));
  }

  // weight error
  void weightInPlace(Eigen::VectorXd& b) const override;

  // weight jacobian matrix and error
  void weightInPlace(std::vector<Eigen::MatrixXd>& As,
                     Eigen::VectorXd& b) const override;
};

// compose two loss function
class ComposedLoss : public LossFunction {
 private:
  std::shared_ptr<LossFunction> l1_, l2_;

  // private constructor
  ComposedLoss(const std::shared_ptr<LossFunction>& l1,
               const std::shared_ptr<LossFunction>& l2)
      : LossFunction(), l1_(l1), l2_(l2) {}

 public:
  // static shared pointer constructors
  static std::shared_ptr<LossFunction> Compose(
      const std::shared_ptr<LossFunction>& l1,
      const std::shared_ptr<LossFunction>& l2) {
    return std::shared_ptr<LossFunction>(new ComposedLoss(l1, l2));
  }

  virtual ~ComposedLoss() = default;

  void print(std::ostream& out = std::cout) const override;

  // weight error
  void weightInPlace(Eigen::VectorXd& b) const override;

  // weight jacobian matrix and error
  void weightInPlace(std::vector<Eigen::MatrixXd>& As,
                     Eigen::VectorXd& b) const override;
};

}  // namespace minisam
