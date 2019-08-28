/**
 * @file    LossFunction.cpp
 * @brief   Gaussian loss function with mahalanobis distance, and robust loss
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#include <minisam/core/LossFunction.h>

#include <Eigen/Dense>

namespace minisam {

/* ************************************************************************** */
std::shared_ptr<LossFunction> GaussianLoss::SqrtInformation(
    const Eigen::MatrixXd& R) {
  assert(R.rows() == R.cols() &&
         "[GaussianLoss::SqrtInformation] non-square sqrt-root matrix");
  return std::shared_ptr<LossFunction>(new GaussianLoss(R));
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> GaussianLoss::Information(
    const Eigen::MatrixXd& I) {
  assert(I.rows() == I.cols() &&
         "[GaussianLoss::Information] non-square information matrix");
  Eigen::LLT<Eigen::MatrixXd> llt(I.selfadjointView<Eigen::Upper>());
  return std::shared_ptr<LossFunction>(new GaussianLoss(llt.matrixU()));
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> GaussianLoss::Covariance(
    const Eigen::MatrixXd& Sigma) {
  assert(Sigma.rows() == Sigma.cols() &&
         "[GaussianLoss::Covariance] non-square covariance matrix");
  return Information(Sigma.inverse());
}

/* ************************************************************************** */
void GaussianLoss::print(std::ostream& out) const {
  out << "Gaussian loss function : R =" << std::endl << sqrt_info_ << std::endl;
}

/* ************************************************************************** */
void GaussianLoss::weightInPlace(Eigen::VectorXd& b) const {
  assert(sqrt_info_.cols() == b.size() &&
         "[GaussianLoss::weightInPlace] error size wrong");
  b = sqrt_info_ * b;
}

/* ************************************************************************** */
void GaussianLoss::weightInPlace(std::vector<Eigen::MatrixXd>& As,
                                 Eigen::VectorXd& b) const {
  assert(sqrt_info_.cols() == b.size() &&
         "[GaussianLoss::weightInPlace] error size wrong");
  b = sqrt_info_ * b;
  for (auto& A : As) {
    assert(sqrt_info_.cols() == A.rows() &&
           "[GaussianLoss::weightInPlace] jacobian size wrong");
    A = sqrt_info_ * A;
  }
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> DiagonalLoss::Precisions(
    const Eigen::VectorXd& I_diag) {
  return std::shared_ptr<LossFunction>(new DiagonalLoss(I_diag.cwiseSqrt()));
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> DiagonalLoss::Sigmas(
    const Eigen::VectorXd& S_diag) {
  return std::shared_ptr<LossFunction>(new DiagonalLoss(S_diag.cwiseInverse()));
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> DiagonalLoss::Variances(
    const Eigen::VectorXd& V_diag) {
  return std::shared_ptr<LossFunction>(
      new DiagonalLoss((V_diag.cwiseInverse()).cwiseSqrt()));
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> DiagonalLoss::Scales(const Eigen::VectorXd& s) {
  return std::shared_ptr<LossFunction>(new DiagonalLoss(s));
}

/* ************************************************************************** */
void DiagonalLoss::print(std::ostream& out) const {
  out << "Diagonal loss function : R_diag = [" << sqrt_info_diag_.transpose()
      << "]'" << std::endl;
}

/* ************************************************************************** */
void DiagonalLoss::weightInPlace(Eigen::VectorXd& b) const {
  assert(sqrt_info_diag_.size() == b.size() &&
         "[DiagonalLoss::weightInPlace] error size wrong");
  b = b.cwiseProduct(sqrt_info_diag_);
}

/* ************************************************************************** */
void DiagonalLoss::weightInPlace(std::vector<Eigen::MatrixXd>& As,
                                 Eigen::VectorXd& b) const {
  assert(sqrt_info_diag_.size() == b.size() &&
         "[DiagonalLoss::weightInPlace] error size wrong");
  b = b.cwiseProduct(sqrt_info_diag_);
  for (auto& A : As) {
    assert(sqrt_info_diag_.size() == A.rows() &&
           "[DiagonalLoss::weightInPlace] jacobian size wrong");
    for (int i = 0; i < A.rows(); i++) {
      A.row(i) *= sqrt_info_diag_(i);
    }
  }
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> ScaleLoss::Precision(double prec) {
  return std::shared_ptr<LossFunction>(new ScaleLoss(std::sqrt(prec)));
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> ScaleLoss::Sigma(double sigma) {
  return std::shared_ptr<LossFunction>(new ScaleLoss(1.0 / sigma));
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> ScaleLoss::Variance(double var) {
  return std::shared_ptr<LossFunction>(new ScaleLoss(1.0 / std::sqrt(var)));
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> ScaleLoss::Scale(double s) {
  return std::shared_ptr<LossFunction>(new ScaleLoss(s));
}

/* ************************************************************************** */
void ScaleLoss::print(std::ostream& out) const {
  out << "Scale loss function : inv_sigma = " << inv_sigma_ << std::endl;
}

/* ************************************************************************** */
void ScaleLoss::weightInPlace(Eigen::VectorXd& b) const { b *= inv_sigma_; }

/* ************************************************************************** */
void ScaleLoss::weightInPlace(std::vector<Eigen::MatrixXd>& As,
                              Eigen::VectorXd& b) const {
  b *= inv_sigma_;
  for (auto& A : As) {
    A *= inv_sigma_;
  }
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> CauchyLoss::Cauchy(double k) {
  return std::shared_ptr<LossFunction>(new CauchyLoss(k));
}

/* ************************************************************************** */
void CauchyLoss::print(std::ostream& out) const {
  out << "Cauchy loss function : k = " << k_ << std::endl;
}

/* ************************************************************************** */
void CauchyLoss::weightInPlace(Eigen::VectorXd& b) const {
  double sqrtw = std::sqrt(weight(b.norm()));
  b *= sqrtw;
}

/* ************************************************************************** */
void CauchyLoss::weightInPlace(std::vector<Eigen::MatrixXd>& As,
                               Eigen::VectorXd& b) const {
  const double sqrtw = std::sqrt(weight(b.norm()));
  b *= sqrtw;
  for (auto& A : As) {
    A *= sqrtw;
  }
}

/* ************************************************************************** */
std::shared_ptr<LossFunction> HuberLoss::Huber(double k) {
  return std::shared_ptr<LossFunction>(new HuberLoss(k));
}

/* ************************************************************************** */
void HuberLoss::print(std::ostream& out) const {
  out << "Huber loss function : k = " << k_ << std::endl;
}

/* ************************************************************************** */
void HuberLoss::weightInPlace(Eigen::VectorXd& b) const {
  double sqrtw = std::sqrt(weight(b.norm()));
  b *= sqrtw;
}

/* ************************************************************************** */
void HuberLoss::weightInPlace(std::vector<Eigen::MatrixXd>& As,
                              Eigen::VectorXd& b) const {
  const double sqrtw = std::sqrt(weight(b.norm()));
  b *= sqrtw;
  for (auto& A : As) {
    A *= sqrtw;
  }
}

/* ************************************************************************** */
void ComposedLoss::print(std::ostream& out) const {
  out << "Composed loss function : " << std::endl << "Loss 1 : ";
  l1_->print(out);
  out << "Loss 2 : ";
  l2_->print(out);
}

/* ************************************************************************** */
void ComposedLoss::weightInPlace(Eigen::VectorXd& b) const {
  l1_->weightInPlace(b);
  l2_->weightInPlace(b);
}

/* ************************************************************************** */
void ComposedLoss::weightInPlace(std::vector<Eigen::MatrixXd>& As,
                                 Eigen::VectorXd& b) const {
  l1_->weightInPlace(As, b);
  l2_->weightInPlace(As, b);
}
}  // namespace minisam
