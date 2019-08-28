/**
 * @file    CalibKD.cpp
 * @brief   calibration with radial distortion, representing a 3x3 K matrix, and
 * radial and tangential distortion
 * @author  Jing Dong
 * @date    Oct 11, 2018
 */

#include <minisam/geometry/CalibKD.h>

namespace minisam {

/* ************************************************************************** */
Eigen::Vector2d CalibKD::project(const Eigen::Vector2d& pc) const {
  //  r2 = x^2 + y^2;
  //  g = (1 + k1*r2 + k2*r2^2);
  //  dp = [2*p1*x*y + p2*(r2 + 2*x^2); 2*p2*x*y + p1*(r2 + 2*y^2)];
  //  pi(:,i) = g * pc(:,i) + dp;

  // Radial component
  const double xy = pc.x() * pc.y();
  const double xx = pc.x() * pc.x();
  const double yy = pc.y() * pc.y();
  const double r2 = xx + yy;
  const double r4 = r2 * r2;
  const double r = 1.0 + k1() * r2 + k2() * r4;  // scaling factor

  // tangential component
  const double tx = 2. * p1() * xy + p2() * (r2 + 2.0 * xx);
  const double ty = 2. * p2() * xy + p1() * (r2 + 2.0 * yy);

  // Radial and tangential distortion applied
  const double dx = r * pc.x() + tx;
  const double dy = r * pc.y() + ty;
  return Eigen::Vector2d(fx() * dx + cx(), fy() * dy + cy());
}

/* ************************************************************************** */
void CalibKD::projectJacobians(const Eigen::Vector2d& pc,
                               Eigen::Matrix<double, 2, 8>& J_K,
                               Eigen::Matrix<double, 2, 2>& J_p) const {
  // Radial component
  const double xy = pc.x() * pc.y();
  const double xx = pc.x() * pc.x();
  const double yy = pc.y() * pc.y();
  const double r2 = xx + yy;
  const double r4 = r2 * r2;
  const double r = 1.0 + k1() * r2 + k2() * r4;  // scaling factor

  // tangential component
  const double tx = 2. * p1() * xy + p2() * (r2 + 2.0 * xx);
  const double ty = 2. * p2() * xy + p1() * (r2 + 2.0 * yy);

  // Radial and tangential distortion applied
  const double dx = r * pc.x() + tx;
  const double dy = r * pc.y() + ty;

  // see gtsam/geometry/Cal3DS2_Base.cpp Cal3DS2_Base::D2dcalibration
  // clang-format off
  J_K <<  dx, 0.0, 1.0, 0.0, fx()*pc.x()*r2, fx()*pc.x()*r4, fx()*2*xy, fx()*(r2+2*xx),
          0.0, dy, 0.0, 1.0, fy()*pc.y()*r2, fy()*pc.y()*r4, fy()*(r2+2*yy), fy()*2*xy;
  // clang-format on

  // see gtsam/geometry/Cal3DS2_Base.cpp Cal3DS2_Base::D2dintrinsic
  const double drdx = 2.0 * pc.x();
  const double drdy = 2.0 * pc.y();
  const double dgdx = k1() * drdx + k2() * 2.0 * r2 * drdx;
  const double dgdy = k1() * drdy + k2() * 2.0 * r2 * drdy;

  // Dx = 2*p1*xy + p2*(rr+2*xx);
  // Dy = 2*p2*xy + p1*(rr+2*yy);
  const double dDxdx = 2.0 * p1() * pc.y() + p2() * (drdx + 4.0 * pc.x());
  const double dDxdy = 2.0 * p1() * pc.x() + p2() * drdy;
  const double dDydx = 2.0 * p2() * pc.y() + p1() * drdx;
  const double dDydy = 2.0 * p2() * pc.x() + p1() * (drdy + 4.0 * pc.y());

  // clang-format off
  J_p <<  fx() * (r + pc.x()*dgdx + dDxdx), fx() * (pc.x()*dgdy + dDxdy),
          fy() * (pc.y()*dgdx + dDydx), fy() * (r + pc.y()*dgdy + dDydy);
  // clang-format on
}

/* ************************************************************************** */
Eigen::Vector2d CalibKD::unproject(const Eigen::Vector2d& pi) const {
  // looking for fixed point of (pc -txy) / r = project(pc)
  // start from invK * pi

  Eigen::Vector2d invKpi, pc;
  invKpi << (pi.x() - cx()) / fx(), (pi.y() - cy()) / fy();
  pc = invKpi;

  // const parameter for fixed point searching
  constexpr int max_iter = 20;
  constexpr double tol = 1e-5;

  int i;
  for (i = 0; i < max_iter; i++) {
    // stop
    if ((project(pc) - pi).norm() < tol) break;

    // Radial component
    const double xy = pc.x() * pc.y(), xx = pc.x() * pc.x(),
                 yy = pc.y() * pc.y();
    const double r2 = xx + yy;
    const double r = 1.0 + k1() * r2 + k2() * r2 * r2;  // scaling factor
    // tangential component
    const double tx = 2. * p1() * xy + p2() * (r2 + 2.0 * xx);
    const double ty = 2. * p2() * xy + p1() * (r2 + 2.0 * yy);

    pc = (Eigen::Vector2d() << invKpi(0) - tx, invKpi(1) - ty).finished() / r;
  }

  if (i == max_iter) {
    throw std::runtime_error("CalibKD::unproject fails to converge");
  }

  return pc;
}

}  // namespace minisam
