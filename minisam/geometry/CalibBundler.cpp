/**
 * @file    CalibBundler.cpp
 * @brief   intrinsic camera calibration with distortion, used by Bundler
 * @author  Jing Dong
 * @date    Oct 18, 2018
 */

#include <minisam/geometry/CalibBundler.h>

namespace minisam {

/* ************************************************************************** */
void CalibBundler::projectJacobians(const Eigen::Vector2d& pc,
                                    Eigen::Matrix<double, 2, 3>& J_K,
                                    Eigen::Matrix<double, 2, 2>& J_p) const {
  const double r2 = pc.squaredNorm();
  const double d = 1.0 + (k1() + k2() * r2) * r2;

  const double r2x = r2 * pc(0);
  const double r2y = r2 * pc(1);
  // clang-format off
  J_K << d * pc(0), f() * r2x, f() * r2 * r2x,
         d * pc(1), f() * r2y, f() * r2 * r2y;
  // clang-format on

  const double a = 2.0 * (k1() + 2.0 * k2() * r2);
  const double axx = a * pc(0) * pc(0);
  const double axy = a * pc(0) * pc(1);
  const double ayy = a * pc(1) * pc(1);
  J_p << d + axx, axy, axy, d + ayy;
  J_p *= f();
}

/* ************************************************************************** */
Eigen::Vector2d CalibBundler::unproject(const Eigen::Vector2d& pi) const {
  // looking for fixed point of (pc -txy) / r = project(pc)
  // start from invK * pi

  Eigen::Vector2d invKpi, pc;
  invKpi << pi(0) / f(), pi(1) / f();
  pc = invKpi;

  // const parameter for fixed point searching
  constexpr int max_iter = 20;
  constexpr double tol = 1e-5;

  int i;
  for (i = 0; i < max_iter; i++) {
    // stop
    if ((project(pc) - pi).norm() < tol) break;
    const double r2 = pc.squaredNorm();
    const double d = 1.0 + (k1() + k2() * r2) * r2;
    pc = invKpi / d;
  }

  if (i == max_iter) {
    throw std::runtime_error("CalibBundler::unproject fails to converge");
  }
  return pc;
}

}  // namespace minisam
