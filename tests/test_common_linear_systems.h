/**
 * @file    test_common_linear_systems.h
 * @brief   some common utils used in unit tests
 * @author  Jing Dong
 * @date    Mar 30, 2019
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace minisam {
namespace test {

// exmaple systems for linear tests
struct ExampleLinearSystems {
  Eigen::SparseMatrix<double> A1, A2, A3, A4;
  Eigen::SparseMatrix<double> R1_exp, R2_exp, R3_exp;
  Eigen::VectorXd b1, b2, b3, b4;
  Eigen::VectorXd x1_exp, x2_exp, x3_exp;

  ExampleLinearSystems() {
    prepare_static_linear_systems_();
  }

private:
  void prepare_static_linear_systems_() {

  A1.resize(2,2);
  A2.resize(2,2);
  A3.resize(3,2);
  A4.resize(2,2);
  R1_exp.resize(2,2);
  R2_exp.resize(2,2);
  R3_exp.resize(2,2);
  b1.resize(2,1);
  b2.resize(2,1);
  b3.resize(3,1);
  b4.resize(2,1);
  x1_exp.resize(2,1);
  x2_exp.resize(2,1);
  x3_exp.resize(2,1);

  // sys 1, identity
  A1.insert(0,0) = 1.0;
  A1.insert(1,1) = 1.0;
  A1.makeCompressed();
  R1_exp.insert(0,0) = 1.0;
  R1_exp.insert(1,1) = 1.0;
  R1_exp.makeCompressed();
  b1 << 2.3, 6.5;
  x1_exp << 2.3, 6.5;

  // sys 2, well-cond, use matlab get ground truth
  A2.insert(0,0) = 3.2;
  A2.insert(0,1) = 4.5;
  A2.insert(1,0) = -1.9;
  A2.insert(1,1) = 7.6;
  A2.makeCompressed();
  R2_exp.insert(0,0) = 3.721558813185679;
  R2_exp.insert(0,1) = -1.074818429800872e-02;
  R2_exp.insert(1,1) = 8.832320446889044;
  R2_exp.makeCompressed();
  b2 << 3.7, -4.5;
  x2_exp << 1.471554609066018, -0.224216610891390;

  // sys 3, over-cond, use matlab get ground truth
  A3.insert(0,0) = 3.2;
  A3.insert(0,1) = 4.5;
  A3.insert(1,0) = -1.9;
  A3.insert(1,1) = 7.6;
  A3.insert(2,0) = 5.5;
  A3.insert(2,1) = 3.4;
  A3.makeCompressed();
  R3_exp.insert(0,0) = 6.640783086353597;
  R3_exp.insert(0,1) = 2.809909578035331;
  R3_exp.insert(1,1) = 9.037389455106231;
  R3_exp.makeCompressed();
  b3 << 3.7, -4.5, 0.2;
  x3_exp << 0.621807917472435, -0.317884735291232;

  // sys 4, ill-cond
  A4.insert(0,0) = 3.2;
  A4.insert(0,1) = 6.4;
  A4.insert(1,0) = -1.9;
  A4.insert(1,1) = -3.8;
  A4.makeCompressed();
  b4 << 3.7, -4.5;
}
};

} // namespace test
} // namespace minisam
