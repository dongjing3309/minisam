/**
 * @file    testAssertions.h
 * @brief   Utils for calculating numerical Jacobians
 * @author  Jing Dong
 * @date    Sep 14, 2018
 */

#pragma once

// Eigen/scalar traits implementation needed for assert_equal
#include <minisam/core/Eigen.h>
#include <minisam/core/Traits.h>
#include <minisam/core/Variables.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

namespace minisam {

/// equal of intergers
template <class T,
          class = typename std::enable_if<std::is_integral<T>::value>::type>
bool assert_equal(T expected, T actual) {
  if (expected != actual) {
    std::cout << "Not equal:" << std::endl
              << "expected: " << expected << std::endl
              << "actual: " << actual << std::endl;
    return false;
  }
  return true;
}

/// equal of manifold types given tolerance
template <class T,
          class = typename std::enable_if<!std::is_integral<T>::value>::type>
bool assert_equal(const T& expected, const T& actual, double tol = 1e-9) {
  static_assert(is_manifold<T>::value,
                "assert_equal<T> request input T type must be a manifold");

  // check dim
  if (traits<T>::Dim(expected) != traits<T>::Dim(actual)) {
    std::cout << "Not equal:" << std::endl
              << "expected dimension: " << traits<T>::Dim(expected) << std::endl
              << "actual dimension: " << traits<T>::Dim(actual) << std::endl;
    return false;
  }
  // check values
  typename traits<T>::TangentVector diff = traits<T>::Local(expected, actual);
  if (diff.norm() > tol) {
    std::cout << "Not equal:" << std::endl << "expected: ";
    traits<T>::Print(expected);
    std::cout << std::endl << "actual: ";
    traits<T>::Print(actual);
    std::cout << std::endl;
    return false;
  }
  return true;
}

/// specializations of non-traits type matrix
template <>
bool assert_equal(const Eigen::MatrixXd& expected,
                  const Eigen::MatrixXd& actual, double tol);

template <>
bool assert_equal(const Eigen::SparseMatrix<double>& expected,
                  const Eigen::SparseMatrix<double>& actual, double tol);

/// specializations of non-traits type Variables
template <>
bool assert_equal(const Variables& expected, const Variables& actual,
                  double tol);

/// matrix assert_equal for fixed-size and mixing cases
bool assert_equal_matrix(const Eigen::MatrixXd& expected,
                         const Eigen::MatrixXd& actual, double tol = 1e-6);

/// compare vector of interger types
template <class T,
          class = typename std::enable_if<std::is_integral<T>::value>::type>
bool assert_equal_vector(const std::vector<T>& expected,
                         const std::vector<T>& actual) {
  // check size
  if (expected.size() != actual.size()) {
    std::cout << "Not equal:" << std::endl
              << "expected size: " << expected.size() << std::endl
              << "actual size: " << actual.size() << std::endl;
    return false;
  }
  // check values
  for (size_t i = 0; i < expected.size(); i++) {
    if (expected[i] != actual[i]) {
      std::cout << "Not equal:" << std::endl;
      std::cout << "expected: ";
      for (size_t j = 0; j < expected.size(); j++)
        std::cout << expected[j] << ",";
      std::cout << std::endl;
      std::cout << "actual: ";
      for (size_t j = 0; j < actual.size(); j++) std::cout << actual[j] << ",";
      std::cout << std::endl;
      return false;
    }
  }
  return true;
}

/// compare vector of manifold types
template <class T,
          class = typename std::enable_if<!std::is_integral<T>::value>::type>
bool assert_equal_vector(const std::vector<T>& expected,
                         const std::vector<T>& actual, double tol = 1e-9) {
  // check size
  if (expected.size() != actual.size()) {
    std::cout << "Not equal:" << std::endl
              << "expected size: " << expected.size() << std::endl
              << "actual size: " << actual.size() << std::endl;
    return false;
  }
  // check values
  for (size_t i = 0; i < expected.size(); i++) {
    if (!assert_equal<T>(expected[i], actual[i], tol)) {
      std::cout << "Not equal index: " << i << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace minisam
