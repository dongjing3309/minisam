// test scalar traits

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>
#include <minisam/nonlinear/numericalJacobian.h>

#include <minisam/core/Scalar.h>

using namespace std;
using namespace minisam;

typedef Eigen::Matrix<double, 1, 1> Vector1d;


// const ref wrapper for numerical jacobians
double wrapperInverse(const double& s) { 
  return traits<double>::Inverse(s); 
}

double wrapperCompose(const double& s1, const double& s2) { 
  return traits<double>::Compose(s1, s2); 
}


/* ************************************************************************** */
TEST_CASE("doubleTraits", "[geometry]") {
  // double traits: defined
  CHECK(has_traits<double>::value);
  CHECK(is_manifold<double>::value);
  CHECK(is_lie_group<double>::value);
  // float traits: not defined
  CHECK_FALSE(has_traits<float>::value);
  CHECK_FALSE(is_manifold<float>::value);
  CHECK_FALSE(is_lie_group<float>::value);
}

/* ************************************************************************** */
TEST_CASE("doubleDim", "[geometry]") {
  CHECK(assert_equal<size_t>(1, traits<double>::Dim(0.0)));
  CHECK(assert_equal<size_t>(1, traits<double>::Dim(1.0)));
}

/* ************************************************************************** */
TEST_CASE("doubleLocal", "[geometry]") {
  CHECK(assert_equal(Vector1d(0.0), traits<double>::Local(0.0, 0.0)));
  CHECK(assert_equal(Vector1d(1.0), traits<double>::Local(0.0, 1.0)));
  CHECK(assert_equal(Vector1d(-1.0), traits<double>::Local(1.0, 0.0)));
}

/* ************************************************************************** */
TEST_CASE("doubleRetract", "[geometry]") {
  CHECK(assert_equal(0.0, traits<double>::Retract(0.0, Vector1d(0.0))));
  CHECK(assert_equal(2.1, traits<double>::Retract(0.0, Vector1d(2.1))));
  CHECK(assert_equal(2.1, traits<double>::Retract(2.1, Vector1d(0.0))));
  CHECK(assert_equal(3.1, traits<double>::Retract(2.1, Vector1d(1.0))));
}

/* ************************************************************************** */
TEST_CASE("doubleIdentity", "[geometry]") {
  CHECK(assert_equal(0.0, traits<double>::Identity(double())));
}

/* ************************************************************************** */
TEST_CASE("doubleInverse", "[geometry]") {

  // identity
  CHECK(assert_equal(traits<double>::Inverse(traits<double>::Identity(double())), 
      traits<double>::Identity(double())));

  CHECK(assert_equal(0.0, traits<double>::Inverse(0.0)));
  CHECK(assert_equal(-1.0, traits<double>::Inverse(1.0)));
  CHECK(assert_equal(32.4, traits<double>::Inverse(-32.4)));

  // jacobians
  Eigen::MatrixXd expected, actual;

  traits<double>::InverseJacobian(0.0, actual);
  expected = numericalJacobian11(wrapperInverse, 0.0);
  CHECK(assert_equal(expected, actual, 1e-6));

  traits<double>::InverseJacobian(32.4, actual);
  expected = numericalJacobian11(wrapperInverse, 32.4);
  CHECK(assert_equal(expected, actual, 1e-6));
}

/* ************************************************************************** */
TEST_CASE("doubleCompose", "[geometry]") {

  CHECK(assert_equal(0.0, traits<double>::Compose(0.0, 0.0)));
  CHECK(assert_equal(32.4, traits<double>::Compose(32.4, 0.0)));
  CHECK(assert_equal(33.4, traits<double>::Compose(32.4, 1.0)));

  // jacobians
  Eigen::MatrixXd expected1, expected2, actual1, actual2;

  traits<double>::ComposeJacobians(0.0, 0.0, actual1, actual2);
  expected1 = numericalJacobian21(wrapperCompose, 0.0, 0.0);
  expected2 = numericalJacobian22(wrapperCompose, 0.0, 0.0);
  CHECK(assert_equal(expected1, actual1, 1e-6));
  CHECK(assert_equal(expected2, actual2, 1e-6));

  traits<double>::ComposeJacobians(32.4, 0.0, actual1, actual2);
  expected1 = numericalJacobian21(wrapperCompose, 32.4, 0.0);
  expected2 = numericalJacobian22(wrapperCompose, 32.4, 0.0);
  CHECK(assert_equal(expected1, actual1, 1e-6));
  CHECK(assert_equal(expected2, actual2, 1e-6));

  traits<double>::ComposeJacobians(32.4, -2.1, actual1, actual2);
  expected1 = numericalJacobian21(wrapperCompose, 32.4, -2.1);
  expected2 = numericalJacobian22(wrapperCompose, 32.4, -2.1);
  CHECK(assert_equal(expected1, actual1, 1e-6));
  CHECK(assert_equal(expected2, actual2, 1e-6));
}

