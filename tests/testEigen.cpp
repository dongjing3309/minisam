// test Eigen vector traits

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>
#include <minisam/nonlinear/numericalJacobian.h>

#include <minisam/core/Eigen.h>

using namespace std;
using namespace minisam;


/* ************************************************************************** */
TEST_CASE("VectorFixedTraits", "[geometry]") {
  CHECK(has_traits<Eigen::Vector2d>::value);
  CHECK(is_manifold<Eigen::Vector2d>::value);
  CHECK(is_lie_group<Eigen::Vector2d>::value);
}

/* ************************************************************************** */
TEST_CASE("VectorFixedDim", "[geometry]") {

  CHECK(assert_equal<size_t>(2, traits<Eigen::Vector2d>::Dim(Eigen::Vector2d())));
  CHECK(assert_equal<size_t>(3, traits<Eigen::Vector3d>::Dim(Eigen::Vector3d())));
}

/* ************************************************************************** */
TEST_CASE("VectorFixedLocal", "[geometry]") {

  CHECK(assert_equal(Eigen::Vector2d(0, 0), traits<Eigen::Vector2d>::Local(
      Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 0))));
  CHECK(assert_equal(Eigen::Vector2d(2.3, 4.1), traits<Eigen::Vector2d>::Local(
      Eigen::Vector2d(0, 0), Eigen::Vector2d(2.3, 4.1))));
  CHECK(assert_equal(Eigen::Vector2d(-0.7, 3.0), traits<Eigen::Vector2d>::Local(
      Eigen::Vector2d(2.3, 4.1), Eigen::Vector2d(1.6, 7.1))));
}

/* ************************************************************************** */
TEST_CASE("VectorFixedRetract", "[geometry]") {

  CHECK(assert_equal(Eigen::Vector2d(0, 0), traits<Eigen::Vector2d>::Retract(
      Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 0))));
  CHECK(assert_equal(Eigen::Vector2d(2.3, 4.1), traits<Eigen::Vector2d>::Retract(
      Eigen::Vector2d(0, 0), Eigen::Vector2d(2.3, 4.1))));
  CHECK(assert_equal(Eigen::Vector2d(2.3, 4.1), traits<Eigen::Vector2d>::Retract(
      Eigen::Vector2d(2.3, 4.1), Eigen::Vector2d(0, 0))));
  CHECK(assert_equal(Eigen::Vector2d(3.9, 11.2), traits<Eigen::Vector2d>::Retract(
      Eigen::Vector2d(2.3, 4.1), Eigen::Vector2d(1.6, 7.1))));
}

/* ************************************************************************** */
TEST_CASE("VectorFixedIdentity", "[geometry]") {

  CHECK(assert_equal(Eigen::Vector2d(0, 0), traits<Eigen::Vector2d>::Identity(Eigen::Vector2d())));
  CHECK(assert_equal(Eigen::Vector3d(0, 0, 0), traits<Eigen::Vector3d>::Identity(Eigen::Vector3d())));
}

/* ************************************************************************** */
TEST_CASE("VectorFixedInverse", "[geometry]") {
  
  // identity
  CHECK(assert_equal(traits<Eigen::Vector2d>::Identity(Eigen::Vector2d()), 
      traits<Eigen::Vector2d>::Inverse(traits<Eigen::Vector2d>::Identity(Eigen::Vector2d()))));

  CHECK(assert_equal(Eigen::Vector2d(0, 0), traits<Eigen::Vector2d>::Inverse(
      Eigen::Vector2d(0, 0))));
  CHECK(assert_equal(Eigen::Vector2d(-2.1, -5.6), traits<Eigen::Vector2d>::Inverse(
      Eigen::Vector2d(2.1, 5.6))));

  // jacobians
  Eigen::MatrixXd expected, actual;

  traits<Eigen::Vector2d>::InverseJacobian(Eigen::Vector2d(0, 0), actual);
  expected = numericalJacobian11(traits<Eigen::Vector2d>::Inverse, Eigen::Vector2d(0, 0));
  CHECK(assert_equal(expected, actual, 1e-6));

  traits<Eigen::Vector2d>::InverseJacobian(Eigen::Vector2d(2.1, 5.6), actual);
  expected = numericalJacobian11(traits<Eigen::Vector2d>::Inverse, Eigen::Vector2d(2.1, 5.6));
  CHECK(assert_equal(expected, actual, 1e-6));
}

/* ************************************************************************** */
TEST_CASE("VectorFixedCompose", "[geometry]") {

  CHECK(assert_equal(Eigen::Vector2d(0, 0), traits<Eigen::Vector2d>::Compose(
      Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 0))));
  CHECK(assert_equal(Eigen::Vector2d(2.1, 5.6), traits<Eigen::Vector2d>::Compose(
      Eigen::Vector2d(2.1, 5.6), Eigen::Vector2d(0, 0))));
  CHECK(assert_equal(Eigen::Vector2d(3.4, 10.4), traits<Eigen::Vector2d>::Compose(
      Eigen::Vector2d(2.1, 5.6), Eigen::Vector2d(1.3, 4.8))));

  // jacobians
  Eigen::MatrixXd expected1, expected2, actual1, actual2;

  traits<Eigen::Vector2d>::ComposeJacobians(Eigen::Vector2d(0, 0), 
      Eigen::Vector2d(0, 0), actual1, actual2);
  expected1 = numericalJacobian21(traits<Eigen::Vector2d>::Compose, 
      Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 0));
  expected2 = numericalJacobian22(traits<Eigen::Vector2d>::Compose, 
      Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 0));
  CHECK(assert_equal(expected1, actual1, 1e-6));
  CHECK(assert_equal(expected2, actual2, 1e-6));

  traits<Eigen::Vector2d>::ComposeJacobians(Eigen::Vector2d(12.5, -8.7), 
      Eigen::Vector2d(-3.6, 6.2), actual1, actual2);
  expected1 = numericalJacobian21(traits<Eigen::Vector2d>::Compose, 
      Eigen::Vector2d(12.5, -8.7), Eigen::Vector2d(-3.6, 6.2));
  expected2 = numericalJacobian22(traits<Eigen::Vector2d>::Compose, 
      Eigen::Vector2d(12.5, -8.7), Eigen::Vector2d(-3.6, 6.2));
  CHECK(assert_equal(expected1, actual1, 1e-6));
  CHECK(assert_equal(expected2, actual2, 1e-6));
}

/* ************************************************************************** */
TEST_CASE("VectorDynamicTraits", "[geometry]") {
  CHECK(has_traits<Eigen::VectorXd>::value);
  CHECK(is_manifold<Eigen::VectorXd>::value);
  CHECK(is_lie_group<Eigen::VectorXd>::value);
}

/* ************************************************************************** */
TEST_CASE("VectorDynamicDim", "[geometry]") {

  CHECK(assert_equal<size_t>(4, traits<Eigen::VectorXd>::Dim((Eigen::VectorXd(4) << 
      1, 2, 3, 4).finished())));
}

/* ************************************************************************** */
TEST_CASE("VectorDynamicLocal", "[geometry]") {

  CHECK(assert_equal((Eigen::VectorXd(2) << 0, 0).finished(), traits<Eigen::VectorXd>::Local(
      (Eigen::VectorXd(2) << 0, 0).finished(), (Eigen::VectorXd(2) << 0, 0).finished())));
  CHECK(assert_equal((Eigen::VectorXd(2) << 2.3, 4.1).finished(), traits<Eigen::VectorXd>::Local(
      (Eigen::VectorXd(2) << 0, 0).finished(), (Eigen::VectorXd(2) << 2.3, 4.1).finished())));
  CHECK(assert_equal((Eigen::VectorXd(2) << -0.7, 3.0).finished(), traits<Eigen::VectorXd>::Local(
      (Eigen::VectorXd(2) << 2.3, 4.1).finished(), (Eigen::VectorXd(2) << 1.6, 7.1).finished())));
}

/* ************************************************************************** */
TEST_CASE("VectorDynamicRetract", "[geometry]") {

  CHECK(assert_equal((Eigen::VectorXd(2) << 0, 0).finished(), traits<Eigen::VectorXd>::Retract(
      (Eigen::VectorXd(2) << 0, 0).finished(), 
      (Eigen::VectorXd(2) << 0, 0).finished())));
  CHECK(assert_equal((Eigen::VectorXd(2) << 2.3, 4.1).finished(), traits<Eigen::VectorXd>::Retract(
      (Eigen::VectorXd(2) << 0, 0).finished(), 
      (Eigen::VectorXd(2) << 2.3, 4.1).finished())));
  CHECK(assert_equal((Eigen::VectorXd(2) << 2.3, 4.1).finished(), traits<Eigen::VectorXd>::Retract(
      (Eigen::VectorXd(2) << 2.3, 4.1).finished(), 
      (Eigen::VectorXd(2) << 0, 0).finished())));
  CHECK(assert_equal((Eigen::VectorXd(2) << 3.9, 11.2).finished(), traits<Eigen::VectorXd>::Retract(
      (Eigen::VectorXd(2) << 2.3, 4.1).finished(), 
      (Eigen::VectorXd(2) << 1.6, 7.1).finished())));
}

/* ************************************************************************** */
TEST_CASE("VectorDynamicIdentity", "[geometry]") {

  CHECK(assert_equal((Eigen::VectorXd(2) << 0, 0).finished(), 
      traits<Eigen::VectorXd>::Identity(Eigen::VectorXd(2))));
  CHECK(assert_equal((Eigen::VectorXd(3) << 0, 0, 0).finished(), 
      traits<Eigen::VectorXd>::Identity(Eigen::VectorXd(3))));
}

/* ************************************************************************** */
TEST_CASE("VectorDynamicInverse", "[geometry]") {
  
  // identity
  CHECK(assert_equal(traits<Eigen::VectorXd>::Identity(Eigen::VectorXd(2)), 
      traits<Eigen::VectorXd>::Inverse(traits<Eigen::VectorXd>::Identity(Eigen::VectorXd(2)))));

  CHECK(assert_equal((Eigen::VectorXd(2) << 0, 0).finished(), traits<Eigen::VectorXd>::Inverse(
      (Eigen::VectorXd(2) << 0, 0).finished())));
  CHECK(assert_equal((Eigen::VectorXd(2) << -2.1, -5.6).finished(), traits<Eigen::VectorXd>::Inverse(
      (Eigen::VectorXd(2) << 2.1, 5.6).finished())));

  // jacobians
  Eigen::MatrixXd expected, actual;

  traits<Eigen::VectorXd>::InverseJacobian((Eigen::VectorXd(2) << 0, 0).finished(), actual);
  expected = numericalJacobian11(traits<Eigen::VectorXd>::Inverse, 
      (Eigen::VectorXd(2) << 0, 0).finished());
  CHECK(assert_equal(expected, actual, 1e-6));

  traits<Eigen::VectorXd>::InverseJacobian((Eigen::VectorXd(2) << 2.1, 5.6).finished(), actual);
  expected = numericalJacobian11(traits<Eigen::VectorXd>::Inverse, 
      (Eigen::VectorXd(2) << 2.1, 5.6).finished());
  CHECK(assert_equal(expected, actual, 1e-6));
}

/* ************************************************************************** */
TEST_CASE("VectorDynamicCompose", "[geometry]") {

  CHECK(assert_equal((Eigen::VectorXd(2) << 0, 0).finished(), traits<Eigen::VectorXd>::Compose(
      (Eigen::VectorXd(2) << 0, 0).finished(), 
      (Eigen::VectorXd(2) << 0, 0).finished())));
  CHECK(assert_equal((Eigen::VectorXd(2) << 2.1, 5.6).finished(), traits<Eigen::VectorXd>::Compose(
      (Eigen::VectorXd(2) << 2.1, 5.6).finished(), 
      (Eigen::VectorXd(2) << 0, 0).finished())));
  CHECK(assert_equal((Eigen::VectorXd(2) << 3.4, 10.4).finished(), traits<Eigen::VectorXd>::Compose(
      (Eigen::VectorXd(2) << 2.1, 5.6).finished(), 
      (Eigen::VectorXd(2) << 1.3, 4.8).finished())));

  // jacobians
  Eigen::MatrixXd expected1, expected2, actual1, actual2;

  traits<Eigen::VectorXd>::ComposeJacobians((Eigen::VectorXd(2) << 0, 0).finished(), 
      (Eigen::VectorXd(2) << 0, 0).finished(), actual1, actual2);
  expected1 = numericalJacobian21(traits<Eigen::VectorXd>::Compose, 
      (Eigen::VectorXd(2) << 0, 0).finished(), (Eigen::VectorXd(2) << 0, 0).finished());
  expected2 = numericalJacobian22(traits<Eigen::VectorXd>::Compose, 
      (Eigen::VectorXd(2) << 0, 0).finished(), (Eigen::VectorXd(2) << 0, 0).finished());
  CHECK(assert_equal(expected1, actual1, 1e-6));
  CHECK(assert_equal(expected2, actual2, 1e-6));

  traits<Eigen::VectorXd>::ComposeJacobians((Eigen::VectorXd(2) << 12.5, -8.7).finished(), 
      (Eigen::VectorXd(2) << -3.6, 6.2).finished(), actual1, actual2);
  expected1 = numericalJacobian21(traits<Eigen::VectorXd>::Compose, 
      (Eigen::VectorXd(2) << 12.5, -8.7).finished(), 
      (Eigen::VectorXd(2) << -3.6, 6.2).finished());
  expected2 = numericalJacobian22(traits<Eigen::VectorXd>::Compose, 
      (Eigen::VectorXd(2) << 12.5, -8.7).finished(), 
      (Eigen::VectorXd(2) << -3.6, 6.2).finished());
  CHECK(assert_equal(expected1, actual1, 1e-6));
  CHECK(assert_equal(expected2, actual2, 1e-6));
}
