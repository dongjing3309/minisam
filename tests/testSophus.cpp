// test Sophus traits

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>
#include <minisam/nonlinear/numericalJacobian.h>

#include <minisam/geometry/Sophus.h>

#include <string>

using namespace std;
using namespace minisam;


// template for testing different Sophus types
template<class T>
void _testTraits(const T& /*i*/) {
  CHECK(has_traits<T>::value);
  CHECK(is_manifold<T>::value);
  CHECK(is_lie_group<T>::value);
}

template<class T>
void _testDim(const T& i) {
  CHECK(assert_equal(size_t(T::DoF), traits<T>::Dim(i)));
}

template<class T>
void _testIdentity(const T& i) {
  CHECK(assert_equal(i, traits<T>::Identity(i)));
}

template<class T>
void _testLocalRetract(const T& i, const T& ni) {
  CHECK(assert_equal(ni, traits<T>::Retract(i, traits<T>::Local(i, ni))));
}

template<class T>
void _testLogExpmap(const T& i, const T& ni) {
  CHECK(assert_equal(i, traits<T>::Expmap(i, traits<T>::Logmap(i))));
  CHECK(assert_equal(ni, traits<T>::Expmap(ni, traits<T>::Logmap(ni))));
}

template<class T>
void _testInverse(const T& i, const T& ni) {
  // identity
  CHECK(assert_equal(traits<T>::Identity(T()), traits<T>::Inverse(traits<T>::Identity(T()))));
  // jacobians
  Eigen::MatrixXd expected, actual;
  traits<T>::InverseJacobian(i, actual);
  expected = numericalJacobian11(traits<T>::Inverse, i);
  CHECK(assert_equal(expected, actual, 1e-6));
  traits<T>::InverseJacobian(ni, actual);
  expected = numericalJacobian11(traits<T>::Inverse, ni);
  CHECK(assert_equal(expected, actual, 1e-6));
}

template<class T>
void _testCompose(const T& i, const T& ni) {
  // identity
  CHECK(assert_equal(traits<T>::Compose(traits<T>::Identity(T()), traits<T>::Identity(T())), 
      traits<T>::Identity(T())));
  // jacobians
  Eigen::MatrixXd expected1, expected2, actual1, actual2;
  traits<T>::ComposeJacobians(i, i, actual1, actual2);
  expected1 = numericalJacobian21(traits<T>::Compose, i, i);
  expected2 = numericalJacobian22(traits<T>::Compose, i, i);
  CHECK(assert_equal(expected1, actual1, 1e-6));
  CHECK(assert_equal(expected2, actual2, 1e-6));
  traits<T>::ComposeJacobians(ni, ni, actual1, actual2);
  expected1 = numericalJacobian21(traits<T>::Compose, ni, ni);
  expected2 = numericalJacobian22(traits<T>::Compose, ni, ni);
  CHECK(assert_equal(expected1, actual1, 1e-6));
  CHECK(assert_equal(expected2, actual2, 1e-6));
}


// non-identity sophus values
const Sophus::SO2d nso2 = Sophus::SO2d(2.1);
const Sophus::SE2d nse2 = Sophus::SE2d(1.3, Eigen::Vector2d(4.5, -9.1));
const Sophus::SO3d nso3 = Sophus::SO3d::rotX(1.3)*Sophus::SO3d::rotY(4.5)*
    Sophus::SO3d::rotZ(-9.1);
const Sophus::SE3d nse3 = Sophus::SE3d::rotX(1.3)*Sophus::SE3d::rotY(4.5)*
    Sophus::SE3d::rotZ(-9.1)*Sophus::SE3d::trans(6.1, 8.9, -0.5);
const Sophus::Sim3d nsim3(Sophus::Sim3d::exp(((Eigen::VectorXd(7) 
    << 4.5, -9.1, 1.3, -1.4, 9.8, 3.3, 0.1).finished())));


/* ************************************************************************** */
TEST_CASE("SophusTraits", "[geometry]") {
  _testTraits<Sophus::SO2d>(Sophus::SO2d());
  _testTraits<Sophus::SE2d>(Sophus::SE2d());
  _testTraits<Sophus::SO3d>(Sophus::SO3d());
  _testTraits<Sophus::SE3d>(Sophus::SE3d());
  _testTraits<Sophus::Sim3d>(Sophus::Sim3d());
}

/* ************************************************************************** */
TEST_CASE("SophusDim", "[geometry]") {
  _testDim<Sophus::SO2d>(Sophus::SO2d());
  _testDim<Sophus::SE2d>(Sophus::SE2d());
  _testDim<Sophus::SO3d>(Sophus::SO3d());
  _testDim<Sophus::SE3d>(Sophus::SE3d());
  _testDim<Sophus::Sim3d>(Sophus::Sim3d());
}

/* ************************************************************************** */
TEST_CASE("SophusIdentity", "[geometry]") {
  _testIdentity<Sophus::SO2d>(Sophus::SO2d());
  _testIdentity<Sophus::SE2d>(Sophus::SE2d());
  _testIdentity<Sophus::SO3d>(Sophus::SO3d());
  _testIdentity<Sophus::SE3d>(Sophus::SE3d());
  _testIdentity<Sophus::Sim3d>(Sophus::Sim3d());
}

/* ************************************************************************** */
TEST_CASE("SophusLocalRetract", "[geometry]") {
  _testLocalRetract<Sophus::SO2d>(Sophus::SO2d(), nso2);
  _testLocalRetract<Sophus::SE2d>(Sophus::SE2d(), nse2);
  _testLocalRetract<Sophus::SO3d>(Sophus::SO3d(), nso3);
  _testLocalRetract<Sophus::SE3d>(Sophus::SE3d(), nse3);
  _testLocalRetract<Sophus::Sim3d>(Sophus::Sim3d(), nsim3);
}

/* ************************************************************************** */
TEST_CASE("SophusLogExpmap", "[geometry]") {
  _testLogExpmap<Sophus::SO2d>(Sophus::SO2d(), nso2);
  _testLogExpmap<Sophus::SE2d>(Sophus::SE2d(), nse2);
  _testLogExpmap<Sophus::SO3d>(Sophus::SO3d(), nso3);
  _testLogExpmap<Sophus::SE3d>(Sophus::SE3d(), nse3);
  _testLogExpmap<Sophus::Sim3d>(Sophus::Sim3d(), nsim3);
}

/* ************************************************************************** */
TEST_CASE("SophusInverse", "[geometry]") {
  _testInverse<Sophus::SO2d>(Sophus::SO2d(), nso2);
  _testInverse<Sophus::SE2d>(Sophus::SE2d(), nse2);
  _testInverse<Sophus::SO3d>(Sophus::SO3d(), nso3);
  _testInverse<Sophus::SE3d>(Sophus::SE3d(), nse3);
  _testInverse<Sophus::Sim3d>(Sophus::Sim3d(), nsim3);
}

/* ************************************************************************** */
TEST_CASE("SophusCompose", "[geometry]") {
  _testCompose<Sophus::SO2d>(Sophus::SO2d(), nso2);
  _testCompose<Sophus::SE2d>(Sophus::SE2d(), nse2);
  _testCompose<Sophus::SO3d>(Sophus::SO3d(), nso3);
  _testCompose<Sophus::SE3d>(Sophus::SE3d(), nse3);
  _testCompose<Sophus::Sim3d>(Sophus::Sim3d(), nsim3);
}
