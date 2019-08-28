// test camera intrinsic calibration

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>
#include <minisam/nonlinear/numericalJacobian.h>

#include <minisam/geometry/CalibK.h>
#include <minisam/geometry/CalibKD.h>
#include <minisam/geometry/CalibBundler.h>

#include <Eigen/Dense>  // inverse

using namespace std;
using namespace Eigen;
using namespace minisam;


// wrapper for numerical jacobians
template<class CALIBRATION>
Vector2d wrapper_project(const CALIBRATION& K, const Vector2d& p) {
  return K.project(p);
}
template<class CALIBRATION>
Vector2d wrapper_unproject(const CALIBRATION& K, const Vector2d& p) {
  return K.unproject(p);
}

// static test vars
CalibK ucal1(1, 1, 0, 0);           // identity
CalibK ucal2(100, 100, 300, 200);   // non-identity
CalibK ucal3(37.4, 28.9, 107.1, -223.3);  // random non-identity

CalibKD dcal1(1, 1, 0, 0, 0, 0, 0 ,0);                // identity
CalibKD dcal2(100, 100, 300, 200, 0.1, 0.01, 0, 0);   // non-identity
CalibKD dcal3(37.4, 28.9, 107.1, -223.3, 0.089, 0.063, -0.03, 0.08);  // random non-identity
CalibKD dcal4(100, 100, 300, 200, -0.01, 0.001, 0, 0);   // non-identity

CalibBundler bcal1(1, 0 ,0);              // identity
CalibBundler bcal2(100, -0.01, 0.001);    // non-identity
CalibBundler bcal3(37.4, 0.089, 0.063);   // random non-identity


/* ************************************************************************** */
TEST_CASE("CalibKTraits", "[geometry]") {
  CHECK(has_traits<CalibK>::value);
  CHECK(is_manifold<CalibK>::value);
  CHECK_FALSE(is_lie_group<CalibK>::value);
}

/* ************************************************************************** */
TEST_CASE("CalibKConstructor", "[geometry]") {
  CHECK(assert_equal(CalibK((Vector4d() << 1, 2, 3, 4).finished()), CalibK(1, 2, 3, 4)));
}

/* ************************************************************************** */
TEST_CASE("CalibKMatrix", "[geometry]") {

  Matrix3d K2, K3;
  K2 << 100, 0, 300, 0, 100, 200, 0, 0, 1;
  K3 << 37.4, 0, 107.1, 0, 28.9, -223.3, 0, 0, 1;

  CHECK(assert_equal_matrix(Matrix3d::Identity(), ucal1.matrix()));
  CHECK(assert_equal_matrix(Matrix3d::Identity(), ucal1.inverse_matrix()));

  CHECK(assert_equal_matrix(K2, ucal2.matrix()));
  CHECK(assert_equal_matrix(K2.inverse(), ucal2.inverse_matrix()));

  CHECK(assert_equal_matrix(K3, ucal3.matrix()));
  CHECK(assert_equal_matrix(K3.inverse(), ucal3.inverse_matrix()));
}

/* ************************************************************************** */
TEST_CASE("CalibKc2i", "[geometry]") {

  Vector2d pc1 = Vector2d(0, 0);
  Vector2d pc3 = Vector2d(23.4, 16.2);

  // point
  CHECK(assert_equal(pc1, ucal1.project(pc1)));
  CHECK(assert_equal(pc3, ucal1.project(pc3)));

  CHECK(assert_equal(Vector2d(300, 200), ucal2.project(pc1)));
  CHECK(assert_equal(Vector2d(2640, 1820), ucal2.project(pc3)));

  // jacobians
  Eigen::Matrix<double, 2, 4> J_K_actual;
  Eigen::Matrix<double, 2, 2> J_p_actual;

  ucal1.projectJacobians(pc1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibK>, ucal1, pc1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibK>, ucal1, pc1), J_p_actual));
  ucal1.projectJacobians(pc3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibK>, ucal1, pc3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibK>, ucal1, pc3), J_p_actual));

  ucal2.projectJacobians(pc1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibK>, ucal2, pc1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibK>, ucal2, pc1), J_p_actual));
  ucal2.projectJacobians(pc3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibK>, ucal2, pc3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibK>, ucal2, pc3), J_p_actual));

  ucal3.projectJacobians(pc1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibK>, ucal3, pc1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibK>, ucal3, pc1), J_p_actual));
  ucal3.projectJacobians(pc3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibK>, ucal3, pc3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibK>, ucal3, pc3), J_p_actual));
}

/* ************************************************************************** */
TEST_CASE("CalibKi2c", "[geometry]") {

  Vector2d pi1 = Vector2d(300, 200);
  Vector2d pi3 = Vector2d(2640, 1820);

  // point
  CHECK(assert_equal(pi1, ucal1.unproject(pi1)));
  CHECK(assert_equal(pi3, ucal1.unproject(pi3)));

  CHECK(assert_equal(Vector2d(0, 0), ucal2.unproject(pi1)));
  CHECK(assert_equal(Vector2d(23.4, 16.2), ucal2.unproject(pi3)));

  // jacobians
  Eigen::Matrix<double, 2, 4> J_K_actual;
  Eigen::Matrix<double, 2, 2> J_p_actual;

  ucal1.unprojectJacobians(pi1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_unproject<CalibK>, ucal1, pi1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_unproject<CalibK>, ucal1, pi1), J_p_actual));
  ucal1.unprojectJacobians(pi3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_unproject<CalibK>, ucal1, pi3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_unproject<CalibK>, ucal1, pi3), J_p_actual));

  ucal2.unprojectJacobians(pi1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_unproject<CalibK>, ucal2, pi1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_unproject<CalibK>, ucal2, pi1), J_p_actual));
  ucal2.unprojectJacobians(pi3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_unproject<CalibK>, ucal2, pi3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_unproject<CalibK>, ucal2, pi3), J_p_actual));

  ucal3.unprojectJacobians(pi1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_unproject<CalibK>, ucal3, pi1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_unproject<CalibK>, ucal3, pi1), J_p_actual));
  ucal3.unprojectJacobians(pi3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_unproject<CalibK>, ucal3, pi3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_unproject<CalibK>, ucal3, pi3), J_p_actual));
}

/* ************************************************************************** */
TEST_CASE("CalibKDTraits", "[geometry]") {
  CHECK(has_traits<CalibKD>::value);
  CHECK(is_manifold<CalibKD>::value);
  CHECK_FALSE(is_lie_group<CalibKD>::value);
}

/* ************************************************************************** */
TEST_CASE("CalibKDConstructor", "[geometry]") {
  CHECK(assert_equal(CalibKD((VectorXd(8) << 1, 2, 3, 4, 5, 6, 7, 8).finished()), 
      CalibKD(1, 2, 3, 4, 5, 6, 7, 8)));
}

/* ************************************************************************** */
TEST_CASE("CalibKDc2i", "[geometry]") {

  Vector2d pc1 = Vector2d(0, 0);
  Vector2d pc3 = Vector2d(2.4, 1.3);

  // point
  CHECK(assert_equal(pc1, dcal1.project(pc1)));
  CHECK(assert_equal(pc3, dcal1.project(pc3)));

  CHECK(assert_equal(Vector2d(300, 200), dcal2.project(pc1)));
  CHECK(assert_equal(Vector2d(852.006, 499.00325), dcal2.project(pc3)));

  CHECK(assert_equal(Vector2d(300, 200), dcal4.project(pc1)));
  CHECK(assert_equal(Vector2d(535.4406, 327.530325), dcal4.project(pc3)));

  // jacobians
  Eigen::Matrix<double, 2, 8> J_K_actual;
  Eigen::Matrix<double, 2, 2> J_p_actual;

  dcal1.projectJacobians(pc1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibKD>, dcal1, pc1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibKD>, dcal1, pc1), J_p_actual));
  dcal1.projectJacobians(pc3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibKD>, dcal1, pc3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibKD>, dcal1, pc3), J_p_actual));

  dcal2.projectJacobians(pc1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibKD>, dcal2, pc1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibKD>, dcal2, pc1), J_p_actual));
  dcal2.projectJacobians(pc3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibKD>, dcal2, pc3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibKD>, dcal2, pc3), J_p_actual));
  
  dcal3.projectJacobians(pc1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibKD>, dcal3, pc1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibKD>, dcal3, pc1), J_p_actual));
  dcal3.projectJacobians(pc3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibKD>, dcal3, pc3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibKD>, dcal3, pc3), J_p_actual));
  
  dcal4.projectJacobians(pc1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibKD>, dcal4, pc1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibKD>, dcal4, pc1), J_p_actual));
  dcal4.projectJacobians(pc3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibKD>, dcal4, pc3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibKD>, dcal4, pc3), J_p_actual));
}

/* ************************************************************************** */
TEST_CASE("CalibKDi2c", "[geometry]") {

  Vector2d pi1 = Vector2d(300, 200);
  Vector2d pi3 = Vector2d(535.4406, 327.530325);

  CHECK(assert_equal(pi1, dcal1.unproject(pi1), 1e-6));
  CHECK(assert_equal(pi3, dcal1.unproject(pi3), 1e-6));

  CHECK(assert_equal(Vector2d(0, 0), dcal4.unproject(pi1), 1e-6));
  CHECK(assert_equal(Vector2d(2.4, 1.3), dcal4.unproject(pi3), 1e-6));
}


/* ************************************************************************** */
TEST_CASE("CalibBundlerTraits", "[geometry]") {
  CHECK(has_traits<CalibBundler>::value);
  CHECK(is_manifold<CalibBundler>::value);
  CHECK_FALSE(is_lie_group<CalibBundler>::value);
}

/* ************************************************************************** */
TEST_CASE("CalibBundlerConstructor", "[geometry]") {
  CHECK(assert_equal(CalibBundler((VectorXd(3) << 1, 2, 3).finished()), 
      CalibBundler(1, 2, 3)));
}

/* ************************************************************************** */
TEST_CASE("CalibBundlerc2i", "[geometry]") {

  Vector2d pc1 = Vector2d(0, 0);
  Vector2d pc3 = Vector2d(2.4, 1.3);

  // point
  CHECK(assert_equal(pc1, bcal1.project(pc1)));
  CHECK(assert_equal(pc3, bcal1.project(pc3)));

  CHECK(assert_equal(Vector2d(0, 0), bcal2.project(pc1)));
  CHECK(assert_equal(Vector2d(235.4406, 127.530325), bcal2.project(pc3)));

  // jacobians
  Eigen::Matrix<double, 2, 3> J_K_actual;
  Eigen::Matrix<double, 2, 2> J_p_actual;

  bcal1.projectJacobians(pc1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibBundler>, bcal1, pc1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibBundler>, bcal1, pc1), J_p_actual));
  bcal1.projectJacobians(pc3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibBundler>, bcal1, pc3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibBundler>, bcal1, pc3), J_p_actual));

  bcal2.projectJacobians(pc1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibBundler>, bcal2, pc1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibBundler>, bcal2, pc1), J_p_actual));
  bcal2.projectJacobians(pc3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibBundler>, bcal2, pc3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibBundler>, bcal2, pc3), J_p_actual));
  
  bcal3.projectJacobians(pc1, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibBundler>, bcal3, pc1), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibBundler>, bcal3, pc1), J_p_actual));
  bcal3.projectJacobians(pc3, J_K_actual, J_p_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(wrapper_project<CalibBundler>, bcal3, pc3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(wrapper_project<CalibBundler>, bcal3, pc3), J_p_actual));
}

/* ************************************************************************** */
TEST_CASE("CalibBundleri2c", "[geometry]") {

  Vector2d pi1 = Vector2d(0, 0);
  Vector2d pi3 = Vector2d(235.4406, 127.530325);

  CHECK(assert_equal(pi1, bcal1.unproject(pi1), 1e-6));
  CHECK(assert_equal(pi3, bcal1.unproject(pi3), 1e-6));

  CHECK(assert_equal(Vector2d(0, 0), bcal2.unproject(pi1), 1e-6));
  CHECK(assert_equal(Vector2d(2.4, 1.3), bcal2.unproject(pi3), 1e-6));
}
