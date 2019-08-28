// test multi-view geometry projection functions

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>
#include <minisam/nonlinear/numericalJacobian.h>

#include <minisam/geometry/projection.h>
#include <minisam/geometry/Sophus.h>
#include <minisam/geometry/CalibK.h>
#include <minisam/geometry/CalibKD.h>
#include <minisam/geometry/CalibBundler.h>


using namespace std;
using namespace Eigen;
using namespace minisam;


// convert a point from homogeneous coordinate to image frame
Vector2d homogeneous(const Vector3d& ph) {
  return Vector2d(ph(0) / ph(2), ph(1) / ph(2));
}

// pose
Sophus::SE3d pose1;
Sophus::SE3d pose2 = Sophus::SE3d::rotZ(M_PI / 2.0);
Sophus::SE3d pose3(Sophus::SO3d::exp(Vector3d(0.22, 0.17, -0.143)), Vector3d(3.4, -2.9, -8.8));

// point
Vector3d pw1(0, 0, 1);
Vector3d pw2(5, -3, 10);

Vector3d ps11 = Vector3d(0, 0, 1);
Vector3d ps21 = Vector3d(0, 0, 1);
Vector3d ps12 = Vector3d(5, -3, 10);
Vector3d ps22 = Vector3d(-3, -5, 10);


/* ************************************************************************** */
TEST_CASE("transform2sensor", "[geometry]") {

  CHECK(assert_equal(ps11, transform2sensor(pose1, pw1)));
  CHECK(assert_equal(ps21, transform2sensor(pose2, pw1)));
  CHECK(assert_equal(ps12, transform2sensor(pose1, pw2)));
  CHECK(assert_equal(ps22, transform2sensor(pose2, pw2)));

  // jacobians
  Eigen::Matrix<double, 3, 6> J_pose_actual;
  Eigen::Matrix<double, 3, 3> J_pw_actual;

  transform2sensorJacobians(pose1, pw1, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2sensor, pose1, pw1), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2sensor, pose1, pw1), J_pw_actual));
  transform2sensorJacobians(pose2, pw1, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2sensor, pose2, pw1), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2sensor, pose2, pw1), J_pw_actual));
  transform2sensorJacobians(pose3, pw1, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2sensor, pose3, pw1), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2sensor, pose3, pw1), J_pw_actual));

  transform2sensorJacobians(pose1, pw2, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2sensor, pose1, pw2), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2sensor, pose1, pw2), J_pw_actual));
  transform2sensorJacobians(pose2, pw2, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2sensor, pose2, pw2), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2sensor, pose2, pw2), J_pw_actual));
  transform2sensorJacobians(pose3, pw2, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2sensor, pose3, pw2), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2sensor, pose3, pw2), J_pw_actual));
}

/* ************************************************************************** */
TEST_CASE("transform2world", "[geometry]") {

  CHECK(assert_equal(pw1, transform2world(pose1, ps11)));
  CHECK(assert_equal(pw1, transform2world(pose2, ps21)));
  CHECK(assert_equal(pw2, transform2world(pose1, ps12)));
  CHECK(assert_equal(pw2, transform2world(pose2, ps22)));

  // jacobians
  Vector3d ps_rnd = Vector3d(1.3, -4.1, 5.6);
  Eigen::Matrix<double, 3, 6> J_pose_actual;
  Eigen::Matrix<double, 3, 3> J_ps_actual;

  transform2worldJacobians(pose1, ps11, J_pose_actual, J_ps_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2world, pose1, ps11), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2world, pose1, ps11), J_ps_actual));
  transform2worldJacobians(pose2, ps21, J_pose_actual, J_ps_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2world, pose2, ps21), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2world, pose2, ps21), J_ps_actual));

  transform2worldJacobians(pose1, ps12, J_pose_actual, J_ps_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2world, pose1, ps12), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2world, pose1, ps12), J_ps_actual));
  transform2worldJacobians(pose2, ps22, J_pose_actual, J_ps_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2world, pose2, ps22), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2world, pose2, ps22), J_ps_actual));

  transform2worldJacobians(pose3, ps_rnd, J_pose_actual, J_ps_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2world, pose3, ps_rnd), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2world, pose3, ps_rnd), J_ps_actual));
}

/* ************************************************************************** */
TEST_CASE("transform2image", "[geometry]") {

  CHECK(assert_equal(homogeneous(ps11), transform2image(pose1, pw1)));
  CHECK(assert_equal(homogeneous(ps21), transform2image(pose2, pw1)));
  CHECK(assert_equal(homogeneous(ps12), transform2image(pose1, pw2)));
  CHECK(assert_equal(homogeneous(ps22), transform2image(pose2, pw2)));

  // jacobians
  Eigen::Matrix<double, 2, 6> J_pose_actual;
  Eigen::Matrix<double, 2, 3> J_pw_actual;

  transform2imageJacobians(pose1, pw1, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2image, pose1, pw1), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2image, pose1, pw1), J_pw_actual));
  transform2imageJacobians(pose2, pw1, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2image, pose2, pw1), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2image, pose2, pw1), J_pw_actual));
  transform2imageJacobians(pose3, pw1, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2image, pose3, pw1), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2image, pose3, pw1), J_pw_actual));

  transform2imageJacobians(pose1, pw2, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2image, pose1, pw2), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2image, pose1, pw2), J_pw_actual));
  transform2imageJacobians(pose2, pw2, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2image, pose2, pw2), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2image, pose2, pw2), J_pw_actual));
  transform2imageJacobians(pose3, pw2, J_pose_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian21(transform2image, pose3, pw2), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian22(transform2image, pose3, pw2), J_pw_actual));
}


// vars for projections

// calibration
CalibK ucal1(1, 1, 0, 0);           // identity
CalibK ucal2(100, 100, 300, 200);   // non-identity

CalibKD dcal1(100, 100, 300, 200, -0.01, 0.001, 0, 0);    // non-identity
CalibKD dcal2(142.5, 129.6, 323.7, 209.8, -0.0123, 0.00143, 0.00342, 0.000983);    // random non-identity

CalibBundler bcal1(100, -0.01, 0.001);    // non-identity
CalibBundler bcal2(142.5, -0.0123, 0.00983);    // random non-identity

// points
Vector3d pw3(0, 0, 10);
Vector3d pw4(24, 13, 10);

/* ************************************************************************** */
TEST_CASE("project_CalibK", "[geometry]") {

  CHECK(assert_equal(Vector2d(0, 0), project(pose1, ucal1, pw3)));
  CHECK(assert_equal(Vector2d(300, 200), project(pose1, ucal2, pw3)));

  CHECK(assert_equal(Vector2d(2.4, 1.3), project(pose1, ucal1, pw4)));
  CHECK(assert_equal(Vector2d(540, 330), project(pose1, ucal2, pw4)));

  CHECK(assert_equal(Vector2d(1.3, -2.4), project(pose2, ucal1, pw4)));
  CHECK(assert_equal(Vector2d(430, -40), project(pose2, ucal2, pw4)));

  // jacobians
  Eigen::Matrix<double, 2, 6> J_pose_actual;
  Eigen::Matrix<double, 2, 3> J_pw_actual;
  Eigen::Matrix<double, 2, 4> J_K_actual;

  projectJacobians(pose1, ucal2, pw3, J_pose_actual, J_K_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(project<CalibK>, pose1, ucal2, pw3), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(project<CalibK>, pose1, ucal2, pw3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(project<CalibK>, pose1, ucal2, pw3), J_pw_actual));
  projectJacobians(pose1, ucal2, pw4, J_pose_actual, J_K_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(project<CalibK>, pose1, ucal2, pw4), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(project<CalibK>, pose1, ucal2, pw4), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(project<CalibK>, pose1, ucal2, pw4), J_pw_actual));

  projectJacobians(pose3, ucal2, pw3, J_pose_actual, J_K_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(project<CalibK>, pose3, ucal2, pw3), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(project<CalibK>, pose3, ucal2, pw3), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(project<CalibK>, pose3, ucal2, pw3), J_pw_actual));
  projectJacobians(pose3, ucal2, pw4, J_pose_actual, J_K_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(project<CalibK>, pose3, ucal2, pw4), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(project<CalibK>, pose3, ucal2, pw4), J_K_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(project<CalibK>, pose3, ucal2, pw4), J_pw_actual));
}

/* ************************************************************************** */
TEST_CASE("project_CalibKD", "[geometry]") {

  CHECK(assert_equal(Vector2d(300, 200), project(pose1, dcal1, pw3)));
  CHECK(assert_equal(Vector2d(535.4406, 327.530325), project(pose1, dcal1, pw4)));
  CHECK(assert_equal(Vector2d(427.530325, -35.4406), project(pose2, dcal1, pw4)));

  // jacobians
  Eigen::Matrix<double, 2, 6> J_pose_actual;
  Eigen::Matrix<double, 2, 3> J_pw_actual;
  Eigen::Matrix<double, 2, 8> J_KD_actual;

  projectJacobians(pose1, dcal2, pw3, J_pose_actual, J_KD_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(project<CalibKD>, pose1, dcal2, pw3), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(project<CalibKD>, pose1, dcal2, pw3), J_KD_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(project<CalibKD>, pose1, dcal2, pw3), J_pw_actual));
  projectJacobians(pose1, dcal2, pw4, J_pose_actual, J_KD_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(project<CalibKD>, pose1, dcal2, pw4), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(project<CalibKD>, pose1, dcal2, pw4), J_KD_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(project<CalibKD>, pose1, dcal2, pw4), J_pw_actual));

  projectJacobians(pose3, dcal2, pw3, J_pose_actual, J_KD_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(project<CalibKD>, pose3, dcal2, pw3), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(project<CalibKD>, pose3, dcal2, pw3), J_KD_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(project<CalibKD>, pose3, dcal2, pw3), J_pw_actual));
  projectJacobians(pose3, dcal2, pw4, J_pose_actual, J_KD_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(project<CalibKD>, pose3, dcal2, pw4), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(project<CalibKD>, pose3, dcal2, pw4), J_KD_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(project<CalibKD>, pose3, dcal2, pw4), J_pw_actual));
}


/* ************************************************************************** */
TEST_CASE("project_CalibBundler", "[geometry]") {

  CHECK(assert_equal(Vector2d(0, 0), projectBundler(pose1, bcal1, pw3)));
  CHECK(assert_equal(Vector2d(-235.4406, -127.530325), projectBundler(pose1, bcal1, pw4)));
  CHECK(assert_equal(Vector2d(127.530325, -235.4406), projectBundler(pose2, bcal1, pw4)));

  // jacobians
  Eigen::Matrix<double, 2, 6> J_pose_actual;
  Eigen::Matrix<double, 2, 3> J_pw_actual;
  Eigen::Matrix<double, 2, 3> J_c_actual;

  projectBundlerJacobians(pose1, bcal2, pw3, J_pose_actual, J_c_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(projectBundler, pose1, bcal2, pw3), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(projectBundler, pose1, bcal2, pw3), J_c_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(projectBundler, pose1, bcal2, pw3), J_pw_actual));
  projectBundlerJacobians(pose1, bcal2, pw4, J_pose_actual, J_c_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(projectBundler, pose1, bcal2, pw4), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(projectBundler, pose1, bcal2, pw4), J_c_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(projectBundler, pose1, bcal2, pw4), J_pw_actual));

  projectBundlerJacobians(pose3, bcal2, pw3, J_pose_actual, J_c_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(projectBundler, pose3, bcal2, pw3), J_pose_actual, 1e-3));  // 1e-6 fails on raspberry pi
  CHECK(assert_equal_matrix(numericalJacobian32(projectBundler, pose3, bcal2, pw3), J_c_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(projectBundler, pose3, bcal2, pw3), J_pw_actual));
  projectBundlerJacobians(pose3, bcal2, pw4, J_pose_actual, J_c_actual, J_pw_actual);
  CHECK(assert_equal_matrix(numericalJacobian31(projectBundler, pose3, bcal2, pw4), J_pose_actual));
  CHECK(assert_equal_matrix(numericalJacobian32(projectBundler, pose3, bcal2, pw4), J_c_actual));
  CHECK(assert_equal_matrix(numericalJacobian33(projectBundler, pose3, bcal2, pw4), J_pw_actual));
}
