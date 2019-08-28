// test CPU Conjugate Gradient solver

#include "test_common_linear_systems.h"

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/linear/ConjugateGradient.h>

using namespace std;
using namespace minisam;

/* ************************************************************************** */
TEST_CASE("cg", "[linear]") {
  // exmaple systems
  test::ExampleLinearSystems data;

  Eigen::VectorXd x_act;
  LinearSolverStatus status;
  ConjugateGradientSolver cg;

  cg.initialize(data.A1.transpose() * data.A1);
  status = cg.solve(data.A1.transpose() * data.A1, data.A1.transpose() * data.b1, x_act);
  CHECK(assert_equal(data.x1_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  cg.initialize(data.A2.transpose() * data.A2);
  status = cg.solve(data.A2.transpose() * data.A2, data.A2.transpose() * data.b2, x_act);
  CHECK(assert_equal(data.x2_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  cg.initialize(data.A3.transpose() * data.A3);
  status = cg.solve(data.A3.transpose() * data.A3, data.A3.transpose() * data.b3, x_act);
  CHECK(assert_equal(data.x3_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  cg.initialize(data.A4.transpose() * data.A4);
  status = cg.solve(data.A4.transpose() * data.A4, data.A4.transpose() * data.b4, x_act);
  //CHECK(status == LinearSolverStatus::RANK_DEFICIENCY);
}

/* ************************************************************************** */
TEST_CASE("cgls", "[linear]") {
  // exmaple systems
  test::ExampleLinearSystems data;

  Eigen::VectorXd x_act;
  LinearSolverStatus status;
  ConjugateGradientLeastSquareSolver cg;

  cg.initialize(data.A1);
  status = cg.solve(data.A1, data.b1, x_act);
  CHECK(assert_equal(data.x1_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  cg.initialize(data.A2);
  status = cg.solve(data.A2, data.b2, x_act);
  CHECK(assert_equal(data.x2_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  cg.initialize(data.A3);
  status = cg.solve(data.A3, data.b3, x_act);
  CHECK(assert_equal(data.x3_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  cg.initialize(data.A4);
  status = cg.solve(data.A4, data.b4, x_act);
  //CHECK(status == LinearSolverStatus::RANK_DEFICIENCY);
}
