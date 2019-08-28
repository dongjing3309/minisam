// test CPU direct solver

#include "test_common_linear_systems.h"

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/linear/cuda/CUDASolver.h>
#include <minisam/config.h>

#include <Eigen/Dense>


using namespace std;
using namespace minisam;

/* ************************************************************************** */
TEST_CASE("cudacholesky", "[linear]") {
  // exmaple systems
  test::ExampleLinearSystems data;

  Eigen::VectorXd x_act;
  LinearSolverStatus status;
  CUDACholeskySolver chol;

  chol.initialize(data.A1.transpose() * data.A1);
  status = chol.solve(data.A1.transpose() * data.A1, data.A1.transpose() * data.b1, x_act);
  CHECK(status == LinearSolverStatus::SUCCESS);
  CHECK(assert_equal(data.x1_exp, x_act));

  chol.initialize(data.A2.transpose() * data.A2);
  status = chol.solve(data.A2.transpose() * data.A2, data.A2.transpose() * data.b2, x_act);
  CHECK(status == LinearSolverStatus::SUCCESS);
  CHECK(assert_equal(data.x2_exp, x_act));

  chol.initialize(data.A3.transpose() * data.A3);
  status = chol.solve(data.A3.transpose() * data.A3, data.A3.transpose() * data.b3, x_act);
  CHECK(status == LinearSolverStatus::SUCCESS);
  CHECK(assert_equal(data.x3_exp, x_act));

  chol.initialize(data.A4.transpose() * data.A4);
  status = chol.solve(data.A4.transpose() * data.A4, data.A4.transpose() * data.b4, x_act);
  CHECK(status == LinearSolverStatus::RANK_DEFICIENCY);
}
