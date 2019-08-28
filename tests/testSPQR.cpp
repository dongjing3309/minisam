// test CPU direct solver

#include "test_common_linear_systems.h"

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/linear/spqr/SPQR.h>

using namespace std;
using namespace minisam;

/* ************************************************************************** */
TEST_CASE("qr", "[linear]") {
  // exmaple systems
  test::ExampleLinearSystems data;

  Eigen::VectorXd x_act;
  LinearSolverStatus status;
  QRSolver qr;

  qr.initialize(data.A1);
  status = qr.solve(data.A1, data.b1, x_act);
  CHECK(assert_equal(data.x1_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  qr.initialize(data.A2);
  status = qr.solve(data.A2, data.b2, x_act);
  CHECK(assert_equal(data.x2_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  // does not work for SPQR in debug mode
  // qr.initialize(data.A3);
  // status = qr.solve(data.A3, data.b3, x_act);
  // CHECK(assert_equal(data.x3_exp, x_act));
  // CHECK(status == LinearSolverStatus::SUCCESS);

  qr.initialize(data.A4);
  status = qr.solve(data.A4, data.b4, x_act);
  // CHECK(status == LinearSolverStatus::RANK_DEFICIENCY);
}

