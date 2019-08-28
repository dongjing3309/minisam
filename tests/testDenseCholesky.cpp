// test CPU direct solver

#include "test_common_linear_systems.h"

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/linear/DenseCholesky.h>

using namespace std;
using namespace minisam;

/* ************************************************************************** */
TEST_CASE("cholesky", "[linear]") {
  // exmaple systems
  test::ExampleLinearSystems data;

  Eigen::VectorXd x_act;
  LinearSolverStatus status;
  DenseCholeskySolver chol;

  // convert to dense
  const Eigen::MatrixXd A1(data.A1);
  const Eigen::MatrixXd A2(data.A2);
  const Eigen::MatrixXd A3(data.A3);
  const Eigen::MatrixXd A4(data.A4);

  status = chol.solve(A1.transpose() * A1, A1.transpose() * data.b1, x_act);
  CHECK(assert_equal(data.x1_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  status = chol.solve(A2.transpose() * A2, A2.transpose() * data.b2, x_act);
  CHECK(assert_equal(data.x2_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  status = chol.solve(A3.transpose() * A3, A3.transpose() * data.b3, x_act);
  CHECK(assert_equal(data.x3_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  status = chol.solve(A4.transpose() * A4, A4.transpose() * data.b4, x_act);
  // CHECK(status == LinearSolverStatus::RANK_DEFICIENCY);
}
