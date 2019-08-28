// test schur complement solver

#include "test_common_linear_systems.h"

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/core/Scalar.h>
#include <minisam/linear/SchurComplementDenseSolver.h>
#include <minisam/linear/DenseCholesky.h>

using namespace std;
using namespace minisam;

/* ************************************************************************** */
TEST_CASE("SchurComplementDenseSolver_no_elimination", "[linear]") {
  // exmaple systems
  test::ExampleLinearSystems data;

  // prepapre a schur complement information without elimination
  VariableOrdering origin_ordering({key('a', 0), key('b', 0)});
  VariablesToEliminate vars_to_eliminate;
  Variables variables;
  variables.add(key('a', 0), 0.0);
  variables.add(key('b', 0), 1.0);

  std::unique_ptr<SchurComplementOrdering> sc_ordering(new SchurComplementOrdering(
      origin_ordering, vars_to_eliminate, variables));

  Eigen::VectorXd x_act;
  LinearSolverStatus status;
  SchurComplementDenseSolver chol(std::unique_ptr<DenseLinearSolver>(new DenseCholeskySolver()));
  chol.setSchurComplementOrdering(std::move(sc_ordering));

  chol.initialize(data.A1.transpose() * data.A1);
  status = chol.solve(data.A1.transpose() * data.A1, data.A1.transpose() * data.b1, x_act);
  CHECK(assert_equal(data.x1_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  chol.initialize(data.A2.transpose() * data.A2);
  status = chol.solve(data.A2.transpose() * data.A2, data.A2.transpose() * data.b2, x_act);
  CHECK(assert_equal(data.x2_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  chol.initialize(data.A3.transpose() * data.A3);
  status = chol.solve(data.A3.transpose() * data.A3, data.A3.transpose() * data.b3, x_act);
  CHECK(assert_equal(data.x3_exp, x_act));
  CHECK(status == LinearSolverStatus::SUCCESS);

  chol.initialize(data.A4.transpose() * data.A4);
  status = chol.solve(data.A4.transpose() * data.A4, data.A4.transpose() * data.b4, x_act);
  // CHECK(status == LinearSolverStatus::RANK_DEFICIENCY);
}

// TODO: add a test with elimination