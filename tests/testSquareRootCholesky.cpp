// test CPU direct R solver

#include "test_common_linear_systems.h"

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/linear/SquareRootCholesky.h>

#include <Eigen/LU> // rank

using namespace minisam;


// exmaple systems
test::ExampleLinearSystems data;
Eigen::SparseMatrix<double> A, H;

/* ************************************************************************** */
TEST_CASE("SquareRootCholesky_prep_static_values", "[linear]") {

  // use random A to generate test, to test whether R'R - ordering.permute(A'A) = 0
  const int A_row = 40, A_col = 20;
  Eigen::MatrixXd A_dense;
  int A_rank;
  do {
    A_dense = Eigen::MatrixXd::Random(A_row, A_col);
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(A_dense);
    A_rank = lu_decomp.rank();
  } while (A_rank < A_col); // make sure full rank

  A = A_dense.sparseView();
  H = A.transpose() * A;
}

/* ************************************************************************** */
TEST_CASE("SquareRootCholesky", "[linear]") {

  Eigen::VectorXd x_act;
  SquareRootSolverStatus status;
  SquareRootSolverCholesky chol(OrderingMethod::NONE);

  Eigen::SparseMatrix<double> R_act(2,2), L_act(2,2), L_exp;

  chol.initialize(data.A1.transpose() * data.A1);
  status = chol.solveR(data.A1.transpose() * data.A1, R_act);
  CHECK(assert_equal(data.R1_exp, R_act));
  CHECK(status == SquareRootSolverStatus::SUCCESS);
  status = chol.solveL(data.A1.transpose() * data.A1, L_act);
  L_exp = data.R1_exp.transpose();
  CHECK(assert_equal(L_exp, L_act));
  CHECK(status == SquareRootSolverStatus::SUCCESS);

  chol.initialize(data.A2.transpose() * data.A2);
  status = chol.solveR(data.A2.transpose() * data.A2, R_act);
  CHECK(assert_equal(data.R2_exp, R_act));
  CHECK(status == SquareRootSolverStatus::SUCCESS);
  status = chol.solveL(data.A2.transpose() * data.A2, L_act);
  L_exp = data.R2_exp.transpose();
  CHECK(assert_equal(L_exp, L_act));
  CHECK(status == SquareRootSolverStatus::SUCCESS);

  chol.initialize(data.A3.transpose() * data.A3);
  status = chol.solveR(data.A3.transpose() * data.A3, R_act);
  CHECK(assert_equal(data.R3_exp, R_act));
  CHECK(status == SquareRootSolverStatus::SUCCESS);
  status = chol.solveL(data.A3.transpose() * data.A3, L_act);
  L_exp = data.R3_exp.transpose();
  CHECK(assert_equal(L_exp, L_act));
  CHECK(status == SquareRootSolverStatus::SUCCESS);

  chol.initialize(data.A4.transpose() * data.A4);
  status = chol.solveR(data.A4.transpose() * data.A4, R_act);
  CHECK(status == SquareRootSolverStatus::RANK_DEFICIENCY);
  status = chol.solveL(data.A4.transpose() * data.A4, R_act);
  CHECK(status == SquareRootSolverStatus::RANK_DEFICIENCY);
}

/* ************************************************************************** */
TEST_CASE("SquareRootCholesky_random", "[linear]") {

  Eigen::SparseMatrix<double> R, L, PHPt_act, PHPt_exp;

  // AMD ordering
  SquareRootSolverStatus status;
  SquareRootSolverCholesky chol;

  status = chol.initialize(H);
  CHECK(status == SquareRootSolverStatus::SUCCESS);
  chol.ordering()->permuteSystemFull(H, PHPt_exp);

  // R'R = ordering.permute(A'A)
  status = chol.solveR(H, R);
  CHECK(status == SquareRootSolverStatus::SUCCESS);
  PHPt_act = R.transpose() * R;
  CHECK(assert_equal(PHPt_exp, PHPt_act));

  // LL' = ordering.permute(A'A)
  status = chol.solveL(H, L);
  CHECK(status == SquareRootSolverStatus::SUCCESS);
  PHPt_act = L * L.transpose();
  CHECK(assert_equal(PHPt_exp, PHPt_act));

  // no ordering
  SquareRootSolverCholesky choln(OrderingMethod::NONE);

  status = choln.initialize(H);
  CHECK(status == SquareRootSolverStatus::SUCCESS);
  
  // R'R = A'A
  status = choln.solveR(H, R);
  CHECK(status == SquareRootSolverStatus::SUCCESS);
  PHPt_act = R.transpose() * R;
  CHECK(assert_equal(H, PHPt_act));

  // LL' = A'A
  status = choln.solveL(H, L);
  CHECK(status == SquareRootSolverStatus::SUCCESS);
  PHPt_act = L * L.transpose();
  CHECK(assert_equal(H, PHPt_act));
}
