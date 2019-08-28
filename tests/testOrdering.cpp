// test natural ordering

#include "test_common_linear_systems.h"

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/linear/Ordering.h>
#include <minisam/linear/AMDOrdering.h>

#include <Eigen/LU> // rank
#include <iostream>

using namespace minisam;


// exmaple systems
test::ExampleLinearSystems data;

// assert_equal for MatrixXi
bool assert_equal_MatrixXi(const Eigen::MatrixXi& expected, const Eigen::MatrixXi& actual) {
  // check dim
  if (expected.rows() != actual.rows() || expected.cols() != actual.cols()) {
    std::cout << "Not equal:" << std::endl 
        << "expected dimension: (" << expected.rows() << ", " << expected.cols() << ")" << std::endl 
        << "actual dimension: (" << actual.rows() << ", " << actual.cols() << ")" << std::endl;
    return false;
  }
  // check values
  for (int i = 0; i < expected.rows(); i++) {
    for (int j = 0; j < expected.cols(); j++) {
      if (expected(i,j) != actual(i,j)) {
        std::cout << "Not equal:" << std::endl << "expected: " << expected 
            << std::endl << "actual: " << actual << std::endl;
        return false;
      }
    }
  }
  return true;
}

/* ************************************************************************** */
TEST_CASE("OrderingBase", "[linear]") {

  Eigen::SparseMatrix<double> H, Hord_exp, Hord_act(2, 2), Hord_tmp;
  Eigen::VectorXd Atbord_exp, Atbord_act, x_act;

  // sys2: identity ordering 
  H = data.A2.transpose() * data.A2;
  Hord_exp = data.A2.transpose() * data.A2;
  Atbord_exp = data.A2.transpose() * data.b2;

  Ordering ordering1(Ordering::PermMatrix((Eigen::VectorXi(2) << 0, 1).finished()));

  ordering1.permuteSystemSelfAdjoint<Eigen::Lower>(H, Hord_act);
  CHECK(assert_equal<Eigen::SparseMatrix<double>>(Hord_exp.triangularView<Eigen::Lower>(), Hord_act));

  ordering1.permuteSystemSelfAdjoint<Eigen::Upper>(H, Hord_act);
  CHECK(assert_equal<Eigen::SparseMatrix<double>>(Hord_exp.triangularView<Eigen::Upper>(), Hord_act));

  ordering1.permuteSystemFull(H, Hord_act);
  CHECK(assert_equal(Hord_exp, Hord_act));

  ordering1.permuteRhs(data.A2.transpose() * data.b2, Atbord_act);
  CHECK(assert_equal(Atbord_exp, Atbord_act));

  ordering1.permuteBackSolution(Eigen::MatrixXd(Hord_act).inverse() * Atbord_act, x_act);
  CHECK(assert_equal(data.x2_exp, x_act));

  // sys2: reverse ordering 
  Hord_exp.setZero();
  Hord_exp.insert(0,0) = 78.0100;
  Hord_exp.insert(0,1) = -0.0400;
  Hord_exp.insert(1,0) = -0.0400;
  Hord_exp.insert(1,1) = 13.8500;
  Hord_exp.makeCompressed();
  Atbord_exp << -17.5500, 20.3900;

  Ordering ordering2(Ordering::PermMatrix((Eigen::VectorXi(2) << 1, 0).finished()));

  // FIXME: only dense assert equal works
  ordering2.permuteSystemSelfAdjoint<Eigen::Lower>(H, Hord_act);
  CHECK(assert_equal<Eigen::MatrixXd>(Hord_exp.triangularView<Eigen::Lower>(), Hord_act));

  ordering2.permuteSystemSelfAdjoint<Eigen::Upper>(H, Hord_act);
  CHECK(assert_equal<Eigen::MatrixXd>(Hord_exp.triangularView<Eigen::Upper>(), Hord_act));

  ordering2.permuteSystemFull(H, Hord_act);
  CHECK(assert_equal(Hord_exp, Hord_act));

  ordering2.permuteRhs(data.A2.transpose() * data.b2, Atbord_act);
  CHECK(assert_equal(Atbord_exp, Atbord_act));

  ordering2.permuteBackSolution(Eigen::MatrixXd(Hord_act).inverse() * Atbord_act, x_act);
  CHECK(assert_equal(data.x2_exp, x_act));
}

/* ************************************************************************** */
TEST_CASE("AMDOrdering", "[linear]") {

  Eigen::SparseMatrix<double> H, Hord_exp, Hord_act;
  Eigen::VectorXd Atbord_exp, Atbord_act, x_act;

  // sys1: identity
  H = data.A1.transpose() * data.A1;
  Hord_act = Eigen::SparseMatrix<double>(H.rows(), H.cols());
  Hord_exp = data.A1.transpose() * data.A1;
  Atbord_exp = data.A1.transpose() * data.b1;

  AMDOrdering ordering1(H);

  ordering1.permuteSystemFull(H, Hord_act);
  CHECK(assert_equal(Hord_exp, Hord_act));

  ordering1.permuteRhs(data.A1.transpose() * data.b1, Atbord_act);
  CHECK(assert_equal(Atbord_exp, Atbord_act));

  ordering1.permuteBackSolution(Eigen::MatrixXd(Hord_act).inverse() * Atbord_act, x_act);
  CHECK(assert_equal(data.x1_exp, x_act));

  // sys1: well-cond
  H = data.A2.transpose() * data.A2;
  Hord_act = Eigen::SparseMatrix<double>(H.rows(), H.cols());
  Hord_exp = data.A2.transpose() * data.A2;
  Atbord_exp = data.A2.transpose() * data.b2;

  AMDOrdering ordering2(H);

  ordering2.permuteSystemFull(H, Hord_act);
  CHECK(assert_equal(Hord_exp, Hord_act));

  ordering2.permuteRhs(data.A2.transpose() * data.b2, Atbord_act);
  CHECK(assert_equal(Atbord_exp, Atbord_act));

  ordering2.permuteBackSolution(Eigen::MatrixXd(Hord_act).inverse() * Atbord_act, x_act);
  CHECK(assert_equal(data.x2_exp, x_act));
}

/* ************************************************************************** */
TEST_CASE("OrderingGivenOrdering", "[linear]") {
  // given indices
  int Hsize = 10;
  Eigen::MatrixXd H = Eigen::MatrixXd::Random(Hsize, Hsize);
  // make H symmetric
  H = 0.5 * (H + H.transpose()).eval();
  Eigen::VectorXd x = Eigen::VectorXd::Random(Hsize);
  Eigen::VectorXi indx(Hsize), indx_reverse(Hsize);
  indx << 4, 7, 1, 3, 0, 9, 5, 8, 6, 2;
  indx_reverse << 4, 2, 9, 3, 0, 6, 8, 1, 7, 5;

  Eigen::MatrixXd H_perm_exp(Hsize, Hsize);
  Eigen::SparseMatrix<double> H_perm_act;
  Eigen::VectorXd x_perm_act, x_perm_exp(Hsize);
  for (int i = 0; i < Hsize; i++)
    for (int j = 0; j < Hsize; j++)
      H_perm_exp(indx(i), indx(j)) = H(i, j);
  for (int i = 0; i < Hsize; i++)
    x_perm_exp(indx(i)) = x(i);
  
  // ordering
  Ordering ordering(indx);
  CHECK(assert_equal_MatrixXi(indx, ordering.indices()));
  CHECK(assert_equal_MatrixXi(indx_reverse, ordering.reversedIndices()));

  // matrix
  ordering.permuteSystemFull(H.sparseView(), H_perm_act);
  CHECK(assert_equal(H_perm_exp, Eigen::MatrixXd(H_perm_act)));

  // vector
  ordering.permuteRhs(x, x_perm_act);
  CHECK(assert_equal(x_perm_exp, x_perm_act));
  ordering.permuteBackSolution(x_perm_exp, x_perm_act);
  CHECK(assert_equal(x, x_perm_act));

  // reverse ordering
  Ordering ordering_reversed(indx_reverse);
  CHECK(assert_equal_MatrixXi(indx_reverse, ordering_reversed.indices()));
  CHECK(assert_equal_MatrixXi(indx, ordering_reversed.reversedIndices()));

  ordering_reversed.permuteSystemFull(H_perm_exp.sparseView(), H_perm_act);
  CHECK(assert_equal(H, Eigen::MatrixXd(H_perm_act)));

  ordering_reversed.permuteRhs(x_perm_exp, x_perm_act);
  CHECK(assert_equal(x, x_perm_act));
  ordering_reversed.permuteBackSolution(x, x_perm_act);
  CHECK(assert_equal(x_perm_exp, x_perm_act));
}

/* ************************************************************************** */
TEST_CASE("OrderingNaturalOrdering", "[linear]") {
  // use random A to generate test, to test whether R'R - ordering.permute(A'A) = 0
  const int A_row = 40, A_col = 20;
  Eigen::MatrixXd A_dense;
  int A_rank;
  do {
    A_dense = Eigen::MatrixXd::Random(A_row, A_col);
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(A_dense);
    A_rank = lu_decomp.rank();
  } while (A_rank < A_col); // make sure full rank

  Eigen::SparseMatrix<double> A, H;
  A = A_dense.sparseView();
  H = A.transpose() * A;
  Eigen::VectorXd x = Eigen::VectorXd::Random(A_col);

  NaturalOrdering ordering(H);

  // indices
  Eigen::VectorXi idx_exp(H.cols());
  for (int i = 0; i < H.cols(); i++)
    idx_exp[i] = i;

  CHECK(ordering.indices() == idx_exp);
  CHECK(ordering.reversedIndices() == idx_exp);

  // perm system
  Eigen::SparseMatrix<double> Hperm;
  Eigen::VectorXd xperm;

  ordering.permuteSystemFull(H, Hperm);
  CHECK(assert_equal(H, Hperm));
  ordering.permuteSystemSelfAdjoint<Eigen::Upper>(H, Hperm);
  CHECK(assert_equal(H, Eigen::SparseMatrix<double>(Hperm.selfadjointView<Eigen::Upper>())));
  ordering.permuteSystemSelfAdjoint<Eigen::Lower>(H, Hperm);
  CHECK(assert_equal(H, Eigen::SparseMatrix<double>(Hperm.selfadjointView<Eigen::Lower>())));
  ordering.permuteRhs(x, xperm);
  CHECK(assert_equal(x, xperm));
  ordering.permuteBackSolution(x, xperm);
  CHECK(assert_equal(x, xperm));
}

