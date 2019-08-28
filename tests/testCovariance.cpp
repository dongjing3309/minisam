// test CPU direct R solver

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/linear/Covariance.h>
#include <minisam/linear/SquareRootCholesky.h>

#include <Eigen/LU> // rank

using namespace minisam;

// exmaple systems
Eigen::SparseMatrix<double> H1(2,2), H2(2,2), H3(2,2), H;
Eigen::MatrixXd Hinv1(2,2), Hinv2(2,2), Hinv3(2,2), Hinv;
const int A_row = 40, A_col = 20;

/* ************************************************************************** */
TEST_CASE("Covariance_prep_static_values", "[linear]") {

  Eigen::SparseMatrix<double> A1(2,2), A2(2,2), A3(3,2);

  // sys 1, identity
  A1.insert(0,0) = 1.0;
  A1.insert(1,1) = 1.0;
  A1.makeCompressed();
  H1 = A1.transpose() * A1;
  Hinv1 = Eigen::MatrixXd::Identity(2, 2);

  // sys 2, well-cond, use matlab get ground truth
  A2.insert(0,0) = 3.2;
  A2.insert(0,1) = 4.5;
  A2.insert(1,0) = -1.9;
  A2.insert(1,1) = 7.6;
  A2.makeCompressed();
  H2 = A2.transpose() * A2;
  Hinv2 <<  7.220227298789961e-02,  3.702206024247995e-05,
            3.702206024247995e-05,  1.281888835895923e-02;

  // sys 3, over-cond, use matlab get ground truth
  A3.insert(0,0) = 3.2;
  A3.insert(0,1) = 4.5;
  A3.insert(1,0) = -1.9;
  A3.insert(1,1) = 7.6;
  A3.insert(2,0) = 5.5;
  A3.insert(2,1) = 3.4;
  A3.makeCompressed();
  H3 = A3.transpose() * A3;
  Hinv3 <<  2.486783565761668e-02,  -5.180683413767191e-03,
            -5.180683413767191e-03, 1.224373732835655e-02;

  // use random A to generate test, to test whether R'R - ordering.permute(A'A) = 0
  Eigen::MatrixXd A_dense;
  int A_rank;
  do {
    A_dense = Eigen::MatrixXd::Random(A_row, A_col);
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(A_dense);
    A_rank = lu_decomp.rank();
  } while (A_rank < A_col); // make sure full rank

  Eigen::MatrixXd H_dense = A_dense.transpose() * A_dense;
  H = H_dense.sparseView();
  Hinv = H_dense.inverse();
}

/* ************************************************************************** */
TEST_CASE("Covariance_ground_truth", "[linear]") {
  // get ground truth from matlab
  SquareRootSolverCholesky sqrsolver(OrderingMethod::NONE);
  Eigen::SparseMatrix<double> L;
  Eigen::MatrixXd Hinv_act; 
  std::vector<int> full_indices = {0, 1};

  sqrsolver.initialize(H1);
  sqrsolver.solveL(H1, L);
  Covariance c1(L);
  Hinv_act = c1.marginalCovariance(full_indices);
  CHECK(assert_equal(Hinv1, Hinv_act));

  sqrsolver.initialize(H2);
  sqrsolver.solveL(H2, L);
  Covariance c2(L);
  Hinv_act = c2.marginalCovariance(full_indices);
  CHECK(assert_equal(Hinv2, Hinv_act));

  sqrsolver.initialize(H3);
  sqrsolver.solveL(H3, L);
  Covariance c3(L);
  Hinv_act = c3.marginalCovariance(full_indices);
  CHECK(assert_equal(Hinv3, Hinv_act));
}

/* ************************************************************************** */
TEST_CASE("Covariance_random", "[linear]") {
  // get ground truth from matlab
  SquareRootSolverCholesky sqrsolver(OrderingMethod::NONE);
  Eigen::SparseMatrix<double> L;
  Eigen::MatrixXd Hinv_act, Hinv_partial;
  std::vector<int> full_indices;

  sqrsolver.initialize(H);
  sqrsolver.solveL(H, L);
  Covariance c(L);

  // block index
  full_indices = {5, 6, 7, 8};
  Hinv_act = c.marginalCovariance(full_indices);
  Hinv_partial = Hinv.block(5, 5, 4, 4);
  CHECK(assert_equal(Hinv_partial, Hinv_act));

  // full index
  full_indices.clear();
  for (int i = 0; i < A_col; i++) full_indices.push_back(i);
  Hinv_act = c.marginalCovariance(full_indices);
  CHECK(assert_equal(Hinv, Hinv_act));
}