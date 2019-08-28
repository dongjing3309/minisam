// test linearization

#include "test_common_factors.h"

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/nonlinear/linearization.h>
#include <minisam/nonlinear/SparsityPattern.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Factor.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/core/Key.h>
#include <minisam/core/Scalar.h>

#include <string>
#include <sstream>

using namespace std;
using namespace minisam;


typedef test::PFactor<double> PFactord;
typedef test::BFactor<double> BFactord;

// unit lossfunc
std::shared_ptr<LossFunction> unitlossfunc1 = nullptr;
std::shared_ptr<LossFunction> unitlossfunc2 = ScaleLoss::Sigma(1);


// prepared graph and values
Variables values_prep;
FactorGraph graph_prep;
VariableOrdering vordering1, vordering2;  // default ordering / a different ordering


/* ************************************************************************** */
TEST_CASE("linearization_prep_static_values", "[nonlinear]") {
  
  values_prep.add<double>(0, 0.5);
  values_prep.add<double>(1, 1.2);
  values_prep.add<double>(2, 2.8);
  values_prep.add<double>(3, 3.3);
  values_prep.add<double>(4, 4.4);

  graph_prep.add(PFactord(0, 0.0, unitlossfunc1));
  graph_prep.add(PFactord(1, 1.0, unitlossfunc1));
  graph_prep.add(PFactord(2, 2.0, unitlossfunc2));
  graph_prep.add(PFactord(3, 3.0, unitlossfunc2));
  graph_prep.add(PFactord(4, 4.0, unitlossfunc2));

  graph_prep.add(BFactord(0, 1, 1.0, unitlossfunc1));
  graph_prep.add(BFactord(1, 2, 1.0, unitlossfunc1));
  graph_prep.add(BFactord(2, 3, 1.0, unitlossfunc2));
  graph_prep.add(BFactord(3, 4, 1.0, unitlossfunc2));

  graph_prep.add(BFactord(1, 0, -1.0, unitlossfunc1));
  graph_prep.add(BFactord(2, 1, -1.0, unitlossfunc2));

  vordering1 = VariableOrdering(vector<Key>{0, 1, 2, 3, 4});
  vordering2 = VariableOrdering(vector<Key>{1, 3, 0, 4, 2});
}

/* ************************************************************************** */
TEST_CASE("linearizationJacobianSparsity", "[nonlinear]") {

  internal::JacobianSparsityPattern cache;


  // default variable ordering
  cache = internal::constructJacobianSparsity(graph_prep, values_prep, vordering1);

  CHECK(assert_equal<int>(11, cache.A_rows));
  CHECK(assert_equal<int>(5, cache.A_cols));

  vector<int> var_col_exp = vector<int>({0, 1, 2, 3, 4});
  CHECK(assert_equal_vector(var_col_exp, cache.var_col));
  
  vector<int> var_dim_exp = vector<int>({1, 1, 1, 1, 1});
  CHECK(assert_equal_vector(var_dim_exp, cache.var_dim));

  vector<int> nnz_cols_exp{3, 5, 4, 3, 2};
  CHECK(assert_equal_vector(nnz_cols_exp, cache.nnz_cols));


  // non-default variable ordering
  cache = internal::constructJacobianSparsity(graph_prep, values_prep, vordering2);

  CHECK(assert_equal<int>(11, cache.A_rows));
  CHECK(assert_equal<int>(5, cache.A_cols));

  var_col_exp = vector<int>({0, 1, 2, 3, 4});
  CHECK(assert_equal_vector(var_col_exp, cache.var_col));
  
  var_dim_exp = vector<int>({1, 1, 1, 1, 1});
  CHECK(assert_equal_vector(var_dim_exp, cache.var_dim));

  nnz_cols_exp = vector<int>({5, 3, 3, 2, 4});
  CHECK(assert_equal_vector(nnz_cols_exp, cache.nnz_cols));
}

/* ************************************************************************** */
TEST_CASE("linearizationLowerHessianSparsity", "[nonlinear]") {

  internal::LowerHessianSparsityPattern cache;


  // default variable ordering
  cache = internal::constructLowerHessianSparsity(graph_prep, values_prep, vordering1);

  CHECK(assert_equal<int>(11, cache.A_rows));
  CHECK(assert_equal<int>(5, cache.A_cols));

  vector<int> nnz_AtA_cols_exp = vector<int>{2, 2, 2, 2, 1};
  CHECK(assert_equal_vector(nnz_AtA_cols_exp, cache.nnz_AtA_cols));

  vector<size_t> nnz_AtA_vars_accum_exp = vector<size_t>{0, 2, 4, 6, 8, 9};
  CHECK(assert_equal_vector(nnz_AtA_vars_accum_exp, cache.nnz_AtA_vars_accum));

  // correlated vars
  auto it = cache.corl_vars[0].begin();

  CHECK(cache.corl_vars.size() == 5);
  CHECK(cache.corl_vars[0].size() == 1);
  it = cache.corl_vars[0].begin();
  CHECK(*it == 1);
  CHECK(cache.corl_vars[1].size() == 1);
  it = cache.corl_vars[1].begin();
  CHECK(*it == 2);
  CHECK(cache.corl_vars[2].size() == 1);
  it = cache.corl_vars[2].begin();
  CHECK(*it == 3);
  CHECK(cache.corl_vars[3].size() == 1);
  it = cache.corl_vars[3].begin();
  CHECK(*it == 4);
  CHECK(cache.corl_vars[4].size() == 0);

  // inner insert map for each var
  CHECK(cache.inner_insert_map.size() == 5);
  CHECK(cache.inner_insert_map[0].size() == 1);
  CHECK(cache.inner_insert_map[0][1] == 0);
  CHECK(cache.inner_insert_map[1].size() == 1);
  CHECK(cache.inner_insert_map[1][2] == 0);
  CHECK(cache.inner_insert_map[2].size() == 1);
  CHECK(cache.inner_insert_map[2][3] == 0);
  CHECK(cache.inner_insert_map[3].size() == 1);
  CHECK(cache.inner_insert_map[3][4] == 0);
  CHECK(cache.inner_insert_map[4].size() == 0);


  // non-default variable ordering
  cache = internal::constructLowerHessianSparsity(graph_prep, values_prep, vordering2);

  CHECK(assert_equal<int>(11, cache.A_rows));
  CHECK(assert_equal<int>(5, cache.A_cols));

  nnz_AtA_cols_exp = vector<int>({3, 3, 1, 1, 1});;
  CHECK(assert_equal_vector(nnz_AtA_cols_exp, cache.nnz_AtA_cols));

  nnz_AtA_vars_accum_exp = vector<size_t>{0, 3, 6, 7, 8, 9};
  CHECK(assert_equal_vector(nnz_AtA_vars_accum_exp, cache.nnz_AtA_vars_accum));

  // correlated vars
  CHECK(cache.corl_vars.size() == 5);
  CHECK(cache.corl_vars[0].size() == 2);
  it = cache.corl_vars[0].begin();
  CHECK(*it == 2);
  it++;
  CHECK(*it == 4);
  CHECK(cache.corl_vars[1].size() == 2);
  it = cache.corl_vars[1].begin();
  CHECK(*it == 3);
  it++;
  CHECK(*it == 4);
  CHECK(cache.corl_vars[2].size() == 0);
  CHECK(cache.corl_vars[3].size() == 0);
  CHECK(cache.corl_vars[4].size() == 0);

  // inner insert map for each var
  CHECK(cache.inner_insert_map.size() == 5);
  CHECK(cache.inner_insert_map[0].size() == 2);
  CHECK(cache.inner_insert_map[0][2] == 0);
  CHECK(cache.inner_insert_map[0][4] == 1);
  CHECK(cache.inner_insert_map[1].size() == 2);
  CHECK(cache.inner_insert_map[1][3] == 0);
  CHECK(cache.inner_insert_map[1][4] == 1);
  CHECK(cache.inner_insert_map[2].size() == 0);
  CHECK(cache.inner_insert_map[3].size() == 0);
  CHECK(cache.inner_insert_map[4].size() == 0);
}

/* ************************************************************************** */
TEST_CASE("linearization", "[nonlinear]") {

  Eigen::SparseMatrix<double> A_exp(11,5), AtA_exp, AtA_lower_exp;
  Eigen::SparseMatrix<double> A_act, AtA_act;
  Eigen::VectorXd b_act, b_exp, Atb_act, Atb_exp;

  internal::JacobianSparsityPattern jcache;
  internal::LowerHessianSparsityPattern hcache;

  // default variable ordering

  // prior
  A_exp.insert(0,0) = 1.0;
  A_exp.insert(1,1) = 1.0;
  A_exp.insert(2,2) = 1.0;
  A_exp.insert(3,3) = 1.0;
  A_exp.insert(4,4) = 1.0;
  // between
  A_exp.insert(5,0) = -1.0;
  A_exp.insert(5,1) = 1.0;
  A_exp.insert(6,1) = -1.0;
  A_exp.insert(6,2) = 1.0;
  A_exp.insert(7,2) = -1.0;
  A_exp.insert(7,3) = 1.0;
  A_exp.insert(8,3) = -1.0;
  A_exp.insert(8,4) = 1.0;
  A_exp.insert(9,1) = -1.0;
  A_exp.insert(9,0) = 1.0;
  A_exp.insert(10,2) = -1.0;
  A_exp.insert(10,1) = 1.0;
  // b
  b_exp = (Eigen::VectorXd(11) << -0.5, -0.2, -0.8, -0.3, -0.4, 0.3, -0.6, 0.5, -0.1, 
      -0.3, 0.6).finished();

  AtA_exp = A_exp.transpose() * A_exp;
  AtA_lower_exp = AtA_exp.triangularView<Eigen::Lower>();
  Atb_exp = A_exp.transpose() * b_exp;

  cout << "AtA_exp = " << endl << AtA_exp << endl;

  jcache = internal::constructJacobianSparsity(graph_prep, values_prep, vordering1);
  hcache = internal::constructLowerHessianSparsity(graph_prep, values_prep, vordering1);

  internal::linearzationJacobian(graph_prep, values_prep, jcache, A_act, b_act);
  CHECK(assert_equal(A_exp, A_act));
  CHECK(assert_equal(b_exp, b_act));

  internal::linearzationLowerHessian(graph_prep, values_prep, hcache, AtA_act, Atb_act);
  CHECK(assert_equal(AtA_lower_exp, AtA_act));
  CHECK(assert_equal(Atb_exp, Atb_act));

  internal::linearzationFullHessian(graph_prep, values_prep, hcache, AtA_act, Atb_act);
  CHECK(assert_equal(AtA_exp, AtA_act));
  CHECK(assert_equal(Atb_exp, Atb_act));

  // non-default variable ordering
  A_exp.setZero();

  // prior
  A_exp.insert(0,2) = 1.0;
  A_exp.insert(1,0) = 1.0;
  A_exp.insert(2,4) = 1.0;
  A_exp.insert(3,1) = 1.0;
  A_exp.insert(4,3) = 1.0;
  // between
  A_exp.insert(5,2) = -1.0;
  A_exp.insert(5,0) = 1.0;
  A_exp.insert(6,0) = -1.0;
  A_exp.insert(6,4) = 1.0;
  A_exp.insert(7,4) = -1.0;
  A_exp.insert(7,1) = 1.0;
  A_exp.insert(8,1) = -1.0;
  A_exp.insert(8,3) = 1.0;
  A_exp.insert(9,0) = -1.0;
  A_exp.insert(9,2) = 1.0;
  A_exp.insert(10,4) = -1.0;
  A_exp.insert(10,0) = 1.0;

  AtA_exp = A_exp.transpose() * A_exp;
  AtA_lower_exp = AtA_exp.triangularView<Eigen::Lower>();
  Atb_exp = A_exp.transpose() * b_exp;

  jcache = internal::constructJacobianSparsity(graph_prep, values_prep, vordering2);
  hcache = internal::constructLowerHessianSparsity(graph_prep, values_prep, vordering2);

  internal::linearzationJacobian(graph_prep, values_prep, jcache, A_act, b_act);
  CHECK(assert_equal(A_exp, A_act));
  CHECK(assert_equal(b_exp, b_act));

  internal::linearzationLowerHessian(graph_prep, values_prep, hcache, AtA_act, Atb_act);
  CHECK(assert_equal(AtA_lower_exp, AtA_act));
  CHECK(assert_equal(Atb_exp, Atb_act));

  internal::linearzationFullHessian(graph_prep, values_prep, hcache, AtA_act, Atb_act);
  CHECK(assert_equal(AtA_exp, AtA_act));
  CHECK(assert_equal(Atb_exp, Atb_act));

}
