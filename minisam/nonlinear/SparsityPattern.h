/**
 * @file    SparsityPattern.h
 * @brief   linear system Ax = b sparsity pattern for fast linearization
 * @author  Jing Dong
 * @date    Oct 25, 2018
 */

#pragma once

#include <minisam/core/VariableOrdering.h>

#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

namespace minisam {

// forward declearation
class FactorGraph;
class Variables;

namespace internal {

// base class for A and A'A sparsity pattern, if variable ordering is fixed,
// only need to be constructed once for different linearzation runs
struct SparsityPatternBase {
  // basic size information
  int A_rows;  // = b size
  int A_cols;  // = A'A size
  VariableOrdering var_ordering;

  // var_dim: dim of each vars (using ordering of var_ordering)
  // var_col: start col of each vars (using ordering of var_ordering)
  std::vector<int> var_dim;
  std::vector<int> var_col;

  // print
  virtual void print(std::ostream& out = std::cout) const;
};

// struct store  given variable ordering
struct JacobianSparsityPattern : SparsityPatternBase {
  // Eigen::Sparse memory allocation information
  // number of non-zeros count for each col of A, use for Eigen sparse matrix A
  // reservation
  std::vector<int> nnz_cols;

  // start row of each factor
  std::vector<int> factor_err_row;

  // print
  void print(std::ostream& out = std::cout) const override;
};

// struct store A'A lower part sparsity pattern given variable ordering
// note this does not apply to full A'A!
struct LowerHessianSparsityPattern : SparsityPatternBase {
  // number of non-zeros count for each col of AtA (each row of A)
  // use for Eigen sparse matrix AtA reserve
  std::vector<int> nnz_AtA_cols;
  int total_nnz_AtA_cols;

  // accumulated nnzs in AtA before each var
  // index: var idx, value: total skip nnz
  std::vector<size_t> nnz_AtA_vars_accum;

  // corl_vars: variable ordering position of all correlated vars of each var
  // (not include self), set must be ordered
  std::vector<std::set<int>> corl_vars;

  // inner index of each coorelated vars, exculde lower triangular part
  // index: corl var idx, value: inner index
  std::vector<std::unordered_map<int, int>> inner_insert_map;

  // sparse matrix inner/outer index
  std::vector<int> inner_index, inner_nnz_index, outer_index;

  // print
  void print(std::ostream& out = std::cout) const override;
};

// construct Ax = b sparsity pattern cache from a factor graph and a set of
// variables
JacobianSparsityPattern constructJacobianSparsity(
    const FactorGraph& graph, const Variables& variables,
    const VariableOrdering& variable_ordering);

// construct A'Ax = A'b sparsity pattern cache from a factor graph and a set of
// variables
LowerHessianSparsityPattern constructLowerHessianSparsity(
    const FactorGraph& graph, const Variables& variables,
    const VariableOrdering& variable_ordering);

}  // namespace internal
}  // namespace minisam
