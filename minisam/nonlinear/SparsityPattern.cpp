/**
 * @file    SparsityPattern.cpp
 * @brief   linear system Ax = b sparsity pattern for fast linearization
 * @author  Jing Dong
 * @date    Oct 25, 2018
 */

#include <minisam/nonlinear/SparsityPattern.h>

#include <minisam/core/Factor.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Variables.h>

#include <iostream>
#include <numeric>

using namespace std;

namespace minisam {
namespace internal {

/* ************************************************************************** */
JacobianSparsityPattern constructJacobianSparsity(
    const FactorGraph& graph, const Variables& variables,
    const VariableOrdering& var_ordering) {
  JacobianSparsityPattern sparsity;

  // A size
  sparsity.A_rows = (int)graph.dim();
  sparsity.A_cols = (int)variables.dim();

  // var_dim: dim of each vars (using ordering of var_ordering)
  // var_col: start col of each vars (using ordering of var_ordering)
  sparsity.var_dim.reserve(var_ordering.size());
  sparsity.var_col.reserve(var_ordering.size());
  int col_counter = 0;

  for (size_t i = 0; i < var_ordering.size(); i++) {
    sparsity.var_col.push_back(col_counter);
    int vdim = (int)variables.at(var_ordering[i])->dim();
    sparsity.var_dim.push_back(vdim);
    col_counter += vdim;
  }

  // counter for row of error
  sparsity.nnz_cols.resize(sparsity.A_cols, 0);
  // counter row of factor
  sparsity.factor_err_row.reserve(graph.size());
  int err_row_counter = 0;

  for (auto f = graph.begin(); f != graph.end(); f++) {
    // factor dim
    int f_dim = (int)(*f)->dim();

    for (auto pkey = (*f)->keys().begin(); pkey != (*f)->keys().end(); pkey++) {
      // A col start index
      Key vkey = *pkey;
      size_t key_order_idx = var_ordering.searchKey(vkey);
      // A col non-zeros
      for (int nz_col = sparsity.var_col[key_order_idx];
           nz_col <
           sparsity.var_col[key_order_idx] + sparsity.var_dim[key_order_idx];
           nz_col++) {
        sparsity.nnz_cols[nz_col] += f_dim;
      }
    }

    sparsity.factor_err_row.push_back(err_row_counter);
    err_row_counter += f_dim;
  }

  // copy var ordering
  sparsity.var_ordering = var_ordering;

  return sparsity;
}

/* ************************************************************************** */
LowerHessianSparsityPattern constructLowerHessianSparsity(
    const FactorGraph& graph, const Variables& variables,
    const VariableOrdering& var_ordering) {
  LowerHessianSparsityPattern sparsity;

  // A size
  sparsity.A_rows = (int)graph.dim();
  sparsity.A_cols = (int)variables.dim();

  // var_dim: dim of each vars (using ordering of var_ordering)
  // var_col: start col of each vars (using ordering of var_ordering)
  sparsity.var_dim.reserve(var_ordering.size());
  sparsity.var_col.reserve(var_ordering.size());
  int col_counter = 0;

  for (size_t i = 0; i < var_ordering.size(); i++) {
    sparsity.var_col.push_back(col_counter);
    int vdim = (int)variables.at(var_ordering[i])->dim();
    sparsity.var_dim.push_back(vdim);
    col_counter += vdim;
  }

  // AtA col correlated vars of lower part
  // does not include itself
  sparsity.corl_vars = vector<set<int>>(var_ordering.size(), set<int>());

  for (const auto& f : graph) {
    vector<size_t> factor_key_order_idx;
    factor_key_order_idx.reserve(f->keys().size());
    for (auto key : f->keys()) {
      factor_key_order_idx.push_back(var_ordering.searchKeyUnsafe(key));
    }
    for (size_t i : factor_key_order_idx) {
      for (size_t j : factor_key_order_idx) {
        // only lower part
        if (i < j) sparsity.corl_vars[i].insert((int)j);
      }
    }
  }

  sparsity.nnz_AtA_cols.resize(sparsity.A_cols, 0);
  sparsity.nnz_AtA_vars_accum.reserve(var_ordering.size() + 1);
  sparsity.nnz_AtA_vars_accum.push_back(0);
  size_t last_nnz_AtA_vars_accum = 0;

  for (size_t var_idx = 0; var_idx < var_ordering.size(); var_idx++) {
    int self_dim = sparsity.var_dim[var_idx];
    int self_col = sparsity.var_col[var_idx];
    // self: lower triangular part
    last_nnz_AtA_vars_accum += ((1 + self_dim) * self_dim) / 2;
    for (int i = 0; i < self_dim; i++) {
      int col = self_col + i;
      sparsity.nnz_AtA_cols[col] += self_dim - i;
    }
    // non-self
    for (auto corl_var_idx : sparsity.corl_vars[var_idx]) {
      last_nnz_AtA_vars_accum += sparsity.var_dim[corl_var_idx] * self_dim;
      for (int col = self_col; col < self_col + self_dim; col++) {
        sparsity.nnz_AtA_cols[col] += sparsity.var_dim[corl_var_idx];
      }
    }
    sparsity.nnz_AtA_vars_accum.push_back(last_nnz_AtA_vars_accum);
  }
  sparsity.total_nnz_AtA_cols =
      accumulate(sparsity.nnz_AtA_cols.begin(), sparsity.nnz_AtA_cols.end(), 0);

  // where to insert nnz element
  sparsity.inner_insert_map.resize(var_ordering.size(),
                                   unordered_map<int, int>());

  for (size_t var1_idx = 0; var1_idx < var_ordering.size(); var1_idx++) {
    int nnzdim_counter = 0;
    // non-self
    for (auto it = sparsity.corl_vars[var1_idx].begin();
         it != sparsity.corl_vars[var1_idx].end(); it++) {
      int var2_idx = *it;
      sparsity.inner_insert_map[var1_idx][var2_idx] = nnzdim_counter;
      nnzdim_counter += sparsity.var_dim[var2_idx];
    }
  }

  // prepare sparse matrix inner/outer index
  sparsity.inner_index.resize(sparsity.total_nnz_AtA_cols);
  sparsity.inner_nnz_index.resize(sparsity.A_cols);
  sparsity.outer_index.resize(sparsity.A_cols + 1);

  int* inner_index_ptr = &sparsity.inner_index[0];
  int* inner_nnz_ptr = &sparsity.inner_nnz_index[0];
  int* outer_index_ptr = &sparsity.outer_index[0];

  int out_counter = 0;
  *outer_index_ptr = 0;
  outer_index_ptr++;

  for (size_t var_idx = 0; var_idx < sparsity.var_dim.size(); var_idx++) {
    for (int i = 0; i < sparsity.var_dim[var_idx]; i++) {
      *inner_nnz_ptr = sparsity.nnz_AtA_cols[i + sparsity.var_col[var_idx]];
      out_counter += sparsity.nnz_AtA_cols[i + sparsity.var_col[var_idx]];
      *outer_index_ptr = out_counter;
      inner_nnz_ptr++;
      outer_index_ptr++;

      // self
      for (int ii = i; ii < sparsity.var_dim[var_idx]; ii++) {
        *inner_index_ptr = sparsity.var_col[var_idx] + ii;
        inner_index_ptr++;
      }
      // non-self
      for (int corl_idx : sparsity.corl_vars[var_idx]) {
        for (int j = 0; j < sparsity.var_dim[corl_idx]; j++) {
          *inner_index_ptr = j + sparsity.var_col[corl_idx];
          inner_index_ptr++;
        }
      }
    }
  }

  // copy
  sparsity.var_ordering = var_ordering;

  return sparsity;
}

/* ************************************************************************** */
void SparsityPatternBase::print(std::ostream& out) const {
  out << "------------------------------------------------------------" << endl;
  out << "Linear system :" << endl;
  out << "A size = (" << A_rows << ", " << A_cols << ")" << endl;
  out << "A'A size = (" << A_cols << ", " << A_cols << ")" << endl;
  out << "b size = " << A_rows << endl;

  out << "------------------------------------------------------------" << endl;
  out << "Variables (sort by ordering) :" << endl;
  var_ordering.print(out);
}

/* ************************************************************************** */
void JacobianSparsityPattern::print(std::ostream& out) const {
  out << "Jacobian Sparsity Pattern :" << endl;
  SparsityPatternBase::print(out);

  out << "------------------------------------------------------------" << endl;
  out << "Non-zeros memory allocation :" << endl;
  out << "A non-zeros each col : " << endl;
  for (int i = 0; i < A_cols; i++) {
    out << nnz_cols[i] << " ";
  }
  out << endl;
}

/* ************************************************************************** */
void LowerHessianSparsityPattern::print(std::ostream& out) const {
  out << "Lower Hessian Sparsity Pattern :" << endl;
  SparsityPatternBase::print(out);

  out << "------------------------------------------------------------" << endl;
  out << "Non-zeros memory allocation :" << endl;
  out << "A'A non-zeros each col: " << endl;
  for (int i = 0; i < A_cols; i++) {
    out << nnz_AtA_cols[i] << " ";
  }
  out << endl;
}

}  // namespace internal
}  // namespace minisam
