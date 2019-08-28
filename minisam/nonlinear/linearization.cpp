/**
 * @file    linearization.cpp
 * @brief   Tools to linearize a nonlinear factor graph to linear system Ax = b
 * @author  Jing Dong
 * @date    Oct 15, 2017
 */

#include <minisam/config.h>

#include <minisam/nonlinear/linearization.h>

#include <minisam/core/Factor.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Variables.h>
#include <minisam/nonlinear/SparsityPattern.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#ifdef MINISAM_WITH_MULTI_THREADS
#include <mutex>
#include <thread>
#endif

using namespace std;

namespace minisam {

/* ************************************************************************** */
void linearzationJacobian(const FactorGraph& graph, const Variables& variables,
                          Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b) {
  linearzationJacobian(graph, variables, A, b,
                       variables.defaultVariableOrdering());
}

/* ************************************************************************** */
void linearzationJacobian(const FactorGraph& graph, const Variables& variables,
                          Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
                          const VariableOrdering& ordering) {
  internal::JacobianSparsityPattern pattern =
      internal::constructJacobianSparsity(graph, variables, ordering);
  internal::linearzationJacobian(graph, variables, pattern, A, b);
}

/* ************************************************************************** */
void linearzationFullHessian(const FactorGraph& graph,
                             const Variables& variables,
                             Eigen::SparseMatrix<double>& AtA,
                             Eigen::VectorXd& Atb) {
  linearzationFullHessian(graph, variables, AtA, Atb,
                          variables.defaultVariableOrdering());
}

/* ************************************************************************** */
void linearzationFullHessian(const FactorGraph& graph,
                             const Variables& variables,
                             Eigen::SparseMatrix<double>& AtA,
                             Eigen::VectorXd& Atb,
                             const VariableOrdering& ordering) {
  internal::LowerHessianSparsityPattern pattern =
      internal::constructLowerHessianSparsity(graph, variables, ordering);
  internal::linearzationFullHessian(graph, variables, pattern, AtA, Atb);
}

namespace internal {

/* ************************************************************************** */
void linearzationJacobian(const FactorGraph& factors, const Variables& values,
                          const JacobianSparsityPattern& sparsity,
                          Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b) {
  // init A and b
  A = Eigen::SparseMatrix<double>(sparsity.A_rows, sparsity.A_cols);
  b = Eigen::VectorXd(sparsity.A_rows);

  // pre-allocate A by number of non-zeros each col
  A.reserve(sparsity.nnz_cols);

  // accumulator for row
  int err_row_counter = 0;

  for (size_t f_idx = 0; f_idx < factors.size(); f_idx++) {
    const std::shared_ptr<Factor>& f = factors.factors()[f_idx];

    // get var index of factors
    vector<int> jacobian_col;
    jacobian_col.reserve(f->keys().size());

    for (auto pkey = f->keys().begin(); pkey != f->keys().end(); pkey++) {
      size_t key_idx = sparsity.var_ordering.searchKey(*pkey);
      jacobian_col.push_back(sparsity.var_col[key_idx]);
    }

    // whiten err and jacobians
    pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> wht_Js_err =
        f->weightedJacobiansError(values);

    const vector<Eigen::MatrixXd>& wht_Js = wht_Js_err.first;
    b.segment(err_row_counter, f->dim()) = -wht_Js_err.second;

    // update jacobian matrix
    for (size_t j_idx = 0; j_idx < wht_Js.size(); j_idx++) {
      // Eigen doesn't allow block write operation
      // write element-wise
      // scan by row for better CPU cache hit
      for (int j = 0; j < wht_Js[j_idx].cols(); j++) {
        for (int i = 0; i < wht_Js[j_idx].rows(); i++) {
          A.insert(i + err_row_counter, j + jacobian_col[j_idx]) =
              wht_Js[j_idx](i, j);
        }
      }
    }

    // update row counter
    err_row_counter += (int)f->dim();
  }

  // always output compressed matrix
  A.makeCompressed();
}

namespace {
/* ************************************************************************** */
// data struct for sort key in
Eigen::MatrixXd stackMatrixCol_(const std::vector<Eigen::MatrixXd>& mats) {
  assert(mats.size() > 0);
  int H_stack_cols = 0;
  for (const auto& H : mats) {
    H_stack_cols += static_cast<int>(H.cols());
  }
  const int rows = static_cast<int>(mats[0].rows());
  Eigen::MatrixXd H_stack(rows, H_stack_cols);
  H_stack_cols = 0;
  for (const auto& H : mats) {
    assert(H.rows() == rows);
    H_stack.block(0, H_stack_cols, rows, H.cols()) = H;
    H_stack_cols += H.cols();
  }
  return H_stack;
}

/* ************************************************************************** */
#ifdef MINISAM_WITH_MULTI_THREADS
void linearzationLowerHessianSingleFactor_(
    const std::shared_ptr<Factor>& f, const Variables& values,
    const LowerHessianSparsityPattern& sparsity,
    Eigen::SparseMatrix<double>& AtA, Eigen::VectorXd& Atb, std::mutex& mutex_A,
    std::mutex& mutex_b) {
#else
void linearzationLowerHessianSingleFactor_(
    const std::shared_ptr<Factor>& f, const Variables& values,
    const LowerHessianSparsityPattern& sparsity,
    Eigen::SparseMatrix<double>& AtA, Eigen::VectorXd& Atb) {
#endif

  // whiten err and jacobians
  vector<size_t> var_idx, jacobian_col, jacobian_col_local;
  var_idx.reserve(f->size());
  jacobian_col.reserve(f->size());
  jacobian_col_local.reserve(f->size());
  size_t local_col = 0;
  for (Key vkey : f->keys()) {
    // A col start index
    size_t key_idx = sparsity.var_ordering.searchKeyUnsafe(vkey);
    var_idx.push_back(key_idx);
    jacobian_col.push_back(sparsity.var_col[key_idx]);
    jacobian_col_local.push_back(local_col);
    local_col += sparsity.var_dim[key_idx];
  }

  const pair<vector<Eigen::MatrixXd>, Eigen::VectorXd> wht_Js_err =
      f->weightedJacobiansError(values);

  const vector<Eigen::MatrixXd>& wht_Js = wht_Js_err.first;
  const Eigen::VectorXd& wht_err = wht_Js_err.second;

  Eigen::MatrixXd stackJ = stackMatrixCol_(wht_Js);

  Eigen::MatrixXd stackJtJ(stackJ.cols(), stackJ.cols());

  // adaptive multiply for better speed
  if (stackJ.cols() > 12) {
    // stackJtJ.setZero();
    memset(stackJtJ.data(), 0, stackJ.cols() * stackJ.cols() * sizeof(double));
    stackJtJ.selfadjointView<Eigen::Lower>().rankUpdate(stackJ.transpose());
  } else {
    stackJtJ.noalias() = stackJ.transpose() * stackJ;
  }

  const Eigen::VectorXd stackJtb = stackJ.transpose() * wht_err;

#ifdef MINISAM_WITH_MULTI_THREADS
  mutex_b.lock();
#endif

  for (size_t j_idx = 0; j_idx < wht_Js.size(); j_idx++) {
    Atb.segment(jacobian_col[j_idx], wht_Js[j_idx].cols()) -=
        stackJtb.segment(jacobian_col_local[j_idx], wht_Js[j_idx].cols());
  }

#ifdef MINISAM_WITH_MULTI_THREADS
  mutex_b.unlock();
  mutex_A.lock();
#endif

  for (size_t j_idx = 0; j_idx < wht_Js.size(); j_idx++) {
    // scan by row
    size_t nnz_AtA_vars_accum_var = sparsity.nnz_AtA_vars_accum[var_idx[j_idx]];
    double* value_ptr = AtA.valuePtr() + nnz_AtA_vars_accum_var;

    for (int j = 0; j < wht_Js[j_idx].cols(); j++) {
      for (int i = j; i < wht_Js[j_idx].cols(); i++) {
        *(value_ptr++) += stackJtJ(jacobian_col_local[j_idx] + i,
                                   jacobian_col_local[j_idx] + j);
      }
      value_ptr += (sparsity.nnz_AtA_cols[jacobian_col[j_idx] + j] -
                    wht_Js[j_idx].cols() + j);
    }
  }

#ifdef MINISAM_WITH_MULTI_THREADS
  mutex_A.unlock();
#endif

  // update lower non-diag hessian blocks
  for (size_t j1_idx = 0; j1_idx < wht_Js.size(); j1_idx++) {
    for (size_t j2_idx = 0; j2_idx < wht_Js.size(); j2_idx++) {
      // we know var_idx[j1_idx] != var_idx[j2_idx]
      // assume var_idx[j1_idx] > var_idx[j2_idx]
      // insert to block location (j1_idx, j2_idx)
      if (var_idx[j1_idx] > var_idx[j2_idx]) {
        size_t nnz_AtA_vars_accum_var2 =
            sparsity.nnz_AtA_vars_accum[var_idx[j2_idx]];
        int var2_dim = sparsity.var_dim[var_idx[j2_idx]];

        int inner_insert_var2_var1 =
            sparsity.inner_insert_map[var_idx[j2_idx]].at(var_idx[j1_idx]);

        double* value_ptr = AtA.valuePtr() + nnz_AtA_vars_accum_var2 +
                            var2_dim + inner_insert_var2_var1;

#ifdef MINISAM_WITH_MULTI_THREADS
        mutex_A.lock();
#endif

        if (j1_idx > j2_idx) {
          for (int j = 0; j < wht_Js[j2_idx].cols(); j++) {
            for (int i = 0; i < wht_Js[j1_idx].cols(); i++) {
              *(value_ptr++) += stackJtJ(jacobian_col_local[j1_idx] + i,
                                         jacobian_col_local[j2_idx] + j);
            }
            value_ptr += (sparsity.nnz_AtA_cols[jacobian_col[j2_idx] + j] - 1 -
                          wht_Js[j1_idx].cols());
          }
        } else {
          for (int j = 0; j < wht_Js[j2_idx].cols(); j++) {
            for (int i = 0; i < wht_Js[j1_idx].cols(); i++) {
              *(value_ptr++) += stackJtJ(jacobian_col_local[j2_idx] + j,
                                         jacobian_col_local[j1_idx] + i);
            }
            value_ptr += (sparsity.nnz_AtA_cols[jacobian_col[j2_idx] + j] - 1 -
                          wht_Js[j1_idx].cols());
          }
        }

#ifdef MINISAM_WITH_MULTI_THREADS
        mutex_A.unlock();
#endif
      }
    }
  }
}

/* ************************************************************************** */
#ifdef MINISAM_WITH_MULTI_THREADS
void linearzationLowerHessianCaller_(
    const FactorGraph& factors, const Variables& values,
    const LowerHessianSparsityPattern& sparsity,
    Eigen::SparseMatrix<double>& AtA, Eigen::VectorXd& Atb, std::mutex& mutex_A,
    std::mutex& mutex_b, int thread_id, int total_thread) {
  for (size_t fidx = thread_id; fidx < factors.size(); fidx += total_thread) {
    linearzationLowerHessianSingleFactor_(factors.factors()[fidx], values,
                                          sparsity, AtA, Atb, mutex_A, mutex_b);
  }
}
#endif
}  // namespace

/* ************************************************************************** */
void linearzationLowerHessian(const FactorGraph& factors,
                              const Variables& values,
                              const LowerHessianSparsityPattern& sparsity,
                              Eigen::SparseMatrix<double>& AtA,
                              Eigen::VectorXd& Atb) {
  // init empty AtA and Atb
  AtA = Eigen::SparseMatrix<double>(sparsity.A_cols, sparsity.A_cols);
  Atb = Eigen::VectorXd::Zero(sparsity.A_cols);

  // pre-allocate AtA by number of non-zeros each col
  AtA.reserve(sparsity.nnz_AtA_cols);

  // prepare empty AtA with zeros
  // depends on IEEE 754 floating point format of 0.0
  memset(AtA.valuePtr(), 0, sparsity.total_nnz_AtA_cols * sizeof(double));
  memcpy(AtA.innerIndexPtr(), &sparsity.inner_index[0],
         sparsity.total_nnz_AtA_cols * sizeof(int));
  memcpy(AtA.innerNonZeroPtr(), &sparsity.inner_nnz_index[0],
         sparsity.A_cols * sizeof(int));
  memcpy(AtA.outerIndexPtr(), &sparsity.outer_index[0],
         sparsity.A_cols * sizeof(int));

// incremental fill-in Hessian
#ifdef MINISAM_WITH_MULTI_THREADS

  mutex mutex_A, mutex_b;  // data mutex

  // init threads
  vector<thread> linthreads;
  linthreads.reserve(MINISAM_WITH_MULTI_THREADS_NUM);
  for (int i = 0; i < MINISAM_WITH_MULTI_THREADS_NUM; i++) {
    linthreads.emplace_back(linearzationLowerHessianCaller_, std::ref(factors),
                            std::ref(values), std::ref(sparsity), std::ref(AtA),
                            std::ref(Atb), std::ref(mutex_A), std::ref(mutex_b),
                            i, MINISAM_WITH_MULTI_THREADS_NUM);
  }

  // wait threads to finish
  for (int i = 0; i < MINISAM_WITH_MULTI_THREADS_NUM; i++) {
    linthreads[i].join();
  }

#else
  for (size_t f_idx = 0; f_idx < factors.size(); f_idx++) {
    const std::shared_ptr<Factor>& f = factors.factors()[f_idx];
    linearzationLowerHessianSingleFactor_(f, values, sparsity, AtA, Atb);
  }
#endif

  // always output compressed matrix
  AtA.makeCompressed();
}

/* ************************************************************************** */
void linearzationFullHessian(const FactorGraph& graph,
                             const Variables& variables,
                             const LowerHessianSparsityPattern& sparsity,
                             Eigen::SparseMatrix<double>& AtA,
                             Eigen::VectorXd& Atb) {
  Eigen::SparseMatrix<double> AtA_lower;
  linearzationLowerHessian(graph, variables, sparsity, AtA_lower, Atb);
  AtA = AtA_lower.selfadjointView<Eigen::Lower>();
}

}  // namespace internal
}  // namespace minisam
