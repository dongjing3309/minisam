/**
 * @file    FactorGraph.cpp
 * @brief   Factor graph class
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#include <minisam/config.h>

#include <minisam/core/FactorGraph.h>

#include <minisam/core/Factor.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Variables.h>

#ifdef MINISAM_WITH_MULTI_THREADS
#include <mutex>
#include <thread>
#endif

namespace minisam {

/* ************************************************************************** */
FactorGraph::FactorGraph(const FactorGraph& graph) {
  factors_.reserve(graph.size());
  for (const auto& f : graph.factors_) {
    factors_.push_back(f->copy());
  }
}

/* ************************************************************************** */
void FactorGraph::print(std::ostream& out) const {
  for (const auto& f : factors_) {
    f->print(out);
  }
}

/* ************************************************************************** */
size_t FactorGraph::dim() const {
  size_t errdim = 0;
  for (const auto& f : factors_) {
    errdim += f->dim();
  }
  return errdim;
}

/* ************************************************************************** */
Eigen::VectorXd FactorGraph::error(const Variables& variables) const {
  Eigen::VectorXd wht_err(dim());
  size_t err_pos = 0;
  for (const auto& f : factors_) {
    wht_err.segment(err_pos, f->dim()) = f->weightedError(variables);
    err_pos += f->dim();
  }
  return wht_err;
}

/* ************************************************************************** */
double FactorGraph::errorSquaredNorm(const Variables& variables) const {
  double err_squared_norm = 0.0;

#ifdef MINISAM_WITH_MULTI_THREADS
  // multi thread implementation
  std::mutex mutex_err;
  std::vector<std::thread> linthreads;
  linthreads.reserve(MINISAM_WITH_MULTI_THREADS_NUM);

  for (int i = 0; i < MINISAM_WITH_MULTI_THREADS_NUM; i++) {
    linthreads.emplace_back([this, &variables, &err_squared_norm, &mutex_err,
                             i]() {
      double err_thread = 0.0;
      for (size_t fidx = i; fidx < size();
           fidx += MINISAM_WITH_MULTI_THREADS_NUM) {
        err_thread += factors_[fidx]->weightedError(variables).squaredNorm();
      }
      mutex_err.lock();
      err_squared_norm += err_thread;
      mutex_err.unlock();
    });
  }
  // wait threads to finish
  for (int i = 0; i < MINISAM_WITH_MULTI_THREADS_NUM; i++) {
    linthreads[i].join();
  }

#else
  // single thread implementation
  for (const auto& f : factors_) {
    err_squared_norm += f->weightedError(variables).squaredNorm();
  }
#endif
  return err_squared_norm;
}
}  // namespace minisam
