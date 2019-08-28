/**
 * @file    linearization.h
 * @brief   Tools to linearize a nonlinear factor graph to linear system Ax = b
 * @author  Jing Dong
 * @date    Oct 15, 2017
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace minisam {

// forward declearation
class FactorGraph;
class Variables;
class VariableOrdering;

namespace internal {
struct JacobianSparsityPattern;
struct LowerHessianSparsityPattern;
}

/**
 * Jacobian linearization
 */

// linearize factor graph to Ax = b with given A sparsity pattern cache
// by default use factor graph default varbiable ordering
void linearzationJacobian(const FactorGraph& graph, const Variables& variables,
                          Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b);

void linearzationJacobian(const FactorGraph& graph, const Variables& variables,
                          Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
                          const VariableOrdering& ordering);

/**
 * Hessian linearization
 */

// give a full (symetric) Hessian
// by default use factor graph default varbiable ordering
void linearzationFullHessian(const FactorGraph& graph,
                             const Variables& variables,
                             Eigen::SparseMatrix<double>& AtA,
                             Eigen::VectorXd& Atb);

void linearzationFullHessian(const FactorGraph& graph,
                             const Variables& variables,
                             Eigen::SparseMatrix<double>& AtA,
                             Eigen::VectorXd& Atb,
                             const VariableOrdering& ordering);

// implementations
namespace internal {

// linearize factor graph to Ax = b with given A sparsity pattern cache
void linearzationJacobian(const FactorGraph& graph, const Variables& variables,
                          const JacobianSparsityPattern& sparsity,
                          Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b);

// linearize factor graph to normal eq AtAx = Atb with given A sparsity pattern
// cache
// give a full (symetric) Hessian
// much slower and memory consuming compare to lower triangular version
void linearzationFullHessian(const FactorGraph& graph,
                             const Variables& variables,
                             const LowerHessianSparsityPattern& sparsity,
                             Eigen::SparseMatrix<double>& AtA,
                             Eigen::VectorXd& Atb);

// linearize factor graph to normal eq AtAx = Atb with given A sparsity pattern
// cache
// only lower part Hessian will be given
// used internally for solvers which only request lower part of Hessian
// faster and more memory efficient than full Hessian version
void linearzationLowerHessian(const FactorGraph& graph,
                              const Variables& variables,
                              const LowerHessianSparsityPattern& sparsity,
                              Eigen::SparseMatrix<double>& AtA,
                              Eigen::VectorXd& Atb);

}  // namespace internal
}  // namespace minisam
