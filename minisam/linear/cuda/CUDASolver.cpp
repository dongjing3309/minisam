/**
 * @file    CUDASolver.cpp
 * @brief   CUDA Direct linear solvers of Ax = b
 * @author  Zhaoyang Lv, Jing Dong
 * @date    Sep 20, 2018
 */

#include <minisam/linear/cuda/CUDASolver.h>
#include <minisam/utils/Timer.h>

#include <cuda_runtime.h>

#include <iostream>

namespace minisam {

// function for checking cuda/cusparse/cusolver error
void checkCudaErrors(cudaError_t status) {
  if (status != cudaSuccess) {
    if (status == cudaErrorMemoryAllocation) {
      throw std::runtime_error(
          "[checkCudaErrors] CUDA cannot allocate enough graphics memory");
    } else {
      throw std::runtime_error("[checkCudaErrors] CUDA internal error");
    }
  }
}

void checkCusolverErrors(cusolverStatus_t status) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    if (status == CUSOLVER_STATUS_ALLOC_FAILED) {
      throw std::runtime_error(
          "[checkCusolverErrors] cusolver cannot allocate enough graphics "
          "memory");
    } else {
      throw std::runtime_error("[checkCusolverErrors] cusolver internal error");
    }
  }
}

void checkCusparseErrors(cusparseStatus_t status) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    if (status == CUSPARSE_STATUS_ALLOC_FAILED) {
      throw std::runtime_error(
          "[checkCusparseErrors] cusparse cannot allocate enough graphics "
          "memory");
    } else {
      throw std::runtime_error("[checkCusparseErrors] cusparse internal error");
    }
  }
}

/* ************************************************************************** */
CUDACholeskySolver::CUDACholeskySolver()
    : cusolverSpH_(NULL),
      descrA_(NULL),
      d_csrRowPtrA_(NULL),
      d_csrColIndA_(NULL),
      d_csrValA_(NULL),
      d_x_(NULL),
      d_b_(NULL),
      h_x_(NULL) {}

/* ************************************************************************** */
CUDACholeskySolver::~CUDACholeskySolver() { free_(); }

/* ************************************************************************** */
LinearSolverStatus CUDACholeskySolver::initialize(
    const Eigen::SparseMatrix<double> &A) {
  // free all existing allocations
  free_();

  // AMD ordering
  amd_ = std::shared_ptr<Ordering>(new AMDOrdering(A));

  checkCusolverErrors(cusolverSpCreate(&cusolverSpH_));
  checkCusparseErrors(cusparseCreateMatDescr(&descrA_));
  checkCusparseErrors(
      cusparseSetMatType(descrA_, CUSPARSE_MATRIX_TYPE_GENERAL));

  // // stream
  // checkCudaErrors(cudaStreamCreate(&stream_));
  // checkCudaErrors(cusolverSpSetStream(cusolverSpH_, stream_));

  // note that Eigen uses CSC format, while cusparse uses CSR format
  // but here since A is symetric, it does not matter in this case
  checkCudaErrors(
      cudaMalloc((void **)&d_csrRowPtrA_, sizeof(int) * (A.cols() + 1)));
  checkCudaErrors(
      cudaMalloc((void **)&d_csrColIndA_, sizeof(int) * A.nonZeros()));
  checkCudaErrors(
      cudaMalloc((void **)&d_csrValA_, sizeof(double) * A.nonZeros()));
  checkCudaErrors(cudaMalloc((void **)&d_x_, sizeof(double) * A.cols()));
  checkCudaErrors(cudaMalloc((void **)&d_b_, sizeof(double) * A.cols()));

  // host
  h_x_ = (double *)malloc(sizeof(double) * A.cols());

  return LinearSolverStatus::SUCCESS;
}

/* ************************************************************************** */
LinearSolverStatus CUDACholeskySolver::solve(
    const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b,
    Eigen::VectorXd &x) {
  // apply AMD ordering
  Eigen::SparseMatrix<double> A_reorderd(A.rows(), A.cols());
  Eigen::VectorXd b_reordered, x_reordered;

  // use Eigen permutation
  // cusolver of lower part of CSR format, which is same as upper part of CSC
  // format
  amd_->permuteSystemSelfAdjoint<Eigen::Upper>(A, A_reorderd);
  amd_->permuteRhs(b, b_reordered);

  // note that Eigen uses CSC format, while cusparse uses CSR format
  // but here since A is symetric, it does not matter in this case
  checkCudaErrors(cudaMemcpy(d_csrValA_, A_reorderd.valuePtr(),
                             A_reorderd.nonZeros() * sizeof(double),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csrRowPtrA_, A_reorderd.outerIndexPtr(),
                             (1 + A_reorderd.cols()) * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csrColIndA_, A_reorderd.innerIndexPtr(),
                             A_reorderd.nonZeros() * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b_, b_reordered.data(),
                             sizeof(double) * A_reorderd.cols(),
                             cudaMemcpyHostToDevice));

  // the constant used in cusolverSp
  // singularity is -1 if A is invertible under tol
  // tol determines the condition of singularity
  int singularity = 0;
  const double tol = 0.0;

  // no internal reordering, so only lower part (upper part of CSC) is used
  checkCusolverErrors(cusolverSpDcsrlsvchol(
      cusolverSpH_, A_reorderd.rows(), A_reorderd.nonZeros(), descrA_,
      d_csrValA_, d_csrRowPtrA_, d_csrColIndA_, d_b_, tol, 0, d_x_,
      &singularity));

  checkCudaErrors(cudaMemcpy(h_x_, d_x_, sizeof(double) * A_reorderd.cols(),
                             cudaMemcpyDeviceToHost));
  x_reordered = Eigen::Map<Eigen::VectorXd>(h_x_, A_reorderd.cols());

  amd_->permuteBackSolution(x_reordered, x);

  if (singularity == -1) {
    return LinearSolverStatus::SUCCESS;
  } else {
    return LinearSolverStatus::RANK_DEFICIENCY;
  }
}

/* ************************************************************************** */
void CUDACholeskySolver::free_() {
  // solver handle
  if (cusolverSpH_) {
    checkCusolverErrors(cusolverSpDestroy(cusolverSpH_));
  }
  if (descrA_) {
    checkCusparseErrors(cusparseDestroyMatDescr(descrA_));
  }

  // device
  if (d_csrValA_) {
    checkCudaErrors(cudaFree(d_csrValA_));
  }
  if (d_csrRowPtrA_) {
    checkCudaErrors(cudaFree(d_csrRowPtrA_));
  }
  if (d_csrColIndA_) {
    checkCudaErrors(cudaFree(d_csrColIndA_));
  }
  if (d_x_) {
    checkCudaErrors(cudaFree(d_x_));
  }
  if (d_b_) {
    checkCudaErrors(cudaFree(d_b_));
  }

  // host
  if (h_x_) {
    free(h_x_);
  }
}
}
