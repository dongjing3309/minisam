/**
 * @file    optimizer.cpp
 * @author  Jing Dong
 * @date    Nov 18, 2017
 */

#include "print.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <minisam/nonlinear/NonlinearOptimizer.h>
#include <minisam/nonlinear/GaussNewtonOptimizer.h>
#include <minisam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <minisam/nonlinear/DoglegOptimizer.h>
#include <minisam/nonlinear/MarginalCovariance.h>
#include <minisam/core/Variables.h>
#include <minisam/core/FactorGraph.h>


using namespace minisam;
namespace py = pybind11;


void wrap_optimizer(py::module& m) {

  // liner solver type
  py::enum_<LinearSolverType>(m, "LinearSolverType", py::arithmetic())
    .value("CHOLESKY", LinearSolverType::CHOLESKY)
    .value("CHOLMOD", LinearSolverType::CHOLMOD)
    .value("QR", LinearSolverType::QR)
    .value("CG", LinearSolverType::CG)
    .value("LSCG", LinearSolverType::LSCG)
    .value("CUDA_CHOLESKY", LinearSolverType::CUDA_CHOLESKY)
    .value("SCHUR_DENSE_CHOLESKY", LinearSolverType::SCHUR_DENSE_CHOLESKY)
    ;

  // verbosity level
  py::enum_<NonlinearOptimizerVerbosityLevel>(m, "NonlinearOptimizerVerbosityLevel", py::arithmetic())
    .value("WARNING", NonlinearOptimizerVerbosityLevel::WARNING)
    .value("ITERATION", NonlinearOptimizerVerbosityLevel::ITERATION)
    .value("SUBITERATION", NonlinearOptimizerVerbosityLevel::SUBITERATION)
    ;

  // status of optimization
  py::enum_<NonlinearOptimizationStatus>(m, "NonlinearOptimizationStatus", py::arithmetic())
    .value("SUCCESS", NonlinearOptimizationStatus::SUCCESS)
    .value("MAX_ITERATION", NonlinearOptimizationStatus::MAX_ITERATION)
    .value("ERROR_INCREASE", NonlinearOptimizationStatus::ERROR_INCREASE)
    .value("RANK_DEFICIENCY", NonlinearOptimizationStatus::RANK_DEFICIENCY)
    .value("INVALID", NonlinearOptimizationStatus::INVALID)
    ;

  // optimizer params base
  py::class_<NonlinearOptimizerParams>(m, "NonlinearOptimizerParams")
    .def(py::init<>())
    .def_readwrite("max_iterations", &NonlinearOptimizerParams::max_iterations)
    .def_readwrite("min_rel_err_decrease", &NonlinearOptimizerParams::min_rel_err_decrease)
    .def_readwrite("min_abs_err_decrease", &NonlinearOptimizerParams::min_abs_err_decrease)
    .def_readwrite("linear_solver_type", &NonlinearOptimizerParams::linear_solver_type)
    .def_readwrite("verbosity_level", &NonlinearOptimizerParams::verbosity_level)
    WRAP_TYPE_PYTHON_PRINT(NonlinearOptimizerParams)
    ;

  // optimizer base
  py::class_<NonlinearOptimizer>(m, "NonlinearOptimizer")

    .def("optimize", [](NonlinearOptimizer &opt, const FactorGraph& graph, 
            const Variables& init_values, Variables& opt_values, 
            const VariablesToEliminate& var_elimiated) {
          return opt.optimize(graph, init_values, opt_values, var_elimiated);
        }, py::call_guard<py::gil_scoped_release>())
    .def("optimize", [](NonlinearOptimizer &opt, const FactorGraph& graph, 
            const Variables& init_values, Variables& opt_values) {
          return opt.optimize(graph, init_values, opt_values);
        }, py::call_guard<py::gil_scoped_release>())

    .def("iterate", &NonlinearOptimizer::iterate)
    .def("iterations", &NonlinearOptimizer::iterations)
    WRAP_TYPE_PYTHON_PRINT(NonlinearOptimizer)
    ;

  // GN
  py::class_<GaussNewtonOptimizerParams, NonlinearOptimizerParams>(m, "GaussNewtonOptimizerParams")
    .def(py::init<>())
    ;

  py::class_<GaussNewtonOptimizer, NonlinearOptimizer>(m, "GaussNewtonOptimizer")
    .def(py::init<>())
    .def(py::init<const GaussNewtonOptimizerParams&>())
    ;

  // LM
  py::class_<LevenbergMarquardtOptimizerParams, NonlinearOptimizerParams>(m, "LevenbergMarquardtOptimizerParams")
    .def(py::init<>())
    .def_readwrite("lambda_init", &LevenbergMarquardtOptimizerParams::lambda_init)
    .def_readwrite("lambda_increase_factor_init", &LevenbergMarquardtOptimizerParams::lambda_increase_factor_init)
    .def_readwrite("lambda_increase_factor_update", &LevenbergMarquardtOptimizerParams::lambda_increase_factor_update)
    .def_readwrite("lambda_decrease_factor_min", &LevenbergMarquardtOptimizerParams::lambda_decrease_factor_min)
    .def_readwrite("lambda_min", &LevenbergMarquardtOptimizerParams::lambda_min)
    .def_readwrite("lambda_max", &LevenbergMarquardtOptimizerParams::lambda_max)
    .def_readwrite("gain_ratio_thresh", &LevenbergMarquardtOptimizerParams::gain_ratio_thresh)
    .def_readwrite("diagonal_damping", &LevenbergMarquardtOptimizerParams::diagonal_damping)
    ;

  py::class_<LevenbergMarquardtOptimizer, NonlinearOptimizer>(m, "LevenbergMarquardtOptimizer")
    .def(py::init<>())
    .def(py::init<const LevenbergMarquardtOptimizerParams&>())
    .def("reset", &LevenbergMarquardtOptimizer::reset)
    ;

  // dogleg
  py::class_<DoglegOptimizerParams, NonlinearOptimizerParams>(m, "DoglegOptimizerParams")
    .def(py::init<>())
    .def_readwrite("radius_init", &DoglegOptimizerParams::radius_init)
    .def_readwrite("radius_min", &DoglegOptimizerParams::radius_min)
    ;

  py::class_<DoglegOptimizer, NonlinearOptimizer>(m, "DoglegOptimizer")
    .def(py::init<>())
    .def(py::init<const DoglegOptimizerParams&>())
    .def("reset", &DoglegOptimizer::reset)
    ;

  // TODO: linearization

  // marginal covariance
  py::enum_<OrderingMethod>(m, "OrderingMethod", py::arithmetic())
    .value("NONE", OrderingMethod::NONE)
    .value("AMD", OrderingMethod::AMD)
    ;
  py::enum_<SquareRootSolverType>(m, "SquareRootSolverType", py::arithmetic())
    .value("CHOLESKY", SquareRootSolverType::CHOLESKY)
    ;
  py::enum_<MarginalCovarianceSolverStatus>(m, "MarginalCovarianceSolverStatus", py::arithmetic())
    .value("SUCCESS", MarginalCovarianceSolverStatus::SUCCESS)
    .value("RANK_DEFICIENCY", MarginalCovarianceSolverStatus::RANK_DEFICIENCY)
    .value("INVALID", MarginalCovarianceSolverStatus::INVALID)
    ;
  py::class_<MarginalCovarianceSolverParams>(m, "MarginalCovarianceSolverParams")
    .def(py::init<>())
    .def_readwrite("sqr_solver_type", &MarginalCovarianceSolverParams::sqr_solver_type)
    .def_readwrite("ordering_method", &MarginalCovarianceSolverParams::ordering_method)
    ;

  py::class_<MarginalCovarianceSolver>(m, "MarginalCovarianceSolver")
    .def(py::init<>())
    .def(py::init<const MarginalCovarianceSolverParams&>())
    .def("initialize", &MarginalCovarianceSolver::initialize, py::call_guard<py::gil_scoped_release>())
    .def("marginalCovariance", &MarginalCovarianceSolver::marginalCovariance)
    .def("jointMarginalCovariance", &MarginalCovarianceSolver::jointMarginalCovariance)
    ;
}