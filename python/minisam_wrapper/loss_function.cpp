/**
 * @file    loss_fuction.cpp
 * @author  Jing Dong
 * @date    Nov 15, 2017
 */

#include "print.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <minisam/core/LossFunction.h>


using namespace minisam;
namespace py = pybind11;


void wrap_loss_function(py::module& m) {

  // loss function base class
  py::class_<LossFunction, std::shared_ptr<LossFunction>>(m, "LossFunction")
    // override-able
    .def("weightError", &LossFunction::weightError)
    .def("weightJacobians", &LossFunction::weightJacobians)
    .def("weightInPlace", [](const std::shared_ptr<LossFunction>& loss, 
            py::EigenDRef<Eigen::VectorXd> b) {
          Eigen::VectorXd b_in = b;
          loss->weightInPlace(b_in);
          b = b_in;
        })
    .def("weightInPlace", [](const std::shared_ptr<LossFunction>& loss, 
            std::vector<py::EigenDRef<Eigen::MatrixXd>>& As, py::EigenDRef<Eigen::VectorXd> b) {
          Eigen::VectorXd b_in = b;
          std::vector<Eigen::MatrixXd> As_in;
          for (size_t i = 0; i < As.size(); i++)
            As_in.push_back(As[i]);
          loss->weightInPlace(As_in, b_in);
          b = b_in;
          for (size_t i = 0; i < As.size(); i++)
            As[i] = As_in[i];
        })
    WRAP_TYPE_PYTHON_PRINT(LossFunction)
    ;

  // derived loss function

  // gaussian classes
  py::class_<GaussianLoss, LossFunction, std::shared_ptr<GaussianLoss>>(m, 
      "GaussianLoss")
    .def_static("SqrtInformation", &GaussianLoss::SqrtInformation)
    .def_static("Information", &GaussianLoss::Information)
    .def_static("Covariance", &GaussianLoss::Covariance)
    ;

  py::class_<DiagonalLoss, LossFunction, std::shared_ptr<DiagonalLoss>>(m, 
      "DiagonalLoss")
    .def_static("Precisions", &DiagonalLoss::Precisions)
    .def_static("Sigmas", &DiagonalLoss::Sigmas)
    .def_static("Variances", &DiagonalLoss::Variances)
    .def_static("Scales", &DiagonalLoss::Scales)
    ;

  py::class_<ScaleLoss, LossFunction, std::shared_ptr<ScaleLoss>>(m, 
      "ScaleLoss")
    .def_static("Precision", &ScaleLoss::Precision)
    .def_static("Sigma", &ScaleLoss::Sigma)
    .def_static("Variance", &ScaleLoss::Variance)
    .def_static("Scale", &ScaleLoss::Scale)
    ;

  // robust kernel
  py::class_<CauchyLoss, LossFunction, std::shared_ptr<CauchyLoss>>(m, 
      "CauchyLoss")
    .def_static("Cauchy", &CauchyLoss::Cauchy)
    ;

  py::class_<HuberLoss, LossFunction, std::shared_ptr<HuberLoss>>(m, 
      "HuberLoss")
    .def_static("Huber", &HuberLoss::Huber)
    ;

  // compose loss
  py::class_<ComposedLoss, LossFunction, std::shared_ptr<ComposedLoss>>(m, 
      "ComposedLoss")
    .def_static("Compose", &ComposedLoss::Compose)
    ;
}