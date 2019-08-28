// test numerical jacobians

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/nonlinear/numericalJacobian.h>
#include <minisam/core/Scalar.h> // use double traits

#include <cmath>

using namespace std;
using namespace minisam;


// example function
double func(double x) {
  return std::exp(x) / (std::sin(x) - x*x);
} 

// analytic diff of example function
double difffunc(double x) {
  return std::exp(x) * (std::sin(x) - std::cos(x) - x*x + 2*x) / std::pow(std::sin(x) - x*x, 2);
} 

/* ************************************************************************** */
TEST_CASE("numericalDiff", "[nonlinear]") {
  // large delta 
  double delta = 0.1, x;
  Eigen::MatrixXd Jexp(1,1);

  x = 0.37;
  Jexp << difffunc(x);

  CHECK(assert_equal(Jexp, numericalJacobian<double, double>(func, x, delta, 
      NumericalJacobianType::CENTRAL), 1e-0));
  CHECK_FALSE(assert_equal(Jexp, numericalJacobian<double, double>(func, x, delta, 
      NumericalJacobianType::CENTRAL), 1e-3));

  CHECK(assert_equal(Jexp, numericalJacobian<double, double>(func, x, delta, 
      NumericalJacobianType::RIDDERS3), 1e-3));
  CHECK_FALSE(assert_equal(Jexp, numericalJacobian<double, double>(func, x, delta, 
      NumericalJacobianType::RIDDERS3), 1e-6));

  CHECK(assert_equal(Jexp, numericalJacobian<double, double>(func, x, delta, 
      NumericalJacobianType::RIDDERS5), 1e-6));
}
