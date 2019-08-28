/**
 * An example of defining a vector space manifold R^2
 * In C++ this is done by specializing minisam::traits<> with
 *   1. minisam type category tag
 *   2. tangent space vector type defs
 *   3. Dim() function returns manifold dimensionality,
 *   4. Local() and Retract() functions defines the local coordinate chart.
 *   5. Print() for printing
 */

#include <minisam/core/FactorGraph.h>
#include <minisam/core/Variables.h>
#include <minisam/nonlinear/GaussNewtonOptimizer.h>
#include <minisam/slam/PriorFactor.h>

#include <iostream>

/* ************************* example custom type **************************** */

class Point2D {
 private:
  double x_, y_;

 public:
  // constructor
  Point2D(double x, double y) : x_(x), y_(y) {}

  // access
  double x() const { return x_; }
  double y() const { return y_; }
};

/* ************************** custom type traits **************************** */

namespace minisam {
template <>
struct traits<Point2D> {
  // minisam type category tag, used for static checking
  // must be manifold_tag or lie_group_tag to be optimizable
  typedef manifold_tag type_category;

  // print
  static void Print(const Point2D& m, std::ostream& out = std::cout) {
    out << "custom 2D point [" << m.x() << ", " << m.y() << "]'";
  }

  // tangent vector type defs
  typedef Eigen::Matrix<double, 2, 1> TangentVector;

  // local coordinate dimension
  static constexpr size_t Dim(const Point2D&) { return 2; }

  // map manifold point s to local coordinate
  static TangentVector Local(const Point2D& origin, const Point2D& s) {
    return Eigen::Matrix<double, 2, 1>(s.x() - origin.x(), s.y() - origin.y());
  }

  // apply changes in local coordinate to manifold, \oplus operator
  static Point2D Retract(const Point2D& origin, const TangentVector& v) {
    return Point2D(origin.x() + v[0], origin.y() + v[1]);
  }
};
}  // namespace minisam

/* ******************* optimizing custom 2D point type ********************** */

using namespace std;
using namespace minisam;

int main() {
  // graph container
  FactorGraph graph;

  // add a single prior on (0, 0)
  graph.add(PriorFactor<Point2D>(key('x', 0), Point2D(0, 0), nullptr));

  graph.print();
  cout << endl;

  // initial variables for optimization
  Variables initials;

  // initial point value set to (2, 3)
  initials.add(key('x', 0), Point2D(2, 3));

  cout << "initials:" << endl;
  initials.print();
  cout << endl;

  // optimize
  GaussNewtonOptimizer opt(GaussNewtonOptimizerParams{});
  Variables results;

  auto status = opt.optimize(graph, initials, results);
  if (status != NonlinearOptimizationStatus::SUCCESS) {
    cout << "optimization error" << endl;
  }

  cout << "optimized:" << endl;
  results.print();
  cout << endl;

  return 0;
}
