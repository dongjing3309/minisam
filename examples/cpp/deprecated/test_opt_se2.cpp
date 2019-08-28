// test Variable related types

#include <minisam/slam/PriorFactor.h>
#include <minisam/slam/BetweenFactor.h>
#include <minisam/nonlinear/GaussNewtonOptimizer.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/Variables.h>
#include <minisam/core/LossFunction.h>
#include <minisam/utils/Timer.h>

#include <minisam/geometry/Sophus.h>  // include when use Sophus classes in optimization

using namespace std;
using namespace minisam;
using namespace Eigen;


// A simple 2D pose-graph SLAM
// The robot moves from x1 to x5, with odometry information between each pair. 
// the robot moves 5 each step, and makes 90 deg right turns at x3 - x5
// At x5, there is a *loop closure* between x2 is avaible
// The graph strcuture is shown:
//
//   p-x1 - x2 - x3
//          |    |
//          x5 - x4 

int main() {

  // graph
  FactorGraph graph;

  // prior
  std::shared_ptr<LossFunction> lossp = DiagonalLoss::Sigmas((VectorXd(3) << 1.0, 1.0, 0.1).finished());

  graph.add(PriorFactor<Sophus::SE2d>(key('x', 1), Sophus::SE2d(0, Vector2d(0, 0)), lossp));

  // between
  std::shared_ptr<LossFunction> lossb = DiagonalLoss::Sigmas((VectorXd(3) << 0.5, 0.5, 0.1).finished());

  graph.add(BetweenFactor<Sophus::SE2d>(key('x', 1), key('x', 2), Sophus::SE2d(0, Vector2d(5, 0)), lossb));
  graph.add(BetweenFactor<Sophus::SE2d>(key('x', 2), key('x', 3), Sophus::SE2d(-M_PI/2, Vector2d(5, 0)), lossb));
  graph.add(BetweenFactor<Sophus::SE2d>(key('x', 3), key('x', 4), Sophus::SE2d(-M_PI/2, Vector2d(5, 0)), lossb));
  graph.add(BetweenFactor<Sophus::SE2d>(key('x', 4), key('x', 5), Sophus::SE2d(-M_PI/2, Vector2d(5, 0)), lossb));

  // loop closure
  std::shared_ptr<LossFunction> lossl = DiagonalLoss::Sigmas((VectorXd(3) << 0.5, 0.5, 0.1).finished());

  graph.add(BetweenFactor<Sophus::SE2d>(key('x', 5), key('x', 2), Sophus::SE2d(-M_PI/2, Vector2d(5, 0)), lossl));

  graph.print();


  // init values
  Variables init_values;

  init_values.add(key('x', 1), Sophus::SE2d(0.2, Vector2d(0.2, -0.3)));
  init_values.add(key('x', 2), Sophus::SE2d(-0.1, Vector2d(5.1, 0.3)));
  init_values.add(key('x', 3), Sophus::SE2d(-M_PI/2 - 0.2, Vector2d(9.9, -0.1)));
  init_values.add(key('x', 4), Sophus::SE2d(-M_PI + 0.1, Vector2d(10.2, -5.0)));
  init_values.add(key('x', 5), Sophus::SE2d(M_PI/2 - 0.1, Vector2d(5.1, -5.1)));

  init_values.print();

  
  // opt
  GaussNewtonOptimizer opt;
  Variables values;

  NonlinearOptimizationStatus status = opt.optimize(graph, init_values, values);

  if (status != NonlinearOptimizationStatus::SUCCESS) {
    cout << "optimization error" << endl;
    return 1;
  }

  cout << "opt values :" << endl;
  values.print();
  cout << "iter = " << opt.iterations() << endl;

  global_timer().print();

  return 0;
}
