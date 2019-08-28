// test Variable related types

#include <minisam/slam/PriorFactor.h>
#include <minisam/slam/BetweenFactor.h>
#include <minisam/nonlinear/GaussNewtonOptimizer.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/Variables.h>
#include <minisam/core/LossFunction.h>

#include <minisam/core/Eigen.h>  // include when use Eigen vector in optimization

using namespace std;
using namespace minisam;


int main() {

  // noisemodel
  std::shared_ptr<LossFunction> loss1 = DiagonalLoss::Sigmas((Eigen::VectorXd(2) << 0.1, 0.1).finished());
  std::shared_ptr<LossFunction> loss2 = DiagonalLoss::Sigmas((Eigen::VectorXd(2) << 0.1, 0.1).finished());

  // graph
  FactorGraph graph;

  graph.add(PriorFactor<Eigen::Vector2d>(key('x', 1), Eigen::Vector2d(0, 0), loss1));
  graph.add(PriorFactor<Eigen::Vector2d>(key('x', 2), Eigen::Vector2d(1, 1), loss1));
  graph.add(PriorFactor<Eigen::Vector2d>(key('x', 3), Eigen::Vector2d(2, 2), loss1));

  graph.add(BetweenFactor<Eigen::Vector2d>(key('x', 1), key('x', 2), Eigen::Vector2d(1, 1), loss2));
  graph.add(BetweenFactor<Eigen::Vector2d>(key('x', 2), key('x', 3), Eigen::Vector2d(1, 1), loss2));
  
  graph.print();
  

  // Variables
  Variables init_values;

  init_values.add(key('x', 1), Eigen::Vector2d(0, 0));
  init_values.add(key('x', 2), Eigen::Vector2d(0.1, 0.1));
  init_values.add(key('x', 3), Eigen::Vector2d(0.3, 0.3));

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

  return 0;
}
