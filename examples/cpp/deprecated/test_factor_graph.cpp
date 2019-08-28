// test Variable related types

#include <minisam/core/Scalar.h>  // include when use double/float in optimization
#include <minisam/core/Eigen.h>  // include when use Eigen vector in optimization

#include <minisam/nonlinear/linearization.h>
#include <minisam/nonlinear/SparsityPattern.h>
#include <minisam/linear/SparseCholesky.h>

#include <minisam/slam/PriorFactor.h>
#include <minisam/slam/BetweenFactor.h>

#include <minisam/core/Variable.h>
#include <minisam/core/Variables.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Factor.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/LossFunction.h>

#include <Eigen/SparseQR>

using namespace std;
using namespace minisam;


int main() {

  // noisemodel
  std::shared_ptr<LossFunction> loss1 = DiagonalLoss::Sigmas(
      (Eigen::VectorXd(2) << 0.1, 0.1).finished());
  std::shared_ptr<LossFunction> loss2 = DiagonalLoss::Sigmas(
      (Eigen::VectorXd(2) << 0.1, 0.1).finished());

  // graph
  FactorGraph graph;
  Variables init_values;

  // Variables
  init_values.add(key('x', 1), Eigen::Vector2d(0, 0));
  init_values.add(key('x', 2), Eigen::Vector2d(0.1, 0.2));
  init_values.add(key('x', 3), Eigen::Vector2d(0.3, 0.4));

  init_values.print();

  // factors
  graph.add(PriorFactor<Eigen::Vector2d>(key('x', 3), Eigen::Vector2d(2, 2), loss1));
  graph.add(PriorFactor<Eigen::Vector2d>(key('x', 1), Eigen::Vector2d(0, 0), loss1));
  graph.add(PriorFactor<Eigen::Vector2d>(key('x', 2), Eigen::Vector2d(1, 1), loss1));
  
  graph.add(BetweenFactor<Eigen::Vector2d>(key('x', 1), key('x', 2), 
      Eigen::Vector2d(1, 1), loss2));
  graph.add(BetweenFactor<Eigen::Vector2d>(key('x', 2), key('x', 3), 
      Eigen::Vector2d(1, 1), loss2));
  // doubled between factors, test AtA and A pattern
  graph.add(BetweenFactor<Eigen::Vector2d>(key('x', 2), key('x', 1), 
      Eigen::Vector2d(-1, -1), loss2));
  graph.add(BetweenFactor<Eigen::Vector2d>(key('x', 3), key('x', 2), 
      Eigen::Vector2d(-1, -1), loss2));

  graph.print();
  

  cout << "graph size = " << graph.size() << endl;
  cout << "graph b dim = " << graph.dim() << endl;

  // keylist
  VariableOrdering keylist = init_values.defaultVariableOrdering();
  cout << "keylist size = " << keylist.size() << endl;
  cout << "keylist : ";
  for (size_t i = 0; i < keylist.size(); i++) {
    cout << keyString(keylist[i]) << ", ";
  }
  cout << endl;

  // test linearzation
  cout << "===============================" << endl;
  internal::JacobianSparsityPattern jsparsity_cache;
  internal::LowerHessianSparsityPattern hsparsity_cache;

  jsparsity_cache = internal::constructJacobianSparsity(graph, init_values, 
      init_values.defaultVariableOrdering());
  hsparsity_cache = internal::constructLowerHessianSparsity(graph, init_values, 
      init_values.defaultVariableOrdering());

  jsparsity_cache.print();
  hsparsity_cache.print();

  Eigen::VectorXd b, dx;
  Eigen::SparseMatrix<double> As;

  internal::linearzationJacobian(graph, init_values, jsparsity_cache, As, b);
  cout << "As = " << endl << As << endl;
  cout << "Ad = " << endl << Eigen::MatrixXd(As) << endl;
  cout << "AtA = " << endl << Eigen::MatrixXd(As).transpose() * Eigen::MatrixXd(As) << endl;
  cout << "b = " << endl << b << endl;
  cout << "Atb = " << endl << Eigen::MatrixXd(As).transpose() * b << endl;

  Eigen::VectorXd Atb;
  Eigen::SparseMatrix<double> AtAs;

  cout << "===============================" << endl;
  internal::linearzationFullHessian(graph, init_values, hsparsity_cache, AtAs, Atb);
  cout << "AtAd = " << endl << Eigen::MatrixXd(AtAs) << endl;
  cout << "Atb = " << endl << Atb << endl;


  cout << "===============================" << endl;
  // solve
  // TODO: what compresse mode do???
/*  
  QRSolver qr;
  qr.initialize(As);
  qr.solve(As, b, dx);
  cout << "dx_qr = " << endl << dx << endl;
*/
  SparseCholeskySolver chol;
  chol.initialize(AtAs);
  chol.solve(AtAs, Atb, dx);
  cout << "dx_chol = " << endl << dx << endl;

  Variables values = init_values.retract(dx, keylist);
  cout << "opt values :" << endl;
  values.print();

  cout << "init values after opt:" << endl;
  init_values.print();

  return 0;
}
