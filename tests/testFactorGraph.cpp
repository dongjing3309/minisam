// test factor graph container

#include "test_common_factors.h"

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/nonlinear/linearization.h>
#include <minisam/nonlinear/NumericalFactor.h>
#include <minisam/core/FactorGraph.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Factor.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/core/Scalar.h>

#include <string>
#include <sstream>

using namespace std;
using namespace minisam;


typedef test::PFactor<double> PFactord;

// numerical prior factor on double
class NPFactor: public NumericalFactor {

private:
  double prior_;

public:
  NPFactor(Key key, double prior, const std::shared_ptr<LossFunction>& lossfunc): 
      NumericalFactor(1, std::vector<Key>{key}, lossfunc), prior_(prior) {}
  virtual ~NPFactor() {}

  std::shared_ptr<Factor> copy() const { return std::shared_ptr<Factor>(new NPFactor(*this)); }

  // error function
  Eigen::VectorXd error(const Variables& values) const {
    return (Eigen::VectorXd(1) << values.at<double>(keys()[0]) - prior_).finished();
  }
};

// unit noisemodel
std::shared_ptr<LossFunction> unitloss1 = nullptr;
std::shared_ptr<LossFunction> unitloss2 = ScaleLoss::Sigma(1.0);


// prepared graph and values
Variables values_prep, values_prep_copy;
FactorGraph graph_prep, num_graph_prep;

/* ************************************************************************** */
TEST_CASE("FactorGraph_prep_static_values", "[core]") {
  
  values_prep.add<double>(0, 0.0);
  values_prep.add<double>(key('b', 1000), 5.5);
  values_prep.add<double>(key('a', 0), 2.2);
  values_prep.add<double>(key('b', 0), 4.4);
  values_prep.add<double>(1000, 1.1);
  values_prep.add<double>(key('a', 1000), 3.3);

  values_prep_copy = Variables(values_prep);
  
  graph_prep.add(PFactord(0, 5.3, unitloss1));
  graph_prep.add(PFactord(key('b', 1000), 6.3, unitloss1));
  graph_prep.add(PFactord(key('a', 0), -4.1, unitloss1));
  graph_prep.add(PFactord(key('b', 0), 0.9, unitloss2));
  graph_prep.add(PFactord(1000, -2.1, unitloss2));
  graph_prep.add(PFactord(key('a', 1000), 7.7, unitloss2));

  num_graph_prep.add(NPFactor(0, 5.3, unitloss1));
  num_graph_prep.add(NPFactor(key('b', 1000), 6.3, unitloss1));
  num_graph_prep.add(NPFactor(key('a', 0), -4.1, unitloss1));
  num_graph_prep.add(NPFactor(key('b', 0), 0.9, unitloss2));
  num_graph_prep.add(NPFactor(1000, -2.1, unitloss2));
  num_graph_prep.add(NPFactor(key('a', 1000), 7.7, unitloss2));
}

/* ************************************************************************** */
TEST_CASE("FactorCopyConstructor", "[core]") {

  FactorGraph graph, graph_copy;
  graph.add(PFactord(0, 1.1, unitloss1));
  graph.add(PFactord(1, 2.2, unitloss2));
  graph_copy.add(PFactord(0, 1.1, unitloss1));
  graph_copy.add(PFactord(1, 2.2, unitloss2));

  // copy graph and change graph
  FactorGraph graph1 = graph;
  FactorGraph graph2(graph);
  dynamic_cast<PFactord*>(graph.factors()[0].get())->keylist_nonconst() = vector<Key>{10};
  dynamic_cast<PFactord*>(graph.factors()[1].get())->keylist_nonconst() = vector<Key>{11};

  for (size_t i = 0; i < graph.size(); i++) {
    CHECK_FALSE(assert_equal_vector(graph.factors()[i]->keys(), graph1.factors()[i]->keys()));
    CHECK_FALSE(assert_equal_vector(graph.factors()[i]->keys(), graph2.factors()[i]->keys()));
    CHECK(assert_equal_vector(graph_copy.factors()[i]->keys(), graph1.factors()[i]->keys()));
    CHECK(assert_equal_vector(graph_copy.factors()[i]->keys(), graph2.factors()[i]->keys()));
  }
}

/* ************************************************************************** */
TEST_CASE("FactorGraphAdd", "[core]") {

  FactorGraph graph;
  CHECK_NOTHROW(graph.add(PFactord(0, 0.0, unitloss1)));
}

/* ************************************************************************** */
TEST_CASE("FactorGraphErase", "[core]") {

  FactorGraph graph;
  graph.add(PFactord(0, 0.0, unitloss1));
  CHECK_NOTHROW(graph.erase(0));
  CHECK(assert_equal<size_t>(0, graph.size()));
}

/* ************************************************************************** */
TEST_CASE("FactorGraphSize", "[core]") {

  FactorGraph graph;
  // empty container
  CHECK(assert_equal<size_t>(0, graph.size()));

  // non-empty container
  const size_t intend_size = 10;
  for (size_t i = 0; i < intend_size; i++)
    graph.add(PFactord(0, 0.0, unitloss1));
  CHECK(assert_equal(intend_size, graph.size()));
}

/* ************************************************************************** */
TEST_CASE("FactorGraphDim", "[core]") {

  FactorGraph graph;
  // empty container
  CHECK(assert_equal<size_t>(0, graph.dim()));

  // non-empty container
  const size_t intend_size = 10;
  for (size_t i = 0; i < intend_size; i++)
    graph.add(PFactord(0, 0.0, unitloss1));
  CHECK(assert_equal(intend_size, graph.dim()));
}

/* ************************************************************************** */
TEST_CASE("FactorGraphError", "[core]") {

  Eigen::VectorXd err_act = graph_prep.error(values_prep);
  Eigen::VectorXd err_exp = (Eigen::VectorXd(6) << -5.3, -0.8, 6.3, 3.5, 3.2, -4.4).finished();
  CHECK(assert_equal(err_exp, err_act));
}

/* ************************************************************************** */
TEST_CASE("FactorGraphErrorNorm", "[core]") {

  double err_act = graph_prep.errorSquaredNorm(values_prep);
  double err_exp = (Eigen::VectorXd(6) << -5.3, -0.8, 6.3, 3.5, 3.2, -4.4).finished().squaredNorm();
  CHECK(assert_equal(err_exp, err_act));
}

/* ************************************************************************** */
TEST_CASE("FactorNumericalJacobians", "[nonlinear]") {

  for (size_t i = 0; i < num_graph_prep.size(); i++) {
    vector<Eigen::MatrixXd> exp_jacobians = graph_prep.at(i)->jacobians(values_prep);
    vector<Eigen::MatrixXd> act_jacobians = num_graph_prep.at(i)->jacobians(values_prep);
    CHECK(assert_equal_vector(exp_jacobians, act_jacobians));

    // check value consistency since numericalJacobians_ involves const_cast
    CHECK(assert_equal(values_prep_copy, values_prep, 1e-20));
  }
}
