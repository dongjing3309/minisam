// test Variables container

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/core/Variables.h>
#include <minisam/core/VariableOrdering.h>
#include <minisam/core/Key.h>
#include <minisam/core/Eigen.h> // Eigen types
#include <minisam/core/Scalar.h> // double

#include <string>
#include <sstream>

using namespace std;
using namespace minisam;


/* ************************************************************************** */
TEST_CASE("VariablesCopyConstructor", "[core]") {
  
  Variables values, values_copy;

  values.add<double>(0, 0.0);
  values.add<double>(1000, 1.1);

  values_copy.add<double>(0, 0.0);
  values_copy.add<double>(1000, 1.1);

  // copy values and change values
  Variables values_test1 = values;
  Variables values_test2(values);
  values.update<double>(0, 100.0);
  values.add<double>(1, 100.1);

  CHECK_FALSE(assert_equal(values, values_test1));
  CHECK_FALSE(assert_equal(values, values_test2));
  CHECK(assert_equal(values_copy, values_test1));
  CHECK(assert_equal(values_copy, values_test2));
}

/* ************************************************************************** */
TEST_CASE("VariablesInsert", "[core]") {

  Variables values;

  CHECK_NOTHROW(values.add<double>(0, 0.0));
  CHECK_THROWS_WITH(values.add<double>(0, 0.0), "[Variables::add] key 0 is already in Variables");
}

/* ************************************************************************** */
TEST_CASE("VariablesErase", "[core]") {

  Variables values;

  values.add<double>(0, 0.0);
  values.erase(0);
  CHECK(assert_equal<size_t>(0, values.size()));

  // should not throw anything if not exist
  CHECK_NOTHROW(values.erase(0));
}

/* ************************************************************************** */
TEST_CASE("VariablesSize", "[core]") {

  Variables values;

  // empty container
  CHECK(assert_equal<size_t>(0, values.size()));

  // non-empty container
  size_t intend_size = 10;
  for (size_t i = 0; i < intend_size; i++)
    values.add<double>(i, 0.0);
  CHECK(assert_equal(intend_size, values.size()));
}

/* ************************************************************************** */
TEST_CASE("VariablesExists", "[core]") {

  Variables values;
  values.add<double>(0, 0.0);
  values.add<Eigen::Vector2d>(key('x', 0), Eigen::Vector2d(0, 0));

  // exists should success
  CHECK(values.exists(0));
  CHECK(values.exists(key('x', 0)));

  // exists should fail
  CHECK_FALSE(values.exists(1));
  CHECK_FALSE(values.exists(key('x', 1)));
  CHECK_FALSE(values.exists(key('y', 0)));
}

/* ************************************************************************** */
TEST_CASE("VariablesAt", "[core]") {

  const double nd = 3.42;
  const Eigen::Vector3d nv3 = Eigen::Vector3d(1.2, 6.7, -3.4);

  Variables values;
  values.add<double>(0, nd);
  values.add<Eigen::Vector3d>(key('x', 0), nv3);

  // at should success
  CHECK(assert_equal(nd, values.at<double>(0)));
  CHECK(assert_equal(nv3, values.at<Eigen::Vector3d>(key('x', 0))));

  // at should throw: key does not exists
  CHECK_THROWS_WITH(values.at<double>(1), "[Variables::at] cannot find key 1 in Variables");
  CHECK_THROWS_WITH(values.at<Eigen::Vector3d>(key('x', 1)), 
      "[Variables::at] cannot find key x1 in Variables");

  // at should throw: type cast fails
  stringstream ss1;
  ss1 << "[Variable::cast] cannot find cast Variable to " << typeid(Eigen::Vector2d).name();
  CHECK_THROWS_WITH(values.at<Eigen::Vector2d>(0), ss1.str());
  stringstream ss2;
  ss2 << "[Variable::cast] cannot find cast Variable to " << typeid(Eigen::Vector2d).name();
  CHECK_THROWS_WITH(values.at<Eigen::Vector2d>(key('x', 0)), ss2.str());
}

/* ************************************************************************** */
TEST_CASE("VariablesUpdate", "[core]") {

  const double nd1 = 3.42, nd2 = 1.83;
  const Eigen::Vector3d nv31 = Eigen::Vector3d(1.2, 6.7, -3.4), nv32 = Eigen::Vector3d(0.6, 7.8, 9.1);

  Variables values;
  values.add<double>(0, nd1);
  values.add<Eigen::Vector3d>(key('x', 0), nv31);

  values.update<double>(0, nd2);
  values.update<Eigen::Vector3d>(key('x', 0), nv32);

  // at should success
  CHECK(assert_equal(nd2, values.at<double>(0)));
  CHECK(assert_equal(nv32, values.at<Eigen::Vector3d>(key('x', 0))));
}

/* ************************************************************************** */
// since Variables use unordered_map, the iterator does not have particular order
// so disable this test (compile but skip)
TEST_CASE("VariablesIterator", "[!hide]") {

  Variables values;

  values.add<double>(0, 0.0);
  values.add<double>(key('b', 1000), 5.5);
  values.add<double>(key('a', 0), 2.2);
  values.add<double>(key('b', 0), 4.4);
  values.add<double>(1000, 1.1);
  values.add<double>(key('a', 1000), 3.3);
  
  auto it = values.begin();
  CHECK(assert_equal(Key(0), it->first));
  CHECK(assert_equal(0.0, it->second->cast<double>()));

  it++;
  CHECK(assert_equal(Key(1000), it->first));
  CHECK(assert_equal(1.1, it->second->cast<double>()));

  it++;
  CHECK(assert_equal(key('a', 0), it->first));
  CHECK(assert_equal(2.2, it->second->cast<double>()));

  it++;
  CHECK(assert_equal(key('a', 1000), it->first));
  CHECK(assert_equal(3.3, it->second->cast<double>()));

  it++;
  CHECK(assert_equal(key('b', 0), it->first));
  CHECK(assert_equal(4.4, it->second->cast<double>()));

  it++;
  CHECK(assert_equal(key('b', 1000), it->first));
  CHECK(assert_equal(5.5, it->second->cast<double>()));

  it++;
  CHECK(it == values.end());
}

/* ************************************************************************** */
TEST_CASE("VariablesDim", "[core]") {

  Variables values;

  // empty container
  CHECK(assert_equal<size_t>(0ul, values.dim()));

  // non-empty container
  size_t intend_size = 10;
  for (size_t i = 0; i < intend_size; i++)
    values.add<double>(i, 0.0);
  CHECK(assert_equal(intend_size * 1, values.dim()));

  values.add<Eigen::Vector3d>(key('a', 0), Eigen::Vector3d());
  CHECK(assert_equal(intend_size * 1 + 3, values.dim()));

  values.add<Eigen::VectorXd>(key('b', 0), Eigen::VectorXd(6));
  CHECK(assert_equal(intend_size * 1 + 3 + 6, values.dim()));
}

/* ************************************************************************** */
TEST_CASE("VariablesRetractLocal", "[core]") {

  Variables values0, values1;
  values0.add<double>(0, 1.2);
  values0.add(1, Eigen::Vector2d(3.2, -7.6));
  values0.add(2, Eigen::Vector3d(-0.9, 1.3, -2.2));
  values1.add<double>(0, 3.1);
  values1.add(1, Eigen::Vector2d(-5.4, 2.1));
  values1.add(2, Eigen::Vector3d(8.9, 0.2, 4.6));

  VariableOrdering vordering(vector<Key>{0, 1, 2});
  Eigen::VectorXd local_exp(6);
  local_exp << 1.9, -8.6, 9.7, 9.8, -1.1, 6.8;

  // local
  const Eigen::VectorXd z6 = Eigen::VectorXd::Zero(6);
  CHECK(assert_equal(z6, values0.local(values0, vordering)));
  CHECK(assert_equal(z6, values1.local(values1, vordering)));
  CHECK(assert_equal(local_exp, values0.local(values1, vordering)));
  // retract
  CHECK(assert_equal(values0, values0.retract(Eigen::VectorXd::Zero(6), vordering)));
  CHECK(assert_equal(values1, values1.retract(Eigen::VectorXd::Zero(6), vordering)));
  CHECK(assert_equal(values1, values0.retract(local_exp, vordering)));

  // test a different variable ordering
  vordering = VariableOrdering(vector<Key>{2, 1, 0});
  local_exp << 9.8, -1.1, 6.8, -8.6, 9.7, 1.9;

  CHECK(assert_equal(local_exp, values0.local(values1, vordering)));
  CHECK(assert_equal(values1, values0.retract(local_exp, vordering)));
}
