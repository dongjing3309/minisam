// test schur complement

#include <minisam/3rdparty/Catch2/catch.hpp>
#include <minisam/utils/testAssertions.h>

#include <minisam/core/SchurComplement.h>

#include <minisam/core/Eigen.h>
#include <minisam/core/Variables.h>

using namespace std;
using namespace minisam;


/* ************************************************************************** */
TEST_CASE("VariablesToEliminate", "[core]") {
  VariablesToEliminate elem_vars;

  CHECK_FALSE(elem_vars.isEliminatedAny());
  CHECK_FALSE(elem_vars.isVariableEliminated('a'));
  CHECK_FALSE(elem_vars.isVariableEliminated('b'));
  CHECK_FALSE(elem_vars.isVariableEliminated('c'));

  elem_vars.eliminate('a');
  CHECK(elem_vars.isEliminatedAny());
  CHECK(elem_vars.isVariableEliminated('a'));
  CHECK_FALSE(elem_vars.isVariableEliminated('b'));
  CHECK_FALSE(elem_vars.isVariableEliminated('c'));

  elem_vars.eliminate('b');
  CHECK(elem_vars.isEliminatedAny());
  CHECK(elem_vars.isVariableEliminated('a'));
  CHECK(elem_vars.isVariableEliminated('b'));
  CHECK_FALSE(elem_vars.isVariableEliminated('c'));
}

/* ************************************************************************** */
TEST_CASE("SchurComplementOrdering", "[core]") {

  std::vector<Key> keys = {key('a', 1), key('a', 2), key('b', 1), key('b', 3), 
      key('c', 4), key('c', 3)};
  VariableOrdering origin_ordering(keys);
  Variables values;
  values.add(key('a', 1), Eigen::Vector2d());
  values.add(key('a', 2), Eigen::Vector2d());
  values.add(key('b', 1), Eigen::Vector3d());
  values.add(key('b', 3), Eigen::Vector3d());
  values.add(key('c', 3), Eigen::Vector3d());
  values.add(key('c', 4), Eigen::Vector3d());

  // eliminate a
  VariablesToEliminate elem_vars;
  elem_vars.eliminate('a');
  SchurComplementOrdering sc_ordering = SchurComplementOrdering(origin_ordering, 
      elem_vars, values);

  CHECK(assert_equal<int>(12, sc_ordering.reducedSysUntilIndex()));
  CHECK(assert_equal<size_t>(2, sc_ordering.eliminatedVariableSize()));
  CHECK(assert_equal<int>(2, sc_ordering.eliminatedVariableDim(0)));
  CHECK(assert_equal<int>(0, sc_ordering.eliminatedVariablePosition(0)));
  CHECK(assert_equal<int>(2, sc_ordering.eliminatedVariableDim(1)));
  CHECK(assert_equal<int>(2, sc_ordering.eliminatedVariablePosition(1)));

  // eliminate b c
  elem_vars = VariablesToEliminate();
  elem_vars.eliminate('b');
  elem_vars.eliminate('c');
  sc_ordering = SchurComplementOrdering(origin_ordering, elem_vars, values);

  CHECK(assert_equal<int>(4, sc_ordering.reducedSysUntilIndex()));
  CHECK(assert_equal<size_t>(4, sc_ordering.eliminatedVariableSize()));
  CHECK(assert_equal<int>(3, sc_ordering.eliminatedVariableDim(0)));
  CHECK(assert_equal<int>(0, sc_ordering.eliminatedVariablePosition(0)));
  CHECK(assert_equal<int>(3, sc_ordering.eliminatedVariableDim(1)));
  CHECK(assert_equal<int>(3, sc_ordering.eliminatedVariablePosition(1)));
  CHECK(assert_equal<int>(3, sc_ordering.eliminatedVariableDim(2)));
  CHECK(assert_equal<int>(6, sc_ordering.eliminatedVariablePosition(2)));
  CHECK(assert_equal<int>(3, sc_ordering.eliminatedVariableDim(3)));
  CHECK(assert_equal<int>(9, sc_ordering.eliminatedVariablePosition(3)));
}
