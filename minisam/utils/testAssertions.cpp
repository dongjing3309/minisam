/**
 * @file    testAssertions.cpp
 * @brief   Utils for calculating numerical Jacobians
 * @author  Jing Dong
 * @date    Sep 14, 2018
 */

#include <minisam/utils/testAssertions.h>

using namespace std;

namespace minisam {

/* ************************************************************************** */
bool assert_equal_matrix(const Eigen::MatrixXd& expected,
                         const Eigen::MatrixXd& actual, double tol) {
  return assert_equal<Eigen::MatrixXd>(expected, actual, tol);
}

/* ************************************************************************** */
template <>
bool assert_equal(const Eigen::MatrixXd& expected,
                  const Eigen::MatrixXd& actual, double tol) {
  // check dim
  if (expected.rows() != actual.rows() || expected.cols() != actual.cols()) {
    cout << "Not equal:" << endl
         << "expected dimension: (" << expected.rows() << ", "
         << expected.cols() << ")" << endl
         << "actual dimension: (" << actual.rows() << ", " << actual.cols()
         << ")" << endl;
    return false;
  }
  // check values
  for (int i = 0; i < expected.rows(); i++) {
    for (int j = 0; j < expected.cols(); j++) {
      double elem_abs = std::fabs(expected(i, j));
      double diff = std::fabs(expected(i, j) - actual(i, j));
      if (diff / elem_abs > tol && diff > tol) {
        cout << "Not equal:" << endl
             << "expected: " << expected << endl
             << "actual: " << actual << endl;
        return false;
      }
    }
  }
  return true;
}

/* ************************************************************************** */
template <>
bool assert_equal(const Eigen::SparseMatrix<double>& expected,
                  const Eigen::SparseMatrix<double>& actual, double tol) {
  // compress before compare
  Eigen::SparseMatrix<double> exp_compressed(expected);
  Eigen::SparseMatrix<double> act_compressed(actual);
  exp_compressed.makeCompressed();
  act_compressed.makeCompressed();
  // check dim and nnz
  if (exp_compressed.rows() != act_compressed.rows() ||
      exp_compressed.cols() != act_compressed.cols() ||
      exp_compressed.nonZeros() != act_compressed.nonZeros()) {
    cout << "Not equal:" << endl
         << "expected dimension: (" << expected.rows() << ", "
         << expected.cols() << "), non-zeros: " << expected.nonZeros() << endl
         << exp_compressed << endl
         << "actual dimension: (" << actual.rows() << ", " << actual.cols()
         << "), non-zeros: " << actual.nonZeros() << endl
         << act_compressed << endl;
    return false;
  }
  // compare structure and values
  for (int i = 0; i < exp_compressed.outerSize(); i++) {
    Eigen::SparseMatrix<double>::InnerIterator it1(exp_compressed, i);
    Eigen::SparseMatrix<double>::InnerIterator it2(act_compressed, i);
    for (; it1 && it2; ++it1, ++it2) {
      double elem_abs = std::fabs(it1.value());
      double diff = std::fabs(it1.value() - it2.value());
      if ((diff / elem_abs > tol && diff > tol) || it1.row() != it2.row() ||
          it1.col() != it2.col() || it1.index() != it2.index()) {
        cout << "Not equal:" << endl
             << "expected: " << exp_compressed << endl
             << "actual: " << act_compressed << endl;
        return false;
      }
    }
  }
  return true;
}

/* ************************************************************************** */
template <>
bool assert_equal(const Variables& expected, const Variables& actual,
                  double tol) {
  // check size
  if (expected.size() != actual.size()) {
    cout << "Not equal:" << endl
         << "expected size: " << expected.size() << endl
         << "actual size: " << actual.size() << endl;
    return false;
  }
  // check key and value
  // does not assume both variables have the same key sequence

  for (auto it_exp = expected.begin(); it_exp != expected.end(); it_exp++) {
    // key
    if (!actual.exists(it_exp->first)) {
      cout << "Not equal:" << endl
           << "expected key: " << keyString(it_exp->first)
           << " does not exist in actual" << endl;
      return false;
    }
    // value
    const shared_ptr<Variable> var_act = actual.at(it_exp->first);
    const Eigen::VectorXd local = it_exp->second->local(*var_act);
    if (local.norm() > tol) {
      cout << "Not equal:" << endl
           << "key:" << keyString(it_exp->first) << endl
           << "expected value: ";
      it_exp->second->print();
      cout << endl << "actual value: ";
      var_act->print();
      cout << endl;
      return false;
    }
  }
  return true;
}

}  // namespace minisam
