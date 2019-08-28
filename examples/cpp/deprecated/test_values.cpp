// test Variable related types

#include <minisam/core/Variables.h>
#include <minisam/core/Scalar.h>  // include when use double/float in optimization
#include <minisam/core/Eigen.h>  // include when use Eigen::Vector in optimization
#include <minisam/geometry/Sophus.h>  // include when use Sophus classes in optimization

#include <sstream>

using namespace std;
using namespace minisam;


int main() {

  // Variables
  Variables values;

  values.add(key('x', 1), Eigen::Vector3d(0.0, 1.0, 2.0));
  values.add(key('x', 2), Eigen::Vector2d(0.1, 1.2));
  values.add(key('x', 3), 0.3);
  //values.add(key('x', 4), -0.3f);
  values.add(key('x', 5), Sophus::SO3d::rotZ(0.1));
  values.add(key('x', 6), Sophus::SE3d::transZ(2.0) * Sophus::SE3d::rotZ(0.2));

  values.print();

  cout << "values[1] = " << values.at<Eigen::Vector3d>(key('x', 1)) << endl;
  cout << "values[2] = " << values.at<Eigen::Vector2d>(key('x', 2)) << endl;
  cout << "values[3] = " << values.at<double>(key('x', 3)) << endl;
  //cout << "values[4] = " << values.at<float>(key('x', 4)) << endl;
  cout << "values[5] = " << values.at<Sophus::SO3d>(key('x', 5)) << endl;
  cout << "values[6] = " << values.at<Sophus::SE3d>(key('x', 6)) << endl;
  //values.at<double>(key('x', 5));

  cout << "values[1].exist = " << values.exists(key('x', 1)) << endl;
  cout << "values[100].exist = " << values.exists(key('x', 100)) << endl;

  cout << "values.size = " << values.size() << endl;

  cout << "values.dim = " << values.dim() << endl;

  // test print to stringstream
  stringstream ss;
  values.print(ss);
  cout << ss.str() << endl;

  return 0;
}
