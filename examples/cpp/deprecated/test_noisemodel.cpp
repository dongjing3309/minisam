// test Value related types

#include <minisam/linear/GaussianNoiseModel.h>

#include <minisam/geometry/Vector.h>  // include when use Eigen vector in optimization

using namespace std;
using namespace minisam;
using namespace Eigen;

int main() {

  // Gaussian
  MatrixXd R = (MatrixXd(2,2) << 1, 1, 0, 1).finished();
  shared_ptr<NoiseModel> noisemodel_g1 = GaussianNoiseModel::SqrtInformation(R);
  noisemodel_g1->print();
  cout << endl;

  MatrixXd I = (MatrixXd(2,2) << 10, 1, 1, 10).finished();
  shared_ptr<NoiseModel> noisemodel_g2 = GaussianNoiseModel::Information(I);
  noisemodel_g2->print();
  cout << endl;

  MatrixXd S = (MatrixXd(2,2) << 0.1010, -0.0101, -0.0101, 0.1010).finished();
  shared_ptr<NoiseModel> noisemodel_g3 = GaussianNoiseModel::Covariance(S);
  noisemodel_g3->print();
  cout << endl;

  return 0;
}
