#include <minisam/config.h>

#include <iostream>

using namespace std;


int main() {

#ifdef MINISAM_USE_CUSOLVER
  cout << "MINISAM_USE_CUSOLVER 1" << endl;
#else
  cout << "MINISAM_USE_CUSOLVER 0" << endl;
#endif

#ifdef CUSOLVER_DOUBLE_PRECISION
  cout << "CUSOLVER_DOUBLE_PRECISION 1" << endl;
#else
  cout << "CUSOLVER_DOUBLE_PRECISION 0" << endl;
#endif

  return 0;
}