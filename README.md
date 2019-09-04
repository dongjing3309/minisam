miniSAM
=====

Website: https://minisam.readthedocs.io/

-------------------------------------------

miniSAM is an open-source C++/Python framework for solving factor graph based least squares problems. The APIs and implementation of miniSAM are heavily inspired and influenced by [GTSAM](https://gtsam.org/), a famous factor graph framework, but miniSAM is a much more lightweight framework with

- Full Python/NumPy API, which enables more agile development and easy binding with existing Python projects, and
- A wide list of sparse linear solvers, including CUDA enabled sparse linear solvers.

miniSAM is developed by [Jing Dong](mailto:thu.dongjing@gmail.com) and [Zhaoyang Lv](mailto:zhaoyang.lv@gatech.edu). This work was initially started as final project of [Math 6644](https://www.cc.gatech.edu/~echow/cse6644-17.html) back to 2017, and mostly finished part-time when both authors were PhD students at College of Computing, Georgia Institute of Technology.

Mandatory Prerequisites
------

- [CMake](https://cmake.org/) 3.4+ (Ubuntu: `sudo apt-get install cmake`), compilation configuration tool.
- [Eigen](http://eigen.tuxfamily.org) 3.3.0+ (Ubuntu: `sudo apt-get install libeigen3-dev`), a C++ template library for linear algebra.

Optional Dependencies
------

- [Sophus](https://github.com/strasdat/Sophus), a C++ implementation of Lie Groups using Eigen. miniSAM uses Sophus for all SLAM/multi-view geometry functionalities.
- [Python](http://www.python.org/) 2.7/3.4+ to use miniSAM Python package.
- [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) (Ubuntu: `sudo apt-get install libsuitesparse-dev`), a suite of sparse matrix algorithms. miniSAM has option to use CHOLMOD and SPQR sparse linear solvers.
- [CUDA](https://developer.nvidia.com/cuda-downloads) 9.0+. miniSAM has option to use cuSOLVER Cholesky sparse linear solver.

Get Started
------

Please refer to https://minisam.readthedocs.io/install.html for more details.

To get and compile the library (on Ubuntu Linux):

```
$ git clone --recurse-submodules https://github.com/dongjing3309/minisam.git
$ mkdir build
$ cd build
$ cmake ..
$ make
$ make check  # optional, run unit tests
```
Tested Compatibility
-----

The miniSAM library is designed to be cross-platform, should be compatible with any modern compiler which supports C++11. It has been tested on Ubuntu Linux and Windows for now.

- Ubuntu: GCC 5.4+, Clang 3.8+
- Windows: Visual C++ 2015.3+


Questions & Bug Reporting
-----

Please use Github issue tracker for general questions and reporting bugs, before submitting an issue please have a look of [this page](https://minisam.readthedocs.io/github_issue.html).

Citing
-----

If you use miniSAM in an academic context, please cite following publications:

```
@article{Dong19ppniv,
  author    = {Jing Dong and Zhaoyang Lv},
  title     = {mini{SAM}: A Flexible Factor Graph Non-linear Least Squares Optimization Framework},
  journal   = {CoRR},
  volume    = {abs/1909.00903},
  year      = {2019},
  url       = {http://arxiv.org/abs/1909.00903}
}
```

License
-----

miniSAM is released under the BSD license, reproduced in the file LICENSE in this directory.
Note that the linked sparse linear solvers have different licenses, see [this page](https://minisam.readthedocs.io/install.html#sparse-solvers-license) for details
