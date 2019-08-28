miniSAM
=====

miniSAM is an open-source C++/Python framework for solving factor graph based least squares problems. The APIs and implementation of miniSAM are heavily inspired and influenced by [GTSAM](https://gtsam.org/), a famous factor graph framework, but miniSAM is a much more lightweight framework with

- Full Python/NumPy API, which enables more agile development and easy binding with existing Python projects, and
- A wide list of sparse linear solvers, including CUDA enabled sparse linear solvers.

miniSAM is developed by [Jing Dong](mailto:thu.dongjing@gmail.com) and [Zhaoyang Lv](mailto:zhaoyang.lv@gatech.edu). This work was initially started as final project of [Math 6644](https://math.gatech.edu/courses/math/6644), and mostly finished part-time when both authors were PhD students at College of Computing, Georgia Institute of Technology.

Mandatory Prerequisites
------

- CMake 3.4+ (Ubuntu: `sudo apt-get install cmake`), compilation configuration tool.
- [Eigen](http://eigen.tuxfamily.org) 3.3.0+ (Ubuntu: `sudo apt-get install libeigen3-dev`), a C++ template library for linear algebra.

Optional Dependencies
------

- [Sophus](https://github.com/strasdat/Sophus), a C++ implementation of Lie Groups using Eigen. miniSAM uses Sophus for all SLAM/multi-view geometry functionalities.
- Python 2.7/3.4+ to use miniSAM Python interface.
- [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) (Ubuntu: `sudo apt-get install libsuitesparse-dev`), a suite of sparse matrix algorithms. miniSAM has option to use CHOLMOD and SPQR sparse linear solvers.
- [CUDA](https://developer.nvidia.com/cuda-downloads) 7.5+. miniSAM has option to use cuSOLVER Cholesky sparse linear solver.

Get Started
------

To get and compile the library:

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

- Ubuntu: GCC >= 5.4, Clang >= 3.8
- Windows: Visual C++ >= 2015.3


Questions & Bug Reporting
-----

Please use Github issue tracker for general questions and reporting bugs.

Citing
-----

If you use miniSAM in an academic context, please cite following publications:

```
@article{Dong19ppniv,
  author    = {Jing Dong and Zhaoyang Lv},
  title     = {mini{SAM}: A Flexible Factor Graph Non-linear Least Squares Optimization Framework},
  year      = {2019},
}
```

License
-----

miniSAM is released under the BSD license, reproduced in the file LICENSE in this directory.