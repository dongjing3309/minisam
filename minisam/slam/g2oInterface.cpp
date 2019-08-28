/**
 * @file    g2oInterface.cpp
 * @brief   File interface of g2o and Toro format
 * @author  Jing Dong
 * @date    Nov 12, 2017
 */

#include <minisam/slam/g2oInterface.h>

#include <minisam/core/FactorGraph.h>
#include <minisam/core/Key.h>
#include <minisam/core/LossFunction.h>
#include <minisam/core/Variables.h>
#include <minisam/geometry/Sophus.h>
#include <minisam/slam/BetweenFactor.h>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <fstream>
#include <iostream>

using namespace std;

namespace minisam {

/* ************************************************************************** */
bool loadG2O(const std::string& filename, FactorGraph& graph,
             Variables& init_values) {
  ifstream g2ofile(filename.c_str(), ifstream::in);
  if (!g2ofile) {
    throw invalid_argument("[loadG2O] ERROR: cannot load file : " + filename);
  }
  bool is_graph_3d = false;  // set to false to avoid uninitialized warning

  // parse each line
  string line;
  while (getline(g2ofile, line)) {
    // reac file string
    stringstream ss(line);
    string strhead;
    ss >> strhead;

    // g2o format 2D
    if (strhead == "VERTEX_SE2") {
      // SE2 init pose
      size_t id;
      double x, y, th;
      ss >> id >> x >> y >> th;

      Sophus::SE2d pose(th, Eigen::Vector2d(x, y));
      init_values.add(key('x', id), pose);
      is_graph_3d = false;

    } else if (strhead == "EDGE_SE2") {
      // SE2 factor
      size_t ido, idi;
      double dx, dy, dth, i11, i12, i13, i22, i23, i33;
      // clang-format off
      ss >> ido >> idi >> dx >> dy >> dth >> i11 >> i12 >> i13 >> i22 >> i23 >> i33;
      Sophus::SE2d dpose(dth, Eigen::Vector2d(dx, dy));
      Eigen::Matrix3d I = (Eigen::Matrix3d() <<
          i11, i12, i13,
          i12, i22, i23,
          i13, i23, i33).finished();
      // clang-format on

      graph.add(BetweenFactor<Sophus::SE2d>(key('x', ido), key('x', idi), dpose,
                                            GaussianLoss::Information(I)));
      is_graph_3d = false;

      // TORO format 2D
    } else if (strhead == "VERTEX2") {
      // SE2 init pose
      size_t id;
      double x, y, th;
      ss >> id >> x >> y >> th;

      Sophus::SE2d pose(th, Eigen::Vector2d(x, y));
      init_values.add(key('x', id), pose);
      is_graph_3d = false;

    } else if (strhead == "EDGE2") {
      // SE2 factor
      size_t ido, idi;
      double dx, dy, dth, i11, i12, i13, i22, i23, i33;
      // clang-format off
      ss >> ido >> idi >> dx >> dy >> dth >> i11 >> i12 >> i22 >> i33 >> i13 >> i23;
      Sophus::SE2d dpose(dth, Eigen::Vector2d(dx, dy));
      Eigen::Matrix3d I = (Eigen::Matrix3d() <<
          i11, i12, i13,
          i12, i22, i23,
          i13, i23, i33).finished();
      // clang-format on

      graph.add(BetweenFactor<Sophus::SE2d>(key('x', ido), key('x', idi), dpose,
                                            GaussianLoss::Information(I)));
      is_graph_3d = false;

    } else if (strhead == "VERTEX_SE3:QUAT") {
      // SE3 init pose
      size_t id;
      double x, y, z, qx, qy, qz, qw;
      ss >> id >> x >> y >> z >> qx >> qy >> qz >> qw;

      Sophus::SE3d pose(Eigen::Quaternion<double>(qw, qx, qy, qz),
                        Eigen::Vector3d(x, y, z));
      init_values.add(key('x', id), pose);
      is_graph_3d = true;

    } else if (strhead == "EDGE_SE3:QUAT") {
      // SE2 factor
      size_t ido, idi;
      double dx, dy, dz, qx, qy, qz, qw, in[21];
      ss >> ido >> idi >> dx >> dy >> dz >> qx >> qy >> qz >> qw;
      for (int i = 0; i < 21; i++) {
        ss >> in[i];
      }

      Sophus::SE3d dpose(Eigen::Quaternion<double>(qw, qx, qy, qz),
                         Eigen::Vector3d(dx, dy, dz));
      // clang-format off
      Eigen::Matrix<double, 6, 6> I = (Eigen::Matrix<double, 6, 6>() <<
          in[0],  in[1],  in[2],  in[3],  in[4],  in[5],
          in[1],  in[6],  in[7],  in[8],  in[9],  in[10],
          in[2],  in[7],  in[11], in[12], in[13], in[14],
          in[3],  in[8],  in[12], in[15], in[16], in[17],
          in[4],  in[9],  in[13], in[16], in[18], in[19],
          in[5],  in[10], in[14], in[17], in[19], in[20]).finished();
      // clang-format on
      graph.add(BetweenFactor<Sophus::SE3d>(key('x', ido), key('x', idi), dpose,
                                            GaussianLoss::Information(I)));
      is_graph_3d = true;

    } else {
      throw invalid_argument("[loadG2O] ERROR: cannot parse file " + filename +
                             " by 2D/3D g2o/TORO format");
    }
  }
  g2ofile.close();
  return is_graph_3d;
}
}
