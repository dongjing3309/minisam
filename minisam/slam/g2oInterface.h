/**
 * @file    g2oInterface.h
 * @brief   File interface of g2o and Toro format
 * @author  Jing Dong
 * @date    Nov 12, 2017
 */

#pragma once

#include <string>

namespace minisam {

// forward decleariation
class FactorGraph;
class Variables;

// load init values and factor graph from g2o file
// return whether the pose graph is a 3D pose graph
bool loadG2O(const std::string& filename, FactorGraph& graph,
             Variables& init_values);

}  // namespace minisam
