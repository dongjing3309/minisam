/**
 * @file    print.h
 * @brief   support python print of minisam classes
 * @author  Jing Dong
 * @date    Nov 15, 2017
 */

#pragma once

#include <sstream>
#include <string>


namespace minisam {

// print object to string, remove last char
template<class T>
std::string printType2String(const T& obj) {
  std::stringstream ss;
  obj.print(ss);
  std::string str = ss.str();
  // remove last new line, which is needed in C++, but will be auto-filled by Python print()
  str.pop_back();
  return str;
}

// print object to string, not remove last char
template<class T>
std::string printType2StringNotRemoveLast(const T& obj) {
  std::stringstream ss;
  obj.print(ss);
  return ss.str();
}

} // namespace minisam

// wrap type to python print
#define WRAP_TYPE_PYTHON_PRINT(T) \
  .def("__repr__", [](const T &obj) { return minisam::printType2String(obj); })

#define WRAP_TYPE_PYTHON_PRINT_NOT_REMOVE_LAST(T) \
  .def("__repr__", [](const T &obj) { return minisam::printType2StringNotRemoveLast(obj); })

