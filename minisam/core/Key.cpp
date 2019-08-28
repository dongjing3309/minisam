/**
 * @file    Key.cpp
 * @brief   Index variable by interger
 * @author  Jing Dong
 * @date    Sep 6, 2018
 */

#include <minisam/core/Key.h>

#include <sstream>

namespace minisam {

/* ************************************************************************** */
std::string keyString(Key key) {
  std::stringstream ss;
  if (keyChar(key) >= 0x20 && keyChar(key) < 0x7F) {
    ss << keyChar(key) << keyIndex(key);
  } else {
    ss << key;
  }
  return ss.str();
}

}  // namespace minisam
