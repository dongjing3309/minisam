/**
 * @file    VariableOrdering.cpp
 * @brief   Class represent variable ordering, and COLAMD method
 * @author  Jing Dong
 * @date    Oct 15, 2017
 */

#include <minisam/core/VariableOrdering.h>

#include <cassert>
#include <sstream>

namespace minisam {

/* ************************************************************************** */
VariableOrdering::VariableOrdering(const std::vector<Key>& keylist)
    : keylist_(keylist) {
  // build index search map
  for (size_t i = 0; i < keylist.size(); i++) {
    assert(keymap_.find(keylist[i]) == keymap_.end() &&
           "[VariableOrdering::VariableOrdering] duplicated key in ctor");
    keymap_.insert(std::make_pair(keylist[i], i));
  }
}

/* ************************************************************************** */
size_t VariableOrdering::push_back(Key key) {
  auto it_key = keymap_.find(key);
  if (it_key != keymap_.end()) {
    return std::distance(keymap_.begin(), it_key);
  } else {
    keymap_[key] = size();
    keylist_.push_back(key);
    return size() - 1;
  }
}

/* ************************************************************************** */
size_t VariableOrdering::searchKey(Key key) const {
  auto itkey = keymap_.find(key);
  if (itkey == keymap_.end()) {
    std::stringstream ss;
    ss << "[VariableOrdering::searchKey] cannot find key " << keyString(key)
       << " in variable ordering";
    throw std::runtime_error(ss.str());
  } else {
    return itkey->second;
  }
}

/* ************************************************************************** */
void VariableOrdering::print(std::ostream& out) const {
  if (size() > 0) {
    out << keyString(keylist_[0]);
    for (size_t i = 1; i < size(); i++) {
      out << ", " << keyString(keylist_[i]);
    }
  } else {
    out << "Empty VariableOrdering";
  }
  out << std::endl;
}

}  // namespace minisam
