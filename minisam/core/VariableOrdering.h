/**
 * @file    VariableOrdering.h
 * @brief   Class represent variable ordering, with fast searching
 * @author  Jing Dong
 * @date    Oct 14, 2017
 */

#pragma once

#include <minisam/core/Key.h>

#include <iostream>
#include <unordered_map>
#include <vector>

namespace minisam {

// variable ordering is a list of keys, with fast searching key list position
class VariableOrdering {
 private:
  std::vector<Key> keylist_;
  // for fast searching key list position
  std::unordered_map<Key, size_t> keymap_;

 public:
  VariableOrdering() = default;
  explicit VariableOrdering(const std::vector<Key>& keylist);

  ~VariableOrdering() = default;

  // size
  size_t size() const { return keylist_.size(); }

  // i-th key
  inline Key key(size_t index) const { return keylist_[index]; }
  inline Key operator[](size_t index) const { return this->key(index); }

  // add a new key to the end of the key list
  // no throw, in key exists in list, the list itself won't change
  // return current position in the key list
  size_t push_back(Key key);

  // access key vector
  const std::vector<Key>& keys() const { return keylist_; }

  // search key's position
  // throw if cannot find
  size_t searchKey(Key key) const;

  // unsafe but fast searchKey, only in C++ and should internal use only
  inline size_t searchKeyUnsafe(Key key) const { return keymap_.at(key); }

  // print
  void print(std::ostream& out = std::cout) const;
};

}  // namespace minisam
