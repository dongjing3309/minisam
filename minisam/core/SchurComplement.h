/**
 * @file    SchurComplement.h
 * @brief   schur complement related
 * @author  Jing Dong
 * @date    Jun 26, 2019
 */

#pragma once

#include <minisam/core/Key.h>
#include <minisam/core/VariableOrdering.h>

#include <array>
#include <iostream>
#include <memory>
#include <vector>

namespace minisam {

// forward declearation
class Variables;

// variables to be eliminated by schur complement
class VariablesToEliminate {
 private:
  bool eliminate_any_;  // whether eliminate any variable
  std::array<bool, 256> map_key_;

 public:
  VariablesToEliminate();

  ~VariablesToEliminate() = default;

  // print
  void print(std::ostream& out = std::cout) const;

  // eliminate variable by char
  inline void eliminate(unsigned char c) {
    eliminate_any_ = true;
    map_key_[c] = true;
  }

  // whether eliminate any variable
  // use to check whether schur complement is needed at all
  inline bool isEliminatedAny() const { return eliminate_any_; }

  // whether a variable is eliminated
  inline bool isVariableEliminated(unsigned char c) const {
    return map_key_[c];
  }
};

// ordering information for schur complement
class SchurComplementOrdering {
 private:
  VariableOrdering ordering_;
  int rs_until_idx_;
  std::vector<int> var_dims_;
  std::vector<int> var_pos_in_esys_;

 public:
  SchurComplementOrdering(const VariableOrdering& origin_ordering,
                          const VariablesToEliminate& vars_to_eliminate,
                          const Variables& variables);

  ~SchurComplementOrdering() = default;

  // access oridering
  // always explicit variables before elimiated variables
  const VariableOrdering& ordering() const { return ordering_; }

  // reduced system until index
  int reducedSysUntilIndex() const { return rs_until_idx_; }

  // eliminated variables size
  size_t eliminatedVariableSize() const { return var_dims_.size(); }

  // eliminated variables dimension in elimiated system
  int eliminatedVariableDim(size_t i) const { return var_dims_[i]; }
  // eliminated variables position in elimiated system
  int eliminatedVariablePosition(size_t i) const { return var_pos_in_esys_[i]; }
};

}  // namespace minisam
