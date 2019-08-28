/**
 * @file    SchurComplement.cpp
 * @brief   schur complement related
 * @author  Jing Dong
 * @date    Jun 26, 2019
 */

#include <minisam/core/SchurComplement.h>

#include <minisam/core/Variables.h>

#include <algorithm>

namespace minisam {

/* ************************************************************************** */
VariablesToEliminate::VariablesToEliminate() : eliminate_any_(false) {
  // by default initialize everyone not eliminate
  for (size_t i = 0; i < 256; i++) {
    map_key_[i] = false;
  }
}

/* ************************************************************************** */
void VariablesToEliminate::print(std::ostream& out) const {
  out << "Variables to be eliminated: ";
  if (isEliminatedAny()) {
    out << "variable key start with: ";
    for (size_t i = 0; i < 256; i++) {
      if (map_key_[i]) {
        out << "\'" << static_cast<unsigned char>(i) << "\', ";
      }
    }
  } else {
    out << "None";
  }
  out << std::endl;
}

/* ************************************************************************** */
SchurComplementOrdering::SchurComplementOrdering(
    const VariableOrdering& origin_ordering,
    const VariablesToEliminate& vars_to_eliminate, const Variables& variables) {
  // sort ordering list to have explicit variables before elimiated variables
  std::vector<Key> keys_ordered = origin_ordering.keys();
  std::sort(keys_ordered.begin(), keys_ordered.end(),
            [&vars_to_eliminate](Key key1, Key key2) {
              // key1 < key2  <->  key1 is not eliminated, key2 is eliminated
              return !vars_to_eliminate.isVariableEliminated(keyChar(key1)) &&
                     vars_to_eliminate.isVariableEliminated(keyChar(key2));
            });
  ordering_ = VariableOrdering(keys_ordered);

  // check first variable must not be elimiated
  assert(!vars_to_eliminate.isVariableEliminated(ordering_[0]));

  // get reduced / eliminated system info
  rs_until_idx_ = 0;
  int var_dim_accumulate = 0;
  bool reduced_sys_done = false;

  for (size_t i = 0; i < ordering_.size(); i++) {
    const Key key = ordering_[i];
    const int var_dim = static_cast<int>(variables.at(key)->dim());

    // check whether done with elimiated system
    if (!reduced_sys_done &&
        vars_to_eliminate.isVariableEliminated(keyChar(key))) {
      reduced_sys_done = true;
      // reserve space once done
      var_dims_.reserve(ordering_.size() - i);
      var_pos_in_esys_.reserve(ordering_.size() - i);
    }

    if (reduced_sys_done) {
      // elem system info
      var_dims_.push_back(var_dim);
      var_pos_in_esys_.push_back(var_dim_accumulate);
      var_dim_accumulate += var_dim;
    } else {
      // reduced system info
      rs_until_idx_ += var_dim;
    }
  }

  // check rs_until_idx_ in normal range
  assert(rs_until_idx_ > 0);
  assert(rs_until_idx_ <= static_cast<int>(variables.dim()));
}

}  // namespace minisam
