/**
 * @file    Key.h
 * @brief   Index variable by interger
 * @author  Jing Dong
 * @date    Oct 15, 2017
 */

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace minisam {

// Key is just size_type unsigned interger
typedef size_t Key;

/**
 * formulate Key in char + uint
 */

// some const
static constexpr size_t indexBits = (sizeof(Key) - sizeof(unsigned char)) * 8;
static constexpr Key charMask = Key(std::numeric_limits<unsigned char>::max())
                                << indexBits;
static constexpr Key indexMask = ~charMask;  // also max index

// convert char + uint form in Key
inline Key key(unsigned char c, size_t i) {
  // char saved first 8 bits
  // check index size
  if (i > indexMask) {
    throw std::invalid_argument("[key] index too large");
  }
  return (static_cast<size_t>(c) << indexBits) | i;
}

// get char from key
inline unsigned char keyChar(Key key) {
  return static_cast<unsigned char>((key & charMask) >> indexBits);
}

// get index from key
inline size_t keyIndex(Key key) { return key & indexMask; }

// convert key to string to format "x0" for print
// only print char if char in 31 < char < 127
// if not in range, just output key itself as index
std::string keyString(Key key);

}  // namespace minisam
