/**
 * @file    Traits.h
 * @brief   miniSAM traits defines manifold and Lie group related behavior
 * @author  Jing Dong
 * @date    Oct 13, 2017
 */

#pragma once

#include <type_traits>

namespace minisam {

// default traits declaration for general manifold / lie groups
template <typename T>
struct traits {};

// traits tags for compile-time type assert
// typedef traits<T>::type_category with approperate tag

// traits for manifold
struct manifold_tag {};

// traits for Lie group, which is belonged to manifold
struct lie_group_tag : manifold_tag {};

// traits for camera intrinsics calibration, which is manifold
struct camera_intrinsics_tag : manifold_tag {};

// checker for type traits exists (by tag)
template <typename T>
class has_traits {
 private:
  template <typename C>
  static auto test_tag_(typename traits<C>::type_category*)
      -> decltype(typename traits<C>::type_category(),
                  std::true_type());  // SFINAE
  template <typename C>
  static std::false_type test_tag_(...);

 public:
  static constexpr bool value =
      std::is_same<std::true_type, decltype(test_tag_<T>(nullptr))>::value;
};

// checkers for type traits is manifold type
template <typename T>
class is_manifold {
 private:
  template <typename C>
  static auto test_tag_(typename traits<C>::type_category*) -> decltype(
      typename std::enable_if<std::is_base_of<
          manifold_tag, typename traits<C>::type_category>::value>::type(),
      std::true_type());  // SFINAE
  template <typename C>
  static std::false_type test_tag_(...);

 public:
  static constexpr bool value =
      std::is_same<std::true_type, decltype(test_tag_<T>(nullptr))>::value;
};

// checkers for type traits is lie group type
template <typename T>
class is_lie_group {
 private:
  template <typename C>
  static auto test_tag_(typename traits<C>::type_category*) -> decltype(
      typename std::enable_if<std::is_base_of<
          lie_group_tag, typename traits<C>::type_category>::value>::type(),
      std::true_type());  // SFINAE
  template <typename C>
  static std::false_type test_tag_(...);

 public:
  static constexpr bool value =
      std::is_same<std::true_type, decltype(test_tag_<T>(nullptr))>::value;
};

// checkers for type traits is camera intrinsics type
template <typename T>
class is_camera_intrinsics {
 private:
  template <typename C>
  static auto test_tag_(typename traits<C>::type_category*) -> decltype(
      typename std::enable_if<
          std::is_base_of<camera_intrinsics_tag,
                          typename traits<C>::type_category>::value>::type(),
      std::true_type());  // SFINAE
  template <typename C>
  static std::false_type test_tag_(...);

 public:
  static constexpr bool value =
      std::is_same<std::true_type, decltype(test_tag_<T>(nullptr))>::value;
};

}  // namespace minisam
