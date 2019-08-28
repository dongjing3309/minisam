/**
 * @file    Timer.h
 * @brief   Timer
 * @author  Zhaoyang Lv, Jing Dong
 * @date    April 11, 2018
 */

#pragma once

#include <minisam/config.h>

#include <algorithm>
#include <chrono>
#include <iomanip>  // std::setw
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>  // std::pair
#include <vector>

namespace minisam {

namespace internal {
typedef std::chrono::high_resolution_clock::time_point Time;

// time now
inline internal::Time clockNow_() {
  return std::chrono::high_resolution_clock::now();
}

// get duration between two time stamp
inline long long getDuration_(const internal::Time& end,
                              const internal::Time& start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
      .count();
}
}  // namespace internal

// type of a single timer
struct Timer {
 private:
  long long t_min_, t_max_, t_last_, t_total_;
  size_t count_;
  internal::Time last_time_;

 public:
  Timer()
      : t_min_(std::numeric_limits<long long>::max()),
        t_max_(std::numeric_limits<long long>::min()),
        t_last_(0),
        t_total_(0),
        count_(0) {}

  // set current time and start counting time
  inline void tic() { last_time_ = internal::clockNow_(); }

  // Count the time of the event which has been instantiated
  inline void toc() {
    const internal::Time now = internal::clockNow_();
    t_last_ = internal::getDuration_(now, last_time_);
    t_min_ = std::min(t_min_, t_last_);
    t_max_ = std::max(t_max_, t_last_);
    t_total_ += t_last_;
    count_++;
  }

  // internal version of tic/toc
  // controlled by CMake flag MINISAM_WITH_INTERNAL_TIMING
  inline void tic_() {
#ifdef MINISAM_WITH_INTERNAL_TIMING
    tic();
#endif
  }

  inline void toc_() {
#ifdef MINISAM_WITH_INTERNAL_TIMING
    toc();
#endif
  }

  // access time statistics like min/max/avg
  inline size_t size() const { return count_; }
  inline long long sum() const { return t_total_; }
  inline long long last() const { return t_last_; }
  inline long long max() const { return t_max_; }
  inline long long min() const { return t_min_; }
};

// global timer
class GlobalTimer {
 private:
  std::map<std::string, Timer> timers_;

 public:
  GlobalTimer() = default;
  ~GlobalTimer() = default;

  // get an existing timer, if not exist add a new one
  Timer* getTimer(const std::string& name) {
    auto insert_result = timers_.insert(std::make_pair(name, Timer()));
    return &(insert_result.first->second);
  }

  // Print all time that has been recorded
  void print(std::ostream& out = std::cout) const;

  // reset timer
  void reset() { timers_.clear(); }

 private:
  // convert a long long nano second to a string with automatic unit
  // (ns/us/ms/s)
  static std::string printTimeString_(long long time_ns, int min_unit = 10);
};

// accesss global timer instance
GlobalTimer& global_timer();

}  // namespace minisam
