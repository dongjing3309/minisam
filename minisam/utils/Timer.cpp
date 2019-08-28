/**
 * @file    Timer.h
 * @brief   Timer
 * @author  Zhaoyang Lv, Jing Dong
 * @date    April 12, 2019
 */

#include <minisam/utils/Timer.h>

namespace minisam {

/* ************************************************************************** */
GlobalTimer& global_timer() {
  static GlobalTimer global_timer;
  return global_timer;
}

/* ************************************************************************** */
void GlobalTimer::print(std::ostream& out) const {
  using std::endl;
  using std::setw;
  using std::left;
  using std::right;

  out << left << setw(32) << "Profiling item" << right << setw(10) << "Total"
      << setw(8) << "Freq" << setw(10) << "Avg" << setw(10) << "Max" << setw(10)
      << "Min" << endl;
  out << std::setfill('-') << setw(80) << "-" << endl;
  out << std::setfill(' ');

  for (auto it = timers_.begin(); it != timers_.end(); it++) {
    if (it->second.size() > 0) {
      out << left << setw(32) << it->first;
      out << right << setw(10) << printTimeString_(it->second.sum());
      out << setw(8) << it->second.size();
      out << setw(10)
          << printTimeString_(it->second.sum() / (long long)it->second.size());
      out << setw(10) << printTimeString_(it->second.max());
      out << setw(10) << printTimeString_(it->second.min());
      out << endl;
    }
  }
}

/* ************************************************************************** */
std::string GlobalTimer::printTimeString_(long long time_ns, int min_unit) {
  std::stringstream ss;
  if (time_ns < min_unit * 1000ll) {
    ss << time_ns << " ns";
  } else if (time_ns < min_unit * 1000000ll) {
    ss << time_ns / 1000ll << " us";
  } else if (time_ns < min_unit * 1000000000ll) {
    ss << std::setprecision(4) << double(time_ns) / 1e6 << " ms";
  } else {
    ss << std::setprecision(4) << double(time_ns) / 1e9 << " s ";
  }
  return ss.str();
}

}  // namespace minisam
