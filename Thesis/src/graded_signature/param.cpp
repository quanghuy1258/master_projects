#include <cmath>

#include "graded_signature/param.h"
#include "graded_signature/utility.h"

namespace graded_signature {

int64_t Param::get_k() { return k; }
int64_t Param::get_N() {
  int64_t N = 1;
  N <<= k;
  return N;
}
int64_t Param::get_n() { return n; }
int64_t Param::get_m() { return 2 * n * get_log2q(); }
int64_t Param::get_q() { return q; }
int64_t Param::get_log2q() { return ceil_log2(q); }
double Param::get_sigma() { return sigma; }
double Param::get_beta2() { return sigma * std::sqrt(2 * get_m()); }
double Param::get_betaInf() { return sigma * std::sqrt(std::log2(get_m())); }
int64_t Param::get_intBetaInf() { return std::floor(get_betaInf()); }
int64_t Param::get_logBetaInf() { return ceil_log2(get_intBetaInf() + 1); }
int64_t Param::get_t() { return t; }

} // namespace graded_signature
