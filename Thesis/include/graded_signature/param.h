#ifndef GRADED_SIGNATURE_PARAM_H
#define GRADED_SIGNATURE_PARAM_H

#include <cstdint>

namespace graded_signature {

struct Param {
  int64_t k;
  int64_t n;
  int64_t q;
  double sigma;
  int64_t t;

  int64_t get_k();
  int64_t get_N();
  int64_t get_n();
  int64_t get_m();
  int64_t get_q();
  double get_sigma();
  double get_beta2();
  double get_betaInf();
  int64_t get_t();
};

} // namespace graded_signature

#endif
