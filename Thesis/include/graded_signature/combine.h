#ifndef GRADED_SIGNATURE_COMBINE_H
#define GRADED_SIGNATURE_COMBINE_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "graded_signature/sign.h"
#include "graded_signature/zkp.h"

namespace graded_signature {

struct UpkSign {
  std::unique_ptr<int64_t[]> *upk;
  USign *s;
};

struct GSign {
  std::unique_ptr<int64_t[]> cv;
  ZKP zkp;
};

std::function<bool(const UpkSign &a, const UpkSign &b)>
compare_UpkSign(int64_t n);

GSign combine(Param &param, Gpk &gpk, int64_t l, std::unique_ptr<int64_t[]> &v,
              std::unique_ptr<int64_t[]> &ov, std::string &msg);
bool verify(Param &param, Gpk &gpk, int64_t l, std::string &msg, GSign &gsign);

} // namespace graded_signature

#endif
