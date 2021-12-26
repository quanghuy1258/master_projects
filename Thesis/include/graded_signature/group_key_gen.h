#ifndef GRADED_SIGNATURE_GROUP_KEY_GEN_H
#define GRADED_SIGNATURE_GROUP_KEY_GEN_H

#include <cstdint>
#include <memory>

#include "graded_signature/param.h"
#include "graded_signature/utility.h"

namespace graded_signature {

struct Gpk {
  std::unique_ptr<int64_t[]> A;
  std::unique_ptr<int64_t[]> A_k;
  std::unique_ptr<int64_t[]> D;
  std::unique_ptr<int64_t[]> D_0;
  std::unique_ptr<int64_t[]> D_1;
  std::unique_ptr<int64_t[]> u;
  std::unique_ptr<int64_t[]> B;
  std::unique_ptr<int64_t[]> F;
  PseudoMatrix comm_matrix;
};

struct Gsk {
  std::unique_ptr<int64_t[]> inv_A;
  std::unique_ptr<int64_t[]> T_A;
  std::unique_ptr<double[]> T_A_;
};

void group_key_gen(Param param, Gpk &gpk, Gsk &gsk);

} // namespace graded_signature

#endif
