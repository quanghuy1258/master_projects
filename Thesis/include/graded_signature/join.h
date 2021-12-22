#ifndef GRADED_SIGNATURE_JOIN_H
#define GRADED_SIGNATURE_JOIN_H

#include <cstdint>
#include <memory>

#include "graded_signature/group_key_gen.h"
#include "graded_signature/param.h"

namespace graded_signature {

struct Cert {
  std::unique_ptr<int64_t[]> id;
  std::unique_ptr<int64_t[]> s;
  std::unique_ptr<int64_t[]> d;
};

Cert join(Param param, Gpk &gpk, Gsk &gsk, std::unique_ptr<int64_t[]> &v);

} // namespace graded_signature

#endif
