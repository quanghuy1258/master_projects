#ifndef GRADED_SIGNATURE_ZKP_H
#define GRADED_SIGNATURE_ZKP_H

#include <cstdint>
#include <functional>
#include <memory>

#include "graded_signature/hash.h"
#include "graded_signature/utility.h"

namespace graded_signature {

struct ZKP {
  std::unique_ptr<int64_t[]> comm;
  std::unique_ptr<int64_t[]> resp;
};

ZKP generate_zkp(int64_t t, int64_t D, int64_t L, int64_t q,
                 std::unique_ptr<int64_t[]> &x,
                 std::function<std::unique_ptr<int64_t[]>(int64_t *)> &P,
                 PseudoMatrix &A, int64_t n, Hash &rom);
bool verify_zkp(int64_t t, int64_t D, int64_t L, int64_t q,
                std::unique_ptr<int64_t[]> &v,
                std::function<std::unique_ptr<int64_t[]>(int64_t *)> &P,
                PseudoMatrix &A, int64_t n, Hash &rom, ZKP &zkp);

} // namespace graded_signature

#endif
