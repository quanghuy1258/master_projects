#ifndef GRADED_SIGNATURE_USER_KEY_GEN_H
#define GRADED_SIGNATURE_USER_KEY_GEN_H

#include <cstdint>
#include <memory>

namespace graded_signature {

void user_key_gen(int64_t n, int64_t q, double sigma,
                  std::unique_ptr<int64_t[]> &F, std::unique_ptr<int64_t[]> &v,
                  std::unique_ptr<int64_t[]> &z);

} // namespace graded_signature

#endif
