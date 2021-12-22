#ifndef GRADED_SIGNATURE_HASH_H
#define GRADED_SIGNATURE_HASH_H

#include <cstdint>
#include <memory>

#include <openssl/sha.h>

namespace graded_signature {

struct Hash {
  SHA512_CTX ctx;

  Hash();
  void update(void *data, int64_t size);
  std::unique_ptr<int64_t[]> digest(int64_t t);
};

} // namespace graded_signature

#endif
