#include <cstring>

#include "graded_signature/hash.h"

namespace graded_signature {

Hash::Hash() { SHA512_Init(&ctx); }

void Hash::update(void *data, int64_t size) { SHA512_Update(&ctx, data, size); }

std::unique_ptr<int64_t[]> Hash::digest(int64_t t) {
  unsigned char md[64];
  SHA512_Final(md, &ctx);

  ctx = SHA512_CTX();
  SHA512_Init(&ctx);

  std::unique_ptr<int64_t[]> h(new int64_t[t]);
  std::memset(h.get(), 0, t * sizeof(int64_t));
  // 323 == std::floor(512 / std::log2(3));
  for (int64_t i = 0; i < 323; i++) {
    int64_t temp = 0;
    for (int64_t j = 0; j < 64; j++) {
      temp <<= 8;
      temp |= md[j];
      md[j] = 0;
      md[j] |= temp / 3;
      temp %= 3;
    }
    h[i % t] = (h[i % t] + temp) % 3;
  }
  return h;
}

} // namespace graded_signature
