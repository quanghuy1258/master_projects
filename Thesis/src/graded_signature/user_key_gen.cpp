#include <cmath>
#include <cstring>

#include "graded_signature/user_key_gen.h"
#include "graded_signature/utility.h"

namespace graded_signature {

void user_key_gen(int64_t n, int64_t q, double sigma,
                  std::unique_ptr<int64_t[]> &F, std::unique_ptr<int64_t[]> &v,
                  std::unique_ptr<int64_t[]> &z) {
  int64_t m = 2 * n * ceil_log2(q);
  double betaInf = sigma * std::sqrt(std::log2(m));
  z.reset(new int64_t[4 * m]);
  for (int64_t i = 0; i < 4 * m; i++) {
    while (1) {
      z[i] = sample_integer(0, sigma, std::sqrt(std::log2(m)));
      if (z[i] > betaInf)
        continue;
      if (z[i] < -betaInf)
        continue;
      break;
    }
  }

  v.reset(new int64_t[4 * n]);
  std::memset(v.get(), 0, 4 * n * sizeof(int64_t));
  for (int64_t i = 0; i < 4 * n; i++) {
    for (int64_t j = 0; j < 4 * m; j++) {
      v[i] += q + F[i * 4 * m + j] * z[j] % q;
      v[i] %= q;
    }
  }
}

} // namespace graded_signature
