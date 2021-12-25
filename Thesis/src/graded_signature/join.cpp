#include <cmath>
#include <cstring>
#include <random>

#include "graded_signature/join.h"
#include "graded_signature/utility.h"

namespace graded_signature {

Cert join(Param param, Gpk &gpk, Gsk &gsk, std::unique_ptr<int64_t[]> &v) {
  std::random_device rd;
  std::uniform_int_distribution<int64_t> UB(0, 1);

  int64_t k = param.get_k();
  int64_t n = param.get_n();
  int64_t m = param.get_m();
  int64_t q = param.get_q();
  int64_t log2q = param.get_log2q();
  double sigma = param.get_sigma();
  double betaInf = param.get_betaInf();

  Cert cert;
  cert.id.reset(new int64_t[k]);
  for (int64_t i = 0; i < k; i++)
    cert.id[i] = UB(rd);
  cert.s.reset(new int64_t[2 * m]);
  for (int64_t i = 0; i < 2 * m; i++) {
    while (1) {
      cert.s[i] = sample_integer(0, sigma, std::sqrt(std::log2(m)));
      if (cert.s[i] > betaInf)
        continue;
      if (cert.s[i] < -betaInf)
        continue;
      break;
    }
  }
  cert.d.reset(new int64_t[2 * m]);
  for (int64_t i = 0; i < m; i++) {
    while (1) {
      cert.d[m + i] = sample_integer(0, sigma, std::sqrt(std::log2(m)));
      if (cert.d[m + i] > betaInf)
        continue;
      if (cert.d[m + i] < -betaInf)
        continue;
      break;
    }
  }

  std::unique_ptr<int64_t[]> w(new int64_t[2 * n]);
  std::memset(w.get(), 0, 2 * n * sizeof(int64_t));
  for (int64_t i = 0; i < 2 * n; i++) {
    for (int64_t j = 0; j < 2 * m; j++) {
      int64_t tv = v[j / log2q];
      tv = get_bit(tv, j % log2q);
      w[i] += gpk.D_0[i * 2 * m + j] * tv;
      w[i] %= q;
      w[i] += q + gpk.D_1[i * 2 * m + j] * cert.s[j] % q;
      w[i] %= q;
    }
  }

  std::unique_ptr<int64_t[]> u(new int64_t[n]);
  std::memcpy(u.get(), gpk.u.get(), n * sizeof(int64_t));
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < m; j++) {
      int64_t tw = w[j / log2q];
      tw = get_bit(tw, j % log2q);
      u[i] += gpk.D[i * m + j] * tw;
      u[i] %= q;
      u[i] += q - gpk.A_k[i * m + j] * cert.d[m + j] % q;
      u[i] %= q;
      for (int64_t h = 0; h < k; h++) {
        if (cert.id[h] == 0)
          continue;
        u[i] += q - gpk.A_k[(h + 1) * n * m + i * m + j] * cert.d[m + j] % q;
        u[i] %= q;
      }
    }
  }

  std::unique_ptr<int64_t[]> x =
      var_sample_gauss(n, q, gsk.inv_A, gsk.T_A, gsk.T_A_, sigma, u);
  std::memcpy(cert.d.get(), x.get(), m * sizeof(int64_t));

  return cert;
}

} // namespace graded_signature
