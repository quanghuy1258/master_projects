#include <cstring>
#include <memory>
#include <random>

#include "graded_signature/sign.h"
#include "graded_signature/utility.h"

namespace graded_signature {

void create_P(int64_t k, int64_t n, int64_t m, int64_t q, int64_t log2q,
              int64_t intBetaInf, int64_t logBetaInf, Gpk &gpk,
              std::function<std::unique_ptr<int64_t[]>(int64_t *)> &P) {
  P = [ k, n, m, q, log2q, intBetaInf, logBetaInf, &
        gpk ](int64_t * x) -> std::unique_ptr<int64_t[]> {
    std::unique_ptr<int64_t[]> v(new int64_t[9 * n]);
    std::memset(v.get(), 0, 9 * n * sizeof(int64_t));

    int64_t *v1 = v.get();
    int64_t *v2 = v.get() + n;
    int64_t *v3 = v.get() + 3 * n;
    int64_t *v4 = v.get() + 7 * n;

    std::unique_ptr<int64_t[]> x_betaInf(new int64_t[m * (k + 8)]);
    for (int64_t i = 0; i < m * (k + 8); i++)
      x_betaInf[i] =
          compose(x + i * (1 + logBetaInf), q, intBetaInf, logBetaInf);
    int64_t *x_s = x_betaInf.get();
    int64_t *x_d1 = x_betaInf.get() + 2 * m;
    int64_t *x_d2 = x_betaInf.get() + 3 * m;
    int64_t *x_id = x_betaInf.get() + 4 * m;
    int64_t *x_z = x_betaInf.get() + m * (k + 4);
    int64_t *x_binv = x + m * (1 + logBetaInf) * (k + 8);
    int64_t *x_ov = x + m * ((1 + logBetaInf) * (k + 8) + 2);
    int64_t *x_w = x + m * ((1 + logBetaInf) * (k + 8) + 4);

    for (int64_t i = 0; i < n; i++) {
      for (int64_t j = 0; j < m; j++) {
        v1[i] += gpk.A[i * m + j] * x_d1[j] % q;
        v1[i] %= q;
        v1[i] += gpk.A_k[i * m + j] * x_d2[j] % q;
        v1[i] %= q;
        for (int64_t h = 0; h < k; h++) {
          v1[i] += gpk.A_k[(h + 1) * n * m + i * m + j] * x_id[h * m + j] % q;
          v1[i] %= q;
        }
        v1[i] += q - gpk.D[i * m + j] * x_w[j] % q;
        v1[i] %= q;
      }
    }
    for (int64_t i = 0; i < 2 * n; i++) {
      for (int64_t j = 0; j < log2q; j++) {
        v2[i] += x_w[i * log2q + j] * (1 << j) % q;
        v2[i] %= q;
      }
      for (int64_t j = 0; j < 2 * m; j++) {
        v2[i] += q - gpk.D_0[i * 2 * m + j] * x_binv[j] % q;
        v2[i] %= q;
        v2[i] += q - gpk.D_1[i * 2 * m + j] * x_s[j] % q;
        v2[i] %= q;
      }
    }
    for (int64_t i = 0; i < 4 * n; i++) {
      for (int64_t j = 0; j < log2q; j++) {
        v3[i] += x_binv[i * log2q + j] * (1 << j) % q;
        v3[i] %= q;
      }
      for (int64_t j = 0; j < 4 * m; j++) {
        v3[i] += q - gpk.F[i * 4 * m + j] * x_z[j] % q;
        v3[i] %= q;
      }
    }
    for (int64_t i = 0; i < 2 * n; i++) {
      for (int64_t j = 0; j < 2 * m; j++) {
        v4[i] += gpk.B[i * 4 * m + j] * x_binv[j] % q;
        v4[i] %= q;
        v4[i] += gpk.B[i * 4 * m + 2 * m + j] * x_ov[j] % q;
        v4[i] %= q;
      }
    }
    return v;
  };
}

USign sign(Param &param, Gpk &gpk, Cert &cert, std::unique_ptr<int64_t[]> &v,
           std::unique_ptr<int64_t[]> &z, std::string &msg) {
  std::random_device rd;
  std::uniform_int_distribution<int64_t> UB(0, 1);

  int64_t k = param.get_k();
  int64_t n = param.get_n();
  int64_t m = param.get_m();
  int64_t q = param.get_q();
  int64_t log2q = param.get_log2q();
  int64_t intBetaInf = param.get_intBetaInf();
  int64_t logBetaInf = param.get_logBetaInf();
  int64_t t = param.get_t();

  int64_t L = m * ((1 + logBetaInf) * (k + 8) + 5);
  std::unique_ptr<int64_t[]> x(new int64_t[2 * L]);
  std::memset(x.get(), 0, L * sizeof(int64_t));
  int64_t *x_ptr = x.get();
  // s
  for (int64_t i = 0; i < 2 * m; i++) {
    decompose(x_ptr, cert.s[i], intBetaInf, logBetaInf);
    x_ptr += 1 + logBetaInf;
  }
  // d
  for (int64_t i = 0; i < 2 * m; i++) {
    decompose(x_ptr, cert.d[i], intBetaInf, logBetaInf);
    x_ptr += 1 + logBetaInf;
  }
  // id
  for (int64_t i = 0; i < k; i++) {
    for (int64_t j = 0; j < m; j++) {
      if (cert.id[i])
        decompose(x_ptr, cert.d[m + j], intBetaInf, logBetaInf);
      x_ptr += 1 + logBetaInf;
    }
  }
  // z
  for (int64_t i = 0; i < 4 * m; i++) {
    decompose(x_ptr, z[i], intBetaInf, logBetaInf);
    x_ptr += 1 + logBetaInf;
  }
  // bin_v
  for (int64_t i = 0; i < 2 * m; i++) {
    int64_t temp = v[i / log2q];
    x_ptr[i] = get_bit(temp, i % log2q);
  }
  // o_v
  for (int64_t i = 0; i < 2 * m; i++)
    x_ptr[2 * m + i] = UB(rd);
  // w
  for (int64_t i = 0; i < 2 * n; i++) {
    int64_t w = 0;
    for (int64_t j = 0; j < 2 * m; j++) {
      w += gpk.D_0[i * 2 * m + j] * x_ptr[j];
      w %= q;
      w += q + gpk.D_1[i * 2 * m + j] * cert.s[j] % q;
      w %= q;
    }
    for (int64_t j = 0; j < log2q; j++)
      x_ptr[4 * m + i * log2q + j] = get_bit(w, j);
  }

  USign usign;
  usign.cv.reset(new int64_t[2 * n]);
  usign.ov.reset(new int64_t[2 * m]);
  // c_v, o_v
  std::memset(usign.cv.get(), 0, 2 * n * sizeof(int64_t));
  for (int64_t i = 0; i < 2 * n; i++) {
    for (int64_t j = 0; j < 2 * m; j++) {
      usign.cv[i] += gpk.B[i * 4 * m + j] * x_ptr[j];
      usign.cv[i] %= q;
      usign.cv[i] += gpk.B[i * 4 * m + 2 * m + j] * x_ptr[2 * m + j];
      usign.cv[i] %= q;
    }
  }
  std::memcpy(usign.ov.get(), x_ptr + 2 * m, 2 * m * sizeof(int64_t));

  // VALID
  x_ptr = x.get();
  for (int64_t i = 0; i < L; i++)
    x[L + i] = 1 - x[i];

  // P
  std::function<std::unique_ptr<int64_t[]>(int64_t *)> P;
  create_P(k, n, m, q, log2q, intBetaInf, logBetaInf, gpk, P);

  Hash rom;
  std::unique_ptr<unsigned char[]> msg_data(new unsigned char[msg.length()]);
  std::memcpy(msg_data.get(), msg.c_str(), msg.length());
  rom.update(msg_data.get(), msg.length());
  rom.update(usign.cv.get(), 2 * n * sizeof(int64_t));

  usign.zkp = generate_zkp(t, 9 * n, L, q, x, P, gpk.comm_matrix, n, rom);

  return usign;
}

bool verify(Param &param, Gpk &gpk, std::string &msg, USign &usign) {
  int64_t k = param.get_k();
  int64_t n = param.get_n();
  int64_t m = param.get_m();
  int64_t q = param.get_q();
  int64_t log2q = param.get_log2q();
  int64_t intBetaInf = param.get_intBetaInf();
  int64_t logBetaInf = param.get_logBetaInf();
  int64_t t = param.get_t();

  int64_t L = m * ((1 + logBetaInf) * (k + 8) + 5);

  std::unique_ptr<int64_t[]> y(new int64_t[9 * n]);
  std::memcpy(y.get(), gpk.u.get(), n * sizeof(int64_t));
  std::memset(y.get() + n, 0, 6 * n * sizeof(int64_t));
  std::memcpy(y.get() + 7 * n, usign.cv.get(), 2 * n * sizeof(int64_t));

  std::function<std::unique_ptr<int64_t[]>(int64_t *)> P;
  create_P(k, n, m, q, log2q, intBetaInf, logBetaInf, gpk, P);

  Hash rom;
  std::unique_ptr<unsigned char[]> msg_data(new unsigned char[msg.length()]);
  std::memcpy(msg_data.get(), msg.c_str(), msg.length());
  rom.update(msg_data.get(), msg.length());
  rom.update(usign.cv.get(), 2 * n * sizeof(int64_t));

  return verify_zkp(t, 9 * n, L, q, y, P, gpk.comm_matrix, n, rom, usign.zkp);
}

} // namespace graded_signature
