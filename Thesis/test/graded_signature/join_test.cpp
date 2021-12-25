#include <cstring>
#include <iostream>
#include <memory>

#include "gtest/gtest.h"

#include "graded_signature/group_key_gen.h"
#include "graded_signature/join.h"
#include "graded_signature/param.h"
#include "graded_signature/user_key_gen.h"
#include "graded_signature/utility.h"

TEST(Join, Join) {
  graded_signature::Param param;
  param.k = 8;
  param.n = 16;
  param.q = 131063;
  param.sigma = 100;
  param.t = 10;
  std::cout << "param: done" << std::endl;

  int64_t k = param.get_k();
  int64_t n = param.get_n();
  int64_t m = param.get_m();
  int64_t q = param.get_q();
  int64_t log2q = param.get_log2q();
  double beta2 = param.get_beta2();
  double betaInf = param.get_betaInf();

  graded_signature::Gpk gpk;
  graded_signature::Gsk gsk;
  graded_signature::group_key_gen(param, gpk, gsk);
  std::cout << "group key gen: done" << std::endl;

  std::unique_ptr<int64_t[]> upk, usk;
  graded_signature::user_key_gen(param.get_n(), param.get_q(),
                                 param.get_sigma(), gpk.F, upk, usk);
  std::cout << "user key gen: done" << std::endl;

  graded_signature::Cert cert = graded_signature::join(param, gpk, gsk, upk);
  std::cout << "join: done" << std::endl;

  std::unique_ptr<int64_t[]> w;
  w.reset(new int64_t[2 * n]);
  std::memset(w.get(), 0, 2 * n * sizeof(int64_t));
  for (int64_t i = 0; i < 2 * n; i++) {
    for (int64_t j = 0; j < 2 * m; j++) {
      int64_t t_upk = upk[j / log2q];
      t_upk = graded_signature::get_bit(t_upk, j % log2q);
      w[i] += gpk.D_0[i * 2 * m + j] * t_upk;
      w[i] %= q;
      w[i] += q + gpk.D_1[i * 2 * m + j] * cert.s[j] % q;
      w[i] %= q;
    }
  }

  for (int64_t i = 0; i < n; i++) {
    int64_t LHS = 0;
    int64_t RHS = gpk.u[i];
    for (int64_t j = 0; j < m; j++) {
      // LHS
      LHS += q + gpk.A[i * m + j] * cert.d[j] % q;
      LHS %= q;
      LHS += q + gpk.A_k[i * m + j] * cert.d[m + j] % q;
      LHS %= q;
      for (int64_t h = 0; h < k; h++) {
        if (cert.id[h] == 0)
          continue;
        LHS += q + gpk.A_k[(h + 1) * m * n + i * m + j] * cert.d[m + j] % q;
        LHS %= q;
      }

      // RHS
      int64_t t_w = w[j / log2q];
      t_w = graded_signature::get_bit(t_w, j % log2q);
      RHS += gpk.D[i * m + j] * t_w;
      RHS %= q;
    }
    EXPECT_EQ(LHS, RHS);
  }

  // cert.id is on {0, 1}
  // cert.s, cert.d are on [-betaInf, betaInf]
  for (int64_t i = 0; i < k; i++)
    EXPECT_TRUE(cert.id[i] == 0 || cert.id[i] == 1);
  double s, d;
  s = d = 0;
  for (int64_t i = 0; i < 2 * m; i++) {
    s += cert.s[i] * cert.s[i];
    d += cert.d[i] * cert.d[i];
    EXPECT_FALSE(cert.s[i] > betaInf || cert.s[i] < -betaInf);
    EXPECT_FALSE(cert.d[i] > betaInf || cert.d[i] < -betaInf);
  }
  EXPECT_LT(s, beta2 * beta2);
  EXPECT_LT(d, beta2 * beta2);
}
