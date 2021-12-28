#include <algorithm>
#include <cstring>
#include <random>

#include "gtest/gtest.h"

#include "graded_signature/combine.h"
#include "graded_signature/group_key_gen.h"
#include "graded_signature/join.h"
#include "graded_signature/param.h"
#include "graded_signature/sign.h"
#include "graded_signature/user_key_gen.h"
#include "graded_signature/utility.h"

TEST(Combine, Comparison) {
  int64_t n = 16;
  int64_t q = 131063;
  int64_t log2q = graded_signature::ceil_log2(q);

  std::random_device rd;
  std::uniform_int_distribution<int64_t> UZq(0, q - 1);

  std::unique_ptr<int64_t[]> upk[8];
  for (int64_t i = 0; i < 5; i++) {
    upk[i].reset(new int64_t[4 * n]);
    for (int64_t j = 0; j < 4 * n; j++)
      upk[i][j] = UZq(rd);
  }
  for (int64_t i = 5; i < 8; i++) {
    upk[i].reset(new int64_t[4 * n]);
    std::memcpy(upk[i].get(), upk[i - 5].get(), 4 * n * sizeof(int64_t));
  }

  graded_signature::UpkSign upk_sign[8];
  for (int64_t i = 0; i < 8; i++) {
    upk_sign[i].upk = upk + i;
    upk_sign[i].s = nullptr;
  }

  std::sort(upk_sign, upk_sign + 8, graded_signature::compare_UpkSign(n));

  for (int64_t i = 0; i < 7; i++) {
    for (int64_t j = 4 * n * log2q - 1; j >= 0; j--) {
      int64_t a = upk_sign[i].upk[0][j / log2q];
      int64_t b = upk_sign[i + 1].upk[0][j / log2q];
      a = graded_signature::get_bit(a, j % log2q);
      b = graded_signature::get_bit(b, j % log2q);
      if (a == b)
        continue;
      EXPECT_LT(a, b);
      break;
    }
  }
}

TEST(Combine, Combine) {
  graded_signature::Param param;
  param.k = 8;
  param.n = 16;
  param.q = 131063;
  param.sigma = 100;
  param.t = 10;
  std::cout << "param: done" << std::endl;

  int64_t n = param.get_n();
  int64_t m = param.get_m();
  int64_t q = param.get_q();
  int64_t log2q = param.get_log2q();

  std::random_device rd;
  std::uniform_int_distribution<int64_t> UZlog2q(0, (1 << log2q) - 1);

  graded_signature::Gpk gpk;
  graded_signature::Gsk gsk;
  graded_signature::group_key_gen(param, gpk, gsk);
  std::cout << "group key gen: done" << std::endl;

  std::unique_ptr<int64_t[]> upk1, usk1;
  graded_signature::user_key_gen(param.get_n(), param.get_q(),
                                 param.get_sigma(), gpk.F, upk1, usk1);
  std::unique_ptr<int64_t[]> upk2, usk2;
  graded_signature::user_key_gen(param.get_n(), param.get_q(),
                                 param.get_sigma(), gpk.F, upk2, usk2);
  std::unique_ptr<int64_t[]> upk3, usk3;
  graded_signature::user_key_gen(param.get_n(), param.get_q(),
                                 param.get_sigma(), gpk.F, upk3, usk3);
  std::cout << "user key gen: done" << std::endl;

  graded_signature::Cert cert1 = graded_signature::join(param, gpk, gsk, upk1);
  graded_signature::Cert cert2 = graded_signature::join(param, gpk, gsk, upk2);
  graded_signature::Cert cert3 = graded_signature::join(param, gpk, gsk, upk3);
  graded_signature::USign sign1, sign2, sign3;
  std::cout << "join: done" << std::endl;

  std::string msg = "Graded Signature";
  graded_signature::UpkSign upk_sign[3];
  upk_sign[0].upk = &upk1;
  upk_sign[0].s = &sign1;
  upk_sign[1].upk = &upk2;
  upk_sign[1].s = &sign2;
  upk_sign[2].upk = &upk3;
  upk_sign[2].s = &sign3;
  std::sort(upk_sign, upk_sign + 3,
            graded_signature::compare_UpkSign(param.get_n()));
  for (int64_t i = 0; i < 3; i++) {
    EXPECT_EQ(upk_sign[i].upk == &upk1, upk_sign[i].s == &sign1);
    EXPECT_EQ(upk_sign[i].upk == &upk2, upk_sign[i].s == &sign2);
    EXPECT_EQ(upk_sign[i].upk == &upk3, upk_sign[i].s == &sign3);
  }
  std::unique_ptr<int64_t[]> v(new int64_t[12 * n]);
  for (int64_t i = 0; i < 3; i++)
    std::memcpy(v.get() + i * 4 * n, upk_sign[i].upk->get(),
                4 * n * sizeof(int64_t));
  std::unique_ptr<int64_t[]> ov(new int64_t[12 * n]);
  for (int64_t i = 0; i < 12 * n; i++)
    ov[i] = UZlog2q(rd);
  graded_signature::GSign gsign =
      graded_signature::combine(param, gpk, 3, v, ov, msg);
  std::cout << "combine: done" << std::endl;

  for (int64_t i = 0; i < 6 * n; i++) {
    int64_t iB = (i % (2 * n)) * 4 * m;
    int64_t ix = (i / (2 * n)) * 4 * n;
    int64_t temp = 0;
    for (int64_t j = 0; j < 2 * m; j++) {
      temp += gpk.B[iB + j] *
              graded_signature::get_bit(v[ix + j / log2q], j % log2q);
      temp %= q;
      temp += gpk.B[iB + 2 * m + j] *
              graded_signature::get_bit(ov[ix + j / log2q], j % log2q);
      temp %= q;
    }
    EXPECT_EQ(gsign.cv[i], temp);
  }
  EXPECT_TRUE(verify(param, gpk, 3, msg, gsign));
}
