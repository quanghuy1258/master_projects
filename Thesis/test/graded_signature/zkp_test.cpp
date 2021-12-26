#include <cstring>
#include <iostream>
#include <random>

#include "gtest/gtest.h"

#include "graded_signature/utility.h"
#include "graded_signature/zkp.h"

TEST(ZKP, ZKP) {
  int64_t t = 10;
  int64_t D = 16;
  int64_t L = 32;
  int64_t q = 131063;
  int64_t log2q = graded_signature::ceil_log2(q);

  std::random_device rd;
  std::uniform_int_distribution<int64_t> UB(0, 1);
  std::uniform_int_distribution<int64_t> UZq(0, q - 1);

  std::unique_ptr<int64_t[]> x(new int64_t[2 * L]);
  for (int64_t i = 0; i < L; i++) {
    x[i] = UB(rd);
    x[i + L] = 1 - x[i];
  }

  std::unique_ptr<int64_t[]> P(new int64_t[D * L]);
  for (int64_t i = 0; i < D * L; i++)
    P[i] = UZq(rd);

  std::function<std::unique_ptr<int64_t[]>(int64_t *)> Pf =
      [ D, L, q, &P ](int64_t * x_ptr) -> std::unique_ptr<int64_t[]> {
    std::unique_ptr<int64_t[]> v(new int64_t[D]);
    std::memset(v.get(), 0, D * sizeof(int64_t));
    for (int64_t i = 0; i < D; i++) {
      for (int64_t j = 0; j < L; j++) {
        v[i] += (P[i * L + j] * x_ptr[j]) % q;
        v[i] %= q;
      }
    }
    return v;
  };
  std::unique_ptr<int64_t[]> v = Pf(x.get());

  graded_signature::PseudoMatrix A;
  A.size_M = D * D * L * L * log2q;
  A.M.reset(new int64_t[A.size_M]);
  for (int64_t i = 0; i < A.size_M; i++)
    A[i] = UZq(rd);
  std::cout << "prepare: done" << std::endl;

  graded_signature::Hash rom_gen, rom_ver;
  graded_signature::ZKP zkp =
      graded_signature::generate_zkp(t, D, L, q, x, Pf, A, D, rom_gen);
  std::cout << "generate: done" << std::endl;

  EXPECT_TRUE(
      graded_signature::verify_zkp(t, D, L, q, v, Pf, A, D, rom_ver, zkp));
}
