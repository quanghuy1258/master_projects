#include <iostream>
#include <map>

#include "gtest/gtest.h"

#include "graded_signature/utility.h"

TEST(Utility, CeilLog2) {
  EXPECT_EQ(graded_signature::ceil_log2(1), 0);
  EXPECT_EQ(graded_signature::ceil_log2(2), 1);
  int64_t x = 4;
  for (int64_t i = 2; i <= 62; i++) {
    EXPECT_EQ(graded_signature::ceil_log2(x), i);
    EXPECT_EQ(graded_signature::ceil_log2(x - 1), i);
    x <<= 1;
  }
}

TEST(Utility, InverseMod) {
  int64_t q = 131063;
  for (int64_t i = 1; i < q; i++)
    EXPECT_EQ((i * graded_signature::inverse_mod(i, q)) % q, 1);
}

TEST(Utility, GenTrapdoor) {
  int64_t n = 16;
  int64_t q = 131063;
  int64_t m = 2 * n * graded_signature::ceil_log2(q);
  std::unique_ptr<int64_t[]> A, T_A;
  graded_signature::gen_trapdoor(n, q, A, T_A);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < m; j++) {
      int64_t t = 0;
      for (int64_t k = 0; k < m; k++) {
        t += q + A[i * m + k] * T_A[k * m + j] % q;
        t %= q;
      }
      EXPECT_EQ(t, 0);
    }
  }
}

TEST(Utility, SampleInteger) {
  double c = 3.14;
  double s = 3.14;
  double t = 2.82;
  std::map<int64_t, int64_t> map;
  for (int64_t n = 0; n < 10000; ++n) {
    int64_t x = graded_signature::sample_integer(c, s, t);
    EXPECT_TRUE(x <= c + s * t);
    EXPECT_TRUE(x >= c - s * t);
    ++map[x];
  }
  for (const auto pair : map)
    std::cout << std::setw(2) << pair.first << " generated " << std::setw(4)
              << pair.second << " times" << std::endl;
}

TEST(Utility, GramSchmidt) {
  int64_t n = 16;
  int64_t q = 131063;
  int64_t m = 2 * n * graded_signature::ceil_log2(q);
  std::unique_ptr<int64_t[]> A, T_A;
  graded_signature::gen_trapdoor(n, q, A, T_A);
  std::unique_ptr<double[]> T_A_ = graded_signature::gram_schmidt(m, T_A);
  for (int64_t j = 0; j < m; j++) {
    for (int64_t k = 0; k < j; k++) {
      double t = 0;
      for (int64_t i = 0; i < m; i++) {
        t += T_A_[i * m + j] * T_A_[i * m + k];
      }
      EXPECT_NEAR(t, 0, 1e-9);
    }
  }
  int64_t max_int = 0;
  int64_t min_int = 0;
  double max_real = 0;
  double min_real = 0;
  for (int64_t i = 0; i < m; i++) {
    for (int64_t j = 0; j < m; j++) {
      max_int = ((max_int < T_A[i * m + j]) ? T_A[i * m + j] : max_int);
      min_int = ((min_int > T_A[i * m + j]) ? T_A[i * m + j] : min_int);
      max_real = ((max_real < T_A_[i * m + j]) ? T_A_[i * m + j] : max_real);
      min_real = ((min_real > T_A_[i * m + j]) ? T_A_[i * m + j] : min_real);
    }
  }
  std::cout << "Max of T_A = " << max_int << std::endl;
  std::cout << "Min of T_A = " << min_int << std::endl;
  std::cout << "Max of T_A_ = " << max_real << std::endl;
  std::cout << "Min of T_A_ = " << min_real << std::endl;
}

TEST(Utility, SampleGauss) {
  int64_t n = 16;
  int64_t q = 131063;
  int64_t m = 2 * n * graded_signature::ceil_log2(q);
  std::unique_ptr<int64_t[]> A, T_A;
  graded_signature::gen_trapdoor(n, q, A, T_A);
  std::unique_ptr<double[]> T_A_ = graded_signature::gram_schmidt(m, T_A);
  double s = 100;
  double beta2 = s * std::sqrt(2 * m);
  double betaInf = s * std::sqrt(std::log2(m));
  std::unique_ptr<int64_t[]> c(new int64_t[m]);
  for (int64_t i = 0; i < m; i++) {
    c[i] = i % q;
  }

  std::cout << "Prepare: Done\n";
  std::unique_ptr<int64_t[]> x =
      graded_signature::sample_gauss(m, T_A, T_A_, s, c);
  std::cout << "Sample: Done\n";

  for (int64_t i = 0; i < n; i++) {
    int64_t t = 0;
    for (int64_t j = 0; j < m; j++) {
      t += q + A[i * m + j] * x[j] % q;
      t %= q;
    }
    EXPECT_EQ(t, 0);
  }

  double u = 0;
  for (int64_t i = 0; i < m; i++) {
    double v = (c[i] - x[i]) * (c[i] - x[i]);
    u += v;
    EXPECT_LT(std::sqrt(v), betaInf);
  }
  EXPECT_LT(std::sqrt(u), beta2);
}

TEST(Utility, SpecialInverseMatrix) {
  int64_t n = 16;
  int64_t q = 131063;
  int64_t m = 2 * n * graded_signature::ceil_log2(q);
  std::unique_ptr<int64_t[]> A, T_A;
  graded_signature::gen_trapdoor(n, q, A, T_A);
  std::unique_ptr<int64_t[]> inv_A =
      graded_signature::sp_inverse_matrix(n, q, A);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < n; j++) {
      int64_t t = 0;
      for (int64_t k = 0; k < m; k++) {
        t += q + A[i * m + k] * inv_A[k * n + j] % q;
        t %= q;
      }
      if (i == j)
        EXPECT_EQ(t, 1);
      else
        EXPECT_EQ(t, 0);
    }
  }
}

TEST(Utility, VariantSampleGauss) {
  int64_t n = 16;
  int64_t q = 131063;
  int64_t m = 2 * n * graded_signature::ceil_log2(q);
  std::unique_ptr<int64_t[]> A, T_A;
  graded_signature::gen_trapdoor(n, q, A, T_A);
  std::unique_ptr<int64_t[]> inv_A =
      graded_signature::sp_inverse_matrix(n, q, A);
  std::unique_ptr<double[]> T_A_ = graded_signature::gram_schmidt(m, T_A);
  double s = 100;
  double beta2 = s * std::sqrt(2 * m);
  double betaInf = s * std::sqrt(std::log2(m));
  std::unique_ptr<int64_t[]> u(new int64_t[n]);
  for (int64_t i = 0; i < n; i++) {
    u[i] = i % q;
  }

  std::cout << "Prepare: Done\n";
  std::unique_ptr<int64_t[]> x =
      graded_signature::var_sample_gauss(n, q, inv_A, T_A, T_A_, s, u);
  std::cout << "Sample: Done\n";

  for (int64_t i = 0; i < n; i++) {
    int64_t t = 0;
    for (int64_t j = 0; j < m; j++) {
      t += q + A[i * m + j] * x[j] % q;
      t %= q;
    }
    EXPECT_EQ(t, u[i]);
  }

  double y = 0;
  for (int64_t i = 0; i < m; i++) {
    double z = x[i] * x[i];
    y += z;
    EXPECT_LT(std::sqrt(z), betaInf);
  }
  EXPECT_LT(std::sqrt(y), beta2);
}
