#include <cstring>
#include <random>

#include "graded_signature/utility.h"

namespace graded_signature {

static std::random_device rd;

int64_t ceil_log2(int64_t x) {
  const int64_t t[6] = {0x7FFFFFFF00000000ll, 0x00000000FFFF0000ll,
                        0x000000000000FF00ll, 0x00000000000000F0ll,
                        0x000000000000000Cll, 0x0000000000000002ll};

  int64_t y = (((x & (x - 1)) == 0) ? 0 : 1);
  int64_t j = 32;

  for (int64_t i = 0; i < 6; i++) {
    int64_t k = (((x & t[i]) == 0) ? 0 : j);
    y += k;
    x >>= k;
    j >>= 1;
  }

  return y;
}

double rho(double x, double c, double s) {
  return std::exp((-pi * (x - c) * (x - c)) / (s * s));
}

int64_t inverse_mod(int64_t x, int64_t p) {
  int64_t q = p - 2;
  int64_t y = 1;
  while (q) {
    y *= ((q & 1) ? x : 1);
    y %= p;
    x = (x * x) % p;
    q >>= 1;
  }
  return y;
}

int64_t get_bit(int64_t x, int64_t i) {
  x >>= i;
  x &= 1;
  return x;
}

void decompose(int64_t *x_ptr, int64_t x, int64_t intBetaInf,
               int64_t logBetaInf) {
  if (x < 0) {
    *x_ptr = 1;
    x += intBetaInf;
  } else
    *x_ptr = 0;
  x_ptr++;
  if (x >= (1 << (logBetaInf - 1))) {
    *x_ptr = 1;
    x -= intBetaInf + 1 - (1 << (logBetaInf - 1));
  } else
    *x_ptr = 0;
  x_ptr++;
  for (int64_t i = 0; i < logBetaInf - 1; i++)
    x_ptr[i] = get_bit(x, i);
}

int64_t compose(int64_t *x_ptr, int64_t q, int64_t intBetaInf,
                int64_t logBetaInf) {
  int64_t x = 0;
  x += x_ptr[0] * (q - intBetaInf) % q;
  x %= q;
  x += x_ptr[1] * (intBetaInf + 1 - (1 << (logBetaInf - 1))) % q;
  x %= q;
  for (int64_t i = 0; i < logBetaInf - 1; i++) {
    x += x_ptr[2 + i] * (1 << i) % q;
    x %= q;
  }
  return x;
}

int64_t &PseudoMatrix::operator[](int64_t i) {
  return M[(size_M + i % size_M) % size_M];
}

// [1] Trapdoors for Lattices: Simpler, Tighter, Faster, Smaller
void gen_trapdoor(int64_t n, int64_t q, std::unique_ptr<int64_t[]> &A,
                  std::unique_ptr<int64_t[]> &T_A) {

  int64_t k = ceil_log2(q);
  // Because the matrix G, the matrix S, the matrix S_k is sparse, we do not
  // store them directly
  // - The matrix G: see page 17 in [1]
  // - The matrix S: see page 17 in [1]
  // - The matrix S_k: see page 20 in [1]
  std::unique_ptr<int64_t[]> gt(new int64_t[k]);
  std::unique_ptr<int64_t[]> qt(new int64_t[k]);
  for (int64_t i = 0; i < k; i++) {
    gt[i] = 1;
    gt[i] <<= i;
    qt[i] = q;
    qt[i] = get_bit(qt[i], i);
  }

  int64_t w = n * k;  // = n * ceil_log2(q);
  int64_t m_ = n * k; // = n * ceil_log2(q);
  int64_t m = m_ + w; // = 2 * n * ceil_log2(q);
  // Randomize the (m_ x w) matrix R on {-1,0,1} with probability 1/4, 1/2, 1/4
  std::discrete_distribution<int64_t> P({1, 2, 1});
  std::unique_ptr<int64_t[]> R(new int64_t[m_ * w]);
  for (int64_t i = 0; i < m_; i++) {
    for (int64_t j = 0; j < w; j++)
      R[i * w + j] = P(rd) - 1;
  }
  // Randomize the (n x m_) matrix A_ on Z_q uniformly
  // Generate the (n x m) matrix A = [A_ | G - A_ * R]
  std::uniform_int_distribution<int64_t> UZq(0, q - 1);
  A.reset(new int64_t[n * m]);
  for (int64_t i = 0; i < n; i++) {
    // A_
    for (int64_t j = 0; j < m_; j++) {
      A[i * m + j] = UZq(rd);
    }
    // G - A_ * R
    for (int64_t j = 0; j < w; j++) {
      // G
      A[i * m + m_ + j] = ((j / k == i) ? gt[j % k] : 0);
      // A_ * R
      for (int64_t l = 0; l < m_; l++) {
        A[i * m + m_ + j] += q - A[i * m + l] * R[l * w + j];
        A[i * m + m_ + j] %= q;
      }
    }
  }

  /* Generate the (m x m) matrix T_A =
  ** /I R\ /I 0\ = /I + R * W     R * S\
  ** \0 I/ \W S/   \    W           S  /
  */
  T_A.reset(new int64_t[m * m]);
  std::memset(T_A.get(), 0, m * m * sizeof(int64_t));
  for (int64_t i = 0; i < w; i++) {
    // W
    for (int64_t j = 0; j < m_; j++) {
      T_A[(m_ + i) * m + j] = A[(i / k) * m + j];
      T_A[(m_ + i) * m + j] = get_bit(T_A[(m_ + i) * m + j], i % k);
      T_A[(m_ + i) * m + j] *= -1;
    }
    // S
    T_A[(m_ + i) * m + m_ + k * (i / k) + (i % k)] = 2;
    if (i % k != 0) {
      T_A[(m_ + i) * m + m_ + k * (i / k) + (i % k) - 1] = -1;
    }
    T_A[(m_ + i) * m + m_ + k * (i / k) + k - 1] = qt[i % k];
  }
  for (int64_t i = 0; i < m_; i++) {
    // I + R * W
    for (int64_t j = 0; j < m_; j++) {
      for (int64_t l = 0; l < w; l++)
        T_A[i * m + j] += R[i * w + l] * T_A[(m_ + l) * m + j];
    }
    T_A[i * m + i] += 1;
    // R * S
    for (int64_t j = 0; j < w; j++) {
      for (int64_t l = 0; l < w; l++)
        T_A[i * m + m_ + j] += R[i * w + l] * T_A[(m_ + l) * m + m_ + j];
    }
  }
}

int64_t sample_integer(double c, double s, double t) {
  int64_t min_int = std::ceil(c - s * t);
  int64_t max_int = std::floor(c + s * t);
  std::unique_ptr<double[]> prob(new double[max_int - min_int + 1]);
  for (int64_t i = min_int; i <= max_int; i++) {
    prob[i - min_int] = rho(i, c, s);
  }
  std::discrete_distribution<int64_t> dist(prob.get(),
                                           prob.get() + max_int - min_int + 1);
  return dist(rd) + min_int;
}

std::unique_ptr<double[]> gram_schmidt(int64_t n,
                                       std::unique_ptr<int64_t[]> &B) {
  std::unique_ptr<double[]> B_(new double[n * n]);
  std::unique_ptr<double[]> B_norm(new double[n]);
  for (int64_t j = 0; j < n; j++) {
    for (int64_t i = 0; i < n; i++) {
      B_[i * n + j] = B[i * n + j];
    }
    for (int64_t k = 0; k < j; k++) {
      double t = 0;
      for (int64_t i = 0; i < n; i++) {
        t += B[i * n + j] * B_[i * n + k];
      }
      for (int64_t i = 0; i < n; i++)
        B_[i * n + j] -= (t / B_norm[k]) * B_[i * n + k];
    }
    B_norm[j] = 0;
    for (int64_t i = 0; i < n; i++)
      B_norm[j] += B_[i * n + j] * B_[i * n + j];
  }
  return B_;
}

std::unique_ptr<int64_t[]> sample_gauss(int64_t n,
                                        std::unique_ptr<int64_t[]> &B,
                                        std::unique_ptr<double[]> &B_, double s,
                                        std::unique_ptr<int64_t[]> &c) {
  std::unique_ptr<int64_t[]> x(new int64_t[n]);
  std::memcpy(x.get(), c.get(), n * sizeof(int64_t));
  for (int64_t j = n - 1; j >= 0; j--) {
    double c_num = 0;
    double c_den = 0;
    for (int64_t i = 0; i < n; i++) {
      c_num += B_[i * n + j] * x[i];
      c_den += B_[i * n + j] * B_[i * n + j];
    }
    int64_t z = sample_integer(c_num / c_den, s / std::sqrt(c_den),
                               std::sqrt(std::log2(n)));
    for (int64_t i = 0; i < n; i++)
      x[i] -= z * B[i * n + j];
  }
  for (int64_t i = 0; i < n; i++) {
    x[i] = c[i] - x[i];
  }
  return x;
}

std::unique_ptr<int64_t[]> sp_inverse_matrix(int64_t n, int64_t q,
                                             std::unique_ptr<int64_t[]> &A) {
  int64_t m = 2 * n * ceil_log2(q);
  std::unique_ptr<int64_t[]> AB(new int64_t[n * (m + n)]);
  std::unique_ptr<int64_t[]> pivot(new int64_t[n]);
  std::memset(AB.get(), 0, n * (m + n) * sizeof(int64_t));
  for (int64_t i = 0; i < n; i++) {
    std::memcpy(AB.get() + i * (m + n), A.get() + i * m, m * sizeof(int64_t));
    AB[i * (m + n) + m + i] = 1;
  }

  int64_t i = 0;
  int64_t j = 0;
  std::unique_ptr<int64_t[]> temp_vec(new int64_t[m + n]);
  while (i < n) {
    int64_t k, l;

    k = i;
    while (k < n) {
      if (AB[k * (m + n) + j])
        break;
      k++;
    }
    if (k == n) {
      j++;
      continue;
    }
    pivot[i] = j;
    if (i != k) {
      std::memcpy(temp_vec.get(), AB.get() + i * (m + n),
                  (m + n) * sizeof(int64_t));
      std::memcpy(AB.get() + i * (m + n), AB.get() + k * (m + n),
                  (m + n) * sizeof(int64_t));
      std::memcpy(AB.get() + k * (m + n), temp_vec.get(),
                  (m + n) * sizeof(int64_t));
    }

    int64_t inv_temp = inverse_mod(AB[i * (m + n) + j], q);
    for (k = j + 1; k < m + n; k++) {
      AB[i * (m + n) + k] *= inv_temp;
      AB[i * (m + n) + k] %= q;
    }
    for (l = 0; l < n; l++) {
      if (l == i)
        continue;
      for (k = j + 1; k < m + n; k++) {
        AB[l * (m + n) + k] +=
            q - AB[l * (m + n) + j] * AB[i * (m + n) + k] % q;
        AB[l * (m + n) + k] %= q;
      }
    }

    i++;
    j++;
  }

  std::unique_ptr<int64_t[]> inv_A(new int64_t[m * n]);
  std::memset(inv_A.get(), 0, m * n * sizeof(int64_t));
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < n; j++)
      inv_A[pivot[i] * n + j] = AB[i * (m + n) + m + j];
  }
  return inv_A;
}

std::unique_ptr<int64_t[]> var_sample_gauss(int64_t n, int64_t q,
                                            std::unique_ptr<int64_t[]> &inv_A,
                                            std::unique_ptr<int64_t[]> &T_A,
                                            std::unique_ptr<double[]> &T_A_,
                                            double s,
                                            std::unique_ptr<int64_t[]> &u) {
  int64_t m = 2 * n * ceil_log2(q);
  std::unique_ptr<int64_t[]> c(new int64_t[m]);
  std::memset(c.get(), 0, m * sizeof(int64_t));
  for (int64_t i = 0; i < m; i++) {
    for (int64_t j = 0; j < n; j++) {
      c[i] += q + inv_A[i * n + j] * u[j] % q;
      c[i] %= q;
    }
  }
  std::unique_ptr<int64_t[]> x = sample_gauss(m, T_A, T_A_, s, c);
  for (int64_t i = 0; i < m; i++) {
    x[i] = c[i] - x[i];
  }
  return x;
}

} // namespace graded_signature
