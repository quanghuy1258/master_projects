#include <cstring>
#include <iostream>

#include "graded_signature/combine.h"
#include "graded_signature/hash.h"
#include "graded_signature/utility.h"

namespace graded_signature {

std::function<bool(const UpkSign &a, const UpkSign &b)>
compare_UpkSign(int64_t n) {
  return [n](const UpkSign &a, const UpkSign &b) -> bool {
    int64_t *a_ptr = a.upk->get();
    int64_t *b_ptr = b.upk->get();
    for (int64_t i = 4 * n - 1; i >= 0; i--) {
      if (a_ptr[i] == b_ptr[i])
        continue;
      return a_ptr[i] < b_ptr[i];
    }
    return false;
  };
}

void create_P(int64_t n, int64_t m, int64_t q, int64_t l,
              std::unique_ptr<int64_t[]> &B,
              std::function<std::unique_ptr<int64_t[]>(int64_t *)> &P) {
  P = [ n, m, q, l, &B ](int64_t * x) -> std::unique_ptr<int64_t[]> {
    int64_t D = l * 2 * n + (l - 1) * 2 * m;

    std::unique_ptr<int64_t[]> v(new int64_t[D]);
    std::memset(v.get(), 0, D * sizeof(int64_t));

    for (int64_t i = 0; i < l * 2 * n; i++) {
      int64_t iB = (i % (2 * n)) * 4 * m;
      int64_t ix = (i / (2 * n)) * 4 * m;
      for (int64_t j = 0; j < 2 * m; j++) {
        v[i] += q + B[iB + j] * x[ix + j] % q;
        v[i] %= q;
        v[i] += q + B[iB + 2 * m + j] * x[ix + 2 * m + j] % q;
        v[i] %= q;
      }
    }
    for (int64_t i = 0; i < l - 1; i++) {
      int64_t *x1 = x + i * 4 * m;
      int64_t *x2 = x + (i + 1) * 4 * m;
      int64_t *x3 = x + l * 4 * m + i * (8 * m - 5);
      int64_t *v_ptr = v.get() + l * 2 * n + i * 2 * m;
      v_ptr[0] = x1[0] - x2[0] + x3[0] + 2 * x3[1] + 2 * x3[2];
      v_ptr[0] = (q + v_ptr[0] % q) % q;
      for (int64_t j = 1; j < 2 * m - 1; j++) {
        v_ptr[j] = x1[j] - x2[j] - 10 * x3[j * 4 - 2] + 2 * x3[j * 4 - 1] +
                   3 * x3[j * 4] + 4 * x3[j * 4 + 1] + 4 * x3[j * 4 + 2];
        v_ptr[j] = (q + v_ptr[j] % q) % q;
      }
      v_ptr[2 * m - 1] = x1[2 * m - 1] - x2[2 * m - 1] - x3[8 * m - 6];
      v_ptr[2 * m - 1] = (q + v_ptr[2 * m - 1] % q) % q;
    }

    return v;
  };
}

GSign combine(Param &param, Gpk &gpk, int64_t l, std::unique_ptr<int64_t[]> &v,
              std::unique_ptr<int64_t[]> &ov, std::string &msg) {
  int64_t n = param.get_n();
  int64_t m = param.get_m();
  int64_t q = param.get_q();
  int64_t log2q = param.get_log2q();
  int64_t t = param.get_t();

  GSign gsign;

  int64_t D = l * 2 * n + (l - 1) * 2 * m;
  int64_t L = l * 4 * m + (l - 1) * (8 * m - 5);
  std::unique_ptr<int64_t[]> x(new int64_t[2 * L]);
  for (int64_t i = 0; i < l; i++) {
    for (int64_t j = 0; j < 2 * m; j++) {
      int64_t temp;
      temp = v[i * 4 * n + j / log2q];
      x[i * 4 * m + j] = get_bit(temp, j % log2q);
      temp = ov[i * 4 * n + j / log2q];
      x[i * 4 * m + 2 * m + j] = get_bit(temp, j % log2q);
    }
  }
  int64_t *x_ptr = x.get() + L - 1;
  for (int64_t i = l - 1; i > 0; i--) {
    if (x[i * 4 * m + 2 * m - 1] < x[(i - 1) * 4 * m + 2 * m - 1])
      break;
    x_ptr[0] = 1 - x[i * 4 * m + 2 * m - 1] + x[(i - 1) * 4 * m + 2 * m - 1];
    x_ptr -= 4;
    bool is_break = false;
    for (int64_t j = 2 * m - 2; j > 0; j--) {
      switch (x[i * 4 * m + j] - x[(i - 1) * 4 * m + j]) {
      case 1:
        x_ptr[0] = 0;
        x_ptr[1] = 0;
        x_ptr[2] = 0;
        x_ptr[3] = ((x_ptr[4]) ? 0 : 1);
        break;
      case 0:
        x_ptr[0] = ((x_ptr[4]) ? 1 : 0);
        x_ptr[1] = ((x_ptr[4]) ? 1 : 0);
        x_ptr[2] = 1;
        x_ptr[3] = ((x_ptr[4]) ? 1 : 0);
        break;
      case -1:
        if (x_ptr[4])
          is_break = true;
        else {
          x_ptr[0] = 0;
          x_ptr[1] = 1;
          x_ptr[2] = 0;
          x_ptr[3] = 0;
        }
        break;
      }
      if (is_break)
        break;
      x_ptr -= 4;
    }
    if (is_break)
      break;
    if (x_ptr[4]) {
      if (x[i * 4 * m] <= x[(i - 1) * 4 * m])
        break;
      x_ptr[2] = 0;
      x_ptr[3] = 0;
    } else {
      switch (x[i * 4 * m] - x[(i - 1) * 4 * m]) {
      case 1:
        x_ptr[2] = 0;
        x_ptr[3] = 1;
        break;
      case 0:
        x_ptr[2] = 1;
        x_ptr[3] = 0;
        break;
      case -1:
        x_ptr[2] = 0;
        x_ptr[3] = 0;
        break;
      }
    }
    x_ptr++;
  }
  if (x_ptr != x.get() + l * 4 * m - 1) {
    std::cout << "Error: v is not valid" << std::endl;
    return gsign;
  }
  for (int64_t i = 0; i < L; i++)
    x[L + i] = 1 - x[i];

  gsign.cv.reset(new int64_t[l * 2 * n]);
  std::memset(gsign.cv.get(), 0, l * 2 * n * sizeof(int64_t));
  for (int64_t i = 0; i < l * 2 * n; i++) {
    int64_t iB = (i % (2 * n)) * 4 * m;
    int64_t ix = (i / (2 * n)) * 4 * m;
    for (int64_t j = 0; j < 2 * m; j++) {
      gsign.cv[i] += gpk.B[iB + j] * x[ix + j];
      gsign.cv[i] %= q;
      gsign.cv[i] += gpk.B[iB + 2 * m + j] * x[ix + 2 * m + j];
      gsign.cv[i] %= q;
    }
  }

  std::function<std::unique_ptr<int64_t[]>(int64_t *)> P;
  create_P(n, m, q, l, gpk.B, P);

  Hash rom;
  std::unique_ptr<unsigned char[]> msg_data(new unsigned char[msg.length()]);
  std::memcpy(msg_data.get(), msg.c_str(), msg.length());
  rom.update(msg_data.get(), msg.length());
  rom.update(gsign.cv.get(), l * 2 * n * sizeof(int64_t));

  gsign.zkp = generate_zkp(t, D, L, q, x, P, gpk.comm_matrix, n, rom);

  return gsign;
}

bool verify(Param &param, Gpk &gpk, int64_t l, std::string &msg, GSign &gsign) {
  int64_t n = param.get_n();
  int64_t m = param.get_m();
  int64_t q = param.get_q();
  int64_t t = param.get_t();

  int64_t D = l * 2 * n + (l - 1) * 2 * m;
  int64_t L = l * 4 * m + (l - 1) * (8 * m - 5);

  std::unique_ptr<int64_t[]> y(new int64_t[D]);
  std::memcpy(y.get(), gsign.cv.get(), l * 2 * n * sizeof(int64_t));
  for (int64_t i = 0; i < l - 1; i++) {
    y[l * 2 * n + i * 2 * m] = 1;
    for (int64_t j = 1; j < 2 * m - 1; j++)
      y[l * 2 * n + i * 2 * m + j] = 3;
    y[l * 2 * n + i * 2 * m + 2 * m - 1] = q - 1;
  }

  std::function<std::unique_ptr<int64_t[]>(int64_t *)> P;
  create_P(n, m, q, l, gpk.B, P);

  Hash rom;
  std::unique_ptr<unsigned char[]> msg_data(new unsigned char[msg.length()]);
  std::memcpy(msg_data.get(), msg.c_str(), msg.length());
  rom.update(msg_data.get(), msg.length());
  rom.update(gsign.cv.get(), l * 2 * n * sizeof(int64_t));

  return verify_zkp(t, D, L, q, y, P, gpk.comm_matrix, n, rom, gsign.zkp);
}

} // namespace graded_signature
