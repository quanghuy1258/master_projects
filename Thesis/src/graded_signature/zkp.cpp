#include <algorithm>
#include <cstring>
#include <random>

#include "graded_signature/hash.h"
#include "graded_signature/utility.h"
#include "graded_signature/zkp.h"

namespace graded_signature {

ZKP generate_zkp(int64_t t, int64_t D, int64_t L, int64_t q,
                 std::unique_ptr<int64_t[]> &x,
                 std::function<std::unique_ptr<int64_t[]>(int64_t *)> &P,
                 PseudoMatrix &A, Hash &rom) {
  int64_t log2q = ceil_log2(q);

  std::random_device rd;
  std::uniform_int_distribution<int64_t> U4(0, 3);
  std::uniform_int_distribution<int64_t> UZq(0, q - 1);
  std::uniform_int_distribution<int64_t> Urho(0, (1 << log2q) - 1);

  ZKP zkp;
  zkp.comm.reset(new int64_t[t * (3 * L + D)]);
  std::memset(zkp.comm.get(), 0, t * (3 * L + D) * sizeof(int64_t));
  zkp.resp.reset(new int64_t[t * (9 * L + D)]);
  for (int64_t i = 0; i < t; i++) {
    int64_t *resp = zkp.resp.get() + i * (9 * L + D);
    // rho_1 = rho_pi || rho_Pr
    // pi, rho_pi
    for (int64_t j = 0; j < L; j++)
      resp[j] = U4(rd);
    // r, rho_2, rho_3
    for (int64_t j = 0; j < 2 * L; j++) {
      resp[L + j] = UZq(rd);
      resp[5 * L + D + j] = Urho(rd);
      resp[7 * L + D + j] = Urho(rd);
    }
    // rho_Pr
    for (int64_t j = 0; j < D; j++)
      resp[5 * L + j] = Urho(rd);

    // Pr
    std::unique_ptr<int64_t[]> Pr = P(resp + L);

    int64_t *comm = zkp.comm.get() + i * (3 * L + D);
    // C_1
    for (int64_t j = 0; j < L + D; j++) {
      for (int64_t k = 0; k < L; k++) {
        comm[j] += A[j * 2 * (L + D * log2q) + k] * (resp[k] & 1);
        comm[j] %= q;
        comm[j] += A[j * 2 * (L + D * log2q) + L + k] * (resp[k] >> 1);
        comm[j] %= q;
      }
      for (int64_t k = 0; k < D * log2q; k++) {
        int64_t Pr0 = Pr[k / log2q];
        Pr0 = get_bit(Pr0, k % log2q);
        comm[j] += A[j * 2 * (L + D * log2q) + 2 * L + k] * Pr0;
        comm[j] %= q;
        int64_t rho_Pr0 = resp[5 * L + k / log2q];
        rho_Pr0 = get_bit(rho_Pr0, k % log2q);
        comm[j] += A[j * 2 * (L + D * log2q) + 2 * L + D * log2q + k] * rho_Pr0;
        comm[j] %= q;
      }
    }
    comm += L + D;
    // C_2
    for (int64_t j = 0; j < L; j++) {
      for (int64_t k = 0; k < L * log2q; k++) {
        int64_t pi_r0 = resp[L + k / log2q];
        int64_t pi_r1 = resp[2 * L + k / log2q];
        if (resp[k / log2q] & 1)
          std::swap(pi_r0, pi_r1);
        pi_r0 = get_bit(pi_r0, k % log2q);
        pi_r1 = get_bit(pi_r1, k % log2q);
        comm[j] += A[j * 4 * L * log2q + k] * pi_r0;
        comm[j] %= q;
        comm[j] += A[j * 4 * L * log2q + L * log2q + k] * pi_r1;
        comm[j] %= q;
        int64_t rho_20 = resp[5 * L + D + k / log2q];
        int64_t rho_21 = resp[6 * L + D + k / log2q];
        rho_20 = get_bit(rho_20, k % log2q);
        rho_21 = get_bit(rho_21, k % log2q);
        comm[j] += A[j * 4 * L * log2q + 2 * L * log2q + k] * rho_20;
        comm[j] %= q;
        comm[j] += A[j * 4 * L * log2q + 3 * L * log2q + k] * rho_21;
        comm[j] %= q;
      }
    }
    comm += L;
    // C_3
    for (int64_t j = 0; j < L; j++) {
      for (int64_t k = 0; k < L * log2q; k++) {
        int64_t pi_xr0 = (x[k / log2q] + resp[L + k / log2q]) % q;
        int64_t pi_xr1 = (x[L + k / log2q] + resp[2 * L + k / log2q]) % q;
        if (resp[k / log2q] & 1)
          std::swap(pi_xr0, pi_xr1);
        pi_xr0 = get_bit(pi_xr0, k % log2q);
        pi_xr1 = get_bit(pi_xr1, k % log2q);
        comm[j] += A[j * 4 * L * log2q + k] * pi_xr0;
        comm[j] %= q;
        comm[j] += A[j * 4 * L * log2q + L * log2q + k] * pi_xr1;
        comm[j] %= q;
        int64_t rho_30 = resp[7 * L + D + k / log2q];
        int64_t rho_31 = resp[8 * L + D + k / log2q];
        rho_30 = get_bit(rho_30, k % log2q);
        rho_31 = get_bit(rho_31, k % log2q);
        comm[j] += A[j * 4 * L * log2q + 2 * L * log2q + k] * rho_30;
        comm[j] %= q;
        comm[j] += A[j * 4 * L * log2q + 3 * L * log2q + k] * rho_31;
        comm[j] %= q;
      }
    }
  }

  rom.update(zkp.comm.get(), t * (3 * L + D) * sizeof(int64_t));
  std::unique_ptr<int64_t[]> chal = rom.digest(t);

  for (int64_t i = 0; i < t; i++) {
    int64_t *resp = zkp.resp.get() + i * (9 * L + D);
    switch (chal[i]) {
    case 0:
      std::memcpy(resp + 3 * L, resp + L, 2 * L * sizeof(int64_t));
      std::memcpy(resp + L, x.get(), 2 * L * sizeof(int64_t));
      for (int64_t j = 0; j < L; j++) {
        if (resp[j] & 1) {
          std::swap(resp[L + j], resp[2 * L + j]);
          std::swap(resp[3 * L + j], resp[4 * L + j]);
        }
      }
      std::memset(resp, 0, L * sizeof(int64_t));
      std::memset(resp + 5 * L, 0, D * sizeof(int64_t));
      break;
    case 1:
      for (int64_t j = 0; j < 2 * L; j++) {
        resp[L + j] += x[j];
        resp[L + j] %= q;
      }
      std::memset(resp + 3 * L, 0, 2 * L * sizeof(int64_t));
      std::memset(resp + 5 * L + D, 0, 2 * L * sizeof(int64_t));
      break;
    case 2:
      std::memset(resp + 3 * L, 0, 2 * L * sizeof(int64_t));
      std::memset(resp + 7 * L + D, 0, 2 * L * sizeof(int64_t));
      break;
    }
  }
  return zkp;
}

bool verify_zkp(int64_t t, int64_t D, int64_t L, int64_t q,
                std::unique_ptr<int64_t[]> &v,
                std::function<std::unique_ptr<int64_t[]>(int64_t *)> &P,
                PseudoMatrix &A, Hash &rom, ZKP &zkp) {
  int64_t log2q = ceil_log2(q);

  rom.update(zkp.comm.get(), t * (3 * L + D) * sizeof(int64_t));
  std::unique_ptr<int64_t[]> chal = rom.digest(t);

  for (int64_t i = 0; i < t; i++) {
    int64_t *comm = zkp.comm.get() + i * (3 * L + D);
    int64_t *resp = zkp.resp.get() + i * (9 * L + D);
    std::unique_ptr<int64_t[]> Py, Pr;
    switch (chal[i]) {
    case 0:
      // pi_x
      for (int64_t j = 0; j < L; j++) {
        if (resp[L + j] >> 1)
          return false;
        if (resp[2 * L + j] >> 1)
          return false;
        if (resp[L + j] + resp[2 * L + j] != 1)
          return false;
      }
      // C_2 = COM(pi_r, rho_2)
      // C_3 = COM(pi_x + pi_r, rho_3)
      for (int64_t j = 0; j < L; j++) {
        int64_t C_2 = 0;
        int64_t C_3 = 0;
        for (int64_t k = 0; k < 2 * L * log2q; k++) {
          int64_t pi_r = resp[3 * L + k / log2q];
          pi_r = get_bit(pi_r, k % log2q);
          C_2 += A[j * 4 * L * log2q + k] * pi_r;
          C_2 %= q;
          int64_t rho_2 = resp[5 * L + D + k / log2q];
          rho_2 = get_bit(rho_2, k % log2q);
          C_2 += A[j * 4 * L * log2q + 2 * L * log2q + k] * rho_2;
          C_2 %= q;
          int64_t pi_xr = (resp[L + k / log2q] + resp[3 * L + k / log2q]) % q;
          pi_xr = get_bit(pi_xr, k % log2q);
          C_3 += A[j * 4 * L * log2q + k] * pi_xr;
          C_3 %= q;
          int64_t rho_3 = resp[7 * L + D + k / log2q];
          rho_3 = get_bit(rho_3, k % log2q);
          C_3 += A[j * 4 * L * log2q + 2 * L * log2q + k] * rho_3;
          C_3 %= q;
        }
        if (C_2 != comm[L + D + j])
          return false;
        if (C_3 != comm[2 * L + D + j])
          return false;
      }
      break;
    case 1:
      Py = P(resp + L);
      // pi, rho_pi
      for (int64_t j = 0; j < L; j++) {
        if (resp[j] >> 2)
          return false;
      }
      // C_1 = COM(pi, Py - v, rho_1) = COM(pi, Pr, rho_1)
      for (int64_t j = 0; j < L + D; j++) {
        int64_t C_1 = 0;
        for (int64_t k = 0; k < L; k++) {
          C_1 += A[j * 2 * (L + D * log2q) + k] * (resp[k] & 1);
          C_1 %= q;
          C_1 += A[j * 2 * (L + D * log2q) + L + k] * (resp[k] >> 1);
          C_1 %= q;
        }
        for (int64_t k = 0; k < D * log2q; k++) {
          int64_t Pr0 = (Py[k / log2q] + q - v[k / log2q]) % q;
          Pr0 = get_bit(Pr0, k % log2q);
          C_1 += A[j * 2 * (L + D * log2q) + 2 * L + k] * Pr0;
          C_1 %= q;
          int64_t rho_Pr0 = resp[5 * L + k / log2q];
          rho_Pr0 = get_bit(rho_Pr0, k % log2q);
          C_1 += A[j * 2 * (L + D * log2q) + 2 * L + D * log2q + k] * rho_Pr0;
          C_1 %= q;
        }
        if (C_1 != comm[j])
          return false;
      }
      // C_3 = COM(pi_xr, rho_3)
      for (int64_t j = 0; j < L; j++) {
        int64_t C_3 = 0;
        for (int64_t k = 0; k < L * log2q; k++) {
          int64_t pi_xr0 = resp[L + k / log2q];
          int64_t pi_xr1 = resp[2 * L + k / log2q];
          if (resp[k / log2q] & 1)
            std::swap(pi_xr0, pi_xr1);
          pi_xr0 = get_bit(pi_xr0, k % log2q);
          pi_xr1 = get_bit(pi_xr1, k % log2q);
          C_3 += A[j * 4 * L * log2q + k] * pi_xr0;
          C_3 %= q;
          C_3 += A[j * 4 * L * log2q + L * log2q + k] * pi_xr1;
          C_3 %= q;
          int64_t rho_30 = resp[7 * L + D + k / log2q];
          int64_t rho_31 = resp[8 * L + D + k / log2q];
          rho_30 = get_bit(rho_30, k % log2q);
          rho_31 = get_bit(rho_31, k % log2q);
          C_3 += A[j * 4 * L * log2q + 2 * L * log2q + k] * rho_30;
          C_3 %= q;
          C_3 += A[j * 4 * L * log2q + 3 * L * log2q + k] * rho_31;
          C_3 %= q;
        }
        if (C_3 != comm[2 * L + D + j])
          return false;
      }
      break;
    case 2:
      Pr = P(resp + L);
      // pi, rho_pi
      for (int64_t j = 0; j < L; j++) {
        if (resp[j] >> 2)
          return false;
      }
      // C_1 = COM(pi, Pr, rho_1)
      for (int64_t j = 0; j < L + D; j++) {
        int64_t C_1 = 0;
        for (int64_t k = 0; k < L; k++) {
          C_1 += A[j * 2 * (L + D * log2q) + k] * (resp[k] & 1);
          C_1 %= q;
          C_1 += A[j * 2 * (L + D * log2q) + L + k] * (resp[k] >> 1);
          C_1 %= q;
        }
        for (int64_t k = 0; k < D * log2q; k++) {
          int64_t Pr0 = Pr[k / log2q];
          Pr0 = get_bit(Pr0, k % log2q);
          C_1 += A[j * 2 * (L + D * log2q) + 2 * L + k] * Pr0;
          C_1 %= q;
          int64_t rho_Pr0 = resp[5 * L + k / log2q];
          rho_Pr0 = get_bit(rho_Pr0, k % log2q);
          C_1 += A[j * 2 * (L + D * log2q) + 2 * L + D * log2q + k] * rho_Pr0;
          C_1 %= q;
        }
        if (C_1 != comm[j])
          return false;
      }
      // C_2 = COM(pi_r, rho_2)
      for (int64_t j = 0; j < L; j++) {
        int64_t C_2 = 0;
        for (int64_t k = 0; k < L * log2q; k++) {
          int64_t pi_r0 = resp[L + k / log2q];
          int64_t pi_r1 = resp[2 * L + k / log2q];
          if (resp[k / log2q] & 1)
            std::swap(pi_r0, pi_r1);
          pi_r0 = get_bit(pi_r0, k % log2q);
          pi_r1 = get_bit(pi_r1, k % log2q);
          C_2 += A[j * 4 * L * log2q + k] * pi_r0;
          C_2 %= q;
          C_2 += A[j * 4 * L * log2q + L * log2q + k] * pi_r1;
          C_2 %= q;
          int64_t rho_20 = resp[5 * L + D + k / log2q];
          int64_t rho_21 = resp[6 * L + D + k / log2q];
          rho_20 = get_bit(rho_20, k % log2q);
          rho_21 = get_bit(rho_21, k % log2q);
          C_2 += A[j * 4 * L * log2q + 2 * L * log2q + k] * rho_20;
          C_2 %= q;
          C_2 += A[j * 4 * L * log2q + 3 * L * log2q + k] * rho_21;
          C_2 %= q;
        }
        if (C_2 != comm[L + D + j])
          return false;
      }
      break;
    }
  }
  return true;
}

} // namespace graded_signature
