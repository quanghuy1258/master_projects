#include <random>

#include "graded_signature/group_key_gen.h"
#include "graded_signature/utility.h"

namespace graded_signature {

void group_key_gen(Param param, Gpk &gpk, Gsk &gsk) {
  int64_t k = param.get_k();
  int64_t n = param.get_n();
  int64_t q = param.get_q();
  int64_t m = param.get_m();
  std::random_device rd;
  std::uniform_int_distribution<int64_t> UZq(0, q - 1);

  gen_trapdoor(n, q, gpk.A, gsk.T_A);
  gsk.T_A_ = gram_schmidt(m, gsk.T_A);
  gsk.inv_A = sp_inverse_matrix(n, q, gpk.A);

  gpk.A_k.reset(new int64_t[(k + 1) * n * m]);
  for (int64_t i = 0; i < (k + 1) * n * m; i++) {
    gpk.A_k[i] = UZq(rd);
  }
  gpk.D.reset(new int64_t[n * m]);
  for (int64_t i = 0; i < n * m; i++) {
    gpk.D[i] = UZq(rd);
  }
  gpk.D_0.reset(new int64_t[4 * n * m]);
  gpk.D_1.reset(new int64_t[4 * n * m]);
  for (int64_t i = 0; i < 4 * n * m; i++) {
    gpk.D_0[i] = UZq(rd);
    gpk.D_1[i] = UZq(rd);
  }
  gpk.u.reset(new int64_t[n]);
  for (int64_t i = 0; i < n; i++) {
    gpk.u[i] = UZq(rd);
  }
  gpk.B.reset(new int64_t[8 * n * m]);
  for (int64_t i = 0; i < 8 * n * m; i++) {
    gpk.B[i] = UZq(rd);
  }
  gpk.F.reset(new int64_t[16 * n * m]);
  for (int64_t i = 0; i < 16 * n * m; i++) {
    gpk.F[i] = UZq(rd);
  }
}

} // namespace graded_signature
