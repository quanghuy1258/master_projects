#ifndef GRADED_SIGNATURE_UTILITY_H
#define GRADED_SIGNATURE_UTILITY_H

#include <cmath>
#include <cstdint>
#include <memory>

namespace graded_signature {

const double pi = std::atan(1) * 4;

int64_t ceil_log2(int64_t x);
double rho(double x, double c, double s);
int64_t inverse_mod(int64_t x, int64_t p);
int64_t get_bit(int64_t x, int64_t i);
void decompose(int64_t *x_ptr, int64_t x, int64_t intBetaInf,
               int64_t logBetaInf);
int64_t compose(int64_t *x_ptr, int64_t q, int64_t intBetaInf,
                int64_t logBetaInf);

struct PseudoMatrix {
  std::unique_ptr<int64_t[]> M;
  int64_t size_M;

  int64_t &operator[](int64_t i);
};

// lattice algorithms
void gen_trapdoor(int64_t n, int64_t q, std::unique_ptr<int64_t[]> &A,
                  std::unique_ptr<int64_t[]> &T_A);
int64_t sample_integer(double c, double s, double t);
std::unique_ptr<double[]> gram_schmidt(int64_t n,
                                       std::unique_ptr<int64_t[]> &B);
std::unique_ptr<int64_t[]> sample_gauss(int64_t n,
                                        std::unique_ptr<int64_t[]> &B,
                                        std::unique_ptr<double[]> &B_, double s,
                                        std::unique_ptr<int64_t[]> &c);
// sp_inverse_matrix: special inverse_matrix
//   m = 2 * n * ceil_log2(q)
std::unique_ptr<int64_t[]> sp_inverse_matrix(int64_t n, int64_t q,
                                             std::unique_ptr<int64_t[]> &A);
// var_sample_gauss: variant sample_gauss
//   return x such that A * x = u mod q
std::unique_ptr<int64_t[]> var_sample_gauss(int64_t n, int64_t q,
                                            std::unique_ptr<int64_t[]> &inv_A,
                                            std::unique_ptr<int64_t[]> &T_A,
                                            std::unique_ptr<double[]> &T_A_,
                                            double s,
                                            std::unique_ptr<int64_t[]> &u);

} // namespace graded_signature

#endif
