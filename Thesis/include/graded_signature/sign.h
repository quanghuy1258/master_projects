#ifndef GRADED_SIGNATURE_SIGN_H
#define GRADED_SIGNATURE_SIGN_H

#include <cstdint>
#include <memory>
#include <string>

#include "graded_signature/group_key_gen.h"
#include "graded_signature/join.h"
#include "graded_signature/param.h"
#include "graded_signature/zkp.h"

namespace graded_signature {

struct USign {
  std::unique_ptr<int64_t[]> cv;
  std::unique_ptr<int64_t[]> ov;
  ZKP zkp;
};

USign sign(Param &param, Gpk &gpk, Cert &cert, std::unique_ptr<int64_t[]> &v,
           std::unique_ptr<int64_t[]> &z, std::string &msg);
bool verify(Param &param, Gpk &gpk, std::string &msg, USign &usign);

} // namespace graded_signature

#endif
