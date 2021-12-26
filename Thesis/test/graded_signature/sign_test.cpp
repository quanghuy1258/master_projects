#include <cstring>
#include <iostream>
#include <memory>

#include "gtest/gtest.h"

#include "graded_signature/group_key_gen.h"
#include "graded_signature/join.h"
#include "graded_signature/param.h"
#include "graded_signature/sign.h"
#include "graded_signature/user_key_gen.h"
#include "graded_signature/utility.h"

TEST(Sign, Sign) {
  graded_signature::Param param;
  param.k = 8;
  param.n = 16;
  param.q = 131063;
  param.sigma = 100;
  param.t = 10;
  std::cout << "param: done" << std::endl;

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

  std::string msg = "Graded Signature";
  graded_signature::USign usign =
      graded_signature::sign(param, gpk, cert, upk, usk, msg);
  std::memset(usign.ov.get(), 0, 2 * param.get_m() * sizeof(int64_t));
  std::cout << "sign: done" << std::endl;

  EXPECT_TRUE(graded_signature::verify(param, gpk, msg, usign));
}
