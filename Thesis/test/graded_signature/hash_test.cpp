#include <iostream>

#include "gtest/gtest.h"

#include "graded_signature/hash.h"

TEST(Hash, Hash) {
  int64_t t = 10;
  char s[] = "Graded Signatures";
  int64_t ans[] = {0, 2, 2, 1, 1, 1, 0, 2, 2, 1};
  graded_signature::Hash hash_obj;
  hash_obj.update(s, sizeof(s) - 1);
  std::unique_ptr<int64_t[]> h = hash_obj.digest(t);
  for (int64_t i = 0; i < 10; i++)
    EXPECT_EQ(h[i], ans[i]);
}
