# How to Add Unittest using gtest(googletest)

### 1. make own test code
```
#include "gtest/gtest.h"

TEST(TFLite_test_case, simple_test)
{
    EXPECT_EQ(1, 1);
}
```

### 2. Find and prepare package `googletest` to your test executable
```
find_nnfw_package(GTest QUITE)
if(NOT GTest_FOUND)
  ## Cannot find and prepare googletest package
  return()
endif(NOT GTest_FOUND)
add_executable($YOURTEST_TARGET yourtest1.cc yourtest2.cc)
```

### 3. Link test executable against libgtest.a and libgtest_main.a (+ pthread)
```
target_link_libraries($YOURTEST_TARGET gtest gtest_main pthread)
```

### 4. Install test executable into Product/out/unittest
```
install(TARGETS $YOURTEST_TARGET DESTINATION unittest)
```
