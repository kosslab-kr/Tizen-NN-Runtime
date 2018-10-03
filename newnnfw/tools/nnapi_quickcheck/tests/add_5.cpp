/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"

#include "support/tflite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/builtin_op_data.h"

#include "env.h"
#include "memory.h"
#include "util/environment.h"

#include "support/tflite/Diff.h"
#include "support/tflite/Quantization.h"
#include "support/tflite/interp/FunctionBuilder.h"

#include <iostream>
#include <cassert>

#include <chrono>
#include <random>

using namespace tflite;
using namespace tflite::ops::builtin;

TEST(NNAPI_Quickcheck_add_5, simple_test)
{
  int verbose = 0;
  int tolerance = 1;

  nnfw::util::env::IntAccessor("VERBOSE").access(verbose);
  nnfw::util::env::IntAccessor("TOLERANCE").access(tolerance);

  // Set random seed
  int SEED = std::chrono::system_clock::now().time_since_epoch().count();

  nnfw::util::env::IntAccessor("SEED").access(SEED);

#define INT_VALUE(NAME, VALUE) IntVar NAME##_Value(#NAME, VALUE);
#include "add_5.lst"
#undef INT_VALUE

  const int32_t LEFT_N = LEFT_N_Value();
  const int32_t LEFT_C = LEFT_C_Value();
  const int32_t LEFT_H = LEFT_H_Value();
  const int32_t LEFT_W = LEFT_W_Value();

  const int32_t RIGHT = RIGHT_Value();

  const int32_t OFM_N = LEFT_N;
  const int32_t OFM_C = LEFT_C;
  const int32_t OFM_H = LEFT_H;
  const int32_t OFM_W = LEFT_W;

  // Initialize random number generator
  std::minstd_rand random(SEED);

  std::cout << "Configurations:" << std::endl;
#define PRINT_NEWLINE()     \
  {                         \
    std::cout << std::endl; \
  }
#define PRINT_VALUE(value)                                       \
  {                                                              \
    std::cout << "  " << #value << ": " << (value) << std::endl; \
  }
  PRINT_VALUE(SEED);
  PRINT_NEWLINE();

  PRINT_VALUE(LEFT_N);
  PRINT_VALUE(LEFT_C);
  PRINT_VALUE(LEFT_H);
  PRINT_VALUE(LEFT_W);
  PRINT_NEWLINE();

  PRINT_VALUE(RIGHT);
  PRINT_NEWLINE();

  PRINT_VALUE(OFM_N);
  PRINT_VALUE(OFM_C);
  PRINT_VALUE(OFM_H);
  PRINT_VALUE(OFM_W);
#undef PRINT_VALUE
#undef PRINT_NEWLINE

  auto setup = [&](Interpreter &interp) {
    // Comment from 'context.h'
    //
    // Parameters for asymmetric quantization. Quantized values can be converted
    // back to float using:
    //    real_value = scale * (quantized_value - zero_point);
    //
    // Q: Is this necessary?
    TfLiteQuantizationParams quantization = make_default_quantization();

    // On AddTensors(N) call, T/F Lite interpreter creates N tensors whose index is [0 ~ N)
    interp.AddTensors(3);

    // Configure output
    interp.SetTensorParametersReadWrite(0, kTfLiteFloat32 /* type */, "output" /* name */,
                                        {OFM_N, OFM_H, OFM_W, OFM_C} /* dims */, quantization);

    // Configure input(s)
    interp.SetTensorParametersReadWrite(1, kTfLiteFloat32 /* type */, "left" /* name */,
                                        {LEFT_N, LEFT_H, LEFT_W, LEFT_C} /* dims */, quantization);

    interp.SetTensorParametersReadWrite(2, kTfLiteFloat32 /* type */, "right" /* name */,
                                        {RIGHT} /* dims */, quantization);

    // Add Convolution Node
    //
    // NOTE AddNodeWithParameters take the ownership of param, and deallocate it with free
    //      So, param should be allocated with malloc
    auto param = make_alloc<TfLiteAddParams>();

    param->activation = kTfLiteActNone;

    // Run Add and store the result into Tensor #0
    //  - Read Left from Tensor #1
    //  - Read Left from Tensor #2,
    interp.AddNodeWithParameters({1, 2}, {0}, nullptr, 0, reinterpret_cast<void *>(param),
                                 BuiltinOpResolver().FindOp(BuiltinOperator_ADD, 1));

    interp.SetInputs({1, 2});
    interp.SetOutputs({0});
  };

  const nnfw::support::tflite::interp::FunctionBuilder builder(setup);

  RandomTestParam param;

  param.verbose = verbose;
  param.tolerance = tolerance;

  int res = RandomTestRunner{SEED, param}.run(builder);

  EXPECT_EQ(res, 0);
}