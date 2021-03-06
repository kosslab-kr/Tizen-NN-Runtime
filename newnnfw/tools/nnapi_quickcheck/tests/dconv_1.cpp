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

TEST(NNAPI_Quickcheck_dconv_1, simple_test)
{
  int verbose = 0;
  int tolerance = 1;

  nnfw::util::env::IntAccessor("VERBOSE").access(verbose);
  nnfw::util::env::IntAccessor("TOLERANCE").access(tolerance);

  // Set random seed
  int SEED = std::chrono::system_clock::now().time_since_epoch().count();

  nnfw::util::env::IntAccessor("SEED").access(SEED);

#define INT_VALUE(NAME, VALUE) IntVar NAME##_Value(#NAME, VALUE);
#include "dconv_1.lst"
#undef INT_VALUE

  const int32_t STRIDE_H = STRIDE_H_Value();
  const int32_t STRIDE_W = STRIDE_W_Value();

  const int32_t IFM_C = IFM_C_Value();
  const int32_t IFM_H = IFM_H_Value();
  const int32_t IFM_W = IFM_W_Value();

  const int32_t KER_C = KER_C_Value();
  const int32_t KER_H = KER_H_Value();
  const int32_t KER_W = KER_W_Value();

  const int32_t OFM_C = KER_C;
  const int32_t OFM_H = (IFM_H - KER_H) / STRIDE_H + 1;
  const int32_t OFM_W = (IFM_W - KER_W) / STRIDE_W + 1;

  const int32_t MULTIPLIER = MULTIPLIER_Value();

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

  PRINT_VALUE(IFM_C);
  PRINT_VALUE(IFM_H);
  PRINT_VALUE(IFM_W);
  PRINT_NEWLINE();

  PRINT_VALUE(KER_C);
  PRINT_VALUE(KER_H);
  PRINT_VALUE(KER_W);
  PRINT_NEWLINE();

  PRINT_VALUE(STRIDE_H);
  PRINT_VALUE(STRIDE_W);
  PRINT_NEWLINE();

  PRINT_VALUE(MULTIPLIER);
#undef PRINT_VALUE
#undef PRINT_NEWLINE

  assert(MULTIPLIER * IFM_C == KER_C);

  // Configure Kernel Data
  const uint32_t kernel_size = KER_C * KER_H * KER_W;
  float kernel_data[kernel_size] = {
      0.0f,
  };

  // Fill kernel data with random data
  {
    std::normal_distribution<float> kernel_dist(-1.0f, +1.0f);

    for (uint32_t off = 0; off < kernel_size; ++off)
    {
      kernel_data[off] = kernel_dist(random);
    }
  }

  // Configure Bias Data
  const auto bias_size = KER_C;
  float bias_data[bias_size] = {
      0.0f,
  };

  // Fill bias data with random data
  {
    std::normal_distribution<float> bias_dist(-1.0f, +1.0f);

    for (uint32_t off = 0; off < bias_size; ++off)
    {
      bias_data[off] = bias_dist(random);
    }
  }

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
    interp.AddTensors(4);

    // Configure OFM
    interp.SetTensorParametersReadWrite(0, kTfLiteFloat32 /* type */, "output" /* name */,
                                        {1 /*N*/, OFM_H, OFM_W, OFM_C} /* dims */, quantization);

    // Configure IFM
    interp.SetTensorParametersReadWrite(1, kTfLiteFloat32 /* type */, "input" /* name */,
                                        {1 /*N*/, IFM_H, IFM_W, IFM_C} /* dims */, quantization);

    // NOTE kernel_data & bias_data should live longer than interpreter!
    interp.SetTensorParametersReadOnly(
        2, kTfLiteFloat32 /* type */, "filter" /* name */, {1, KER_H, KER_W, KER_C} /* dims */,
        quantization, reinterpret_cast<const char *>(kernel_data), kernel_size * sizeof(float));

    interp.SetTensorParametersReadOnly(
        3, kTfLiteFloat32 /* type */, "bias" /* name */, {bias_size} /* dims */, quantization,
        reinterpret_cast<const char *>(bias_data), bias_size * sizeof(float));

    // Add Convolution Node
    //
    // NOTE AddNodeWithParameters take the ownership of param, and deallocate it with free
    //      So, param should be allocated with malloc
    auto param = make_alloc<TfLiteDepthwiseConvParams>();

    param->padding = kTfLitePaddingValid;
    param->stride_width = STRIDE_W;
    param->stride_height = STRIDE_H;
    param->depth_multiplier = MULTIPLIER;
    param->activation = kTfLiteActRelu;

    // Run Convolution and store its result into Tensor #0
    //  - Read IFM from Tensor #1
    //  - Read Filter from Tensor #2,
    //  - Read Bias from Tensor #3
    interp.AddNodeWithParameters({1, 2, 3}, {0}, nullptr, 0, reinterpret_cast<void *>(param),
                                 BuiltinOpResolver().FindOp(BuiltinOperator_DEPTHWISE_CONV_2D, 1));

    // Set Tensor #1 as Input #0, and Tensor #0 as Output #0
    interp.SetInputs({1});
    interp.SetOutputs({0});
  };

  const nnfw::support::tflite::interp::FunctionBuilder builder(setup);

  RandomTestParam param;

  param.verbose = verbose;
  param.tolerance = tolerance;

  int res = RandomTestRunner{SEED, param}.run(builder);

  EXPECT_EQ(res, 0);
}
