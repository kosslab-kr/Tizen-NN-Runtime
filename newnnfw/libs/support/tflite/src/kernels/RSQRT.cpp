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

#include "support/tflite/kernels/RSQRT.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"

#include <cmath>
#include <iostream>

namespace tflite
{
namespace ops
{
namespace custom
{
namespace nnfw
{
namespace RSQRT
{

void *InitRSQRT(TfLiteContext *context, const char *buffer, size_t length) { return nullptr; }

void FreeRSQRT(TfLiteContext *context, void *buffer) {}

TfLiteStatus PrepareRSQRT(TfLiteContext *context, TfLiteNode *node)
{
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor *input = GetInput(context, node, 0);
  TfLiteTensor *output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  // Quantized float is not supported yet.
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  return context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims));
}

inline TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node, float float_func(float))
{
  const TfLiteTensor *input = GetInput(context, node, 0);
  TfLiteTensor *output = GetOutput(context, node, 0);
  switch (input->type)
  {
    case kTfLiteFloat32:
    {
      size_t elements = NumElements(input);
      const float *in = input->data.f;
      const float *in_end = in + elements;
      float *out = output->data.f;
      for (; in < in_end; in++, out++)
        *out = float_func(*in);
      return kTfLiteOk;
    }
    default:
    {
      context->ReportError(context, "Input type is %d, requires float32", input->type);
      return kTfLiteError;
    }
  }
}

TfLiteStatus EvalRSQRT(TfLiteContext *context, TfLiteNode *node)
{
  return Eval(context, node, [](float f) { return 1.f / std::sqrt(f); });
}

} // namespace RSQRT
} // namespace nnfw
} // namespace custom
} // namespace ops
} // namespace tflite
