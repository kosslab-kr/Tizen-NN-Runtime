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

#ifndef __NNFW_SUPPORT_TFLITE_KERNELS_RSQRT_H__
#define __NNFW_SUPPORT_TFLITE_KERNELS_RSQRT_H__

#include "tensorflow/contrib/lite/context.h"

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

  void *InitRSQRT(TfLiteContext *context, const char *buffer, size_t length);
  void FreeRSQRT(TfLiteContext *context, void *buffer);
  TfLiteStatus PrepareRSQRT(TfLiteContext *context, TfLiteNode *node);
  TfLiteStatus EvalRSQRT(TfLiteContext *context, TfLiteNode *node);

} // namespace RSQRT
} // namespace nnfw
} // namespace custom
} // namespace ops
} // namespace tflite

#endif
