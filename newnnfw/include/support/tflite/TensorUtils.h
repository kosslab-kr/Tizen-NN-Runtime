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

#ifndef __NNFW_SUPPORT_TFLITE_TENSOR_UTILS_H__
#define __NNFW_SUPPORT_TFLITE_TENSOR_UTILS_H__

#include <tensorflow/contrib/lite/context.h>

namespace nnfw
{
namespace support
{
namespace tflite
{

inline bool isFloatTensor(const TfLiteTensor *tensor)
{
  return tensor->type == kTfLiteFloat32;
}

inline bool isFeatureTensor(const TfLiteTensor *tensor)
{
  return (tensor->dims->size == 4) && (tensor->dims->data[0] == 1);
}

} // namespace tflite
} // namespace support
} // namespace nnfw

#endif // __NNFW_SUPPORT_TFLITE_TENSOR_UTILS_H__
