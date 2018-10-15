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

#ifndef __NNFW_SUPPORT_TFLITE_KERNELS_CUSTOM_OP_H__
#define __NNFW_SUPPORT_TFLITE_KERNELS_CUSTOM_OP_H__

#include "tensorflow/contrib/lite/context.h"
#include "support/tflite/kernels/TensorFlowMax.h"
#include "support/tflite/kernels/RSQRT.h"
#include "support/tflite/kernels/SquaredDifference.h"

namespace tflite
{
namespace ops
{
namespace custom
{
namespace nnfw
{

#define REGISTER_FUNCTION(Name)                                                                  \
  TfLiteRegistration *Register_##Name(void)                                                      \
  {                                                                                              \
    static TfLiteRegistration r = { Name::Init##Name , Name::Free##Name , Name::Prepare##Name ,  \
                                   Name::Eval##Name , 0, #Name};                                 \
    return &r;                                                                                   \
  }

REGISTER_FUNCTION(TensorFlowMax)
REGISTER_FUNCTION(RSQRT)
REGISTER_FUNCTION(SquaredDifference)
#undef REGISTER_FUNCTION

}  // namespace nnfw
}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif // __NNFW_SUPPORT_TFLITE_KERNELS_CUSTOM_OP_H__
