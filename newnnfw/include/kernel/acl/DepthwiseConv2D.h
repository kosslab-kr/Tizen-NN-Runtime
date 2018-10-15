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

#ifndef __NNFW_KERNEL_ACL_DEPTHWISECONV2D_H__
#define __NNFW_KERNEL_ACL_DEPTHWISECONV2D_H__

#include <OperationsUtils.h>

namespace nnfw {
namespace kernel {
namespace acl {

bool depthwiseConvFloat32(const float* inputData, const nnfw::rt::Shape& inputShape,
                          const float* filterData, const nnfw::rt::Shape& filterShape,
                          const float* biasData, const nnfw::rt::Shape& biasShape,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
                          int32_t depth_multiplier, int32_t activation,
                          float* outputData, const nnfw::rt::Shape& outputShape);

namespace neon {

bool depthwiseConvFloat32(const float* inputData, const nnfw::rt::Shape& inputShape,
                          const float* filterData, const nnfw::rt::Shape& filterShape,
                          const float* biasData, const nnfw::rt::Shape& biasShape,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
                          int32_t depth_multiplier, int32_t activation,
                          float* outputData, const nnfw::rt::Shape& outputShape);

} // namespace neon

} // namespace acl
} // namespace kernel
} // namespace nnfw

#endif // __NNFW_KERNEL_ACL_DEPTHWISECONV2D_H__
