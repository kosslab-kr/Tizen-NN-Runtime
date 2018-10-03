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

#ifndef __NNFW_KERNEL_ACL_RESHAPE_H__
#define __NNFW_KERNEL_ACL_RESHAPE_H__

#include <OperationsUtils.h>

namespace nnfw {
namespace kernel {
namespace acl {

bool reshapeGeneric(const void* inputData, const nnfw::rt::Shape& inputShape,
                    void* outputData, const nnfw::rt::Shape& outputShape);
namespace neon {
bool reshapeGeneric(const void* inputData, const nnfw::rt::Shape& inputShape,
                    void* outputData, const nnfw::rt::Shape& outputShape);
} // namespace neon

} // namespace acl
} // namespace kernel
} // namespace nnfw

#endif // __NNFW_KERNEL_ACL_RESHAPE_H__