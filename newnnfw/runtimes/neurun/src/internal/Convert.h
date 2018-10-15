/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __INTERNAL_CONVERT_H__
#define __INTERNAL_CONVERT_H__

#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>

#include "util/feature/Shape.h"
#include "util/kernel/Shape.h"

namespace internal
{

::arm_compute::TensorShape asTensorShape(int32_t h, int32_t w);
::arm_compute::TensorShape asTensorShape(const nnfw::util::feature::Shape &shape);
::arm_compute::TensorShape asTensorShape(const nnfw::util::kernel::Shape &shape);

::arm_compute::TensorInfo asTensorInfo(const nnfw::util::feature::Shape &shape);
::arm_compute::TensorInfo asTensorInfo(const nnfw::util::kernel::Shape &shape);
::arm_compute::TensorInfo asTensorInfo(int32_t size);
::arm_compute::TensorInfo asTensorInfo(int32_t h, int32_t w);

} // namespace internal

#endif // __INTERNAL_CONVERT_H__
