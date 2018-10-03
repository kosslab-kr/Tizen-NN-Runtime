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

#include "Convert.h"

namespace internal
{

::arm_compute::TensorShape asTensorShape(int32_t h, int32_t w)
{
  return ::arm_compute::TensorShape(w, h);
}

::arm_compute::TensorShape asTensorShape(const nnfw::util::feature::Shape &shape)
{
  return ::arm_compute::TensorShape(shape.W, shape.H, shape.C, shape.N);
}

::arm_compute::TensorShape asTensorShape(const nnfw::util::kernel::Shape &shape)
{
  return ::arm_compute::TensorShape(shape.W, shape.H, shape.C, shape.N);
}

::arm_compute::TensorInfo asTensorInfo(const nnfw::util::feature::Shape &shape)
{
  return ::arm_compute::TensorInfo(asTensorShape(shape), 1, ::arm_compute::DataType::F32);
}

::arm_compute::TensorInfo asTensorInfo(const nnfw::util::kernel::Shape &shape)
{
  return ::arm_compute::TensorInfo(asTensorShape(shape), 1, ::arm_compute::DataType::F32);
}

::arm_compute::TensorInfo asTensorInfo(int32_t size)
{
  return ::arm_compute::TensorInfo(::arm_compute::TensorShape(size), 1,
                                   ::arm_compute::DataType::F32);
}

::arm_compute::TensorInfo asTensorInfo(int32_t h, int32_t w)
{
  return ::arm_compute::TensorInfo(::arm_compute::TensorShape(w, h), 1,
                                   ::arm_compute::DataType::F32);
}

} // namespace internal
