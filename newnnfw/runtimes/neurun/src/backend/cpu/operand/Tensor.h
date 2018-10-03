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

#ifndef __NEURUN_BACKEND_CPU_OPERAND_TENSOR_H__
#define __NEURUN_BACKEND_CPU_OPERAND_TENSOR_H__

#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/TensorInfo.h>

namespace neurun
{
namespace backend
{
namespace cpu
{
namespace operand
{

class Tensor : public ::arm_compute::ITensor
{
public:
  Tensor() = default;

  Tensor(::arm_compute::TensorInfo info) : _info(info)
  {
    // TODO Do not allocate buffer here. This tensor is just an abstract Tensor object for cpu.
    uint32_t size = _info.total_size(); // NOTE This size may not be accurate
    _buffer = new uint8_t[size];        // NOTE The allocated buffer is never deallocated.
  }

  Tensor(uint8_t *buffer) : _buffer(buffer)
  {
    // DO NOTHING
  }

public:
  void setBuffer(uint8_t *buffer) { _buffer = buffer; }

public:
  ::arm_compute::TensorInfo *info() const override
  {
    return const_cast<::arm_compute::TensorInfo *>(&_info);
  }

  ::arm_compute::TensorInfo *info() override { return &_info; }

  uint8_t *buffer() const override { return _buffer; }

private:
  ::arm_compute::TensorInfo _info;
  uint8_t *_buffer = nullptr;
};

} // namespace operand
} // namespace cpu
} // namespace backend
} // namespace neurun

#endif // __NEURUN_BACKEND_CPU_OPERAND_TENSOR_H__
