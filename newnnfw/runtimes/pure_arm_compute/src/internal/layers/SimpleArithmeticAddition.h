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

#ifndef __SIMPLE_ARITHMETIC_ADDITION_H__
#define __SIMPLE_ARITHMETIC_ADDITION_H__

#include "internal/arm_compute.h"
#include <arm_compute/core/ITensor.h>

class SimpleArithmeticAddition : public ::arm_compute::IFunction
{
public:
  void configure(::arm_compute::ITensor *lhs, ::arm_compute::ITensor *rhs,
                 ::arm_compute::ITensor *out)
  {
    _lhs = lhs;
    _rhs = rhs;
    _out = out;
  }

public:
  void run(void) override
  {
    if (::internal::arm_compute::isGpuMode())
    {
      auto &q = ::arm_compute::CLScheduler::get().queue();

      CAST_CL(_lhs)->map(q);
      CAST_CL(_rhs)->map(q);
      CAST_CL(_out)->map(q);
    }

    arm_compute::Window window;
    window.use_tensor_dimensions(_out->info()->tensor_shape());

    execute_window_loop(window, [this](const arm_compute::Coordinates &id) {
      // NOTE Must be two input tensors of identical type
      //      Must be output tensor of the same type as input0.
      assert(_lhs->info()->data_type() == _rhs->info()->data_type());
      assert(_lhs->info()->data_type() == _out->info()->data_type());

      switch (_lhs->info()->data_type())
      {
        case ::arm_compute::DataType::F32:
        {
          const auto lhs_value = *reinterpret_cast<float *>(_lhs->ptr_to_element(id));
          const auto rhs_value = *reinterpret_cast<float *>(_rhs->ptr_to_element(id));
          *reinterpret_cast<float *>(_out->ptr_to_element(id)) = lhs_value + rhs_value;
          break;
        }
        case ::arm_compute::DataType::S32:
        {
          const auto lhs_value = *reinterpret_cast<int32_t *>(_lhs->ptr_to_element(id));
          const auto rhs_value = *reinterpret_cast<int32_t *>(_rhs->ptr_to_element(id));
          *reinterpret_cast<int32_t *>(_out->ptr_to_element(id)) = lhs_value + rhs_value;
          break;
        }
        case ::arm_compute::DataType::U32:
        {
          const auto lhs_value = *reinterpret_cast<uint32_t *>(_lhs->ptr_to_element(id));
          const auto rhs_value = *reinterpret_cast<uint32_t *>(_rhs->ptr_to_element(id));
          *reinterpret_cast<uint32_t *>(_out->ptr_to_element(id)) = lhs_value + rhs_value;
          break;
        }
        case ::arm_compute::DataType::QASYMM8:
        {
          const auto lhs_value = *reinterpret_cast<uint8_t *>(_lhs->ptr_to_element(id));
          const auto rhs_value = *reinterpret_cast<uint8_t *>(_rhs->ptr_to_element(id));
          // How to handle with overflow?
          *reinterpret_cast<uint8_t *>(_out->ptr_to_element(id)) = lhs_value + rhs_value;
          break;
        }
        default:
          throw std::runtime_error("Not supported, yet");
          break;
      }
    });

    if (::internal::arm_compute::isGpuMode())
    {
      auto &q = ::arm_compute::CLScheduler::get().queue();

      CAST_CL(_out)->unmap(q);
      CAST_CL(_rhs)->unmap(q);
      CAST_CL(_lhs)->unmap(q);
    }
  }

private:
  ::arm_compute::ITensor *_lhs;
  ::arm_compute::ITensor *_rhs;
  ::arm_compute::ITensor *_out;
};

#endif // __SIMPLE_ARITHMETIC_ADDITION_H__
