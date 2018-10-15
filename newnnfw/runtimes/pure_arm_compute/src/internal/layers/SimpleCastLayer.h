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

#ifndef __SIMPLE_CAST_LAYER_H__
#define __SIMPLE_CAST_LAYER_H__

#include <arm_compute/core/ITensor.h>

#include "internal/arm_compute.h"
#include "internal/op/Cast.h"

class SimpleCastLayer : public ::arm_compute::IFunction
{
public:
  void configure(::arm_compute::ITensor *in, ::arm_compute::ITensor *out)
  {
    _in = in;
    _out = out;
  }

public:
  void run(void) override
  {
    if (::internal::arm_compute::isGpuMode())
    {
      auto &q = ::arm_compute::CLScheduler::get().queue();
      CAST_CL(_in)->map(q);
      CAST_CL(_out)->map(q);
    }

    arm_compute::Window window;
    window.use_tensor_dimensions(_out->info()->tensor_shape());

    execute_window_loop(window,
                        [this](const arm_compute::Coordinates &id) { castData(_in, _out, id); });

    if (::internal::arm_compute::isGpuMode())
    {
      auto &q = ::arm_compute::CLScheduler::get().queue();
      CAST_CL(_out)->unmap(q);
      CAST_CL(_in)->unmap(q);
    }
  }

  void castData(::arm_compute::ITensor *in, ::arm_compute::ITensor *out,
                const arm_compute::Coordinates &id)
  {
    switch (in->info()->data_type())
    {
      case ::arm_compute::DataType::F32:
      {
        copyCast(*reinterpret_cast<float *>(in->ptr_to_element(id)), out, id);
        break;
      }
      case ::arm_compute::DataType::S32:
      {
        copyCast(*reinterpret_cast<int32_t *>(in->ptr_to_element(id)), out, id);
        break;
      }
      case ::arm_compute::DataType::U32:
      {
        copyCast(*reinterpret_cast<uint32_t *>(in->ptr_to_element(id)), out, id);
        break;
      }
      case ::arm_compute::DataType::QASYMM8:
      {
        const uint8_t quantizedValue = *(in->ptr_to_element(id));
        copyCast(in->info()->quantization_info().dequantize(quantizedValue), out, id);
        break;
      }
      default:
        throw std::runtime_error("Not supported, yet");
        break;
    }
  }

private:
  ::arm_compute::ITensor *_in;
  ::arm_compute::ITensor *_out;
};

#endif // __SIMPLE_CAST_LAYER_H__
