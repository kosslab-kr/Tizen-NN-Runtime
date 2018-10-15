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

#ifndef __FEATURE_LOGGING_LAYER_H__
#define __FEATURE_LOGGING_LAYER_H__

#include <arm_compute/core/ITensor.h>
#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include <iostream>
#include <iomanip>
#include <limits>

#include "internal/arm_compute.h"

class FeatureLoggingLayer : public ::arm_compute::IFunction
{
public:
  void configure(const std::string &tag, ::arm_compute::ITensor *target)
  {
    _tag = tag;
    _target = target;
  }

public:
  void run(void) override
  {
    if (::internal::arm_compute::isGpuMode())
    {
      auto &q = ::arm_compute::CLScheduler::get().queue();
      CAST_CL(_target)->map(q);
    }

    const size_t W = _target->info()->dimension(0);
    const size_t H = _target->info()->dimension(1);
    const size_t C = _target->info()->dimension(2);

    std::cout << _tag << std::endl;

    for (size_t ch = 0; ch < C; ++ch)
    {
      std::cout << "Channel #" << ch << std::endl;
      for (size_t row = 0; row < H; ++row)
      {
        for (size_t col = 0; col < W; ++col)
        {
          const arm_compute::Coordinates id{col, row, ch};
          const auto value = *reinterpret_cast<float *>(_target->ptr_to_element(id));

          // TODO Generalize this to integer types
          std::cout << std::setprecision(2);
          std::cout << std::setw(7);
          std::cout << std::setfill(' ');
          std::cout << std::fixed;
          std::cout << value << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }

    if (::internal::arm_compute::isGpuMode())
    {
      auto &q = ::arm_compute::CLScheduler::get().queue();
      CAST_CL(_target)->unmap(q);
    }
  }

private:
  std::string _tag;
  ::arm_compute::ITensor *_target;
};

#endif // __FEATURE_LOGGING_LAYER_H__
