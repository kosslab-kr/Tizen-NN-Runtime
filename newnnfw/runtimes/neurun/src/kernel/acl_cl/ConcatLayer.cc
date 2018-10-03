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

#include "ConcatLayer.h"

#include <arm_compute/runtime/CL/CLScheduler.h>

#include "backend/acl_cl/kernel/View.h"
#include "logging.h"

namespace
{

bool matchSizeExceptAxis(const ::arm_compute::ICLTensor *t1, const ::arm_compute::ICLTensor *t2,
                         uint32_t axis)
{
  assert(t1->info()->num_dimensions() <= 4);
  assert(t2->info()->num_dimensions() <= 4);

  for (uint32_t i = 0; i < 4; i++)
  {
    if (axis == i)
      continue;
    if (t1->info()->dimension(i) != t2->info()->dimension(i))
      return false;
  }
  return true;
}

} // namespace {anonymous}

namespace neurun
{
namespace kernel
{
namespace acl_cl
{

ConcatLayer::ConcatLayer()
    : _input_allocs(), _output_alloc(nullptr), _axis(0), _input_type(OperandType::SCALAR_FLOAT32)
{
  // DO NOTHING
}

bool ConcatLayer::concatenationFloat32()
{
  // Input and output size check
  {
    // NOTE Support only tensor with dimension 4 or less

    uint32_t axis_sum = 0;

    for (auto input : _input_allocs)
    {
      assert(matchSizeExceptAxis(_output_alloc, input, _axis));
      axis_sum += input->info()->dimension(_axis);
    }

    assert(_output_alloc->info()->dimension(_axis) == axis_sum);
  }

  VERBOSE(Concat_RUN) << "START Concat" << std::endl;

  // Perform operation
  {
    uint32_t axis_offset = 0;

    auto &queue = ::arm_compute::CLScheduler::get().queue();

    _output_alloc->map(queue);
    ::internal::arm_compute::kernel::View<float> output_view{_output_alloc};

    for (auto input : _input_allocs)
    {
      input->map(queue);
      const ::internal::arm_compute::kernel::View<float> input_reader{input};

      for (uint32_t n = 0; n < input_reader.shape().N; n++)
      {
        for (uint32_t c = 0; c < input_reader.shape().C; c++)
        {
          for (uint32_t h = 0; h < input_reader.shape().H; h++)
          {
            for (uint32_t w = 0; w < input_reader.shape().W; w++)
            {
              uint32_t no = (_axis == 3) ? axis_offset : 0;
              uint32_t co = (_axis == 2) ? axis_offset : 0;
              uint32_t ho = (_axis == 1) ? axis_offset : 0;
              uint32_t wo = (_axis == 0) ? axis_offset : 0;
              output_view.at(n + no, c + co, h + ho, w + wo) = input_reader.at(n, c, h, w);
            }
          }
        }
      }
      if (_axis == 3)
        axis_offset += input_reader.shape().N;
      if (_axis == 2)
        axis_offset += input_reader.shape().C;
      if (_axis == 1)
        axis_offset += input_reader.shape().H;
      if (_axis == 0)
        axis_offset += input_reader.shape().W;

      input->unmap(queue);
    }
    _output_alloc->unmap(queue);
  }

  VERBOSE(Concat_RUN) << "End   Concat" << std::endl;

  return true;
}

void ConcatLayer::configure(const std::vector<::arm_compute::ICLTensor *> &input_allocs,
                            int32_t axis, ::arm_compute::ICLTensor *output_alloc)
{
  _input_allocs = input_allocs;
  _output_alloc = output_alloc;

  assert(axis < 4);

  // This map converts NHWC to NCHW(reversed)
  // NHWC -> WHCN
  static const uint32_t axis_map[] = {3, 1, 0, 2};
  _axis = axis_map[axis];

  // TODO Support Quant8
  _input_type = OperandType::TENSOR_FLOAT32;
}

void ConcatLayer::run()
{
  if (_input_type == OperandType::TENSOR_FLOAT32)
  {
    concatenationFloat32();
  }
  else if (_input_type == OperandType::TENSOR_QUANT8_ASYMM)
  {
    throw std::runtime_error("NYI - concatenationQuant8()");
  }
}

} // namespace acl_cl
} // namespace kernel
} // namespace neurun
