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

#include "internal/layers/SimpleSpaceToDepth.h"

#include <arm_compute/runtime/CL/CLScheduler.h>

void SimpleSpaceToDepth::configure(::arm_compute::ITensor *input, ::arm_compute::ITensor *output,
                                   int32_t block_size,
                                   const ::arm_compute::Coordinates &axises = {3, 1, 0, 2})
{
  assert(input->info()->num_dimensions() == 4);
  assert(output->info()->num_dimensions() == 4);
  const auto rank = axises.num_dimensions();
  assert(rank == 4);
  for (int i = 0; i < rank; ++i)
  {
    assert(axises[i] >= 0);
    assert(axises[i] < rank);
  }

  _input = input;
  _output = output;
  _block_size = block_size;
  _axises = axises;
}

inline int32_t Offset4D(const ::arm_compute::TensorShape &shape, int32_t b, int32_t h, int32_t w,
                        int32_t d, const ::arm_compute::Coordinates &axises)
{
  // b, h, w, d >= 0
  size_t indexes[4];
  indexes[axises[0]] = b;
  indexes[axises[1]] = h;
  indexes[axises[2]] = w;
  indexes[axises[3]] = d;

  int32_t offset = indexes[3] * shape[2] * shape[1] * shape[0];
  offset += indexes[2] * shape[1] * shape[0];
  offset += indexes[1] * shape[0];
  offset += indexes[0];
  return offset;
}

template <typename T>
inline void SpaceToDepth(const T *input_data, const ::arm_compute::TensorShape &input_shape,
                         int32_t block_size, T *output_data,
                         const ::arm_compute::TensorShape &output_shape,
                         const ::arm_compute::Coordinates &axises)
{
  const int input_batch = input_shape[axises[0]];
  const int input_height = input_shape[axises[1]];
  const int input_width = input_shape[axises[2]];
  const int input_depth = input_shape[axises[3]];

  const int output_batch = output_shape[axises[0]];
  const int output_height = output_shape[axises[1]];
  const int output_width = output_shape[axises[2]];
  const int output_depth = output_shape[axises[3]];

  assert(input_batch == output_batch);
  assert(input_height == output_height * block_size);
  assert(input_width == output_width * block_size);
  assert(input_depth * block_size * block_size == output_depth);

  for (int in_b = 0; in_b < input_batch; ++in_b)
  {
    for (int in_h = 0; in_h < input_height; ++in_h)
    {
      for (int in_w = 0; in_w < input_width; ++in_w)
      {
        for (int in_d = 0; in_d < input_depth; ++in_d)
        {
          const int out_b = in_b;
          const int out_h = in_h / block_size;
          const int out_w = in_w / block_size;
          const int out_d =
              in_d + ((in_h % block_size) * block_size + in_w % block_size) * input_depth;

          const int input_index = Offset4D(input_shape, in_b, in_h, in_w, in_d, axises);
          const int output_index = Offset4D(output_shape, out_b, out_h, out_w, out_d, axises);

          output_data[output_index] = input_data[input_index];
        }
      }
    }
  }
}

void SimpleSpaceToDepth::run()
{
  if (::internal::arm_compute::isGpuMode())
  {
    auto &q = ::arm_compute::CLScheduler::get().queue();

    CAST_CL(_input)->map(q);
    CAST_CL(_output)->map(q);
  }

  auto input_buf = _input->buffer();
  auto output_buf = _output->buffer();
  switch (_input->info()->data_type())
  {
    case ::arm_compute::DataType::U8:
    case ::arm_compute::DataType::QASYMM8:
      SpaceToDepth(reinterpret_cast<const uint8_t *>(input_buf), _input->info()->tensor_shape(),
                   _block_size, reinterpret_cast<uint8_t *>(output_buf),
                   _output->info()->tensor_shape(), _axises);
      break;
    case ::arm_compute::DataType::S8:
      SpaceToDepth(reinterpret_cast<const int8_t *>(input_buf), _input->info()->tensor_shape(),
                   _block_size, reinterpret_cast<int8_t *>(output_buf),
                   _output->info()->tensor_shape(), _axises);
      break;
    case ::arm_compute::DataType::U32:
      SpaceToDepth(reinterpret_cast<const uint32_t *>(input_buf), _input->info()->tensor_shape(),
                   _block_size, reinterpret_cast<uint32_t *>(output_buf),
                   _output->info()->tensor_shape(), _axises);
      break;
    case ::arm_compute::DataType::S32:
      SpaceToDepth(reinterpret_cast<const int32_t *>(input_buf), _input->info()->tensor_shape(),
                   _block_size, reinterpret_cast<int32_t *>(output_buf),
                   _output->info()->tensor_shape(), _axises);
      break;
    case ::arm_compute::DataType::F32:
      SpaceToDepth(reinterpret_cast<const float *>(input_buf), _input->info()->tensor_shape(),
                   _block_size, reinterpret_cast<float *>(output_buf),
                   _output->info()->tensor_shape(), _axises);
      break;
    default:
      ARM_COMPUTE_ERROR("DataType not supported");
      break;
  }

  if (::internal::arm_compute::isGpuMode())
  {
    auto &q = ::arm_compute::CLScheduler::get().queue();

    CAST_CL(_input)->unmap(q);
    CAST_CL(_output)->unmap(q);
  }
}
