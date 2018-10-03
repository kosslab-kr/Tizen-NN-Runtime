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

#include <iostream>
#include "PadLayer.h"
#include <arm_compute/runtime/CL/CLScheduler.h>

void PadLayer::configure(::arm_compute::ICLTensor *input, ::arm_compute::ICLTensor *output,
                         unsigned int border_width)
{
  _input = input;
  _output = output;
  _border_width = border_width;
  _output_height = _output->info()->dimension(0);
  _output_width = _output->info()->dimension(1);

  uint8_t constant_border_value = 0;
  ::arm_compute::PixelValue constant_pixel_value = ::arm_compute::PixelValue(constant_border_value);

  unsigned int padding_size = _border_width;
  input->info()->extend_padding(::arm_compute::PaddingSize{padding_size});
  _fillborderkernel.configure(input, _border_width, ::arm_compute::BorderMode::CONSTANT,
                              constant_pixel_value);
}

void PadLayer::run(void)
{
  _fillborderkernel.run();

  ::arm_compute::Coordinates coordinates =
      ::arm_compute::Coordinates(-_border_width, -_border_width);
  ::arm_compute::TensorShape new_tensor_shape =
      ::arm_compute::TensorShape(_output_height, _output_width);

  /* NOTE: The cl kernel fills the data in the borders(not in the tensor).
           Once the tensor is received back at NNAPI, we are adjusting
           the valid region in such a way that the padding becomes part of the tensor itself
           and matches the size of output. */
  _input->info()->set_valid_region(::arm_compute::ValidRegion(coordinates, new_tensor_shape));

  /* NOTE: Since cl kernel does not have an argument for output tensor while NNAPI does.
           We need to map the input (tensor that is passed to the cl kernel) back to
           output. */

  // TODO: Write a modified CLCopy kernel to do this job.
  populateOutput();
}

void PadLayer::populateOutput()
{
  auto &queue = ::arm_compute::CLScheduler::get().queue();
  _input->map(queue);
  _output->map(queue);

  auto input_tensor = static_cast<::arm_compute::ITensor *>(_input);
  auto const source_data = input_tensor->buffer();

  auto output_tensor = static_cast<::arm_compute::ITensor *>(_output);
  auto dst_data = output_tensor->buffer();

  memmove(dst_data, source_data, _output_height * _output_width * 4);

  _input->unmap(queue);
  _output->unmap(queue);
}
