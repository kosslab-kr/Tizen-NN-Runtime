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

//
// THIS FILE IS UNUSED BUT LEFT FOR FUTURE REFERNCE
//

#if 0

#include "TensorConvertFromCommonLayer.h"

#include "internal/nnapi/feature/Reader.h"
#include "backend/acl_cl/feature/View.h"

#include <util/feature/IndexIterator.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

namespace neurun
{
namespace kernel
{
namespace acl_cl
{

bool TensorConvertFromCommonLayer::convert()
{
  auto inputBuffer = _inputTensor->buffer();
  auto inputSize = _inputTensor->info()->total_size();

  auto &queue = ::arm_compute::CLScheduler::get().queue();

  _outputTensor->map(queue);

  if (_tensorShape.rank() == 2)
  {
    const auto len = _tensorShape.dim(1);

    auto base = reinterpret_cast<const float *>(inputBuffer);

    for (int32_t n = 0; n < len; ++n)
    {
      auto from = base + n;
      auto into =
          reinterpret_cast<float *>(_outputTensor->ptr_to_element(::arm_compute::Coordinates{n}));

      *into = *from;
    }
  }
  else if (_tensorShape.rank() == 4)
  {
    auto featureShape = _tensorShape.asFeature();

    const ::internal::nnapi::feature::Reader<float> from{featureShape, inputBuffer, inputSize};
    ::internal::arm_compute::feature::View<float> into{_outputTensor};

    ::nnfw::util::feature::iterate(featureShape)
        << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
             const auto value = from.at(batch, ch, row, col);
             into.at(batch, ch, row, col) = value;
           };
  }

  _outputTensor->unmap(queue);
}

void TensorConvertFromCommonLayer::configure(::internal::common::Tensor *inputTensor,
                                             ::arm_compute::ICLTensor *outputTensor,
                                             const ::neurun::graph::operand::Shape &tensorShape)
{
  _inputTensor = inputTensor;
  _outputTensor = outputTensor;
  _tensorShape = tensorShape;
}

void TensorConvertFromCommonLayer::run() { convert(); }

} // namespace acl_cl
} // namespace kernel
} // namespace neurun

#endif
