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

#include "TensorConvertToCommonLayer.h"

#include "internal/nnapi/feature/Reader.h"
#include "internal/nnapi/feature/View.h"

#include <util/feature/IndexIterator.h>

namespace neurun
{
namespace kernel
{
namespace cpu
{

bool TensorConvertToCommonLayer::convert()
{
  auto inputBuffer = _inputTensor->buffer();
  auto inputSize = _inputTensor->info()->total_size();

  auto outputBuffer = _outputTensor->buffer();
  auto outputSize = _outputTensor->info()->total_size();

  if (_tensorShape.rank() == 2)
  {
    const auto len = _tensorShape.dim(1);

    auto base = reinterpret_cast<float *>(outputBuffer);

    for (int32_t n = 0; n < len; ++n)
    {
      auto from = reinterpret_cast<const float *>(
          _inputTensor->ptr_to_element(::arm_compute::Coordinates{n}));
      auto into = base + n;

      *into = *from;
    }
  }
  else if (_tensorShape.rank() == 4)
  {
    auto featureShape = _tensorShape.asFeature();

    const ::internal::nnapi::feature::Reader<float> from{featureShape, inputBuffer, inputSize};
    ::internal::nnapi::feature::View<float> into{featureShape, outputBuffer, outputSize};

    ::nnfw::util::feature::iterate(featureShape)
        << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
             const auto value = from.at(batch, ch, row, col);
             into.at(batch, ch, row, col) = value;
           };
  }
}

void TensorConvertToCommonLayer::configure(::internal::cpu::Tensor *inputTensor,
                                           ::internal::common::Tensor *outputTensor,
                                           const Shape &tensorShape)
{
  _inputTensor = inputTensor;
  _outputTensor = outputTensor;
  _tensorShape = tensorShape;
}

void TensorConvertToCommonLayer::run() { convert(); }

} // namespace cpu
} // namespace kernel
} // namespace neurun

#endif
