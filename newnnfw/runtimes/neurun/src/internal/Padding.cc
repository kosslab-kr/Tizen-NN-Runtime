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

#include "internal/Padding.h"

#include <algorithm>

namespace internal
{

Padding valid_padding(void)
{
  //
  // ANEURALNETWORKS_PADDING_VALID
  //
  // VALID padding. No padding.
  //
  // When the input size is not evenly divisible by the filter size,
  // the input at the end that could not fill the whole filter tile
  // will simply be ignored.
  //
  Padding padding;

  padding.top = 0;
  padding.bottom = 0;
  padding.left = 0;
  padding.right = 0;

  return padding;
}

Padding same_padding(const nnfw::util::feature::Shape &ifm_shape,
                     const nnfw::util::feature::Shape &ofm_shape, const Stride &stride, uint32_t kw,
                     uint32_t kh)
{
  Padding padding;

  // ANEURALNETWORKS_PADDING_SAME (from NNAPI spec)
  //
  // SAME padding. Padding on both ends are the "same":
  //
  //	padding_to_beginning = total_padding / 2
  //  padding_to_end = (total_padding + 1)/2.
  //
  const int32_t vertical_needed_input = (ofm_shape.H - 1) * stride.vertical + kh;
  const int32_t vertical_total_padding = std::max(0, vertical_needed_input - ifm_shape.H);

  const int32_t horizontal_needed_input = (ofm_shape.W - 1) * stride.horizontal + kw;
  const int32_t horizontal_total_padding = std::max(0, horizontal_needed_input - ifm_shape.W);

  padding.top = vertical_total_padding / 2;
  padding.bottom = (vertical_total_padding + 1) / 2;
  padding.left = horizontal_total_padding / 2;
  padding.right = (horizontal_total_padding + 1) / 2;

  return padding;
}

} // namespace internal
