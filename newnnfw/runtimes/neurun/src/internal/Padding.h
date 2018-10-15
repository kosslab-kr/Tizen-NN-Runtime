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

#ifndef __INTERNAL_PADDING_H__
#define __INTERNAL_PADDING_H__

#include <stdint.h>

#include <util/feature/Shape.h>

namespace internal
{

struct Padding
{
  uint32_t top;
  uint32_t bottom;
  uint32_t left;
  uint32_t right;
};

struct Stride
{
  uint32_t vertical;
  uint32_t horizontal;
};

Padding valid_padding(void);
Padding same_padding(const nnfw::util::feature::Shape &ifm_shape,
                     const nnfw::util::feature::Shape &ofm_shape, const Stride &stride, uint32_t kw,
                     uint32_t kh);

} // namespace internal

#endif // __INTERNAL_PADDING_H__
