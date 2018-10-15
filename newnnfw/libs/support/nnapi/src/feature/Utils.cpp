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

#include "support/nnapi/feature/Utils.h"

namespace nnfw
{
namespace support
{
namespace nnapi
{
namespace feature
{

uint32_t indexOf(const nnfw::util::feature::Shape &shape, uint32_t ch, uint32_t row, uint32_t col)
{
  uint32_t res = 0;

  // NNAPI assumes that NHWC ordering for feature map
  res += row * shape.W * shape.C;
  res += col * shape.C;
  res += ch;

  return res;
}

} // namespace feature
} // namespace nnapi
} // namespace support
} // namespace nnfw
