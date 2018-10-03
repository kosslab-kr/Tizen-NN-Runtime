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

#ifndef __NNFW_UTIL_TENSOR_COMPARATOR_H__
#define __NNFW_UTIL_TENSOR_COMPARATOR_H__

#include "util/tensor/Index.h"
#include "util/tensor/Shape.h"
#include "util/tensor/Reader.h"
#include "util/tensor/Diff.h"

#include <functional>

#include <vector>

namespace nnfw
{
namespace util
{
namespace tensor
{

class Comparator
{
public:
  Comparator(const std::function<bool (float lhs, float rhs)> &fn) : _compare_fn{fn}
  {
    // DO NOTHING
  }

public:
  struct Observer
  {
    virtual void notify(const Index &index, float expected, float obtained) = 0;
  };

public:
  // NOTE Observer should live longer than comparator
  std::vector<Diff<float>> compare(const Shape &shape,
                                        const Reader<float> &expected,
                                        const Reader<float> &obtained,
                                        Observer *observer = nullptr) const;

private:
  std::function<bool (float lhs, float rhs)> _compare_fn;
};

} // namespace tensor
} // namespace util
} // namespace nnfw

#endif // __NNFW_UTIL_TENSOR_COMPARATOR_H__
