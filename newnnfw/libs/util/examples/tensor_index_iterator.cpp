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

#include "util/tensor/IndexIterator.h"

#include <array>

#include <iostream>
#include <algorithm>

#include <cassert>

void test_iterate(void)
{
  const nnfw::util::tensor::Shape shape{3, 4, 7};

  std::array<int, 3 * 4 * 7> array;

  array.fill(0);

  using nnfw::util::tensor::iterate;
  using nnfw::util::tensor::Index;

  iterate(shape) << [&](const Index &index) {
    assert(index.rank() == shape.rank());

    const size_t rank = index.rank();

    uint32_t offset = index.at(0);

    for (size_t axis = 1; axis < rank; ++axis)
    {
      offset *= shape.dim(axis);
      offset += index.at(axis);
    }

    array[offset] += 1;
  };

  assert(std::all_of(array.begin(), array.end(), [](int num) { return num == 1; }));
}

int main(int argc, char **argv)
{
  test_iterate();

  nnfw::util::tensor::Shape shape{3, 4, 3, 4};

  std::cout << "Iterate over tensor{3, 4, 3, 4}" << std::endl;

  nnfw::util::tensor::iterate(shape) << [](const nnfw::util::tensor::Index &index) {
    std::cout << "rank: " << index.rank() << std::endl;

    for (size_t d = 0; d < index.rank(); ++d)
    {
      std::cout << "  offset(" << d << ") = " << index.at(d) << std::endl;
    }
  };

  return 0;
}
