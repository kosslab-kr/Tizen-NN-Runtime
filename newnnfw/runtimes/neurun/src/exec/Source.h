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

#ifndef __NEURUN_EXEC_SOURCE_H__
#define __NEURUN_EXEC_SOURCE_H__

#include <cassert>

#include <arm_compute/runtime/CL/CLTensor.h>

#include <util/feature/Shape.h>
#include <util/feature/IndexIterator.h>

#include "backend/cpu/operand/Tensor.h" // TODO Remove this dependency to backend
#include "internal/nnapi/feature/Reader.h"
#include "internal/nnapi/feature/View.h"

#include "backend/acl_cl/feature/View.h"

namespace neurun
{
namespace exec
{

struct Source
{
  virtual ~Source() = default;

  virtual void push(::arm_compute::ITensor &tensor) const = 0;
};

//
// VectorSource
//
class VectorSource final : public Source
{
public:
  VectorSource(const int32_t vlen, const uint8_t *base, const size_t size)
      : _vlen{vlen}, _base{base}
  {
    (void)size; // Workaround for unused variable in release mode
    assert(size >= _vlen * sizeof(float));
  }

public:
  void push(::arm_compute::ITensor &tensor) const override
  {
    auto base = reinterpret_cast<const float *>(_base);

    for (int32_t n = 0; n < _vlen; ++n)
    {
      auto from = base + n;
      auto into = reinterpret_cast<float *>(tensor.ptr_to_element(::arm_compute::Coordinates{n}));

      *into = *from;
    }
  }

private:
  const int32_t _vlen;
  const uint8_t *const _base;
};

//
// FeatureSource
//
class FeatureSource final : public Source
{
public:
  FeatureSource(const nnfw::util::feature::Shape &shape, const uint8_t *base, const size_t size)
      : _shape{shape}, _base{base}, _size{size}
  {
    // DO NOTHING
  }

public:
  void push(::arm_compute::ITensor &tensor) const override
  {
    // TODO: This is just workaround codes, It needs to refactor.
    if (typeid(tensor) == typeid(neurun::backend::cpu::operand::Tensor))
    {
      const ::internal::nnapi::feature::Reader<float> from{_shape, _base, _size};
      ::internal::nnapi::feature::View<float> into{_shape, tensor.buffer(), _size};

      ::nnfw::util::feature::iterate(_shape)
          << [&](uint32_t bat, uint32_t ch, uint32_t row, uint32_t col) {
               const auto value = from.at(bat, ch, row, col);
               into.at(bat, ch, row, col) = value;
             };
    }
    else if (typeid(tensor) == typeid(::arm_compute::CLTensor))
    {
      const ::internal::nnapi::feature::Reader<float> from{_shape, _base, _size};
      ::internal::arm_compute::feature::View<float> into{&tensor};

      ::nnfw::util::feature::iterate(_shape)
          << [&](uint32_t bat, uint32_t ch, uint32_t row, uint32_t col) {
               const auto value = from.at(bat, ch, row, col);
               into.at(bat, ch, row, col) = value;
             };
    }
  }

private:
  const nnfw::util::feature::Shape _shape;
  const uint8_t *const _base;
  const size_t _size;
};

} // namespace exec
} // namespace neurun

#endif // __NEURUN_EXEC_SOURCE_H__
