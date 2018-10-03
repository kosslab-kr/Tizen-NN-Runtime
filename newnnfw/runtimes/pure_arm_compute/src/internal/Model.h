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

#ifndef __INTERNAL_MODEL_H__
#define __INTERNAL_MODEL_H__

namespace internal
{
namespace tflite
{
namespace operand
{

class Index
{
public:
  explicit Index(int value) : _value{value}
  {
    // DO NOTHING
  }

public:
  int asInt(void) const { return _value; }

private:
  int _value;
};

} // namespace operand
} // namespace tflite
} // namespace internal

#include <vector>
#include <cstdint>

#include "util/feature/Shape.h"
#include "util/matrix/Shape.h"
#include "util/kernel/Shape.h"
#include "util/tensor/Shape.h"

namespace internal
{
namespace tflite
{
namespace operand
{

struct Shape : public nnfw::util::tensor::Shape
{
public:
  Shape(uint32_t rank);

public:
  int32_t asVector(void) const;
  nnfw::util::feature::Shape asFeature(void) const;
  nnfw::util::matrix::Shape asMatrix(void) const;
  nnfw::util::kernel::Shape asKernel(void) const;
  nnfw::util::tensor::Shape asTensor(void) const;

public:
  void extendRank(size_t);
};

} // namespace operand
} // namespace tflite
} // namespace internal

#include <algorithm>

namespace internal
{
namespace tflite
{
namespace operand
{

struct Data
{
  virtual ~Data() = default;

  virtual size_t size(void) const = 0;
  virtual const uint8_t *base(void) const = 0;
};

class CachedData final : public Data
{
public:
  CachedData(const uint8_t *base, size_t size) : _base{new uint8_t[size]}, _size{size}
  {
    std::copy(base, base + size, _base);
  }

public:
  ~CachedData() { delete[] _base; }

public:
  size_t size(void) const override { return _size; }
  const uint8_t *base(void) const override { return _base; }

private:
  uint8_t *_base;
  size_t _size;
};

class ExternalData final : public Data
{
public:
  ExternalData(const uint8_t *base, size_t size) : _base{base}, _size{size}
  {
    // DO NOTHING
  }

public:
  size_t size(void) const override { return _size; }
  const uint8_t *base(void) const override { return _base; }

private:
  const uint8_t *_base;
  const size_t _size;
};

} // namespace operand
} // namespace tflite
} // namespace internal

#include <memory>
#include <cassert>
#include <functional>
#include "internal/Swizzle.h"

namespace internal
{
namespace tflite
{
namespace operand
{

class Object
{
public:
  explicit Object(const Shape &shape, const int32_t type, const float scale,
                  const int32_t zeroPoint)
      : _shape{shape}, _type{type}, _scale{scale}, _zeroPoint{zeroPoint}
  {
    // DO NOTHING
  }

public:
  const Shape &shape(void) const { return _shape; }
  const int32_t type(void) const { return _type; }
  const float scale(void) const { return _scale; }
  const int32_t zeroPoint(void) const { return _zeroPoint; }

private:
  void data(std::unique_ptr<Data> &&data) { _data = std::move(data); }

public:
  const Data &data(void) const { return *_data; }
  bool hasData(void) const { return _data != nullptr; }

public:
  template <typename T, typename... Args> void data(Args &&... args)
  {
    data(std::unique_ptr<T>(new T{std::forward<Args>(args)...}));
  }

public:
  template <typename T> T asScalar(void) const
  {
    assert((_shape.rank() == 0) || ((_shape.rank() == 1) && (_shape.dim(0) == 1)));
    assert(_data != nullptr);
    assert((_data->base() != nullptr) && (_data->size() == sizeof(T)));

    return *(reinterpret_cast<const T *>(_data->base()));
  }

public:
  template <typename T> T asReorderBits(size_t numOfBits) const
  {
    assert((_shape.rank() == 0) || ((_shape.rank() == 1) && (_shape.dim(0) == 1)));
    assert(_data != nullptr);
    assert((_data->base() != nullptr) && (_data->size() == sizeof(T)));

    return ReorderBits<T>(asScalar<T>(), numOfBits);
  }

private:
  const Shape _shape;
  const int32_t _type;
  const float _scale;
  const int32_t _zeroPoint;
  std::unique_ptr<Data> _data;
};

} // namespace operand
} // namespace tflite
} // namespace internal

#include <memory>

namespace internal
{
namespace tflite
{
namespace operand
{

class Set
{
public:
  void iterate(const std::function<void(const Index &)> &fn)
  {
    for (uint32_t n = 0; n < _objects.size(); ++n)
    {
      const Index operand_index{static_cast<int>(n)};
      fn(operand_index);
    }
  }

public:
  Index append(const Shape &, int32_t type, float scale, int32_t zeroPoint);

public:
  const Object &at(const Index &) const;
  Object &at(const Index &);
  size_t size(void) const { return _objects.size(); }

private:
  std::vector<std::unique_ptr<Object>> _objects;
};

} // namespace operand
} // namespace tflite
} // namespace internal

#include "internal/op/NodeVisitor.h"

namespace internal
{
namespace tflite
{
namespace op
{

class Sequence
{
public:
  Sequence() = default;

public:
  uint32_t size(void) const { return _ops.size(); }

public:
  op::Node &at(uint32_t nth) { return *(_ops.at(nth)); }
  const op::Node &at(uint32_t nth) const { return *(_ops.at(nth)); }

private:
  Sequence &emplace_back(std::unique_ptr<op::Node> &&node)
  {
    _ops.emplace_back(std::move(node));
    return (*this);
  }

public:
  template <typename T, typename... Args> Sequence &emplace_back(Args &&... args)
  {
    return emplace_back(std::unique_ptr<T>(new T{std::forward<Args>(args)...}));
  }

private:
  std::vector<std::unique_ptr<op::Node>> _ops;
};

} // namespace op
} // namespace tflite
} // namespace internal

namespace internal
{
namespace tflite
{

class Model
{
public:
  operand::Set &operands(void) { return _operands; }
  const operand::Set &operands(void) const { return _operands; }

public:
  op::Sequence &operations(void) { return _operations; }
  const op::Sequence &operations(void) const { return _operations; }

private:
  operand::Set _operands;
  op::Sequence _operations;

public:
  // TODO Hide these fields
  std::vector<operand::Index> inputs;
  std::vector<operand::Index> outputs;
};

} // namespace tflite
} // namespace internal

#endif // __INTERNAL_MODEL_H__
