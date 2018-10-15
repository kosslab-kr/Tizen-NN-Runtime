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

#ifndef __NNFW_SUPPORT_TFLITE_TENSOR_VIEW_H__
#define __NNFW_SUPPORT_TFLITE_TENSOR_VIEW_H__

#include "tensorflow/contrib/lite/interpreter.h"

#include "util/tensor/Shape.h"
#include "util/tensor/Index.h"
#include "util/tensor/Reader.h"
#include "util/tensor/NonIncreasingStride.h"

namespace nnfw
{
namespace support
{
namespace tflite
{

template<typename T> class TensorView final : public nnfw::util::tensor::Reader<T>
{
public:
  TensorView(const nnfw::util::tensor::Shape &shape, T *base)
      : _shape{shape}, _base{base}
  {
    // Set 'stride'
    _stride.init(_shape);
  }

public:
  const nnfw::util::tensor::Shape &shape(void) const { return _shape; }

public:
  T at(const nnfw::util::tensor::Index &index) const override
  {
    const auto offset = _stride.offset(index);
    return *(_base + offset);
  }

public:
  T &at(const nnfw::util::tensor::Index &index)
  {
    const auto offset = _stride.offset(index);
    return *(_base + offset);
  }

private:
  nnfw::util::tensor::Shape _shape;

public:
  T *_base;
  nnfw::util::tensor::NonIncreasingStride _stride;

public:
  // TODO Introduce Operand ID class
  static TensorView<T> make(::tflite::Interpreter &interp, int tensor_index)
  {
    auto tensor_ptr = interp.tensor(tensor_index);

    // Set 'shape'
    nnfw::util::tensor::Shape shape(tensor_ptr->dims->size);

    for (uint32_t axis = 0; axis < shape.rank(); ++axis)
    {
      shape.dim(axis) = tensor_ptr->dims->data[axis];
    }

    return TensorView<T>(shape, interp.typed_tensor<T>(tensor_index));
  }
};

} // namespace tflite
} // namespace support
} // namespace nnfw

#endif // __NNFW_SUPPORT_TFLITE_TENSOR_VIEW_H__
