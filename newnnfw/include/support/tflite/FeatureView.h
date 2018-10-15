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

#ifndef __NNFW_SUPPORT_TFLITE_FEATURE_VIEW_H__
#define __NNFW_SUPPORT_TFLITE_FEATURE_VIEW_H__

#include "tensorflow/contrib/lite/interpreter.h"

#include "support/tflite/InputIndex.h"
#include "support/tflite/OutputIndex.h"

#include "util/feature/Shape.h"
#include "util/feature/Reader.h"

namespace nnfw
{
namespace support
{
namespace tflite
{

template<typename T> class FeatureView;

template<> class FeatureView<float> : public nnfw::util::feature::Reader<float>
{
public:
  FeatureView(::tflite::Interpreter &interp, const InputIndex &index);
  FeatureView(::tflite::Interpreter &interp, const OutputIndex &index);

public:
  float at(uint32_t ch, uint32_t row, uint32_t col) const;
  float &at(uint32_t ch, uint32_t row, uint32_t col);

private:
  uint32_t getElementOffset(uint32_t ch, uint32_t row, uint32_t col) const
  {
    uint32_t res = 0;

    // TensorFlow Lite assumes that NHWC ordering for tessor
    res += row * _shape.W * _shape.C;
    res += col * _shape.C;
    res += ch;

    return res;
  }

private:
  nnfw::util::feature::Shape _shape;
  float *_base;
};

} // namespace tflite
} // namespace support
} // namespace nnfw

#endif // __NNFW_SUPPORT_TFLITE_FEATURE_VIEW_H__
