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

#ifndef __NNFW_SUPPORT_TFLITE_COMPARE_H__
#define __NNFW_SUPPORT_TFLITE_COMPARE_H__

#include "tensorflow/contrib/lite/interpreter.h"

#include "util/tensor/Index.h"
#include "util/tensor/Diff.h"
#include "util/tensor/Shape.h"
#include "util/tensor/Comparator.h"

#include "support/tflite/TensorView.h"

#include <functional>
#include <vector>

class TfLiteInterpMatchApp
{
public:
  TfLiteInterpMatchApp(const nnfw::util::tensor::Comparator &comparator)
    : _verbose{false}, _comparator(comparator)
  {
    // DO NOTHING
  }

public:
  int &verbose(void) { return _verbose; }

private:
  int _verbose;

public:
  bool run(::tflite::Interpreter &pure, ::tflite::Interpreter &nnapi) const;
  template <typename T>
  bool compareSingleTensorView(const nnfw::support::tflite::TensorView<T> &expected,
                               const nnfw::support::tflite::TensorView<T> &obtained,
                               int id) const;

private:
  const nnfw::util::tensor::Comparator &_comparator;
};

#include "support/tflite/interp/Builder.h"
#include "support/tflite/Quantization.h"

#include <random>

class RandomGenerator
{
public:
  RandomGenerator(int seed, float mean, float stddev,
                  const TfLiteQuantizationParams quantization = make_default_quantization())
      : _rand{seed}, _dist{mean, stddev}, _quantization{quantization}
  {
    // DO NOTHING
  }

public:
  template <typename T>
  T generate(const ::nnfw::util::tensor::Shape &, const ::nnfw::util::tensor::Index &)
  {
    return generate<T>();
  }

  template <typename T> T generate(void)
  {
    return _dist(_rand);
  }

private:
  std::minstd_rand _rand;
  std::normal_distribution<float> _dist;
  const TfLiteQuantizationParams _quantization;
};

template <>
uint8_t RandomGenerator::generate<uint8_t>(void);

// For NNAPI testing
struct RandomTestParam
{
  int verbose;
  int tolerance;
  int tensor_logging = 0;
  std::string log_path = ""; // meaningful only when tensor_logging is 1
};

class RandomTestRunner
{
public:
  RandomTestRunner(int seed, const RandomTestParam &param,
                   const TfLiteQuantizationParams quantization = make_default_quantization())
      : _randgen{seed, 0.0f, 2.0f, quantization}, _param{param}
  {
    // DO NOTHING
  }

public:
  // NOTE this method updates '_rand'
  // Return 0 if test succeeds
  int run(const nnfw::support::tflite::interp::Builder &builder);

public:
  RandomGenerator &generator() { return _randgen; };

private:
  RandomGenerator _randgen;
  const RandomTestParam _param;

public:
  static RandomTestRunner make(int seed);
};

#endif // __NNFW_SUPPORT_TFLITE_COMPARE_H__
