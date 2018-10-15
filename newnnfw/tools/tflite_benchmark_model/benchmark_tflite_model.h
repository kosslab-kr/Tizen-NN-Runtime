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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef __TFLITE_BENCHMARK_MODEL_BENCHMARK_TFLITE_MODEL_H__
#define __TFLITE_BENCHMARK_MODEL_BENCHMARK_TFLITE_MODEL_H__

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/profiling/profile_summarizer.h"
#include "benchmark_model.h"

namespace nnfw {
namespace benchmark {

// Dumps profiling events if profiling is enabled
class ProfilingListener : public BenchmarkListener {
 public:
  explicit ProfilingListener() : interpreter_(nullptr), has_profiles_(false) {}

  void SetInterpreter(tflite::Interpreter* interpreter);

  void OnSingleRunStart(RunType run_type) override;

  void OnSingleRunEnd() override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;

 private:
  tflite::Interpreter* interpreter_;
  tflite::profiling::Profiler profiler_;
  tflite::profiling::ProfileSummarizer summarizer_;
  bool has_profiles_;
};

// Benchmarks a TFLite model by running tflite interpreter.
class BenchmarkTfLiteModel : public BenchmarkModel {
 public:
  BenchmarkTfLiteModel();
  BenchmarkTfLiteModel(BenchmarkParams params);

  std::vector<Flag> GetFlags() override;
  void LogFlags() override;
  bool ValidateFlags() override;
  uint64_t ComputeInputBytes() override;
  void Init() override;
  void RunImpl() override;
  virtual ~BenchmarkTfLiteModel() {}

  struct InputLayerInfo {
    std::string name;
    std::vector<int> shape;
  };

 private:
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::vector<InputLayerInfo> inputs;
  ProfilingListener profiling_listener_;
};

}  // namespace benchmark
}  // namespace nnfw

#endif  //__TFLITE_BENCHMARK_MODEL_BENCHMARK_TFLITE_MODEL_H__
