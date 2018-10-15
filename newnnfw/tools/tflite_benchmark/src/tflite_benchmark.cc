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

#include "support/tflite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"

#include "support/tflite/Assert.h"
#include "support/tflite/Session.h"
#include "support/tflite/InterpreterSession.h"
#include "support/tflite/NNAPISession.h"
#include "support/tflite/Diff.h"
#include "util/tensor/IndexIterator.h"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>

#include <iostream>

#include "util/environment.h"
#include "util/benchmark.h"

using namespace tflite;
using namespace tflite::ops::builtin;

void help(std::ostream &out, const int argc, char **argv)
{
  std::string cmd = argv[0];
  auto pos = cmd.find_last_of("/");
  if (pos != std::string::npos)
    cmd = cmd.substr(pos + 1);

  out << "use:" << std::endl << cmd << " <model file name>" << std::endl;
}

bool checkParams(const int argc, char **argv)
{
  if (argc < 2)
  {
    help(std::cerr, argc, argv);
    return false;
  }
  return true;
}

int main(const int argc, char **argv)
{

  if (!checkParams(argc, argv))
  {
    return -1;
  }

  const auto filename = argv[1];

  const bool use_nnapi = nnfw::util::get_env_bool("USE_NNAPI");
  const auto thread_count = nnfw::util::get_env_int("THREAD", -1);

  std::cout << "Num threads: " << thread_count << std::endl;
  if (use_nnapi)
  {
    std::cout << "Use NNAPI" << std::endl;
  }

  StderrReporter error_reporter;

  auto model = FlatBufferModel::BuildFromFile(filename, &error_reporter);
  if (model == nullptr)
  {
    std::cerr << "Cannot create model" << std::endl;
    return -1;
  }

  BuiltinOpResolver resolver;

  InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<Interpreter> interpreter;

  TFLITE_ENSURE(builder(&interpreter));

  // Show inputs
  for (uint32_t n = 0; n < interpreter->inputs().size(); ++n)
  {
    // TODO Print shape
    auto tensor_id = interpreter->inputs().at(n);
    auto tensor_ptr = interpreter->tensor(tensor_id);

    std::cout << "Input #" << n << ":" << std::endl;
    std::cout << "  Name: " << tensor_ptr->name << std::endl;
  }

  // Show outputs
  for (uint32_t n = 0; n < interpreter->outputs().size(); ++n)
  {
    // TODO Print shape
    auto tensor_id = interpreter->outputs().at(n);
    auto tensor_ptr = interpreter->tensor(tensor_id);

    std::cout << "Output #" << n << ":" << std::endl;
    std::cout << "  Name: " << tensor_ptr->name << std::endl;
  }

  interpreter->SetNumThreads(thread_count);

  std::shared_ptr<nnfw::support::tflite::Session> sess;

  if (use_nnapi)
  {
    sess = std::make_shared<nnfw::support::tflite::NNAPISession>(interpreter.get());
  }
  else
  {
    sess = std::make_shared<nnfw::support::tflite::InterpreterSession>(interpreter.get());
  }

  //
  // Warming-up
  //
  for (uint32_t n = 0; n < 3; ++n)
  {
    std::chrono::milliseconds elapsed(0);

    sess->prepare();

    for (const auto &id : interpreter->inputs())
    {
      TfLiteTensor *tensor = interpreter->tensor(id);
      if (tensor->type == kTfLiteInt32)
      {
        // Generate singed 32-bit integer (s32) input
        auto tensor_view = nnfw::support::tflite::TensorView<int32_t>::make(*interpreter, id);

        int32_t value = 0;

        nnfw::util::tensor::iterate(tensor_view.shape())
            << [&](const nnfw::util::tensor::Index &ind) {
                 // TODO Generate random values
                 // Gather operation: index should be within input coverage.
                 tensor_view.at(ind) = value;
                 value++;
               };
      }
      else if (tensor->type == kTfLiteUInt8)
      {
        // Generate unsigned 8-bit integer input
        auto tensor_view = nnfw::support::tflite::TensorView<uint8_t>::make(*interpreter, id);

        uint8_t value = 0;

        nnfw::util::tensor::iterate(tensor_view.shape())
            << [&](const nnfw::util::tensor::Index &ind) {
                 // TODO Generate random values
                 tensor_view.at(ind) = value;
                 value = (value + 1) & 0xFF;
               };
      }
      else
      {
        assert(tensor->type == kTfLiteFloat32);

        const int seed = 1; /* TODO Add an option for seed value */
        RandomGenerator randgen{seed, 0.0f, 0.2f};
        const float *end = reinterpret_cast<const float *>(tensor->data.raw_const + tensor->bytes);
        for (float *ptr = tensor->data.f; ptr < end; ptr++)
        {
          *ptr = randgen.generate<float>();
        }
      }
    }

    nnfw::util::benchmark::measure(elapsed) << [&](void) {
      if (!sess->run())
      {
        assert(0 && "run failed");
      }
    };
    sess->teardown();

    std::cout << "Warming-up " << n << ": " << elapsed.count() << "ms" << std::endl;
  }

  //
  // Measure
  //
  const auto cnt = nnfw::util::get_env_int("COUNT", 1);

  using namespace boost::accumulators;

  accumulator_set<double, stats<tag::mean, tag::min, tag::max>> acc;

  for (int n = 0; n < cnt; ++n)
  {
    std::chrono::milliseconds elapsed(0);

    sess->prepare();
    nnfw::util::benchmark::measure(elapsed) << [&](void) {
      if (!sess->run())
      {
        assert(0 && "run failed");
      }
    };
    sess->teardown();

    acc(elapsed.count());

    std::cout << "Iteration " << n << ": " << elapsed.count() << "ms" << std::endl;
  }

  std::cout << "--------" << std::endl;
  std::cout << "Min: " << min(acc) << "ms" << std::endl;
  std::cout << "Max: " << max(acc) << "ms" << std::endl;
  std::cout << "Mean: " << mean(acc) << "ms" << std::endl;

  return 0;
}
