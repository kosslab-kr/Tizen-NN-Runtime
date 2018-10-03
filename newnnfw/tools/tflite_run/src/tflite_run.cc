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

#include "bin_image.h"
#include "args.h"
#include "tensor_dumper.h"
#include "tensor_loader.h"
#include "util/benchmark.h"
#include "util/environment.h"
#include "util/fp32.h"
#include "support/tflite/Diff.h"
#include "support/tflite/Assert.h"
#include "support/tflite/Session.h"
#include "support/tflite/InterpreterSession.h"
#include "support/tflite/NNAPISession.h"
#include "util/tensor/IndexIterator.h"

#include <iostream>
#include <chrono>
#include <algorithm>

using namespace tflite;
using namespace tflite::ops::builtin;

void print_max_idx(float *f, int size)
{
  float *p = std::max_element(f, f + size);
  std::cout << "max:" << p - f;
}

int main(const int argc, char **argv)
{
  bool use_nnapi = false;

  if (std::getenv("USE_NNAPI") != nullptr)
  {
    use_nnapi = true;
  }

  StderrReporter error_reporter;

  TFLiteRun::Args args(argc, argv);

  auto model = FlatBufferModel::BuildFromFile(args.getTFLiteFilename().c_str(), &error_reporter);
  std::unique_ptr<Interpreter> interpreter;

  std::chrono::milliseconds t_prepare(0);
  std::chrono::milliseconds t_invoke(0);

  nnfw::util::benchmark::measure(t_prepare) << [&](void) {
    BuiltinOpResolver resolver;

    InterpreterBuilder builder(*model, resolver);

    TFLITE_ENSURE(builder(&interpreter))

    interpreter->SetNumThreads(1);
  };

  std::shared_ptr<nnfw::support::tflite::Session> sess;

  if (use_nnapi)
  {
    sess = std::make_shared<nnfw::support::tflite::NNAPISession>(interpreter.get());
  }
  else
  {
    sess = std::make_shared<nnfw::support::tflite::InterpreterSession>(interpreter.get());
  }

  sess->prepare();

  TFLiteRun::TensorLoader tensor_loader(*interpreter);

  // Load input from image or dumped tensor file. Two options are exclusive and will be checked
  // from Args.
  if (args.getInputFilename().size() > 0)
  {
    BinImage image(299, 299, 3);
    image.loadImage(args.getInputFilename());

    for (const auto &o : interpreter->inputs())
    {
      image.AssignTensor(interpreter->tensor(o));
    }
  }
  else if (!args.getCompareFilename().empty())
  {
    tensor_loader.load(args.getCompareFilename());

    for (const auto &o : interpreter->inputs())
    {
      const auto &tensor_view = tensor_loader.get(o);
      TfLiteTensor *tensor = interpreter->tensor(o);

      memcpy(reinterpret_cast<void *>(tensor->data.f),
             reinterpret_cast<const void *>(tensor_view._base), tensor->bytes);
    }
  }
  else
  {
    // No input specified. So we fill the input tensors with random values.
    for (const auto &o : interpreter->inputs())
    {
      TfLiteTensor *tensor = interpreter->tensor(o);
      if (tensor->type == kTfLiteInt32)
      {
        // Generate singed 32-bit integer (s32) input
        auto tensor_view = nnfw::support::tflite::TensorView<int32_t>::make(*interpreter, o);

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
        auto tensor_view = nnfw::support::tflite::TensorView<uint8_t>::make(*interpreter, o);

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
  }

  TFLiteRun::TensorDumper tensor_dumper;
  // Must be called before `interpreter->Invoke()`
  tensor_dumper.addTensors(*interpreter, interpreter->inputs());

  std::cout << "input tensor indices = [";
  for (const auto &o : interpreter->inputs())
  {
    std::cout << o << ",";
  }
  std::cout << "]" << std::endl;

  nnfw::util::benchmark::measure(t_invoke) << [&sess](void) {
    if (!sess->run())
    {
      assert(0 && "run failed!");
    }
  };

  sess->teardown();

  // Must be called after `interpreter->Invoke()`
  tensor_dumper.addTensors(*interpreter, interpreter->outputs());

  std::cout << "output tensor indices = [";
  for (const auto &o : interpreter->outputs())
  {
    std::cout << o << "(";

    print_max_idx(interpreter->tensor(o)->data.f, interpreter->tensor(o)->bytes / sizeof(float));

    std::cout << "),";
  }
  std::cout << "]" << std::endl;

  std::cout << "Prepare takes " << t_prepare.count() / 1000.0 << " seconds" << std::endl;
  std::cout << "Invoke takes " << t_invoke.count() / 1000.0 << " seconds" << std::endl;

  if (!args.getDumpFilename().empty())
  {
    const std::string &dump_filename = args.getDumpFilename();
    tensor_dumper.dump(dump_filename);
    std::cout << "Input/output tensors have been dumped to file \"" << dump_filename << "\"."
              << std::endl;
  }

  if (!args.getCompareFilename().empty())
  {
    const std::string &compare_filename = args.getCompareFilename();
    std::cout << "========================================" << std::endl;
    std::cout << "Comparing the results with \"" << compare_filename << "\"." << std::endl;
    std::cout << "========================================" << std::endl;

    // TODO Code duplication (copied from RandomTestRunner)

    int tolerance = 1;
    nnfw::util::env::IntAccessor("TOLERANCE").access(tolerance);

    auto equals = [tolerance](float lhs, float rhs) {
      // NOTE Hybrid approach
      // TODO Allow users to set tolerance for absolute_epsilon_equal
      if (nnfw::util::fp32::absolute_epsilon_equal(lhs, rhs))
      {
        return true;
      }

      return nnfw::util::fp32::epsilon_equal(lhs, rhs, tolerance);
    };

    nnfw::util::tensor::Comparator comparator(equals);
    TfLiteInterpMatchApp app(comparator);
    bool res = true;

    for (const auto &o : interpreter->outputs())
    {
      auto expected = tensor_loader.get(o);
      auto obtained = nnfw::support::tflite::TensorView<float>::make(*interpreter, o);

      res = res && app.compareSingleTensorView(expected, obtained, o);
    }

    if (!res)
    {
      return 255;
    }
  }

  return 0;
}
