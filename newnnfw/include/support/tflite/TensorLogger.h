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

#ifndef __NNFW_SUPPORT_TFLITE_TENSOR_LOGGER_H__
#define __NNFW_SUPPORT_TFLITE_TENSOR_LOGGER_H__

#include "util/tensor/IndexIterator.h"
#include "support/tflite/TensorView.h"

#include <tensorflow/contrib/lite/interpreter.h>
#include <tensorflow/contrib/lite/context.h>
#include <fstream>
#include <iomanip>

namespace nnfw
{
namespace support
{
namespace tflite
{

/*
This is a utility to write input and output value / shape into a file in python form.
any python app can load this value by running the python code below:

  exec(open(filename).read())

generated python code looks like the following:

# ------------- test name -------------
tensor_shape_gen = []
tensor_value_gen = []

tensor_shape_gen.append("{2, 1, 2}")
tensor_value_gen.append([1, 2, 3, 4])

tensor_shape_gen.append("{2}")
tensor_value_gen.append([1, 2])

tensor_shape_gen.append("{2, 1, 2}")
tensor_value_gen.append([1, 4, 3, 8])
# -----------------------------------------
*/

class TensorLogger
{
  private:
    std::ofstream _outfile;

  public:

    static TensorLogger &instance()
    {
        static TensorLogger instance;
        return instance;
    }

    void save(const std::string &path, ::tflite::Interpreter &interp)
    {
      open(path);

      int log_index = 0;
      for (const auto id : interp.inputs())
      {
        _outfile << "# input tensors" << std::endl;
        printTensor(interp, id, log_index++);
      }
      for (const auto id : interp.outputs())
      {
        _outfile << "# output tensors" << std::endl;
        printTensor(interp, id, log_index++);
      }
      close();
    }

  private:

    void open(const std::string &path)
    {
      if (! _outfile.is_open())
        _outfile.open(path, std::ios_base::out);

      _outfile << "# ------ file: " << path << " ------" << std::endl
               << "tensor_shape_gen = []" << std::endl
               << "tensor_value_gen = []" << std::endl << std::endl;
    }

    void printTensor(::tflite::Interpreter &interp, const int id, const int log_index)
    {
        const TfLiteTensor* tensor = interp.tensor(id);

        _outfile << "# tensor name: " << tensor->name << std::endl;
        _outfile << "# tflite::interpreter.tensor("<< id <<") -> "
                    "tensor_value_gen["<< log_index << "]" << std::endl;

        if (tensor->type == kTfLiteInt32)
        {
          printTensorShape(tensor);
          printTensorValue<int32_t>(tensor, tensor->data.i32);
        }
        else if (interp.tensor(id)->type == kTfLiteUInt8)
        {
          printTensorShape(tensor);
          printTensorValue<uint8_t>(tensor, tensor->data.uint8);
        }
        else if (tensor->type == kTfLiteFloat32)
        {
          printTensorShape(tensor);
          printTensorValue<float>(tensor, tensor->data.f);
        }
    }

    void printTensorShape(const TfLiteTensor* tensor)
    {
      _outfile << "tensor_shape_gen.append('{";

      size_t r = 0;
      for (; r < tensor->dims->size - 1; r++)
      {
        _outfile << tensor->dims->data[r]
                 << ", ";
      }
      _outfile << tensor->dims->data[r];

      _outfile << "}')" << std::endl;
    }

    template <typename T>
    void printTensorValue(const TfLiteTensor* tensor, T* tensor_data_ptr)
    {
      _outfile << "tensor_value_gen.append([";

      _outfile << std::fixed << std::setprecision(10);

      const T *end = reinterpret_cast<const T *>(tensor->data.raw_const + tensor->bytes);
      for (T *ptr = tensor_data_ptr; ptr < end; ptr++)
        _outfile << *ptr << ", ";

      _outfile << "])" << std::endl << std::endl;
    }

    void close()
    {
      _outfile << "# --------- tensor shape and value defined above ---------" << std::endl;
      _outfile.close();
    }
};

} // namespace tflite
} // namespace support
} // namespace nnfw

#endif // __NNFW_SUPPORT_TFLITE_TENSOR_LOGGER_H__
