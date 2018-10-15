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

/*
 * Copyright (c) 2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "io_accessor.h"

InputAccessor::InputAccessor(const float* inputData, const Shape& inputShape)
  : _inputData(inputData)
  , _inputShape(inputShape)
{
}

WeightAccessor::WeightAccessor(const float* filterData, const Shape& filterShape)
  : _filterData(filterData)
  , _filterShape(filterShape)
{
}

BiasAccessor::BiasAccessor(const float* biasData, const Shape& biasShape)
  : _biasData(biasData)
  , _biasShape(biasShape)
{
}

OutputAccessor::OutputAccessor(float* outputData, const Shape& outputShape)
  : _outputData(outputData)
  , _outputShape(outputShape)
{
}

bool InputAccessor::access_tensor(arm_compute::ITensor &tensor)
{
  arm_compute::Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());

  execute_window_loop(window, [&](const arm_compute::Coordinates& id)
  {
    uint32_t width  = getSizeOfDimension(_inputShape, 2);
    uint32_t offset = id.y() * width + id.x();
    *reinterpret_cast<float *>(tensor.ptr_to_element(id)) =
        *(_inputData + offset);
  });
  return true;
}

bool WeightAccessor::access_tensor(arm_compute::ITensor &tensor)
{
  arm_compute::Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());

  execute_window_loop(window, [&](const arm_compute::Coordinates& id)
  {
    uint32_t width  = getSizeOfDimension(_filterShape, 2);
    uint32_t offset = id.y() * width + id.x();
    *reinterpret_cast<float *>(tensor.ptr_to_element(id)) =
        *(_filterData + offset);
  });
  return true;
}

bool BiasAccessor::access_tensor(arm_compute::ITensor &tensor)
{
  arm_compute::Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());

  execute_window_loop(window, [&](const arm_compute::Coordinates& id)
  {
    uint32_t width  = getSizeOfDimension(_biasShape, 2);
    uint32_t offset = id.y() * width + id.x();
    *reinterpret_cast<float *>(tensor.ptr_to_element(id)) =
        *(_biasData + offset);
  });
  return true;
}

bool OutputAccessor::access_tensor(arm_compute::ITensor &tensor)
{
  arm_compute::Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());

  execute_window_loop(window, [&](const arm_compute::Coordinates& id)
  {
    uint32_t width  = getSizeOfDimension(_outputShape, 2);
    uint32_t offset = id.y() * width + id.x();
    *(_outputData + offset) =
        *reinterpret_cast<float *>(tensor.ptr_to_element(id));
  });
  return false; // end the network
}
