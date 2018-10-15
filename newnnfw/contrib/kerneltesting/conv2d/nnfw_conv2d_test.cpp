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

#include <iostream>
#include <vector>
#include <cassert>

#include <Eigen/Core>
#include <gemmlowp.h>

#include "types.h"
#include "common.h"
#include "optimized_ops.h"
#include "OperationUtils.h"

#include <arm_compute/graph.h>

#include <arm_compute/runtime/CL/CLFunctions.h>
#include <arm_compute/runtime/CL/functions/CLConvolution.h>

#include "io_accessor.h"
#include "util/environment.h"

static constexpr int kStaticBufferSize = 1605632;
static char static_scratch_buffer[kStaticBufferSize];

#define ANDROID_NN_CONV_PARAMETERS(Type)                                        \
    uint32_t height       = getSizeOfDimension(inputShape, 1);                  \
    uint32_t width        = getSizeOfDimension(inputShape, 2);                  \
    uint32_t filterHeight = getSizeOfDimension(filterShape, 1);                 \
    uint32_t filterWidth  = getSizeOfDimension(filterShape, 2);                 \
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);                 \
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);                 \
    uint32_t inDepth      = getSizeOfDimension(inputShape, 3);                  \
                                                                                \
    uint32_t paddingHeight = (uint32_t)padding_top;                             \
    uint32_t paddingWidth = (uint32_t)padding_left;                             \
                                                                                \
    Dims<4> im2colDim;                                                          \
    im2colDim.sizes[3] = (int)getSizeOfDimension(outputShape, 0);               \
    im2colDim.sizes[2] = (int)getSizeOfDimension(outputShape, 1);               \
    im2colDim.sizes[1] = (int)getSizeOfDimension(outputShape, 2);               \
    im2colDim.sizes[0] = (int)inDepth * filterHeight * filterWidth;             \
                                                                                \
    im2colDim.strides[0] = 1;                                                   \
    for (int i=1; i<4; i++) {                                                   \
        im2colDim.strides[i] = im2colDim.strides[i-1] * im2colDim.sizes[i-1];   \
    }                                                                           \
                                                                                \
    Type* im2colData = nullptr;                                                 \
    int im2colByteSize = sizeof(Type);                                          \
    for (int i=0; i<4; i++) {                                                   \
        im2colByteSize *= im2colDim.sizes[i];                                   \
    }                                                                           \
    if (im2colByteSize <= kStaticBufferSize) {                                  \
        im2colData = reinterpret_cast<Type *>(static_scratch_buffer);           \
    } else {                                                                    \
        im2colData = new (std::nothrow) Type[im2colByteSize / sizeof(Type)];    \
    }


bool convFloat32(const float* inputData, const Shape& inputShape,
                 const float* filterData, const Shape& filterShape,
                 const float* biasData, const Shape& biasShape,
                 int32_t padding_left, int32_t padding_right,
                 int32_t padding_top, int32_t padding_bottom,
                 int32_t stride_width, int32_t stride_height,
                 int32_t activation,
                 float* outputData, const Shape& outputShape) {

    ANDROID_NN_CONV_PARAMETERS(float)

    #define ANDROID_NN_CONV(activation)                                        \
        Conv<FusedActivationFunctionType::activation>(                         \
            inputData, convertShapeToDims(inputShape),                         \
            filterData, convertShapeToDims(filterShape),                       \
            biasData, convertShapeToDims(biasShape),                           \
            stride_width, stride_height, paddingWidth, paddingHeight,          \
            outputData, convertShapeToDims(outputShape),                       \
            im2colData, im2colDim)

    ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_CONV)

    #undef ANDROID_NN_CONV

    if (im2colByteSize > kStaticBufferSize) {
        delete[] im2colData;
    }
    return true;
}

//-----------------------------------------------------------------------------

using arm_compute::DataType;
using arm_compute::graph::Target;
using arm_compute::graph::TensorDescriptor;
using arm_compute::TensorShape;
using arm_compute::graph::frontend::InputLayer;
using arm_compute::graph::frontend::OutputLayer;

namespace acl_graph {

bool convFloat32(const float* inputData, const Shape& inputShape,
                 const float* filterData, const Shape& filterShape,
                 const float* biasData, const Shape& biasShape,
                 int32_t padding_left, int32_t padding_right,
                 int32_t padding_top, int32_t padding_bottom,
                 int32_t stride_width, int32_t stride_height,
                 int32_t activation,
                 float* outputData, const Shape& outputShape)
{
  // Try with simple build-run with ACL Layer
  arm_compute::graph::frontend::Stream graph{0, "ACL_CONV2D_TEST"};

  Target target_hint = nnfw::util::get_env_int("NNFW_ACL_USENEON")
                           ? Target::NEON : Target::CL;

  // Not sure about which index is which value
  uint32_t tsi_c = getSizeOfDimension(inputShape, 0);
  uint32_t tsi_h = getSizeOfDimension(inputShape, 1);
  uint32_t tsi_w = getSizeOfDimension(inputShape, 2);
  uint32_t tsi_n = getSizeOfDimension(inputShape, 3);

  uint32_t tsk_h = getSizeOfDimension(filterShape, 1);
  uint32_t tsk_w = getSizeOfDimension(filterShape, 2);
  uint32_t tsk_n = getSizeOfDimension(filterShape, 3);

  graph << target_hint
        << InputLayer(TensorDescriptor(TensorShape(tsi_w, tsi_h, tsi_c, tsi_n), DataType::F32),
                  std::unique_ptr<InputAccessor>(new InputAccessor(inputData, inputShape)))
        << arm_compute::graph::frontend::ConvolutionLayer(
              tsk_w, tsk_h, tsk_n,
              std::unique_ptr<WeightAccessor>(new WeightAccessor(filterData, filterShape)),
              std::unique_ptr<BiasAccessor>(new BiasAccessor(biasData, biasShape)),
              arm_compute::PadStrideInfo(stride_width, stride_height, padding_top, padding_bottom))
        ;
  if (activation != static_cast<int32_t>(FusedActivationFunc::NONE)) {
    arm_compute::ActivationLayerInfo::ActivationFunction actFunc =
        arm_compute::ActivationLayerInfo::ActivationFunction::RELU;

    graph << arm_compute::graph::frontend::ActivationLayer(arm_compute::ActivationLayerInfo(actFunc));
    // Activation does not provide output Tensor and makes next layer fail to add to graph
    // when it's the last(output) layer. To solve this, need to add a dummy layer.
    uint32_t tso_c = getSizeOfDimension(outputShape, 0);
    uint32_t tso_h = getSizeOfDimension(outputShape, 1);
    uint32_t tso_w = getSizeOfDimension(outputShape, 2);
    uint32_t tso_n = getSizeOfDimension(outputShape, 3);
    graph << arm_compute::graph::frontend::ReshapeLayer(TensorShape(tso_w, tso_h, tso_c, tso_n));
  }
  graph << OutputLayer(std::unique_ptr<OutputAccessor>(new OutputAccessor(outputData, outputShape)))
        ;

  graph.run();

  return true;
}

} // namespace acl_graph

//-----------------------------------------------------------------------------

using arm_compute::TensorInfo;

namespace acl_runtime {

TensorShape calculate_convolution_layer_output_shape(
    const arm_compute::TensorShape &input_shape,
    const arm_compute::TensorShape &weights_shape,
    const arm_compute::PadStrideInfo &conv_info)
{
    unsigned int output_width  = 0;
    unsigned int output_height = 0;

    // Get output width and height
    std::tie(output_width, output_height) =
        arm_compute::scaled_dimensions(
            input_shape.x(), input_shape.y(),
            weights_shape.x(), weights_shape.y(),
            conv_info);

    // Create output shape
    TensorShape output_shape = input_shape;
    output_shape.set(0, output_width);
    output_shape.set(1, output_height);
    output_shape.set(2, weights_shape[3]);

    return output_shape;
}

bool convFloat32(const float* inputData, const Shape& inputShape,
                 const float* filterData, const Shape& filterShape,
                 const float* biasData, const Shape& biasShape,
                 int32_t padding_left, int32_t padding_right,
                 int32_t padding_top, int32_t padding_bottom,
                 int32_t stride_width, int32_t stride_height,
                 int32_t activation,
                 float* outputData, const Shape& outputShape)
{
  arm_compute::CLScheduler::get().default_init();

  uint32_t tsi_c = getSizeOfDimension(inputShape, 0);
  uint32_t tsi_h = getSizeOfDimension(inputShape, 1);
  uint32_t tsi_w = getSizeOfDimension(inputShape, 2);
  uint32_t tsi_n = getSizeOfDimension(inputShape, 3);

  uint32_t tsk_h = getSizeOfDimension(filterShape, 1);
  uint32_t tsk_w = getSizeOfDimension(filterShape, 2);
  uint32_t tsk_n = getSizeOfDimension(filterShape, 3);

  TensorShape input_shape = TensorShape(tsi_w, tsi_h, tsi_c, tsi_n);
  TensorShape filter_shape = TensorShape(tsi_w, tsi_h, tsi_c, tsi_n);
  arm_compute::PadStrideInfo conv_info =
      arm_compute::PadStrideInfo(stride_width, stride_height, padding_top, padding_bottom);

  TensorShape output_shape = calculate_convolution_layer_output_shape(
                                input_shape, filter_shape, conv_info);

  uint32_t tso_c = output_shape[0];
  uint32_t tso_w = output_shape[1];
  uint32_t tso_h = output_shape[2];
  uint32_t tso_n = output_shape[3];

  arm_compute::CLTensor input, output, bias, filter;

  input.allocator()->init(TensorInfo(tsi_w, tsi_h, arm_compute::Format::F32));
  output.allocator()->init(TensorInfo(tso_w, tso_h, arm_compute::Format::F32));
  bias.allocator()->init(TensorInfo(tso_w, tso_h, arm_compute::Format::F32));
  filter.allocator()->init(TensorInfo(tsk_w, tsk_h, arm_compute::Format::F32));

  input.allocator()->allocate();
  output.allocator()->allocate();
  bias.allocator()->allocate();
  filter.allocator()->allocate();

  input.map();
  InputAccessor ia(inputData, inputShape);
  ia.access_tensor(input);
  input.unmap();

  bias.map();
  BiasAccessor ba(biasData, biasShape);
  ba.access_tensor(bias);
  bias.unmap();

  filter.map();
  WeightAccessor fa(filterData, filterShape);
  fa.access_tensor(filter);
  filter.unmap();

  arm_compute::CLConvolutionLayer conv_f;
  conv_f.configure(&input, &filter, &bias, &output, conv_info);

  arm_compute::CLScheduler::get().sync();

  conv_f.run();

  output.map();
  OutputAccessor oa(outputData, outputShape);
  oa.access_tensor(output);
  output.unmap();

  return true;
}

} // namespace acl_runtime

//-----------------------------------------------------------------------------

enum COMPUTE_TYPE {
  COMPUTE_DEFAULT = 0,
  COMPUTE_ACLGRAPH,
  COMPUTE_ACLRT
};

bool convFloat32(const float* inputData, const Shape& inputShape,
                 const float* filterData, const Shape& filterShape,
                 const float* biasData, const Shape& biasShape,
                 int32_t padding_left, int32_t padding_right,
                 int32_t padding_top, int32_t padding_bottom,
                 int32_t stride_width, int32_t stride_height,
                 int32_t activation,
                 float* outputData, const Shape& outputShape,
                 COMPUTE_TYPE compType) {

  switch (compType)
  {
  case COMPUTE_DEFAULT :
    return convFloat32(inputData, inputShape, filterData, filterShape,
                       biasData, biasShape, padding_left, padding_right,
                       padding_top, padding_bottom, stride_width, stride_height,
                       activation, outputData, outputShape);

  case COMPUTE_ACLGRAPH :
    return acl_graph::convFloat32(inputData, inputShape, filterData, filterShape,
                       biasData, biasShape, padding_left, padding_right,
                       padding_top, padding_bottom, stride_width, stride_height,
                       activation, outputData, outputShape);

  case COMPUTE_ACLRT :
    return acl_runtime::convFloat32(inputData, inputShape, filterData, filterShape,
                       biasData, biasShape, padding_left, padding_right,
                       padding_top, padding_bottom, stride_width, stride_height,
                       activation, outputData, outputShape);
  }
  return false;
}

//-----------------------------------------------------------------------------

void dumpData(const char* name, const float* data, const Shape& shape)
{
  uint32_t height = getSizeOfDimension(shape, 1);
  uint32_t width  = getSizeOfDimension(shape, 2);

  std::cout << "---" << name << "---" << std::endl;
  for (int h = 0; h < height; h++) {
    std::cout << "H=" << h << " | ";
    for (int w = 0; w < width; w++) {
      std::cout << data[h * width + w] << ",";
    }
    std::cout << std::endl;
  }
}

void initData(float* outputData, int num, float value)
{
  for (int i = 0; i < num; i++) {
    *(outputData + i) = value;
  }
}

void initDataSeq(float* outputData, int num, float value)
{
  for (int i = 0; i < num; i++) {
    *(outputData + i) = value;
    value += 1.0;
  }
}

// compareData
// return true if result == expected with the shape info,
// otherwise false
bool compareData(const float* result, const float* expected, const Shape& shape)
{
  NN_CHECK_EQ(shape.dimensions.size(), 4);

  uint32_t height = getSizeOfDimension(shape, 1);
  uint32_t width  = getSizeOfDimension(shape, 2);
  uint32_t numitems = height * width;
  for (int item = 0; item < numitems; item++) {
    if (*(result + item) != *(expected + item)) {
      LOG(ERROR) << "compareData failed: result " << *(result + item)
                 << ", expected " << *(expected + item) << std::endl;
      return false;
    }
  }
  return true;
}

int test_3x3_1x1_one(COMPUTE_TYPE comptype)
{
  float inputData[9];
  const Shape inputShape = { OperandType::FLOAT32, {1,3,3,1}, 1.0, 0 };
  float filterData[9];
  const Shape filterShape = { OperandType::FLOAT32, {1,3,3,1}, 1.0, 0 };
  float biasData[1] = { 1.0 };
  const Shape biasShape = { OperandType::FLOAT32, {1,1,1,1}, 1.0, 0 };
  int32_t padding_left = 0;
  int32_t padding_right = 0;
  int32_t padding_top = 0;
  int32_t padding_bottom = 0;
  int32_t stride_width = 1;
  int32_t stride_height = 1;
  int32_t activation = static_cast<int32_t>(FusedActivationFunc::RELU);
  float* outputData = new float[9];
  const Shape outputShape = { OperandType::FLOAT32, {1,1,1,1}, 1.0, 0 };
  float* expectData = new float[9];
  bool bret;

  initData(inputData, sizeof(inputData) / sizeof(inputData[0]), 1.0);
  initData(filterData, sizeof(filterData) / sizeof(filterData[0]), 1.0);
  initData(outputData, sizeof(outputData) / sizeof(outputData[0]), 0.0);
  initData(expectData, sizeof(expectData) / sizeof(expectData[0]), 0.0);

  bret = convFloat32(inputData, inputShape,
                     filterData, filterShape,
                     biasData, biasShape,
                     padding_left, padding_right,
                     padding_top, padding_bottom,
                     stride_width, stride_height,
                     activation,
                     expectData, outputShape,
                     COMPUTE_DEFAULT);

  bret = convFloat32(inputData, inputShape,
                     filterData, filterShape,
                     biasData, biasShape,
                     padding_left, padding_right,
                     padding_top, padding_bottom,
                     stride_width, stride_height,
                     activation,
                     outputData, outputShape,
                     comptype);

  dumpData("Input  ", inputData, inputShape);
  dumpData("Filter ", filterData, filterShape);
  dumpData("Bias   ", biasData, biasShape);
  dumpData("Output ", outputData, outputShape);
  std::cout << std::endl;

  bret = compareData(outputData, expectData, outputShape);

  delete outputData;
  delete expectData;

  if (!bret)
  {
    LOG(ERROR) << "TEST FAILED " << __FUNCTION__ << std::endl;
    return -1;
  }
  return 0;
}

int test_3x3_3x3_one(void)
{
  float inputData[9];
  const Shape inputShape = { OperandType::FLOAT32, {1,3,3,1}, 1.0, 0 };
  float filterData[9];
  const Shape filterShape = { OperandType::FLOAT32, {1,3,3,1}, 1.0, 0 };
  float biasData[1] = { 1.0 };
  const Shape biasShape = { OperandType::FLOAT32, {1,1,1,1}, 1.0, 0 };
  int32_t padding_left = 1;
  int32_t padding_right = 1;
  int32_t padding_top = 1;
  int32_t padding_bottom = 1;
  int32_t stride_width = 1;
  int32_t stride_height = 1;
  int32_t activation = static_cast<int32_t>(FusedActivationFunc::RELU);
  float* outputData = new float[9];
  const Shape outputShape = { OperandType::FLOAT32, {1,3,3,1}, 1.0, 0 };
  float* expectData = new float[9];
  bool bret;

  initData(inputData, sizeof(inputData) / sizeof(inputData[0]), 1.0);
  initData(filterData, sizeof(filterData) / sizeof(filterData[0]), 1.0);
  initData(outputData, sizeof(outputData) / sizeof(outputData[0]), 0.0);
  initData(expectData, sizeof(expectData) / sizeof(expectData[0]), 0.0);

  bret = convFloat32(inputData, inputShape,
                     filterData, filterShape,
                     biasData, biasShape,
                     padding_left, padding_right,
                     padding_top, padding_bottom,
                     stride_width, stride_height,
                     activation,
                     expectData, outputShape,
                     COMPUTE_DEFAULT);

  bret = convFloat32(inputData, inputShape,
                     filterData, filterShape,
                     biasData, biasShape,
                     padding_left, padding_right,
                     padding_top, padding_bottom,
                     stride_width, stride_height,
                     activation,
                     outputData, outputShape,
                     COMPUTE_ACLGRAPH);

  dumpData("Input  ", inputData, inputShape);
  dumpData("Filter ", filterData, filterShape);
  dumpData("Bias   ", biasData, biasShape);
  dumpData("Output ", outputData, outputShape);
  std::cout << std::endl;

  bret = compareData(outputData, expectData, outputShape);

  delete outputData;
  delete expectData;

  if (!bret)
  {
    LOG(ERROR) << "TEST FAILED " << __FUNCTION__ << std::endl;
    return -1;
  }
  return 0;
}

int test_3x3_3x3_seq(void)
{
  float inputData[9];
  const Shape inputShape = { OperandType::FLOAT32, {1,3,3,1}, 1.0, 0 };
  float filterData[9];
  const Shape filterShape = { OperandType::FLOAT32, {1,3,3,1}, 1.0, 0 };
  float biasData[1] = { 1.0 };
  const Shape biasShape = { OperandType::FLOAT32, {1,1,1,1}, 1.0, 0 };
  int32_t padding_left = 1;
  int32_t padding_right = 1;
  int32_t padding_top = 1;
  int32_t padding_bottom = 1;
  int32_t stride_width = 1;
  int32_t stride_height = 1;
  int32_t activation = static_cast<int32_t>(FusedActivationFunc::RELU);
  float* outputData = new float[9];
  const Shape outputShape = { OperandType::FLOAT32, {1,3,3,1}, 1.0, 0 };
  float* expectData = new float[9];
  bool bret;

  initDataSeq(inputData, sizeof(inputData) / sizeof(inputData[0]), 1.0);
  initDataSeq(filterData, sizeof(filterData) / sizeof(filterData[0]), 1.0);
  initDataSeq(outputData, sizeof(outputData) / sizeof(outputData[0]), 0.0);
  initData(expectData, sizeof(expectData) / sizeof(expectData[0]), 0.0);

  bret = convFloat32(inputData, inputShape,
                     filterData, filterShape,
                     biasData, biasShape,
                     padding_left, padding_right,
                     padding_top, padding_bottom,
                     stride_width, stride_height,
                     activation,
                     expectData, outputShape,
                     COMPUTE_DEFAULT);

  bret = convFloat32(inputData, inputShape,
                     filterData, filterShape,
                     biasData, biasShape,
                     padding_left, padding_right,
                     padding_top, padding_bottom,
                     stride_width, stride_height,
                     activation,
                     outputData, outputShape,
                     COMPUTE_ACLGRAPH);

  dumpData("Input  ", inputData, inputShape);
  dumpData("Filter ", filterData, filterShape);
  dumpData("Bias   ", biasData, biasShape);
  dumpData("Output ", outputData, outputShape);
  std::cout << std::endl;

  bret = compareData(outputData, expectData, outputShape);

  delete outputData;
  delete expectData;

  if (!bret)
  {
    LOG(ERROR) << "TEST FAILED " << __FUNCTION__ << std::endl;
    return -1;
  }
  return 0;
}

int main(int argc, char* argv[])
{
  int result;

  // input 3x3, output 1x1, all data 1.0
  result = test_3x3_1x1_one(COMPUTE_ACLGRAPH);
  if (result) return result;
  result = test_3x3_1x1_one(COMPUTE_ACLRT);
  if (result) return result;

  // input 3x3, output 3x3, all data 1.0
  result = test_3x3_3x3_one();
  if (result) return result;

  result = test_3x3_3x3_seq();
  if (result) return result;

  return result;
}
