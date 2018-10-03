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

#include "support/tflite/kernels/TensorFlowMax.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"

#include <iostream>

namespace tflite
{
namespace ops
{
namespace custom
{
namespace nnfw
{
namespace TensorFlowMax
{

struct TensorFlowMaxOp
{
  TensorFlowMaxOp(TfLiteContext *context, TfLiteNode *node)
  {
    input = tflite::GetInput(context, node, 0);
    axis = tflite::GetInput(context, node, 1);
    output = tflite::GetOutput(context, node, 0);
  }
  const TfLiteTensor *input;
  const TfLiteTensor *axis;
  TfLiteTensor *output;
};

void *InitTensorFlowMax(TfLiteContext *context, const char *buffer, size_t length)
{
  // Creates two temp tensors to store index and axis for internal
  // implementation only.
  auto *scratch_tensor_index = new int;
  context->AddTensors(context, 2, scratch_tensor_index);
  return scratch_tensor_index;
}

void FreeTensorFlowMax(TfLiteContext *context, void *buffer)
{
  delete static_cast<TensorFlowMaxOp *>(buffer);
}

// Resizes the temp tensor that stores resolved axis.
TfLiteStatus ResizeTempAxis(TfLiteContext *context, TensorFlowMaxOp *op_context,
                            TfLiteTensor *resolved_axis)
{
  TfLiteIntArray *axis_size = TfLiteIntArrayCreate(1);
  axis_size->data[0] = static_cast<int>(tflite::NumElements(op_context->axis));
  return context->ResizeTensor(context, resolved_axis, axis_size);
}

// Resizes output array based on the input size and resolved axis.
TfLiteStatus ResizeOutputTensor(TfLiteContext *context, TensorFlowMaxOp *op_context)
{
  size_t num_axis = tflite::NumElements(op_context->axis);
  const TfLiteIntArray *input_dims = op_context->input->dims;
  int input_num_dims = tflite::NumDimensions(op_context->input);
  const int *axis = op_context->axis->data.i32;

  {
    // Calculates size of reducing axis.
    int num_reduce_axis = num_axis;
    for (int i = 0; i < num_axis; ++i)
    {
      int current = axis[i];
      if (current < 0)
      {
        current += input_num_dims;
      }
      TF_LITE_ENSURE(context, current >= 0 && current < input_num_dims);
      for (int j = 0; j < i; ++j)
      {
        int previous = axis[j];
        if (previous < 0)
        {
          previous += input_num_dims;
        }
        if (current == previous)
        {
          --num_reduce_axis;
          break;
        }
      }
    }
    // Determines output dimensions.
    TfLiteIntArray *output_dims = TfLiteIntArrayCreate(input_num_dims - num_reduce_axis);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx)
    {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx)
      {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx)
        {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis)
      {
        output_dims->data[idx - num_skip_axis] = input_dims->data[idx];
      }
    }
    return context->ResizeTensor(context, op_context->output, output_dims);
  }
}

// Initializes temp tensors to store index and resolved axis.
TfLiteStatus InitializeTemporaries(TfLiteContext *context, TfLiteNode *node,
                                   TensorFlowMaxOp *op_context)
{
  // Creates a temp index to iterate through input data.
  int *scratch_tensor_index = reinterpret_cast<int *>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(2);
  node->temporaries->data[0] = *scratch_tensor_index;
  TfLiteTensor *scratch_tensor = &context->tensors[node->temporaries->data[0]];
  scratch_tensor->type = kTfLiteInt32;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TfLiteIntArray *index_size = TfLiteIntArrayCreate(1);
  index_size->data[0] = tflite::NumDimensions(op_context->input);
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_tensor, index_size));

  // Creates a temp tensor to store resolved axis given input data.
  node->temporaries->data[1] = *scratch_tensor_index + 1;
  TfLiteTensor *resolved_axis = &context->tensors[node->temporaries->data[1]];
  resolved_axis->type = kTfLiteInt32;
  return kTfLiteOk;
}

TfLiteStatus PrepareTensorFlowMax(TfLiteContext *context, TfLiteNode *node)
{
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  TensorFlowMaxOp op_context(context, node);
  TF_LITE_ENSURE_OK(context, InitializeTemporaries(context, node, &op_context));

  TfLiteTensor *resolved_axis = &context->tensors[node->temporaries->data[1]];
  // Leaves work to Eval if axis is not constant; else resizes output.
  if (!tflite::IsConstantTensor(op_context.axis))
  {
    tflite::SetTensorToDynamic(op_context.output);
    tflite::SetTensorToDynamic(resolved_axis);
    return kTfLiteOk;
  }
  resolved_axis->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, ResizeTempAxis(context, &op_context, resolved_axis));
  return ResizeOutputTensor(context, &op_context);
}

// Gets offset of index if expanded on axis. When expanded, the flattened offset
// will not change, if the output index changes on the given axis. For example,
// if you have a 2D tensor and you are expanding to 3D on axis 0,
// then index (0, 1, 2) and index (1, 1, 2) will map from the same flattened
// offset.
inline size_t ExpandedInputOffset(const int num_dims, const int *dims, const int *index,
                                  const int num_axis, const int *axis)
{
  size_t offset = 0;
  int out_idx = 0;
  for (int in_idx = 0; in_idx < num_dims; ++in_idx)
  {
    // if we need to expand this axis
    bool is_axis = false;
    if (axis != nullptr)
    {
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx)
      {
        if (in_idx == axis[axis_idx])
        {
          is_axis = true;
          break;
        }
      }
    }
    if (!is_axis)
    {
      offset = offset * static_cast<size_t>(dims[in_idx]) + static_cast<size_t>(index[out_idx]);
      out_idx++;
    }
    else
    {
      offset = offset * static_cast<size_t>(dims[in_idx]);
    }
  }
  return offset;
}

// Gets offset of index if reducing on axis. When reducing, the flattened offset
// will not change, if the input index changes on the given axis. For example,
// if you have a 3D tensor and you are reducing to 2D by eliminating axis 0,
// then index (0, 1, 2) and index (1, 1, 2) will map to the same flattened
// offset.
// TODO(kanlig): uses Dims to represent dimensions.
inline size_t ReducedOutputOffset(const int num_dims, const int *dims, const int *index,
                                  const int num_axis, const int *axis)
{
  size_t offset = 0;
  for (int idx = 0; idx < num_dims; ++idx)
  {
    // if we need to skip this axis
    bool is_axis = false;
    if (axis != nullptr)
    {
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx)
      {
        if (idx == axis[axis_idx])
        {
          is_axis = true;
          break;
        }
      }
    }
    if (!is_axis)
    {
      offset = offset * static_cast<size_t>(dims[idx]) + static_cast<size_t>(index[idx]);
    }
  }
  return offset;
}

// Gets next index to iterate through a multidimensional array.
inline bool NextIndex(TfLiteContext *context, const int num_dims, const int *dims, int *current)
{
  int carry = 1;
  for (int idx = num_dims - 1; idx >= 0; --idx)
  {
    int current_val = current[idx] + carry;
    TF_LITE_ENSURE(context, (dims[idx] >= current_val));
    if (dims[idx] == current_val)
    {
      current[idx] = 0;
    }
    else
    {
      current[idx] = current_val;
      carry = 0;
      break;
    }
  }
  return (carry == 0);
}

template <typename T>
inline TfLiteStatus
CustomMax(TfLiteContext *context, T *input_data, const int *input_dims, const int input_num_dims,
          T *output_data, const int *output_dims, const int output_num_dims, const int *axis,
          const int num_axis_dimensions, bool keep_dims, int *temp_index, int *resolved_axis)
{
  // resolves axis.
  int num_resolved_axis = 0;
  for (int idx = 0; idx < num_axis_dimensions; ++idx)
  {
    int current = axis[idx];
    TF_LITE_ENSURE(context, (current < input_num_dims && current + input_num_dims >= 0));
    if (current < 0)
    {
      current += input_num_dims;
    }
    bool is_dup = false;
    for (int j = 0; j < num_resolved_axis; ++j)
    {
      if (resolved_axis[j] == current)
      {
        is_dup = true;
        break;
      }
    }
    if (!is_dup)
    {
      resolved_axis[num_resolved_axis++] = current;
    }
  }

  TF_LITE_ENSURE(context, (input_num_dims > 0));
  TF_LITE_ENSURE(context, (input_dims != nullptr));
  TF_LITE_ENSURE(context, (temp_index != nullptr));

  // resets output data.
  for (int idx = 0; idx < output_num_dims; ++idx)
  {
    temp_index[idx] = 0;
  }
  for (bool has_next = true; has_next;
       has_next = NextIndex(context, output_num_dims, output_dims, temp_index))
  {
    size_t output_offset =
        ReducedOutputOffset(output_num_dims, output_dims, temp_index, 0, nullptr);
    size_t input_offset = ExpandedInputOffset(input_num_dims, input_dims, temp_index,
                                              num_resolved_axis, resolved_axis);
    output_data[output_offset] = input_data[input_offset];
  }

  // resets temp index.
  for (int idx = 0; idx < input_num_dims; ++idx)
  {
    temp_index[idx] = 0;
  }

  // iterates through input_data.
  for (bool has_next = true; has_next;
       has_next = NextIndex(context, input_num_dims, input_dims, temp_index))
  {
    size_t input_offset = ReducedOutputOffset(input_num_dims, input_dims, temp_index, 0, nullptr);
    size_t output_offset = ReducedOutputOffset(input_num_dims, input_dims, temp_index,
                                               num_resolved_axis, resolved_axis);
    if (output_data[output_offset] < input_data[input_offset])
    {
      output_data[output_offset] = input_data[input_offset];
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalTensorFlowMax(TfLiteContext *context, TfLiteNode *node)
{

  TensorFlowMaxOp op_context(context, node);
  int num_axis = static_cast<int>(tflite::NumElements(op_context.axis));
  TfLiteTensor *temp_index = &context->tensors[node->temporaries->data[0]];
  TfLiteTensor *resolved_axis = &context->tensors[node->temporaries->data[1]];
  // Resize the output tensor if the output tensor is dynamic.
  if (tflite::IsDynamicTensor(op_context.output))
  {
    TF_LITE_ENSURE_OK(context, ResizeTempAxis(context, &op_context, resolved_axis));
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }

  TfLiteStatus returnStatus = kTfLiteOk;
  switch (op_context.input->type)
  {
    case kTfLiteFloat32:
      returnStatus = CustomMax<float>(
          context, op_context.input->data.f, op_context.input->dims->data,
          op_context.input->dims->size, op_context.output->data.f, op_context.output->dims->data,
          op_context.output->dims->size, op_context.axis->data.i32, num_axis, false,
          temp_index->data.i32, resolved_axis->data.i32);
      break;
    case kTfLiteInt32:
      returnStatus = CustomMax<int>(context, op_context.input->data.i32,
                                    op_context.input->dims->data, op_context.input->dims->size,
                                    op_context.output->data.i32, op_context.output->dims->data,
                                    op_context.output->dims->size, op_context.axis->data.i32,
                                    num_axis, false, temp_index->data.i32, resolved_axis->data.i32);
      break;
    case kTfLiteUInt8:
      returnStatus = CustomMax<uint8_t>(
          context, op_context.input->data.uint8, op_context.input->dims->data,
          op_context.input->dims->size, op_context.output->data.uint8,
          op_context.output->dims->data, op_context.output->dims->size, op_context.axis->data.i32,
          num_axis, false, temp_index->data.i32, resolved_axis->data.i32);
      break;
    case kTfLiteInt64:
      returnStatus = CustomMax<int64_t>(
          context, op_context.input->data.i64, op_context.input->dims->data,
          op_context.input->dims->size, op_context.output->data.i64, op_context.output->dims->data,
          op_context.output->dims->size, op_context.axis->data.i32, num_axis, false,
          temp_index->data.i32, resolved_axis->data.i32);
      break;
    default:
      returnStatus = kTfLiteError;
  }

  return returnStatus;
}
} // namespace TensorFlowMax
} // namespace nnfw
} // namespace custom
} // namespace ops
} // namespace tflite
