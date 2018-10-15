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

/*******************************************************************************
 * Copyright (c) 2008-2015 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

#include "arm_compute/core/CL/OpenCL.h"

#include <iostream>
#include <vector>

void printDeviceInfo(int n, cl::Device &device, cl::Device &default_device)
{
  bool is_default = (device() == default_device());
  std::cout << "\t\t\t#" << n << " Device: (id: " << device() << ") "
            << (is_default ? " -> default" : "") << "\n";

  const auto name = device.getInfo<CL_DEVICE_NAME>();
  std::cout << "\t\t\t\tName: " << name << "\n";

  const auto compute_unit = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  std::cout << "\t\t\t\tMax Compute Unit: " << compute_unit << "\n";

  const auto max_work_item_size = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  std::cout << "\t\t\t\tMax Work Item Size: [";
  for (auto size : max_work_item_size)
    std::cout << size << ",";
  std::cout << "]\n";

  const auto max_work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  std::cout << "\t\t\t\tMax Work Grpup Size: " << max_work_group_size << "\n";

  const auto max_clock_frequency = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  std::cout << "\t\t\t\tMax Clock Frequency: " << max_clock_frequency << "\n";

  std::cout << "\n";
}

void checkContextMem()
{
  // get context, devices
  //
  std::cout << "\nChecking if devices in GPU shares the same memory address:\n\n";

  cl_int cl_error;

  cl::Platform platform = cl::Platform::getDefault();
  cl::Context context;
  try
  {
    cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0
      };

    context = cl::Context(CL_DEVICE_TYPE_GPU, properties, NULL, NULL, &cl_error);
  }
  catch (cl::Error &err) // thrown when there is no Context for this platform
  {
    std::cout << "\t\t No Context Found\n";
    return;
  }

  std::cout << "\nDevices in GPU:\n\n";

  auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
  auto default_device = cl::Device::getDefault();

  int d = 0;
  for (auto device : devices)
    printDeviceInfo(++d, device, default_device);

  if (d < 2)
  {
    std::cout << "\t\t This options works when there are n (>= 2) devices.\n";
    return;
  }

  // allocate and map memory

  typedef cl_int T;
  const int items_per_device = 128;
  const int length = items_per_device * devices.size();

  std::vector<T> input(length);
  std::vector<T> output(length, 0);

  for (int i = 0; i < length; i++)
    input[i] = i;

  cl::Buffer input_buf(context, (cl_mem_flags)CL_MEM_USE_HOST_PTR, length*sizeof(T), input.data(), &cl_error);
  cl::Buffer output_buf(context, (cl_mem_flags)CL_MEM_USE_HOST_PTR, length*sizeof(T), input.data(), &cl_error);

  // compile test cl code

  std::string kernel_source {
    "typedef int T;                                                 \n" \
    "kernel void memory_test(                                       \n" \
    "   const int dev_id,                                           \n" \
    "   global T* input,                                            \n" \
    "   global T* output,                                           \n" \
    "   const int start_idx,                                        \n" \
    "   const int count)                                            \n" \
    "{                                                              \n" \
    "   int input_idx = get_global_id(0);                           \n" \
    "   if(input_idx < count)                                       \n" \
    "   {                                                           \n" \
    "       int output_idx = start_idx + input_idx;                 \n" \
    "       output[output_idx] = input[input_idx] + dev_id;         \n" \
    "   }                                                           \n" \
    "}                                                              \n"
    };

  std::vector<std::string> programStrings {kernel_source};

  cl::Program program(context, programStrings);

  try
  {
    program.build("-cl-std=CL1.2");
  }
  catch (cl::Error &err)
  {
    cl_int buildErr = CL_SUCCESS;
    auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
    for (auto &pair : buildInfo) {
        std::cerr << pair.second << std::endl << std::endl;
    }
  }

  try
  {
    auto kernel_functor = cl::KernelFunctor<cl_int, cl::Buffer, cl::Buffer, cl_int, cl_int>
                              (program, "memory_test"); // name should be same as cl function name

    // create a queue per device and queue a kernel job

    std::vector<cl::CommandQueue*> queues;

    for (int dev_id = 0; dev_id < devices.size(); dev_id++)
    {
      cl::CommandQueue* que = new cl::CommandQueue(context, devices[dev_id]);
      queues.emplace_back(que);

      kernel_functor(
          cl::EnqueueArgs(
              *que,
              cl::NDRange(items_per_device)),
          (cl_int)dev_id, // dev id
          input_buf,
          output_buf,
          (cl_int)(items_per_device * dev_id), // start index
          (cl_int)(items_per_device), // count
          cl_error
          );
    }

    // sync

    for (d = 0; d < devices.size(); d++)
      (queues.at(d))->finish();

    // check if memory state changed by all devices

    cl::copy(*(queues.at(0)), output_buf, begin(output), end(output));

    bool use_same_memory = true;

    for (int dev_id = 0; dev_id < devices.size(); dev_id++)
    {
      for (int i = 0; i < items_per_device; ++i)
      {
        int output_idx = items_per_device * dev_id + i;
        if (output[output_idx] != input[i] + dev_id)
        {
          std::cout << "Output[" << output_idx << "] : "
                    << "expected = "  << input[i] + dev_id
                    << "; actual = " << output[output_idx] << "\n";
          use_same_memory = false;
          break;
        }
      }
    }

    if (use_same_memory)
      std::cout << "\n=> Mapped memory addresses used by devices in GPU are same.\n\n";
    else
      std::cout << "\n=> Mapped memory addresses used by devices in GPU are different.\n\n";

    for (auto q : queues)
      delete q;
  }
  catch (cl::Error &err)
  {
    std::cerr << "error: code: " << err.err() << ", what: " << err.what() << std::endl;
  }
}

void printHelp()
{
    std::cout << "opencl information: \n\n";
    std::cout << "\t -h : help\n";
    std::cout << "\t -g : print if memory map is shared among devices in GPU (in default platform)\n\n";
}

int main(const int argc, char **argv)
{
  if (argc < 2)
    printHelp();
  else
  {
    std::string option = argv[1];

    if (option == "-h")
      printHelp();
    else if (option == "-g")
      checkContextMem();
  }
  return 0;
}
