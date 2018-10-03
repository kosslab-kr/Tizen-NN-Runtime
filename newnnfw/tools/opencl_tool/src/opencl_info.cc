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
}

void printContext(int n, cl::Platform &plat, int device_type, cl::Context &default_context)
{
  if (device_type == CL_DEVICE_TYPE_DEFAULT)
    std::cout << "\t #" << n << " context when CL_DEVICE_TYPE_DEFAULT";
  else if (device_type == CL_DEVICE_TYPE_GPU)
    std::cout << "\t #" << n << " context when CL_DEVICE_TYPE_GPU";
  else if (device_type == CL_DEVICE_TYPE_CPU)
    std::cout << "\t #" << n << " context when CL_DEVICE_TYPE_CPU";
  else if (device_type == CL_DEVICE_TYPE_ACCELERATOR)
    std::cout << "\t #" << n << " context when CL_DEVICE_TYPE_ACCELERATOR";
  else if (device_type == CL_DEVICE_TYPE_CUSTOM)
    std::cout << "\t #" << n << " context when CL_DEVICE_TYPE_CUSTOM";
  else if (device_type == CL_DEVICE_TYPE_ALL)
    std::cout << "\t #" << n << " context when CL_DEVICE_TYPE_ALL";

  cl::Context context;

  try
  {
    cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)plat(), 0};

    cl_int default_error;

    context = cl::Context(device_type, properties, NULL, NULL, &default_error);
  }
  catch (cl::Error &err) // thrown when there is no Context for this platform
  {
    std::cout << "\t\t No Context Found\n";
    return;
  }

  bool is_default = (context() == default_context());

  std::cout << " (id: " << context() << ") " << (is_default ? " -> default" : "") << "\n";

  const auto device_num = context.getInfo<CL_CONTEXT_NUM_DEVICES>();
  std::cout << "\t\t\tDevice num: " << device_num << "\n";
  if (device_num == 0)
    return;

  auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
  auto default_device = cl::Device::getDefault();

  int d = 0;
  for (auto device : devices)
    printDeviceInfo(++d, device, default_device);
}

void printPlatform(int n, cl::Platform &plat, cl::Platform &default_platform)
{
  bool is_default = (plat() == default_platform());

  std::cout << "#" << n << ". Platform: (id: " << plat() << ") "
            << (is_default ? " -> default" : "") << "\n";

  cl::Context default_context = cl::Context::getDefault();
  std::cout << "\t"
            << "default context: " << default_context() << "\n";

  int x = 0;
  printContext(++x, plat, CL_DEVICE_TYPE_DEFAULT, default_context);
  printContext(++x, plat, CL_DEVICE_TYPE_GPU, default_context);
  printContext(++x, plat, CL_DEVICE_TYPE_CPU, default_context);
  printContext(++x, plat, CL_DEVICE_TYPE_ACCELERATOR, default_context);
  printContext(++x, plat, CL_DEVICE_TYPE_CUSTOM, default_context);
  printContext(++x, plat, CL_DEVICE_TYPE_ALL, default_context);
}

int main(const int argc, char **argv)
{
  std::cout << "\nOpenCL Platform, Context, Device Info are as follows:\n\n";

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  cl::Platform defaultPlatform = cl::Platform::getDefault();

  int n = 0;
  for (auto &p : platforms)
  {
    printPlatform(++n, p, defaultPlatform);
  }

  return 0;
}
