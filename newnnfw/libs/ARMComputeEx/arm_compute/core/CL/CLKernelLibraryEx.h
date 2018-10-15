/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2016-2018 ARM Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __ARM_COMPUTE_CLKERNELLIBRARY_EX_H__
#define __ARM_COMPUTE_CLKERNELLIBRARY_EX_H__

#include "arm_compute/core/CL/OpenCL.h"

#include <map>
#include <set>
#include <string>
#include <utility>

namespace arm_compute
{

/** CLKernelLibrary class */
class CLKernelLibraryEx
{
  using StringSet = std::set<std::string>;

private:
  /** Default Constructor. */
  CLKernelLibraryEx();

public:
  /** Prevent instances of this class from being copied */
  CLKernelLibraryEx(const CLKernelLibraryEx &) = delete;
  /** Prevent instances of this class from being copied */
  const CLKernelLibraryEx &operator=(const CLKernelLibraryEx &) = delete;
  /** Access the KernelLibrary singleton.
   * @return The KernelLibrary instance.
   */
  static CLKernelLibraryEx &get();
  /** Initialises the kernel library.
   *
   * @param[in] kernel_path (Optional) Path of the directory from which kernel sources are loaded.
   * @param[in] context     (Optional) CL context used to create programs.
   * @param[in] device      (Optional) CL device for which the programs are created.
   */
  void init(std::string kernel_path = ".", cl::Context context = cl::Context::getDefault(),
            cl::Device device = cl::Device::getDefault())
  {
    _kernel_path = std::move(kernel_path);
    _context = std::move(context);
    _device = std::move(device);
  }
  /** Sets the path that the kernels reside in.
   *
   * @param[in] kernel_path Path of the kernel.
   */
  void set_kernel_path(const std::string &kernel_path) { _kernel_path = kernel_path; };
  /** Gets the path that the kernels reside in.
   */
  std::string get_kernel_path() { return _kernel_path; };
  /** Gets the source of the selected program.
   *
   * @param[in] program_name Program name.
   *
   * @return Source of the selected program.
   */
  std::string get_program_source(const std::string &program_name);
  /** Sets the CL context used to create programs.
   *
   * @note Setting the context also resets the device to the
   *       first one available in the new context.
   *
   * @param[in] context A CL context.
   */
  void set_context(cl::Context context)
  {
    _context = std::move(context);
    if (_context.get() == nullptr)
    {
      _device = cl::Device();
    }
    else
    {
      const auto cl_devices = _context.getInfo<CL_CONTEXT_DEVICES>();

      if (cl_devices.empty())
      {
        _device = cl::Device();
      }
      else
      {
        _device = cl_devices[0];
      }
    }
  }

  /** Accessor for the associated CL context.
   *
   * @return A CL context.
   */
  cl::Context &context() { return _context; }

  /** Sets the CL device for which the programs are created.
   *
   * @param[in] device A CL device.
   */
  void set_device(cl::Device device) { _device = std::move(device); }

  /** Return the device version
   *
   * @return The content of CL_DEVICE_VERSION
   */
  std::string get_device_version();
  /** Creates a kernel from the kernel library.
   *
   * @param[in] kernel_name       Kernel name.
   * @param[in] build_options_set Kernel build options as a set.
   *
   * @return The created kernel.
   */
  Kernel create_kernel(const std::string &kernel_name,
                       const StringSet &build_options_set = {}) const;
  /** Find the maximum number of local work items in a workgroup can be supported for the kernel.
   *
   */
  size_t max_local_workgroup_size(const cl::Kernel &kernel) const;
  /** Return the default NDRange for the device.
   *
   */
  cl::NDRange default_ndrange() const;

  /** Clear the library's cache of binary programs
   */
  void clear_programs_cache()
  {
    _programs_map.clear();
    _built_programs_map.clear();
  }

  /** Access the cache of built OpenCL programs */
  const std::map<std::string, cl::Program> &get_built_programs() const
  {
    return _built_programs_map;
  }

  /** Add a new built program to the cache
   *
   * @param[in] built_program_name Name of the program
   * @param[in] program            Built program to add to the cache
   */
  void add_built_program(const std::string &built_program_name, cl::Program program);

private:
  /** Load program and its dependencies.
   *
   * @param[in] program_name Name of the program to load.
   */
  const Program &load_program(const std::string &program_name) const;
  /** Concatenates contents of a set into a single string.
   *
   * @param[in] s Input set to concatenate.
   *
   * @return Concatenated string.
   */
  std::string stringify_set(const StringSet &s) const;

  cl::Context _context;     /**< Underlying CL context. */
  cl::Device _device;       /**< Underlying CL device. */
  std::string _kernel_path; /**< Path to the kernels folder. */
  mutable std::map<std::string, const Program>
      _programs_map; /**< Map with all already loaded program data. */
  mutable std::map<std::string, cl::Program>
      _built_programs_map; /**< Map with all already built program data. */
  static const std::map<std::string, std::string>
      _kernel_program_map; /**< Map that associates kernel names with programs. */
  static const std::map<std::string, std::string>
      _program_source_map; /**< Contains sources for all programs.
                                Used for compile-time kernel inclusion. >*/
};
}
#endif /* __ARM_COMPUTE_CLKERNELLIBRARY_EX_H__ */
