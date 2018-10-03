/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __INTERNAL_CPU_MEMORY_ALLOCATOR_H__
#define __INTERNAL_CPU_MEMORY_ALLOCATOR_H__

#include "arm_compute/runtime/ITensorAllocator.h"
#include "arm_compute/runtime/Memory.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace arm_compute
{
class Coordinates;
class TensorInfo;
class Tensor;
};

/** Basic implementation of a CPU memory tensor allocator. */
class TensorAllocator : public ITensorAllocator
{
public:
  /** Default constructor. */
  TensorAllocator(Tensor *owner = nullptr);
  /** Default destructor */
  ~TensorAllocator();

  /** Make ITensorAllocator's init methods available */
  using ITensorAllocator::init;

  /** Shares the same backing memory with another tensor allocator, while the tensor info might be
   * different.
   *  In other words this can be used to create a sub-tensor from another tensor while sharing the
   * same memory.
   *
   * @note TensorAllocator have to be of the same specialized type.
   *
   * @param[in] allocator The allocator that owns the backing memory to be shared. Ownership becomes
   * shared afterwards.
   * @param[in] coords    The starting coordinates of the new tensor inside the parent tensor.
   * @param[in] sub_info  The new tensor information (e.g. shape etc)
   */
  void init(const TensorAllocator &allocator, const Coordinates &coords, TensorInfo sub_info);

  /** Returns the pointer to the allocated data. */
  uint8_t *data() const;

  /** Allocate size specified by TensorInfo of CPU memory.
   *
   * @note The tensor must not already be allocated when calling this function.
   *
   */
  void allocate() override;

  /** Free allocated CPU memory.
   *
   * @note The tensor must have been allocated when calling this function.
   *
   */
  void free() override;
  /** Import an existing memory as a tensor's backing memory
   *
   * @warning If the tensor is flagged to be managed by a memory manager,
   *          this call will lead to an error.
   * @warning Ownership of memory depends on the way the @ref Memory object was constructed
   * @note    Calling free on a tensor with imported memory will just clear
   *          the internal pointer value.
   *
   * @param[in] memory Memory to import
   *
   * @return error status
   */
  arm_compute::Status import_memory(Memory memory);
  /** Associates the tensor with a memory group
   *
   * @param[in] associated_memory_group Memory group to associate the tensor with
   */
  void set_associated_memory_group(MemoryGroup *associated_memory_group);

protected:
  /** No-op for CPU memory
   *
   * @return A pointer to the beginning of the tensor's allocation.
   */
  uint8_t *lock() override;

  /** No-op for CPU memory. */
  void unlock() override;

private:
  MemoryGroup *_associated_memory_group; /**< Registered memory manager */
  Memory _memory;                        /**< CPU memory */
  Tensor *_owner;                        /**< Owner of the allocator */
};

namespace internal
{
namespace cpu
{

class MemoryAllocator : public
{
};

} // namespace cpu
} // namespace internal

#endif // __INTERNAL_CPU_MEMORY_ALLOCATOR_H__
