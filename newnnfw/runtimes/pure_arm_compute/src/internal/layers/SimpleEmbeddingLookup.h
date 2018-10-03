#ifndef __SIMPLE_EMBEDDING_LOOKUP_H__
#define __SIMPLE_EMBEDDING_LOOKUP_H__

#include "internal/arm_compute.h"
#include <arm_compute/core/ITensor.h>
#include <arm_compute/runtime/IFunction.h>

class SimpleEmbeddingLookup : public ::arm_compute::IFunction
{
public:
  void configure(::arm_compute::ITensor *lookups, ::arm_compute::ITensor *values,
                 ::arm_compute::ITensor *output);

  void run() override;

private:
  ::arm_compute::ITensor *_lookups;
  ::arm_compute::ITensor *_values;
  ::arm_compute::ITensor *_output;
};

#endif /*__SIMPLE_EMBEDDING_LOOKUP_H__ */
