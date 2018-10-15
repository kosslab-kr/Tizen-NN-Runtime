#ifndef __SQUARED_DIFFERENCE_OPERATION_H__
#define __SQUARED_DIFFERENCE_OPERATION_H__

#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/CL/CLTensor.h>

#include <arm_compute/runtime/CL/functions/CLArithmeticSubtraction.h>
#include <arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h>
#include <arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h>

class SquaredDifferenceOperation : public ::arm_compute::IFunction
{
public:
  void configure(::arm_compute::ITensor *input1, ::arm_compute::ITensor *input2,
                 ::arm_compute::ITensor *output, ::arm_compute::ConvertPolicy ConvertPolicy,
                 float scale, ::arm_compute::RoundingPolicy RoundingPolicy);

public:
  void run(void) override;

private:
  ::arm_compute::ITensor *_input1;
  ::arm_compute::ITensor *_input2;

  ::arm_compute::ITensor *_output;

private:
  ::arm_compute::CLArithmeticSubtraction _cl_sub;
  ::arm_compute::CLPixelWiseMultiplication _cl_mul;

  ::arm_compute::NEArithmeticSubtraction _neon_sub;
  ::arm_compute::NEPixelWiseMultiplication _neon_mul;
};
#endif // __SQUARED_DIFFERENCE_OPERATION_H__
