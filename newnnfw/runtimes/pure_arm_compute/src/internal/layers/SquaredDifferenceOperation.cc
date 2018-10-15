#include "SquaredDifferenceOperation.h"
#include "internal/arm_compute.h"

void SquaredDifferenceOperation::configure(::arm_compute::ITensor *input1,
                                           ::arm_compute::ITensor *input2,
                                           ::arm_compute::ITensor *output,
                                           ::arm_compute::ConvertPolicy ConvertPolicy, float scale,
                                           ::arm_compute::RoundingPolicy RoundingPolicy)
{
  _input1 = input1;
  _input2 = input2;
  _output = output;

  if (::internal::arm_compute::isGpuMode())
  {
    _cl_sub.configure(CAST_CL(input1), CAST_CL(input2), CAST_CL(output), ConvertPolicy);
    _cl_mul.configure(CAST_CL(output), CAST_CL(output), CAST_CL(output), scale, ConvertPolicy,
                      RoundingPolicy);
  }
  else
  {
    _neon_sub.configure(CAST_NE(input1), CAST_NE(input2), CAST_NE(output), ConvertPolicy);
    _neon_mul.configure(CAST_NE(output), CAST_NE(output), CAST_NE(output), scale, ConvertPolicy,
                        RoundingPolicy);
  }
}

void SquaredDifferenceOperation::run(void)
{
  if (::internal::arm_compute::isGpuMode())
  {
    _cl_sub.run();
    _cl_mul.run();
  }
  else
  {
    _neon_sub.run();
    _neon_mul.run();
  }
}
