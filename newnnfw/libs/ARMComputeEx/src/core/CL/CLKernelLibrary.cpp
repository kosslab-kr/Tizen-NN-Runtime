/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2016-2018 ARM Limited.
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
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

using namespace arm_compute;

const std::map<std::string, std::string> CLKernelLibraryEx::_kernel_program_map = {
    {"absdiff", "absdiff.cl"},
    {"accumulate", "accumulate.cl"},
    {"accumulate_squared", "accumulate.cl"},
    {"accumulate_weighted", "accumulate.cl"},
    {"activation_layer", "activation_layer.cl"},
    {"activation_layer_qa8", "activation_layer_qa8.cl"},
    {"activation_layer_logistic_qa8", "activation_layer_qa8.cl"},
    {"arithmetic_add", "arithmetic_op.cl"},
    {"arithmetic_sub", "arithmetic_op.cl"},
    {"arithmetic_add_qasymm8", "arithmetic_op_quantized.cl"},
    {"batchnormalization_layer_nchw", "batchnormalization_layer.cl"},
    {"batchnormalization_layer_nhwc", "batchnormalization_layer.cl"},
    {"bitwise_or", "bitwise_op.cl"},
    {"bitwise_and", "bitwise_op.cl"},
    {"bitwise_xor", "bitwise_op.cl"},
    {"bitwise_not", "bitwise_op.cl"},
    {"cast", "cast.cl"},
    {"cast_qasymm_in", "cast.cl"},
    {"cast_qasymm_out", "cast.cl"},
    {"channel_combine_NV", "channel_combine.cl"},
    {"channel_combine_RGB888", "channel_combine.cl"},
    {"channel_combine_RGBA8888", "channel_combine.cl"},
    {"channel_combine_UYVY422", "channel_combine.cl"},
    {"channel_combine_YUYV422", "channel_combine.cl"},
    {"channel_shuffle_nchw", "channel_shuffle.cl"},
    {"channel_extract_NV12", "channel_extract.cl"},
    {"channel_extract_NV21", "channel_extract.cl"},
    {"channel_extract_RGB888", "channel_extract.cl"},
    {"channel_extract_RGBA8888", "channel_extract.cl"},
    {"channel_extract_UYVY422", "channel_extract.cl"},
    {"channel_extract_YUYV422", "channel_extract.cl"},
    {"combine_gradients_L1", "canny.cl"},
    {"combine_gradients_L2", "canny.cl"},
    {"concatenate_depth", "concatenate.cl"},
    {"concatenate_width", "concatenate.cl"},
    {"convolution_rectangle", "convolution_rectangle.cl"},
    {"col2im", "col2im.cl"},
    {"convert_depth_down", "depth_convert.cl"},
    {"convert_depth_up", "depth_convert.cl"},
    {"convert_fc_weights", "convert_fc_weights.cl"},
    {"convolution3x3_static", "convolution3x3.cl"},
    {"convolution5x5_static", "convolution5x5.cl"},
    {"convolution7x7_static", "convolution7x7.cl"},
    {"convolution9x9_static", "convolution9x9.cl"},
    {"convolution_separable1x5_static", "convolution5x5.cl"},
    {"convolution_separable5x1_static", "convolution5x5.cl"},
    {"convolution_separable1x7_static", "convolution7x7.cl"},
    {"convolution_separable7x1_static", "convolution7x7.cl"},
    {"convolution_separable1x9_static", "convolution9x9.cl"},
    {"convolution_separable9x1_static", "convolution9x9.cl"},
    {"copy_tensor", "copy_tensor.cl"},
    {"copy_plane", "channel_extract.cl"},
    {"copy_planes_3p", "channel_combine.cl"},
    {"copy_to_keypoint", "fast_corners.cl"},
    {"deconvolution_upsample", "deconvolution_layer.cl"},
    {"depthwise_convolution_3x3", "depthwise_convolution.cl"},
    {"depthwise_convolution_3x3_f16", "depthwise_convolution.cl"},
    {"depthwise_convolution_3x3_quantized_nchw", "depthwise_convolution_quantized.cl"},
    {"depthwise_convolution_3x3_quantized_nhwc_stride1", "depthwise_convolution_quantized.cl"},
    {"depthwise_convolution_3x3_quantized_nhwc_stride2", "depthwise_convolution_quantized.cl"},
    {"depthwise_convolution_3x3_stridex1_stridey1_bifrost_f16", "depthwise_convolution.cl"},
    {"depthwise_convolution_3x3_stridex2_stridey2_bifrost_f16", "depthwise_convolution.cl"},
    {"depthwise_convolution_3x3_stridex1_stridey1_bifrost_f32", "depthwise_convolution.cl"},
    {"depthwise_convolution_3x3_stridex2_stridey2_bifrost_f32", "depthwise_convolution.cl"},
    {"depthwise_im2col", "depthwise_convolution.cl"},
    {"depthwise_vector_to_tensor", "depthwise_convolution.cl"},
    {"depthwise_weights_reshape", "depthwise_convolution.cl"},
    {"dequantization_layer", "dequantization_layer.cl"},
    {"derivative", "derivative.cl"},
    {"dilate", "dilate.cl"},
    {"direct_convolution1x1", "direct_convolution1x1.cl"},
    {"direct_convolution1x1_f32_bifrost", "direct_convolution1x1.cl"},
    {"direct_convolution3x3", "direct_convolution3x3.cl"},
    {"direct_convolution3x3_f32_bifrost", "direct_convolution3x3.cl"},
    {"direct_convolution5x5", "direct_convolution5x5.cl"},
    {"direct_convolution5x5_f32_bifrost", "direct_convolution5x5.cl"},
    {"direct_convolution_1x1_3x3_5x5_quantized", "direct_convolution_1x1_3x3_5x5_quantized.cl"},
    {"erode", "erode.cl"},
    {"fast_corners", "fast_corners.cl"},
    {"fill_image_borders_constant", "fill_border.cl"},
    {"fill_image_borders_replicate", "fill_border.cl"},
    {"finalize", "optical_flow_pyramid_lk.cl"},
    {"floor_layer", "floor.cl"},
    {"gather", "gather.cl"},
    {"gather_1d", "gather.cl"},
    {"gather_1d_out", "gather.cl"},
    {"gaussian1x5_sub_x", "gaussian_pyramid.cl"},
    {"gaussian5x1_sub_y", "gaussian_pyramid.cl"},
    {"gemm_accumulate_biases", "gemm.cl"},
    {"gemm_interleave4x4", "gemm.cl"},
    {"gemm_ma_f16", "gemm.cl"},
    {"gemm_ma_f32", "gemm.cl"},
    {"gemm_ma_qs8", "gemm.cl"},
    {"gemm_ma_qs16", "gemm.cl"},
    {"gemm_mv", "gemv.cl"},
    {"gemm_mv_quantized", "gemv.cl"},
    {"gemm_mm_interleaved_transposed_f16", "gemm.cl"},
    {"gemm_mm_interleaved_transposed_f16_bifrost", "gemm.cl"},
    {"gemm_mm_interleaved_transposed_f32", "gemm.cl"},
    {"gemm_mm_interleaved_transposed_f32_bifrost", "gemm.cl"},
    {"gemm_mm_interleaved_transposed_qs8", "gemm.cl"},
    {"gemm_mm_interleaved_transposed_qs16", "gemm.cl"},
    {"gemm_mm_floating_point", "gemm.cl"},
    {"gemm_mm_floating_point_f16_bifrost", "gemm.cl"},
    {"gemm_mm_floating_point_f32_bifrost", "gemm.cl"},
    {"gemm_mm_floating_point_f32_bifrost_1000", "gemm.cl"},
    {"gemm_mm_qs8", "gemm.cl"},
    {"gemm_mm_qs16", "gemm.cl"},
    {"gemm_lc_vm_f32", "gemm.cl"},
    {"gemm_transpose1xW", "gemm.cl"},
    {"gemmlowp_matrix_a_reduction", "gemmlowp.cl"},
    {"gemmlowp_matrix_b_reduction", "gemmlowp.cl"},
    {"gemmlowp_mm_bifrost", "gemmlowp.cl"},
    {"gemmlowp_mm_midgard", "gemmlowp.cl"},
    {"gemmlowp_mm_interleaved_transposed_bifrost", "gemmlowp.cl"},
    {"gemmlowp_mm_interleaved_transposed_midgard", "gemmlowp.cl"},
    {"gemmlowp_offset_contribution", "gemmlowp.cl"},
    {"gemmlowp_output_stage_quantize_down", "gemmlowp.cl"},
    {"gemmlowp_output_stage_quantize_down_fixedpoint", "gemmlowp.cl"},
    {"harris_score_3x3", "harris_corners.cl"},
    {"harris_score_5x5", "harris_corners.cl"},
    {"harris_score_7x7", "harris_corners.cl"},
    {"hist_border_kernel", "histogram.cl"},
    {"hist_border_kernel_fixed", "histogram.cl"},
    {"hist_local_kernel", "histogram.cl"},
    {"hist_local_kernel_fixed", "histogram.cl"},
    {"hog_block_normalization", "hog.cl"},
    {"hog_detector", "hog.cl"},
    {"hog_orientation_binning", "hog.cl"},
    {"hysteresis", "canny.cl"},
    {"im2col1x1_stridex1_dchw", "im2col.cl"},
    {"im2col3x3_dchw", "im2col.cl"},
    {"im2col5x5_dchw", "im2col.cl"},
    {"im2col11x11_padx0_pady0_dchw", "im2col.cl"},
    {"im2col_generic_dchw", "im2col.cl"},
    {"im2col_generic_padx0_pady0_dchw", "im2col.cl"},
    {"im2col_reduced_dchw", "im2col.cl"},
    {"init_level", "optical_flow_pyramid_lk.cl"},
    {"init_level_max", "optical_flow_pyramid_lk.cl"},
    {"init_level_max_initial_estimate", "optical_flow_pyramid_lk.cl"},
    {"integral_horizontal", "integral_image.cl"},
    {"integral_vertical", "integral_image.cl"},
    {"IYUV_to_NV12_bt709", "color_convert.cl"},
    {"IYUV_to_RGB888_bt709", "color_convert.cl"},
    {"IYUV_to_RGBA8888_bt709", "color_convert.cl"},
    {"IYUV_to_YUV444_bt709", "color_convert.cl"},
    {"l2_normalize", "l2_normalize.cl"},
    {"lktracker_stage0", "optical_flow_pyramid_lk.cl"},
    {"lktracker_stage1", "optical_flow_pyramid_lk.cl"},
    {"magnitude_phase", "magnitude_phase.cl"},
    {"mean_stddev_accumulate", "mean_stddev.cl"},
    {"minmax", "minmaxloc.cl"},
    {"minmax_border", "minmaxloc.cl"},
    {"minmax_layer", "minmax_layer.cl"},
    {"minmaxloc", "minmaxloc.cl"},
    {"non_linear_filter_box3x3", "non_linear_filter3x3.cl"},
    {"non_linear_filter_cross3x3", "non_linear_filter3x3.cl"},
    {"non_linear_filter_disk3x3", "non_linear_filter3x3.cl"},
    {"non_linear_filter_box5x5", "non_linear_filter5x5.cl"},
    {"non_linear_filter_cross5x5", "non_linear_filter5x5.cl"},
    {"non_linear_filter_disk5x5", "non_linear_filter5x5.cl"},
    {"non_max_suppression", "nonmax.cl"},
    {"normalization_layer_cross_map", "normalization_layer.cl"},
    {"normalization_layer_in_map", "normalization_layer.cl"},
    {"NV12_to_IYUV_bt709", "color_convert.cl"},
    {"NV12_to_RGB888_bt709", "color_convert.cl"},
    {"NV12_to_RGBA8888_bt709", "color_convert.cl"},
    {"NV12_to_YUV444_bt709", "color_convert.cl"},
    {"NV21_to_IYUV_bt709", "color_convert.cl"},
    {"NV21_to_RGB888_bt709", "color_convert.cl"},
    {"NV21_to_RGBA8888_bt709", "color_convert.cl"},
    {"NV21_to_YUV444_bt709", "color_convert.cl"},
    {"output_stage_quantized", "direct_convolution_1x1_3x3_5x5_quantized.cl"},
    {"permute_201", "permute.cl"},
    {"permute_120", "permute.cl"},
    {"permute_3201", "permute.cl"},
    {"pixelwise_mul_float", "pixelwise_mul_float.cl"},
    {"pixelwise_mul_int", "pixelwise_mul_int.cl"},
    {"pixelwise_mul_qasymm8", "pixelwise_mul_quantized.cl"},
    {"pixelwise_div_float", "pixelwise_div_float.cl"},
    {"pixelwise_div_int", "pixelwise_div_int.cl"},
    {"pooling_layer_2", "pooling_layer.cl"},
    {"pooling_layer_3", "pooling_layer.cl"},
    {"pooling_layer_optimized_3", "pooling_layer.cl"},
    {"pooling_layer_7", "pooling_layer.cl"},
    {"pooling_layer_MxN_nchw", "pooling_layer.cl"},
    {"pooling_layer_MxN_nhwc", "pooling_layer.cl"},
    {"pooling_layer_MxN_quantized_nhwc", "pooling_layer_quantized.cl"},
    {"pooling_layer_MxN_quantized_nchw", "pooling_layer_quantized.cl"},
    {"quantization_layer", "quantization_layer.cl"},
    {"reduce_max", "reduce_max.cl"},
    {"reduction_operation", "reduction_operation.cl"},
    {"reduction_mean", "reduction_mean.cl"},
    {"remap_nearest_neighbour", "remap.cl"},
    {"remap_bilinear", "remap.cl"},
    {"reshape_layer", "reshape_layer.cl"},
    {"reshape_to_columns", "convolution_layer.cl"},
    {"RGB888_to_IYUV_bt709", "color_convert.cl"},
    {"RGB888_to_NV12_bt709", "color_convert.cl"},
    {"RGB888_to_RGBA8888_bt709", "color_convert.cl"},
    {"RGB888_to_YUV444_bt709", "color_convert.cl"},
    {"RGBA8888_to_IYUV_bt709", "color_convert.cl"},
    {"RGBA8888_to_NV12_bt709", "color_convert.cl"},
    {"RGBA8888_to_RGB888_bt709", "color_convert.cl"},
    {"RGBA8888_to_YUV444_bt709", "color_convert.cl"},
    {"roi_pooling_layer", "roi_pooling_layer.cl"},
    {"scale_nearest_neighbour", "scale.cl"},
    {"scale_bilinear", "scale.cl"},
    {"scharr3x3", "scharr_filter.cl"},
    {"sobel3x3", "sobel_filter.cl"},
    {"sobel_separable5x1", "sobel_filter.cl"},
    {"sobel_separable1x5", "sobel_filter.cl"},
    {"sobel_separable7x1", "sobel_filter.cl"},
    {"sobel_separable1x7", "sobel_filter.cl"},
    {"softmax_layer_norm", "softmax_layer.cl"},
    {"softmax_layer_norm_quantized", "softmax_layer_quantized.cl"},
    {"softmax_layer_max_shift_exp_sum_quantized_serial", "softmax_layer_quantized.cl"},
    {"softmax_layer_max_shift_exp_sum_quantized_parallel", "softmax_layer_quantized.cl"},
    {"softmax_layer_max_shift_exp_sum_serial", "softmax_layer.cl"},
    {"softmax_layer_max_shift_exp_sum_parallel", "softmax_layer.cl"},
    {"strided_slice", "strided_slice.cl"},
    {"suppress_non_maximum", "canny.cl"},
    {"tablelookup_U8", "tablelookup.cl"},
    {"tablelookup_S16", "tablelookup.cl"},
    {"threshold_binary", "threshold.cl"},
    {"threshold_range", "threshold.cl"},
    {"transpose", "transpose.cl"},
    {"UYVY422_to_IYUV_bt709", "color_convert.cl"},
    {"UYVY422_to_NV12_bt709", "color_convert.cl"},
    {"UYVY422_to_RGB888_bt709", "color_convert.cl"},
    {"UYVY422_to_RGBA8888_bt709", "color_convert.cl"},
    {"warp_affine_nearest_neighbour", "warp_affine.cl"},
    {"warp_affine_bilinear", "warp_affine.cl"},
    {"warp_perspective_nearest_neighbour", "warp_perspective.cl"},
    {"warp_perspective_bilinear", "warp_perspective.cl"},
    {"winograd_filter_transform_2x2_3x3_nchw", "winograd.cl"},
    {"winograd_filter_transform_4x4_3x3_nchw", "winograd.cl"},
    {"winograd_filter_transform_4x4_5x5_nchw", "winograd.cl"},
    {"winograd_input_transform_4x4_5x5_stepz1_nchw", "winograd.cl"},
    {"winograd_input_transform_2x2_3x3_stepz1_nchw", "winograd.cl"},
    {"winograd_input_transform_2x2_3x3_stepz2_nchw", "winograd.cl"},
    {"winograd_input_transform_4x4_3x3_stepz1_nchw", "winograd.cl"},
    {"winograd_output_transform_2x2_3x3_nchw", "winograd.cl"},
    {"winograd_output_transform_4x4_3x3_nchw", "winograd.cl"},
    {"winograd_output_transform_4x4_5x5_nchw", "winograd.cl"},
    {"YUYV422_to_IYUV_bt709", "color_convert.cl"},
    {"YUYV422_to_NV12_bt709", "color_convert.cl"},
    {"YUYV422_to_RGB888_bt709", "color_convert.cl"},
    {"YUYV422_to_RGBA8888_bt709", "color_convert.cl"},
    {"topkv2_init", "topkv2.cl"},
    {"topkv2_find_first_negative", "topkv2.cl"},
    {"topkv2_reorder_negatives", "topkv2.cl"},
    {"topkv2_store", "topkv2.cl"},
    {"radixsort_histogram", "topkv2_radixsort.cl"},
    {"radixsort_scanhistograms", "topkv2_radixsort.cl"},
    {"radixsort_pastehistograms", "topkv2_radixsort.cl"},
    {"radixsort_reorder", "topkv2_radixsort.cl"},
    {"topkv2_quicksort", "topkv2_quicksort.cl"},
};

const std::map<std::string, std::string> CLKernelLibraryEx::_program_source_map = {
#ifdef EMBEDDED_KERNELS
    {
        "cast.cl",
#include "./cl_kernels/cast.clembed"
    },
    {
        "fixed_point.h",
#include "./cl_kernels/fixed_point.hembed"
    },
    {
        "gather.cl",
#include "./cl_kernels/gather.clembed"
    },
    {
        "helpers.h",
#include "./cl_kernels/helpers.hembed"
    },
    {
        "helpers_asymm.h",
#include "./cl_kernels/helpers_asymm.hembed"
    },
    {
        "pixelwise_div_float.cl",
#include "./cl_kernels/pixelwise_div_float.clembed"
    },
    {
        "pixelwise_div_int.cl",
#include "./cl_kernels/pixelwise_div_int.clembed"
    },
    {
        "reduce_max.cl",
#include "./cl_kernels/reduce_max.clembed"
    },
    {
        "reduction_mean.cl",
#include "./cl_kernels/reduction_mean.clembed"
    },
    {
        "strided_slice.cl",
#include "./cl_kernels/strided_slice.clembed"
    },
    {
        "topkv2.cl",
#include "./cl_kernels/topkv2.clembed"
    },
    {
        "topkv2_radixsort.cl",
#include "./cl_kernels/topkv2_radixsort.clembed"
    },
    {
        "topkv2_quicksort.cl",
#include "./cl_kernels/topkv2_quicksort.clembed"
    },
#endif /* EMBEDDED_KERNELS */
};

CLKernelLibraryEx::CLKernelLibraryEx()
    : _context(), _device(), _kernel_path("."), _programs_map(), _built_programs_map()
{
  opencl_is_available(); // Make sure the OpenCL symbols are initialised *before* the
                         // CLKernelLibrary is built
}

CLKernelLibraryEx &CLKernelLibraryEx::get()
{
  static CLKernelLibraryEx _kernel_library;
  return _kernel_library;
}

Kernel CLKernelLibraryEx::create_kernel(const std::string &kernel_name,
                                        const StringSet &build_options_set) const
{
  // Find which program contains the kernel
  auto kernel_program_it = _kernel_program_map.find(kernel_name);

  if (_kernel_program_map.end() == kernel_program_it)
  {
    ARM_COMPUTE_ERROR("Kernel %s not found in the CLKernelLibrary", kernel_name.c_str());
  }
  std::string concat_str;

  if (fp16_supported(_device))
  {
    concat_str += " -DARM_COMPUTE_OPENCL_FP16_ENABLED=1 ";
  }

  if (get_cl_version(_device) == CLVersion::CL20)
  {
    concat_str += " -cl-std=CL2.0 ";
  }
  else if (arm_non_uniform_workgroup_supported(_device))
  {
    concat_str += " -cl-arm-non-uniform-work-group-size ";
  }
  else
  {
    ARM_COMPUTE_ERROR("Non uniform workgroup size is not supported!!");
  }

  // Check if the program has been built before with same build options.
  const std::string program_name = kernel_program_it->second;
  const std::string build_options = stringify_set(build_options_set) + concat_str;

  const std::string built_program_name = program_name + "_" + build_options;
  auto built_program_it = _built_programs_map.find(built_program_name);

  cl::Program cl_program;

  if (_built_programs_map.end() != built_program_it)
  {
    // If program has been built, retrieve to create kernel from it
    cl_program = built_program_it->second;
  }
  else
  {
    // Get program
    Program program = load_program(program_name);

    // Build program
    cl_program = program.build(build_options);

    // Add built program to internal map
    _built_programs_map.emplace(built_program_name, cl_program);
  }

  // Create and return kernel
  return Kernel(kernel_name, cl_program);
}

void CLKernelLibraryEx::add_built_program(const std::string &built_program_name,
                                          cl::Program program)
{
  _built_programs_map.emplace(built_program_name, program);
}

const Program &CLKernelLibraryEx::load_program(const std::string &program_name) const
{
  const auto program_it = _programs_map.find(program_name);

  if (program_it != _programs_map.end())
  {
    return program_it->second;
  }

  Program program;

#ifdef EMBEDDED_KERNELS
  const auto program_source_it = _program_source_map.find(program_name);

  if (_program_source_map.end() == program_source_it)
  {
    ARM_COMPUTE_ERROR("Embedded program for %s does not exist.", program_name.c_str());
  }

  program = Program(_context, program_name, program_source_it->second);
#else  /* EMBEDDED_KERNELS */
  // Check for binary
  std::string source_name = _kernel_path + program_name;
  std::string binary_name = source_name + "bin";

  if (std::ifstream(binary_name).is_open())
  {
    const std::string program_binary = read_file(binary_name, true);
    program = Program(_context, _device, program_name,
                      std::vector<unsigned char>(program_binary.begin(), program_binary.end()));
  }
  else if (std::ifstream(source_name).is_open())
  {
    program = Program(_context, program_name, read_file(source_name, false));
  }
  else
  {
    ARM_COMPUTE_ERROR("Kernel file %s does not exist.", source_name.c_str());
  }
#endif /* EMBEDDED_KERNELS */

  // Insert program to program map
  const auto new_program = _programs_map.emplace(program_name, std::move(program));

  return new_program.first->second;
}

std::string CLKernelLibraryEx::stringify_set(const StringSet &s) const
{
  std::string concat_set;

#ifndef EMBEDDED_KERNELS
  concat_set += "-I" + _kernel_path + " ";
#endif /* EMBEDDED_KERNELS */

  // Concatenate set
  for (const auto &el : s)
  {
    concat_set += " " + el;
  }

  return concat_set;
}

std::string CLKernelLibraryEx::get_program_source(const std::string &program_name)
{
  const auto program_source_it = _program_source_map.find(program_name);

  if (program_source_it == _program_source_map.end())
  {
    ARM_COMPUTE_ERROR("Embedded program for %s does not exist.", program_name.c_str());
  }

  return program_source_it->second;
}

size_t CLKernelLibraryEx::max_local_workgroup_size(const cl::Kernel &kernel) const
{
  size_t result;

  size_t err = kernel.getWorkGroupInfo(_device, CL_KERNEL_WORK_GROUP_SIZE, &result);
  ARM_COMPUTE_ERROR_ON_MSG(
      err != 0,
      "clGetKernelWorkGroupInfo failed to return the maximum workgroup size for the kernel");
  ARM_COMPUTE_UNUSED(err);

  return result;
}

cl::NDRange CLKernelLibraryEx::default_ndrange() const
{
  cl::Device device = cl::Device::getDefault();
  GPUTarget _target = get_target_from_device(device);
  cl::NDRange default_range;

  switch (_target)
  {
    case GPUTarget::MIDGARD:
    case GPUTarget::T600:
    case GPUTarget::T700:
    case GPUTarget::T800:
      default_range = cl::NDRange(128u, 1);
      break;
    default:
      default_range = cl::NullRange;
  }

  return default_range;
}

std::string CLKernelLibraryEx::get_device_version() { return _device.getInfo<CL_DEVICE_VERSION>(); }
