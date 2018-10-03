#include "internal/layers/SimpleEmbeddingLookup.h"

#include <arm_compute/runtime/CL/CLScheduler.h>

void SimpleEmbeddingLookup::configure(::arm_compute::ITensor *lookups,
                                      ::arm_compute::ITensor *values,
                                      ::arm_compute::ITensor *output)
{
  // Assume that verification of operands are already done at Planner::visit()
  _lookups = lookups;
  _values = values;
  _output = output;
}

void SimpleEmbeddingLookup::run()
{
  if (::internal::arm_compute::isGpuMode())
  {
    auto &q = ::arm_compute::CLScheduler::get().queue();

    CAST_CL(_lookups)->map(q);
    CAST_CL(_values)->map(q);
    CAST_CL(_output)->map(q);
  }

  // type of elements of lookups is always integer
  const int32_t *lookups_buf = reinterpret_cast<int32_t *>(_lookups->buffer());
  const auto values_buf = _values->buffer();
  auto output_buf = _output->buffer();

  const auto lookups_info = _lookups->info();
  const auto values_info = _values->info();
  const auto output_info = _output->info();

  // TODO Refactor below duplicated code!
  const auto values_rank = values_info->num_dimensions();
  switch (values_rank)
  {
    case 2:
      // (H,W) in nnapi -> (W,H) in acl
      {
        const size_t row_size = values_info->dimension(1);
        const size_t row_bytes = values_info->total_size() / row_size;
        for (size_t i = 0; i < lookups_info->dimension(0); ++i)
        {
          if (lookups_buf[i] < 0 || lookups_buf[i] >= row_size)
            throw std::runtime_error("Embedding Lookup: index out of bounds.");

          size_t idx = lookups_buf[i];
          size_t row_offset_by_idx = values_info->offset_element_in_bytes({0, idx});
          size_t row_offset_by_i = output_info->offset_element_in_bytes({0, i});

          unsigned char *sink_addr = output_buf + row_offset_by_i;
          unsigned char *source_addr = values_buf + row_offset_by_idx;
          memcpy(sink_addr, source_addr, row_bytes);
        }
      }
      break;
    case 3:
      // (B,H,W) in nnapi -> (W,H,B) in acl
      {
        const size_t row_size = values_info->dimension(2);
        const size_t row_bytes = values_info->total_size() / row_size;
        for (size_t i = 0; i < lookups_info->dimension(0); ++i)
        {
          if (lookups_buf[i] < 0 || lookups_buf[i] >= row_size)
            throw std::runtime_error("Embedding Lookup: index out of bounds.");

          size_t idx = lookups_buf[i];
          size_t row_offset_by_idx = values_info->offset_element_in_bytes({0, 0, idx});
          size_t row_offset_by_i = output_info->offset_element_in_bytes({0, 0, i});

          unsigned char *sink_addr = output_buf + row_offset_by_i;
          unsigned char *source_addr = values_buf + row_offset_by_idx;
          memcpy(sink_addr, source_addr, row_bytes);
        }
      }
      break;
    case 4:
      // (N,H,W,C) in nnapi -> (N,C,H,W) in acl
      {
        const size_t row_size = values_info->dimension(3);
        const size_t row_bytes = values_info->total_size() / row_size;
        for (size_t i = 0; i < lookups_info->dimension(0); ++i)
        {
          if (lookups_buf[i] < 0 || lookups_buf[i] >= row_size)
            throw std::runtime_error("Embedding Lookup: index out of bounds.");

          size_t idx = lookups_buf[i];
          size_t row_offset_by_idx = values_info->offset_element_in_bytes({0, 0, 0, idx});
          size_t row_offset_by_i = output_info->offset_element_in_bytes({0, 0, 0, i});

          unsigned char *sink_addr = output_buf + row_offset_by_i;
          unsigned char *source_addr = values_buf + row_offset_by_idx;
          memcpy(sink_addr, source_addr, row_bytes);
        }
      }
      break;
    case 1:
      // In this case, shape of values actually is matrix but the height(row size) is 1 in acl. If
      // row size is 1, this op is not needed and it means this situtation could be wrong.
      throw std::runtime_error("Wrong usage of EmbeddingLookup op!");
    default:
      throw std::runtime_error("Not supported rank!");
  }

  if (::internal::arm_compute::isGpuMode())
  {
    auto &q = ::arm_compute::CLScheduler::get().queue();

    CAST_CL(_lookups)->unmap(q);
    CAST_CL(_values)->unmap(q);
    CAST_CL(_output)->unmap(q);
  }
}
