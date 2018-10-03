#include "tensor_loader.h"

#include <assert.h>

#include <fstream>

#include "util/tensor/Shape.h"

namespace TFLiteRun
{

TensorLoader::TensorLoader(tflite::Interpreter &interpreter)
    : _interpreter(interpreter), _raw_data(nullptr)
{
}

void TensorLoader::load(const std::string &filename)
{
  // TODO Handle file open/read error
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  uint32_t num_tensors = 0;
  file.read(reinterpret_cast<char *>(&num_tensors), sizeof(num_tensors));

  int tensor_indices_raw[num_tensors];
  file.read(reinterpret_cast<char *>(tensor_indices_raw), sizeof(tensor_indices_raw));
  std::vector<int> tensor_indices(tensor_indices_raw, tensor_indices_raw + num_tensors);

  _raw_data = std::unique_ptr<float>(new float[file_size]);
  file.read(reinterpret_cast<char *>(_raw_data.get()), file_size);

  size_t offset = 0;
  for (const auto &o : tensor_indices)
  {
    const TfLiteTensor *tensor = _interpreter.tensor(o);

    // Convert tensor shape to `Shape` from `tensor->dims`
    nnfw::util::tensor::Shape shape(static_cast<size_t>(tensor->dims->size));
    for (int d = 0; d < tensor->dims->size; d++)
    {
      shape.dim(d) = tensor->dims->data[d];
    }

    float *base = _raw_data.get() + offset;

    assert(tensor->bytes % sizeof(float) == 0);
    offset += (tensor->bytes / sizeof(float));

    _tensor_map.insert(std::make_pair(o, nnfw::support::tflite::TensorView<float>(shape, base)));
  }

  // The file size and total output tensor size must match
  assert(file_size == sizeof(num_tensors) + sizeof(tensor_indices_raw) + offset * sizeof(float));

  file.close();
}

const nnfw::support::tflite::TensorView<float> &TensorLoader::get(int tensor_idx) const
{
  auto found = _tensor_map.find(tensor_idx);
  assert(found != _tensor_map.end());
  return found->second;
}

} // end of namespace TFLiteRun
