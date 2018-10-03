#include "tensor_dumper.h"

#include <fstream>
#include <iostream>
#include <cstring>

#include "tensorflow/contrib/lite/interpreter.h"

namespace TFLiteRun
{

TensorDumper::TensorDumper()
{
  // DO NOTHING
}

void TensorDumper::addTensors(tflite::Interpreter &interpreter, const std::vector<int> &indices)
{
  for (const auto &o : indices)
  {
    const TfLiteTensor *tensor = interpreter.tensor(o);
    int size = tensor->bytes;
    std::vector<char> buffer;
    buffer.resize(size);
    memcpy(buffer.data(), tensor->data.raw, size);
    _tensors.emplace_back(o, std::move(buffer));
  }
}

void TensorDumper::dump(const std::string &filename) const
{
  // TODO Handle file open/write error
  std::ofstream file(filename, std::ios::out | std::ios::binary);

  // Write number of tensors
  uint32_t num_tensors = static_cast<uint32_t>(_tensors.size());
  file.write(reinterpret_cast<const char *>(&num_tensors), sizeof(num_tensors));

  // Write tensor indices
  for (const auto &t : _tensors)
  {
    file.write(reinterpret_cast<const char *>(&t._index), sizeof(int));
  }

  // Write data
  for (const auto &t : _tensors)
  {
    file.write(t._data.data(), t._data.size());
  }

  file.close();
}

} // end of namespace TFLiteRun
