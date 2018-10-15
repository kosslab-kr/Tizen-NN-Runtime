#ifndef __TFLITE_RUN_TENSOR_LOADER_H__
#define __TFLITE_RUN_TENSOR_LOADER_H__

#include <sys/mman.h>

#include <string>
#include <unordered_map>

#include "support/tflite/TensorView.h"

namespace tflite
{
class Interpreter;
}

namespace TFLiteRun
{

class TensorLoader
{
public:
  TensorLoader(tflite::Interpreter &interpreter);
  void load(const std::string &filename);
  const nnfw::support::tflite::TensorView<float> &get(int tensor_idx) const;
  size_t getNums() const { return _tensor_map.size(); }

private:
  tflite::Interpreter &_interpreter;
  std::unique_ptr<float> _raw_data;
  std::unordered_map<int, nnfw::support::tflite::TensorView<float>> _tensor_map;
};

} // end of namespace TFLiteRun

#endif // __TFLITE_RUN_TENSOR_LOADER_H__
