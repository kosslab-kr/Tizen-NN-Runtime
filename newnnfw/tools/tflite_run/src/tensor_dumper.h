#ifndef __TFLITE_RUN_TENSOR_DUMPER_H__
#define __TFLITE_RUN_TENSOR_DUMPER_H__

#include <memory>
#include <string>
#include <vector>

namespace tflite
{
class Interpreter;
}

namespace TFLiteRun
{

class TensorDumper
{
private:
  struct Tensor
  {
    int _index;
    std::vector<char> _data;

    Tensor(int index, std::vector<char> &&data) : _index(index), _data(std::move(data)) {}
  };

public:
  TensorDumper();
  void addTensors(tflite::Interpreter &interpreter, const std::vector<int> &indices);
  void dump(const std::string &filename) const;

private:
  std::vector<Tensor> _tensors;
};

} // end of namespace TFLiteRun

#endif // __TFLITE_RUN_TENSOR_DUMPER_H__
