#include <tensorflow/core/public/session.h>

#include <iostream>
#include <stdexcept>

#include <cassert>
#include <cstring>

#include "util/benchmark.h"

#define CHECK_TF(e) {                             \
  if(!(e).ok())                                   \
  {                                               \
    throw std::runtime_error{"'" #e "' FAILED"};  \
  }                                               \
}

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cerr << "USAGE: " << argv[0] << " [T/F model path] [output 0] [output 1] ..." << std::endl;
    return 255;
  }

  std::vector<std::string> output_nodes;

  for (int argn = 2; argn < argc; ++argn)
  {
    output_nodes.emplace_back(argv[argn]);
  }

  tensorflow::Session* sess;

  CHECK_TF(tensorflow::NewSession(tensorflow::SessionOptions(), &sess));

  tensorflow::GraphDef graph_def;

  CHECK_TF(ReadBinaryProto(tensorflow::Env::Default(), argv[1], &graph_def));
  CHECK_TF(sess->Create(graph_def));

  tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 320, 320, 3}));
  std::vector<tensorflow::Tensor> outputs;

  for (uint32_t n = 0; n < 5; ++n)
  {
    std::chrono::milliseconds elapsed(0);

    nnfw::util::benchmark::measure(elapsed) << [&] (void) {
      CHECK_TF(sess->Run({{"input_node", input}}, output_nodes, {}, &outputs));
    };

    std::cout << "Takes " << elapsed.count() << "ms" << std::endl;
  }

  return 0;
}
