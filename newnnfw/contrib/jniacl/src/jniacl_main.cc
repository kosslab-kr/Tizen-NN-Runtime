#include <jni.h>
#include <string>

#include <arm_compute/graph/Graph.h>
#include <arm_compute/graph/Nodes.h>

#include "io_accessor.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_samsung_testaclexec_ActivityMain_RunACLJNI(JNIEnv *env, jobject)
{
    using arm_compute::DataType;
    using arm_compute::graph::Tensor;
    using arm_compute::graph::TargetHint;
    using arm_compute::graph::Graph;
    using arm_compute::TensorInfo;
    using arm_compute::TensorShape;

    arm_compute::graph::Graph graph;
    TargetHint target_hint = TargetHint::OPENCL;
    bool autoinc = true;

    graph << target_hint
        << Tensor(TensorInfo(TensorShape(3U, 3U, 1U, 1U), 1, DataType::F32),
                  std::unique_ptr<InputAccessor>(new InputAccessor(autoinc)))
        << arm_compute::graph::ConvolutionLayer(
              3U, 3U, 1U,
              std::unique_ptr<WeightAccessor>(new WeightAccessor(autoinc)),
              std::unique_ptr<BiasAccessor>(new BiasAccessor()),
              arm_compute::PadStrideInfo(1, 1, 0, 0))
        << Tensor(std::unique_ptr<OutputAccessor>(new OutputAccessor()));
        ;

    graph.run();

    std::string hello = "SoftMax Run OK";

    return env->NewStringUTF(hello.c_str());
}
