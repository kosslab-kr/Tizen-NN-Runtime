#include <iostream>
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <string>
#include <vector>
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
//#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

int main()
{
    const char graph_path[30] = "mobilenet_v1_192.tflite\0";
    const int num_threads = 1;
    std::string input_layer_type = "float";
    std::vector<int> sizes = {192 * 192}; // 192 * 192 * 3
    float x,y;
    
    // to use deep learning, load model into FlatBufferModel and then inference using interpreter
    std::unique_ptr<tflite::FlatBufferModel> model(tflite::FlatBufferModel::BuildFromFile(graph_path));
    
    if(!model){
        printf("Failed to mmap model\n");
        exit(0);
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if(!interpreter){
        printf("Failed to construct interpreter\n");
        exit(0);
    }
    
    
    if(num_threads != 1){
        interpreter->SetNumThreads(num_threads);
    }
    
    
    interpreter->ResizeInputTensor(0, sizes);
    
    
    float* input = interpreter->typed_input_tensor<float>(0);
    
    
    if(interpreter->AllocateTensors() != kTfLiteOk){
        printf("Failed to allocate tensors\n");
        exit(0);
    }
    
    std::cout << "OpenCV Version : " << CV_VERSION << std::endl;
    cv::Mat first_frame, frame, resized_frame;
    int original_width, original_height;
    cv::namedWindow("EXAMPLE02");
    cv::VideoCapture cap;
    
    cap.open(1);
    
    if(cap.isOpened())
    {
        std::cout << "cap opened" << std::endl;
        cap >> first_frame;
        original_width = first_frame.cols;
        original_height = first_frame.rows;
    }
    else
        std::cout << "cap not opened" << std::endl;
    
    
    while(true) {
        
        cap >> frame;
        cv::resize(frame, resized_frame, cv::Size(192, 192)); // resize for cnn
        //cv::resize(resized_frame, frame, cv::Size(original_width, original_height));
        
        uchar *data = resized_frame.data;
        
        // test...  it needs to change input and type value
        interpreter->typed_input_tensor<uchar*>(0)[0] = data
        /*
         interpreter->typed_input_tensor<float>(0)[0] = x;
         interpreter->typed_input_tensor<float>(0)[1] = y;
         */
        if(interpreter->Invoke() != kTfLiteOk)
        {
            std::printf("Failed to invoke!\n");
            exit(0);
        }
        float* output = interpreter->typed_output_tensor<float>(0);
        printf("output = %f\n", output[0]);
        
        cv::imshow("EXAMPLE02", frame);
        if(cv::waitKey(10) == 27)
        {
            break;
        }
    }
    
    cv::destroyWindow("EXAMPLE02");
    
    return 0;
}
