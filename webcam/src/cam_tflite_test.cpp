#include <iostream>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <string>
#include <vector>
#include <time.h>
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/interpreter.h"
//#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"


template <class T>
void GetTopN(T* prediction, int prediction_size, int num_results,
                float threshold, std::vector<std::pair<float, int>>* top_results, bool input_floating);

// explicit instantiation so that we can use them otherwhere
template void GetTopN<uint8_t>(uint8_t*, int, int, float,
                                 std::vector<std::pair<float, int>>*, bool);
template void GetTopN<float>(float*, int, int, float,
                               std::vector<std::pair<float, int>>*, bool);

void LoadLabels(const char* filename, std::vector<std::string>* label_strings);

int main(int argc, char *argv[])
{
    const int num_threads = 4;
 
    if(argc != 6) {
        fprintf(stderr, "Usage: <device#> <model> <input_width> <input_height> <use NNAPI>\n");
        return 1;
    }else{
        std::cout << "Device#: " << argv[1] << std::endl;
        std::cout << "Model from: " << argv[2] << std::endl;
        std::cout << "Input Width: " << argv[3] << std::endl;
        std::cout << "Input Height: " << argv[4] << std::endl;
        std::cout << "Use NNAPI: " << argv[5] << std::endl;
    }

    long conv = strtol(argv[1], NULL, 10);
    const int cam_num = conv;
    
    conv = strtol(argv[3], NULL, 10);
    const int width = conv;
    
    conv = strtol(argv[4], NULL, 10);
    const int height = conv;
    
    conv = strtol(argv[5], NULL, 10);
    const int use_nnapi = conv;

    if(use_nnapi != 0 && use_nnapi != 1)
    {
        fprintf(stderr, "param <use NNAPI> is NOT 0 OR 1\n");
        return 1;
    }


    const char* filename = argv[2];
   
    // to use deep learning, load model into FlatBufferModel and then inference using interpreter
    std::unique_ptr<tflite::FlatBufferModel> model(tflite::FlatBufferModel::BuildFromFile(filename));
    //std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
    
    if(!model){
        printf("Failed to mmap model\n");
        exit(0);
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::vector<std::string> labels;
    LoadLabels("labels.txt", &labels);
    
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    interpreter->UseNNAPI(use_nnapi);

    int t_size = interpreter->tensors_size();

    std::cout << "tensors_size: " << t_size << "\n";
    
    if(!interpreter){
        printf("Failed to construct interpreter\n");
        exit(0);
    }
    
    
    if(num_threads != 1){
        interpreter->SetNumThreads(num_threads);
    }
    
    //interpreter->ResizeInputTensor(interpreter->inputs()[0], std::vector<int>{1, 224, 224, 3});
    
    if(interpreter->AllocateTensors() != kTfLiteOk){
        printf("Failed to allocate tensors\n");
        exit(0);
    }

    std::cout << "OpenCV Version : " << CV_VERSION << std::endl;
    cv::Mat frame, resized_frame, rgb_frame;
    cv::VideoCapture cap;
    
    cap.open(cam_num);

    const float input_std = 255.0f;
   
    if(cap.isOpened())
    {
        std::cout << "cap opened[" << cam_num << "]" << std::endl;
        
        int input = interpreter->inputs()[0];
        int output = interpreter->outputs()[0];
        std::cout << "input : " << input << std::endl;
        std::cout << "output : " << output << std::endl;
        
        const int output_size = (int)labels.size();
        const int kNumResults = 5;
        const float kThreshold = 0.01f;  
        double st, end, fps, elapsed;
        st = static_cast<double>(cv::getTickCount());
        
        while(true) {
            cap >> frame;

            cv::resize(frame, resized_frame, cv::Size(width, height)); // resize for cnn       
            cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);
            
            switch (interpreter->tensor(input)->type) {
                case kTfLiteFloat32:
                    for(int i = 0; i < width * height * 3; ++i)
                        interpreter->typed_input_tensor<float>(0)[i] = rgb_frame.data[i] / input_std;
                    break;
                case kTfLiteUInt8:
                    for(int i = 0; i < width * height * 3; ++i)
                        interpreter->typed_input_tensor<uint8_t>(0)[i] = rgb_frame.data[i];
                    break;
                default:
                    std::cout << "cannot handle input type "
                         << interpreter->tensor(0)->type << " yet";
                    exit(-1);
            }
          
            if(interpreter->Invoke() != kTfLiteOk)
            {
                std::printf("Failed to invoke!\n");
                exit(0);
            }

            std::vector<std::pair<float, int>> top_results;
            
            switch (interpreter->tensor(output)->type) {
                case kTfLiteFloat32:
                    GetTopN(interpreter->typed_output_tensor<float>(0), output_size, kNumResults, kThreshold, &top_results, true);
                    break;
                case kTfLiteUInt8:
                    GetTopN(interpreter->typed_output_tensor<uint8_t>(0), output_size, kNumResults, kThreshold, &top_results, false);
                    break;
                default:
                    std::cout << "cannot handle input type "
                         << interpreter->tensor(0)->type << " yet";
                    exit(-1);
            }         

            int i = 1;
            for (const auto& result : top_results) {               
                std::cout << "[" << i << ", " << labels[result.second] << ", " << result.first << "]" << std::endl;
                i++;
            }
                       
            end = static_cast<double>(cv::getTickCount());
            elapsed = (end - st) / cv::getTickFrequency() * 1000;
            fps = 1000 / elapsed;
            std::cout << "elapsed time : " << elapsed << "ms" << std::endl;
            std::cout << "fps : " << fps << std::endl;
            std::cout << std::endl;
            
            st = static_cast<double>(cv::getTickCount());
            //usleep(10);
/*
            if(cv::waitKey(10) == 27)
            {
                break;
            }
*/
        }
    }
    else
        std::cout << "cap not opened" << std::endl;
    
    return 0;
}

void LoadLabels(const char* filename, std::vector<std::string>* label_strings) {
    std::ifstream t;
    t.open(filename, std::ifstream::in);
    std::string line;
        while (t) {
        std::getline(t, line);
        if (line.length()){
            label_strings->push_back(line);
        }
    }
    t.close();
}

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
template <class T>
void GetTopN(T* prediction, int prediction_size, int num_results,
                float threshold, std::vector<std::pair<float, int>>* top_results, bool input_floating) {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> top_result_pq;

    const long count = prediction_size;
    for (int i = 0; i < count; ++i) {
        float value;
        
        if (input_floating)
            value = prediction[i];
        else
            value = prediction[i] / 255.0;
      
        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        if (value < threshold) {
            continue;
        }

        top_result_pq.push(std::pair<float, int>(value, i));

        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) {
            top_result_pq.pop();
        }
  }

    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
}

