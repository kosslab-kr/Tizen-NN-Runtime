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
//#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"


#define    SIZE 224

void GetTopN(const float* prediction, const int prediction_size, const int num_results, const float threshold, std::vector<std::pair<float, int>>* top_results);
void LoadLabels(const char* filename, std::vector<std::string>* label_strings);

int main(int argc, char *argv[])
{
    const int num_threads = 1;
    std::vector<int> sizes = {SIZE * SIZE * 3}; // 192 * 192 * 3
 
    if(argc != 3) {
        fprintf(stderr, "Usage: <device#> <model>\n");
        return 1;
    }else{
        std::cout << "Reading device# from: " << argv[1] << std::endl;
        std::cout << "Reading model from: " << argv[2] << std::endl;
    }

    long conv = strtol(argv[1], NULL, 10);
    const int cam_num = conv;
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

    interpreter->UseNNAPI(0);

    int t_size = interpreter->tensors_size();

    std::cout << "tensors_size: " << t_size << "\n";
    
    if(!interpreter){
        printf("Failed to construct interpreter\n");
        exit(0);
    }
    
    
    if(num_threads != 1){
        interpreter->SetNumThreads(num_threads);
    }
    
    interpreter->ResizeInputTensor(0, sizes);
    
    if(interpreter->AllocateTensors() != kTfLiteOk){
        printf("Failed to allocate tensors\n");
        exit(0);
    }

    std::cout << "OpenCV Version : " << CV_VERSION << std::endl;
    cv::Mat frame, resized_frame, rgb_frame;
    cv::VideoCapture cap;
    
    cap.open(cam_num);

//    const float input_mean = 0.0f;
    const float input_std = 255.0f;
//    int wanted_input_height = SIZE;
//    int wanted_input_width = SIZE;
//    int wanted_input_channels = 3;
    float* out = interpreter->typed_input_tensor<float>(0);
    
    if(cap.isOpened())
    {
        std::cout << "cap opened[" << cam_num << "]" << std::endl;
        clock_t past_time = clock();
        
        while(true) {
        
            cap >> frame;

            cv::resize(frame, resized_frame, cv::Size(SIZE, SIZE)); // resize for cnn       
            cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);
//            uint8_t* in = rgb_frame.data;
            
            for(int i = 0; i < SIZE * SIZE * 3; ++i)
                out[i] = rgb_frame.data[i] / input_std;
/*            
            for (int y = 0; y < SIZE; ++y) {
                uint8_t* in_row = in + (y * SIZE * 3);
                float* out_row = out + (y * SIZE * 3);
                for (int x = 0; x < SIZE; ++x) {
                    uint8_t* in_pixel = in_row + (x * 3);
                    float* out_pixel = out_row + (x * 3);
                    for (int c = 0; c < 3; ++c) {
                        out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
                    }
                }
            }
*/            
            if(interpreter->Invoke() != kTfLiteOk)
            {
                std::printf("Failed to invoke!\n");
                exit(0);
            }

            const int output_size = (int)labels.size();
            const int kNumResults = 5;
            const float kThreshold = 0.1f;
            std::vector<std::pair<float, int>> top_results;
            float* output = interpreter->typed_output_tensor<float>(0);
            GetTopN(output, output_size, kNumResults, kThreshold, &top_results);

            //std::vector<std::pair<float, std::string>> newValues;
            int i = 1;
            for (const auto& result : top_results) {
                //std::pair<float, std::string> item;
                //item.first = result.first;
                //item.second = labels[result.second];
                
                //newValues.push_back(item);
                
                std::cout << "[" << i << ", " << labels[result.second] << ", " << result.first << "]" << std::endl;
                i++;
            }
            clock_t cur_time = clock();
            clock_t tmp_time = (cur_time - past_time) / 1000.0f;
            tmp_time = tmp_time / 1.0f;
            std::cout << "elapsed time : " << tmp_time << "ms" << std::endl;
            std::cout << "fps : " << 1000.0 / tmp_time << std::endl;
            
            past_time = cur_time;
            
            std::cout << std::endl;
            
            usleep(10);
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
void GetTopN(const float* prediction, const int prediction_size, const int num_results,
                    const float threshold, std::vector<std::pair<float, int>>* top_results) {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> top_result_pq;

    const long count = prediction_size;
    for (int i = 0; i < count; ++i) {
        const float value = prediction[i];
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
