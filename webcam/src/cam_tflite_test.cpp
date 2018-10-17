#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <string>
#include <vector>
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
//#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

#define	SIZE 224

int main(int argc, char *argv[])
{
    const int num_threads = 1;
    //std::string input_layer_type = "float";
    std::vector<int> sizes = {SIZE * SIZE * 3}; // 192 * 192 * 3
 
    if(argc != 2) {
        fprintf(stderr, "Usage: <model>\n");
        return 1;
    }else{
        std::cout << "Reading model from: " << argv[1] << std::endl;
        //std::cout << "Reading image from: " << argv[2] << std::endl;
    }

    const char* filename = argv[1];
    //const char* imagefile = argv[2];
   
    // to use deep learning, load model into FlatBufferModel and then inference using interpreter
    std::unique_ptr<tflite::FlatBufferModel> model(tflite::FlatBufferModel::BuildFromFile(filename));
    //std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
    
    if(!model){
        printf("Failed to mmap model\n");
        exit(0);
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
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
    
    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();

    std::cout << "number of inputs: " << inputs.size() << "\n";
    std::cout << "number of outputs: " << outputs.size() << "\n";

    
    interpreter->ResizeInputTensor(0, sizes);
    
    if(interpreter->AllocateTensors() != kTfLiteOk){
        printf("Failed to allocate tensors\n");
        exit(0);
    }

    std::cout << "OpenCV Version : " << CV_VERSION << std::endl;
    cv::Mat frame, resized_frame, rgb_frame;
//    int original_width, original_height;
    //cv::namedWindow("EXAMPLE02");
    cv::VideoCapture cap;
    
    int cam_num;
    std::cout << "input cam device #: ";
    std::cin >> cam_num;
    cap.open(cam_num);

    const float input_mean = 0.0f;
    const float input_std = 255.0f;
    int wanted_input_height = SIZE;
    int wanted_input_width = SIZE;
    int wanted_input_channels = 3;
    float* out = interpreter->typed_input_tensor<float>(0);
    
    if(cap.isOpened())
    {
        std::cout << "cap opened" << std::endl;
//        cap >> frame;
//        original_width = first_frame.cols;
//        original_height = first_frame.rows;

//        imwrite("img.jpg", first_frame);
//        std::cout << "saved img.jpg" << std::endl;

        while(true) {
        
            cap >> frame;

            cv::resize(frame, resized_frame, cv::Size(SIZE, SIZE)); // resize for cnn
        
            // test...  it needs to change input and type value
            //interpreter->typed_input_tensor<uchar>(0) = data;
            //int input = interpreter->inputs()[0];

            cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);
            uint8_t* in = rgb_frame.data;

            for (int y = 0; y < wanted_input_height; ++y) {
                uint8_t* in_row = in + (y * SIZE * 3);
                float* out_row = out + (y * wanted_input_width * wanted_input_channels);
                for (int x = 0; x < wanted_input_width; ++x) {
                    uint8_t* in_pixel = in_row + (x * 3);
                    float* out_pixel = out_row + (x * wanted_input_channels);
                    for (int c = 0; c < wanted_input_channels; ++c) {
                        out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
                    }
                }
            }

//            for(int i = 0; i < SIZE * SIZE * 3; i++)
//                interpreter->typed_input_tensor<float>(0)[i] = resized_frame.data[i];


/*
            cv::Mat bgr[3];
            cv::split(resized_frame, bgr);

            for(int i = 0; i < SIZE * SIZE; i++)
            {
                for(int j = 0; j < 3; j++)
                    interpreter->typed_input_tensor<uint_8>(0)[i * 3 + j] = bgr[(j - 2) * (-1)].data[i];
//                    interpreter->typed_input_tensor<float>(0)[j * SIZE * SIZE + i] = bgr[j].data[i];
            }
*/
            
            if(interpreter->Invoke() != kTfLiteOk)
            {
                std::printf("Failed to invoke!\n");
                exit(0);
            }

            //float* output = interpreter->typed_output_tensor<float>(0);
            //printf("output = %f\n", output[0]);

            std::vector<std::pair<float, int> > v;
            for(int i = 0; i <= 1000; i++)
                v.push_back(std::pair<float, int>(interpreter->typed_output_tensor<float>(0)[i], i));    

            std::sort(v.begin(), v.end());
           
            for(int i = 1000; i >= 996; i--){
                std::cout << "[" << v.at(i).second << "," << v.at(i).first << "]" << std::endl;
            }
            std::cout << std::endl;

            //cv::imshow("EXAMPLE02", frame);
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
    
    
 
    
    //cv::destroyWindow("EXAMPLE02");
    
    return 0;
}
