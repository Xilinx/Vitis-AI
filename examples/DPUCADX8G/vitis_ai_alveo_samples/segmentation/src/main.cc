/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>
// Header files for APIs
#include <dpu/dpu_runner.hpp>

//GraphInfo shapes;
using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace vitis;
using namespace ai;

uint8_t colorB[] = {128, 232, 70, 156, 153, 153, 30,  0,   35, 152,
                    180, 60,  0,  142, 70,  100, 100, 230, 32};
uint8_t colorG[] = {64, 35, 70, 102, 153, 153, 170, 220, 142, 251,
                    130, 20, 0, 0, 0, 60, 80, 0, 11};
uint8_t colorR[] = {128, 244, 70,  102, 190, 153, 250, 220, 107, 152,
                    70,  220, 255, 0,   0,   0,   0,   0,   119};

// comparison algorithm for priority_queue
class Compare {
    public:
    bool operator()(const pair<int, Mat> &n1, const pair<int, Mat> &n2) const {
        return n1.first > n2.first;
    }
};

// input video
VideoCapture video;

// flags for each thread
bool is_reading = true;
bool is_running_1 = true;
bool is_running_2 = true;
bool is_displaying = true;

queue<pair<int, Mat>> read_queue;                                               // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare> display_queue;  // display queue
mutex mtx_read_queue;                                                           // mutex of read queue
mutex mtx_display_queue;                                                        // mutex of display queue
int read_index = 0;                                                             // frame index of input video
int display_index = 0;                                                          // frame index to display


/**
 * @brief entry routine of segmentation, and put image into display queue
 *
 * @param task - pointer to Segmentation Task
 * @param is_running - status flag of the thread
 *
 * @return none
 */
void runSegmentation(vitis::ai::DpuRunner *runner, bool &is_running) {
	
    while (is_running) {
        // Get an image from read queue
        int index;
        Mat img;
        mtx_read_queue.lock();
        if (read_queue.empty()) {
            mtx_read_queue.unlock();
            if (is_reading) {
                continue;
            } else {
                is_running = false;
                break;
            }
        } else {
            index = read_queue.front().first;
            img = read_queue.front().second;
            read_queue.pop();
            mtx_read_queue.unlock();
        }
        //get in/out tensor
        auto outputTensors = runner->get_output_tensors();
        auto inputTensors = runner->get_input_tensors();
		int outSize = outputTensors[0]->get_element_num() 
      / outputTensors[0]->get_dim_size(0);
	  int batchSize = 1 ;
	//init out data
        float *result = new float[outSize*batchSize];

	//get tensor shape info
        int outHeight;
        int outWidth ;
		int outChannel;
	if (runner->get_tensor_format() == DpuRunner::TensorFormat::NCHW) {
      outHeight = outputTensors[0]->get_dim_size(2);
      outWidth = outputTensors[0]->get_dim_size(3);
	  outChannel = outputTensors[0]->get_dim_size(1);
    } else {
      outHeight = outputTensors[0]->get_dim_size(1);
      outWidth = outputTensors[0]->get_dim_size(2);
	  outChannel = outputTensors[0]->get_dim_size(3);
    }
        int inHeight;
        int inWidth;
        int inSize;
		int inChannel;
		
	if (runner->get_tensor_format() == DpuRunner::TensorFormat::NCHW) {
      inHeight = inputTensors[0]->get_dim_size(2);
      inWidth = inputTensors[0]->get_dim_size(3);
	  inChannel = inputTensors[0]->get_dim_size(1);
    } else {
      inHeight = inputTensors[0]->get_dim_size(1);
      inWidth = inputTensors[0]->get_dim_size(2);
	  inChannel=inputTensors[0]->get_dim_size(3);
    }
	inSize = inputTensors[0]->get_element_num() 
      / inputTensors[0]->get_dim_size(0);

	// image pre-process
	
	float mean[3] = {104,117,123};
    Mat image2 = cv::Mat(inHeight,inWidth,CV_8SC3);
    float *imageInputs = new float[inSize];
    resize(img, image2, Size(inWidth, inHeight), 0, 0, INTER_LINEAR); 
	if(runner->get_tensor_format() == DpuRunner::TensorFormat::NHWC) {
		
		
        for (int h =0; h< inHeight;h++)
            for (int w =0; w< inWidth;w++)
                for (int c =0; c< 3; c++)
                    imageInputs[h*inWidth*3+w*3+c] =img.at<Vec3b>(h,w)[c]-mean[c];
        } else {
            for (int c =0; c< 3; c++)
                for (int h =0; h< inHeight;h++)
                    for (int w =0; w< inWidth;w++)
                        imageInputs[c*inWidth*inHeight+h*inWidth+w] =image2.at<Vec3b>(h,w)[c]-mean[c]; 
		}
	//tensor buffer prepare
        std::vector<vitis::ai::CpuFlatTensorBuffer>inputs, outputs;
        inputs.push_back(ai::CpuFlatTensorBuffer(imageInputs,inputTensors[0]));
        outputs.push_back(ai::CpuFlatTensorBuffer(result,outputTensors[0]));
        std::vector<vitis::ai::TensorBuffer*> inputsPtr, outputsPtr;
        inputsPtr.push_back(&inputs[0]);
        outputsPtr.push_back(&outputs[0]);

	// run
        auto job_id = runner->execute_async(inputsPtr,outputsPtr);
        runner->wait(job_id.first,-1);

        Mat segMat(outHeight, outWidth, CV_8UC3);
        Mat showMat(inHeight, inWidth, CV_8UC3);
	if(runner->get_tensor_format() == DpuRunner::TensorFormat::NHWC) {
            for (int row = 0; row < outHeight; row++) {
                for (int col = 0; col < outWidth; col++) {
                    int i = row * outWidth * 19 + col * 19;
                    auto max_ind = max_element(result + i, result + i + 19);
                    int posit = distance(result + i, max_ind);
                    segMat.at<Vec3b>(row, col) = Vec3b(colorB[posit], colorG[posit], colorR[posit]);
                }
            }
        } else{ 
            for (int row = 0; row < outHeight; row++) {
				
                for (int col = 0; col < outWidth; col++) {
	                float pix_val[19];
		            for( int ch=0;ch<19;ch++){
                        int i = ch*outHeight*outWidth + row*outWidth + col;
                        pix_val[ch] = result[i];
                    }
                    auto max_ind = max_element(pix_val, pix_val + 19);
                    int posit = distance(pix_val, max_ind);
                    segMat.at<Vec3b>(row, col) = Vec3b(colorB[posit], colorG[posit], colorR[posit]);
	            }
            }
        }
        // resize to original scale and overlay for displaying
        resize(segMat, showMat, Size(inWidth, inHeight), 0, 0, INTER_NEAREST);
        for (int i = 0; i < showMat.rows * showMat.cols * 3; i++) {
            img.data[i] = img.data[i] * 0.4 + showMat.data[i] * 0.6;
        }

        // Put image into display queue
        mtx_display_queue.lock();
        display_queue.push(make_pair(index, img));
        mtx_display_queue.unlock();
    }
}

/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(bool &is_reading) {
    while (is_reading) {
        Mat img;
        if (read_queue.size() < 30) {
            if (!video.read(img)) {
                cout << "Finish reading the video." << endl;
                is_reading = false;
                is_running_1 = false;
                is_displaying = false;
                break;
            }
            mtx_read_queue.lock();
            read_queue.push(make_pair(read_index++, img));
            mtx_read_queue.unlock();
        } else {
            usleep(20);
        }
    }
}

/**
 * @brief Display frames in display queue
 *
 * @param is_displaying - status flag of Display thread
 *
 * @return none
 */
void Display(bool &is_displaying) {
	static int fm_cnt=0;
    while (is_displaying) {
        mtx_display_queue.lock();
        if (display_queue.empty()) {
            if (is_running_1 || is_running_2) {
                mtx_display_queue.unlock();
                usleep(20);
            } else {
                is_displaying = false;
                break;
            }
        } else if (display_index == display_queue.top().first) {
            // Display image
            Mat frame = display_queue.top().second;
			char img_name[20];
            sprintf(img_name,"output/image%d.jpg",fm_cnt); // writing inference output as image in output folder
            cv::imwrite(img_name,frame);
	    if(fm_cnt==0)
            cout<<" Saving results to output folder as .jpg images " <<endl;
	    fm_cnt++;
            display_index++;
            display_queue.pop();
            mtx_display_queue.unlock();
        } else {
            mtx_display_queue.unlock();
        }
    }
}

/**
 * @brief Entry for running Segmentation neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */
int main(int argc, char **argv) {

    // Check args
    if (argc != 3) {
        cout << "Usage of segmentation demo: ./segmentaion file_name[string] path(for json file)" << endl;
        cout << "\tfile_name: path to your video file" << endl;
        return -1;
    }

    // Initializations
    string file_name = argv[1];
    cout << "Detect video: " << file_name << endl;
    video.open(file_name);
    if (!video.isOpened()) {
        cout << "Failed to open video: " << file_name;
        return -1;
    }
    // create runner
    auto runners = vitis::ai::DpuRunner::create_dpu_runner(argv[2]);
    auto runner = runners[0].get();
   
    // Run tasks
    array<thread, 3> threads = {thread(Read, ref(is_reading)),
                                thread(runSegmentation, runner, ref(is_running_1)), 
                                thread(Display, ref(is_displaying))};

    for (int i = 0; i < 3; ++i) {
        threads[i].join();
    }
    video.release();
    return 0;
}
