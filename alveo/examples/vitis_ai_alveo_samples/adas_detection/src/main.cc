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

#include <algorithm>
#include <vector>
#include <atomic>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>
#include <dpu/dpu_runner.hpp>

#include "utils.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace vitis;
using namespace ai;

// input video
VideoCapture video;
// flags for each thread
bool is_reading = true;
bool is_running = true;
bool is_displaying = true;

#define NMS_THRESHOLD 0.3f


int idxInputImage = 0;  // frame index of input video
int idxShowImage = 0;   // next frame index to be displayed
bool bReading = true;   // flag of reading input frame
chrono::system_clock::time_point start_time;

typedef pair<int, Mat> imagePair;
class paircomp {
    public:
    bool operator()(const imagePair &n1, const imagePair &n2) const {
        if (n1.first == n2.first) {
            return (n1.first > n2.first);
        }

        return n1.first > n2.first;
    }
};

// mutex for protection of input frames queue
mutex mtxQueueInput;
// mutex for protection of display frmaes queue
mutex mtxQueueShow;
// input frames queue
queue<pair<int, Mat>> queueInput;
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;
//GraphInfo shapes;
/**
 * @brief Feed input frame into DPU for process
 *
 * @param task - pointer to DPU Task for YOLO-v3 network
 * @param frame - pointer to input frame
 * @param mean - mean value for YOLO-v3 network
 *
 * @return none
 */
void setInputImageForYOLO(vitis::ai::DpuRunner* runner, float *data, const Mat& frame, float* mean) {
    Mat img_copy;
	int width,height;
	auto  inputTensors = runner->get_input_tensors();
   
	if (runner->get_tensor_format() == DpuRunner::TensorFormat::NCHW) {
      height = inputTensors[0]->get_dim_size(2);
      width = inputTensors[0]->get_dim_size(3);
    } else {
      height = inputTensors[0]->get_dim_size(1);
      width = inputTensors[0]->get_dim_size(2);
    }
	
  
	int size = inputTensors[0]->get_element_num() / inputTensors[0]->get_dim_size(0);
    image img_new = load_image_cv(frame);
    image img_yolo = letterbox_image(img_new, width, height);

    vector<float> bb(size);
    if (runner->get_tensor_format() == DpuRunner::TensorFormat::NHWC) {

        for(int b = 0; b < height; ++b) {
            for(int c = 0; c < width; ++c) {
                for(int a = 0; a < 3; ++a) {
                    bb[b*width*3 + c*3 + a] = img_yolo.data[a*height*width + b*width + c];
                }
            }
        }
    } else {
        for(int a = 0; a < 3; ++a) {
            for(int b = 0; b < height; ++b) {
                for(int c = 0; c < width; ++c) {
                    bb[a*width*height + b*width + c] = img_yolo.data[a*height*width + b*width + c];
                }
            }
        }
    }
    float scale = pow(2, 7);
    for(int i = 0; i < size; ++i) {
        data[i] = bb.data()[i];
        if(data[i] < 0) data[i] = (float)(127/scale);
    }
    free_image(img_new);
    free_image(img_yolo);
}


/**
 * @brief Thread entry for reading image frame from the input video file
 *
 * @param fileName - pointer to video file name
 *
 * @return none
 */
void readFrame(bool &is_reading) {
	
	while (is_reading) {
        Mat img;
        if (queueInput.size() < 30) {
            if (!video.read(img)) {
                cout << "Finish reading the video." << endl;
                is_reading = false;
                is_running = false;
                is_displaying = false;
                break;
            }
            mtxQueueInput.lock();
            queueInput.push(make_pair(idxInputImage++, img));
            mtxQueueInput.unlock();
        } else {
            usleep(20);
        }
    }
}
/**
 * @brief Thread entry for displaying image frames
 *
 * @param  none
 * @return none
 *
 */
void displayFrame(bool &is_displaying) {
    Mat frame;
    static int fm_cnt=0;
    while (is_displaying) {
        mtxQueueShow.lock();
        if (queueShow.empty()) {
            if (is_running) {
                mtxQueueShow.unlock();
                usleep(10);
            } else {
                is_displaying = false;
                break;
            }
        } else if (idxShowImage == queueShow.top().first) {
            auto show_time = chrono::system_clock::now();
            stringstream buffer;
            frame = queueShow.top().second;
            auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
            buffer << fixed << setprecision(1)
                   << (float)queueShow.top().first / (dura / 1000000.f);
            string a = buffer.str() + " FPS";
            cv::putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{240, 240, 240},1);
            
	    char img_name[20];
            sprintf(img_name,"output/image%d.jpg",fm_cnt); // writing inference output as image in output folder
            cv::imwrite(img_name,frame);
	    if(fm_cnt==0)
	    	cout<<"Saving results to output folder as .jpg images "<<endl;
            idxShowImage++;
            fm_cnt++;
            queueShow.pop();
            mtxQueueShow.unlock();
        } else {
            mtxQueueShow.unlock();
        }
    }
}

/**
 * @brief Post process after the running of DPU for YOLO-v3 network
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param frame
 * @param sWidth
 * @param sHeight
 *
 * @return none
 */
void postProcess(vitis::ai::DpuRunner* runner, Mat& frame,vector<float*> results, int sWidth, int sHeight){
    const string classes[3] = {"car", "person", "cycle"};

    vector<vector<float>> boxes;
    auto  outputTensors = runner->get_output_tensors();
	
    for(int i = 0; i < 4; i++) {
        
		
        int width;
        int height;
        int channel;
		
	if (runner->get_tensor_format() == DpuRunner::TensorFormat::NCHW) {
      height = outputTensors[i]->get_dim_size(2);
      width = outputTensors[i]->get_dim_size(3);
	  channel = outputTensors[i]->get_dim_size(1);
    } else {
      height = outputTensors[i]->get_dim_size(1);
      width = outputTensors[i]->get_dim_size(2);
	  channel = outputTensors[i]->get_dim_size(3);
    }
	int sizeOut = channel * width * height;
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
		if (runner->get_tensor_format() == DpuRunner::TensorFormat::NHWC) {

        get_output(results[i], sizeOut,  channel, height, width, result);
    } else {
        get_output_nchw(results[i], sizeOut,  channel, height, width, result);
    }
        

        /* Store the object detection frames as coordinate information  */
        detect(boxes, result, channel, height, width, i, sHeight, sWidth);
    }

    /* Restore the correct coordinate frame of the original image */
    correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);

    /* Apply the computation for NMS */
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    float h = frame.rows;
    float w = frame.cols;
    for(size_t i = 0; i < res.size(); ++i) {
        float xmin = (res[i][0] - res[i][2]/2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3]/2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2]/2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3]/2.0) * h + 1.0;

        if(res[i][res[i][4] + 6] > CONF ) {
            int type = res[i][4];
            string classname = classes[type];

            if (type==0) {
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 0, 255), 1, 1, 0);
            }
            else if (type==1) {
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 0, 0), 1, 1, 0);
            }
            else {
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0 ,255, 255), 1, 1, 0);
            }
        }
    }
}

/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 *
 * @return none
 */
void runYOLO(vitis::ai::DpuRunner* runner,bool &is_running1) {
    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};
	
    auto  inputTensors = runner->get_input_tensors();
    
    auto outputTensors = runner->get_output_tensors();
	
	auto out_dims = outputTensors[0]->get_dims();
    auto in_dims = inputTensors[0]->get_dims();
	
	/*get shape info*/
    int outSize0 = outputTensors[0]->get_element_num() 
      / outputTensors[0]->get_dim_size(0);
	int outSize1 = outputTensors[1]->get_element_num() 
      / outputTensors[1]->get_dim_size(0);
	int outSize2 = outputTensors[2]->get_element_num() 
      / outputTensors[2]->get_dim_size(0);
	int outSize3 = outputTensors[3]->get_element_num() 
      / outputTensors[3]->get_dim_size(0);
	  
    int inSize = inputTensors[0]->get_element_num() 
      / inputTensors[0]->get_dim_size(0);
	  
    int inHeight = 0;
    int inWidth = 0;
	
    if (runner->get_tensor_format() == DpuRunner::TensorFormat::NCHW) {
      inHeight = inputTensors[0]->get_dim_size(2);
      inWidth = inputTensors[0]->get_dim_size(3);
    } else {
      inHeight = inputTensors[0]->get_dim_size(1);
      inWidth = inputTensors[0]->get_dim_size(2);
    }
	int batchSize = 1 ;
	
    // input/output data define 
    float *data = new float[inSize*batchSize];
    float *result0 = new float[outSize0*batchSize];
    float *result1 = new float[outSize1*batchSize];
    float *result2 = new float[outSize2*batchSize];
    float *result3 = new float[outSize3*batchSize];
    vector<float*> result;
    result.push_back(result0);
    result.push_back(result1);
    result.push_back(result2);
    result.push_back(result3);

    while (is_running1) {
        pair<int, Mat> pairIndexImage;

        mtxQueueInput.lock();
        if (queueInput.empty()) {
            mtxQueueInput.unlock();
            if (is_reading)
            {
                continue;
            } else {
				is_running = false;
                break;
            }
        } else {
            /* get an input frame from input frames queue */
            pairIndexImage = queueInput.front();
            queueInput.pop();
            mtxQueueInput.unlock();
        }
        /* feed input frame into DPU Task with mean value */

        setInputImageForYOLO(runner, data, pairIndexImage.second, mean);
        // input/output tensorbuffer prepare
        std::vector<vitis::ai::CpuFlatTensorBuffer>inputs, outputs;
        inputs.push_back(vitis::ai::CpuFlatTensorBuffer(data,inputTensors[0]));
        outputs.push_back(vitis::ai::CpuFlatTensorBuffer(result0,outputTensors[0]));
        outputs.push_back(vitis::ai::CpuFlatTensorBuffer(result1,outputTensors[1]));
        outputs.push_back(vitis::ai::CpuFlatTensorBuffer(result2,outputTensors[2]));
        outputs.push_back(vitis::ai::CpuFlatTensorBuffer(result3,outputTensors[3]));
        std::vector<vitis::ai::TensorBuffer*> inputsPtr, outputsPtr;
        inputsPtr.push_back(&inputs[0]);
        outputsPtr.push_back(&outputs[0]);
        outputsPtr.push_back(&outputs[1]);
        outputsPtr.push_back(&outputs[2]);
        outputsPtr.push_back(&outputs[3]);
        /* invoke the running of DPU for YOLO-v3 */
        auto job_id = runner->execute_async(inputsPtr,outputsPtr);
        runner->wait(job_id.first,-1);

        postProcess(runner, pairIndexImage.second, result, inWidth, inHeight);
        mtxQueueShow.lock();

        /* push the image into display frame queue */
        queueShow.push(pairIndexImage);
        mtxQueueShow.unlock();
    }
}

/**
 * @brief Entry for running YOLO-v3 neural network for ADAS object detection
 *
 */
int main(const int argc, const char** argv) {
    if (argc != 3) {
        cout << "Usage of ADAS detection: ./adas video-file path(for json file)" << endl;
        return -1;
    }


	string file_name = argv[2];
    cout << "Detect video: " << file_name << endl;
    video.open(file_name);
    if (!video.isOpened()) {
        cout << "Failed to open video: " << file_name;
        return -1;
    }
	
    /* Create 1 XDNN Tasks for YOLO-v3 network model */

    /* Spawn 3 threads:
    - 1 thread for reading video frame
    - 1 identical threads for running YOLO-v3 network model
    - 1 thread for displaying frame in monitor
    */	
    auto runners = vitis::ai::DpuRunner::create_dpu_runner(argv[1]);
    auto runner = runners[0].get();
       array<thread, 3> threadsList = {
    thread(readFrame, ref(is_reading)),
    thread(runYOLO, runner,ref(is_running)),
	thread(displayFrame,ref(is_displaying))

    };
    for (int i = 0; i < 3; i++) {
        threadsList[i].join();
    }
    video.release();
    return 0;
}

