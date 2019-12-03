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

#include <dnndk/dnndk.h>
#include "dputils.h"
#include "utils.h"


using namespace std;
using namespace cv;
using namespace std::chrono;


#define NMS_THRESHOLD 0.3f
#define INPUT_NODE "layer0_conv"

int idxInputImage = 0;  // frame index of input video
int idxShowImage = 0;   // next frame index to be displayed
bool bReading = true;   // flag of reding input frame
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

/**
 * @brief Feed input frame into DPU for process
 *
 * @param task - pointer to DPU Task for YOLO-v3 network
 * @param frame - pointer to input frame
 * @param mean - mean value for YOLO-v3 network
 *
 * @return none
 */
void setInputImageForYOLO(DPUTask* task, const Mat& frame, float* mean) {
    Mat img_copy;
    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);
    int size = dpuGetInputTensorSize(task, INPUT_NODE);
    int8_t* data = dpuGetInputTensorAddress(task, INPUT_NODE);

    image img_new = load_image_cv(frame);
    image img_yolo = letterbox_image(img_new, width, height);

    vector<float> bb(size);
    for(int b = 0; b < height; ++b) {
        for(int c = 0; c < width; ++c) {
            for(int a = 0; a < 3; ++a) {
                bb[b*width*3 + c*3 + a] = img_yolo.data[a*height*width + b*width + c];
            }
        }
    }

    float scale = pow(2, 7);

    for(int i = 0; i < size; ++i) {
        data[i] = int(bb.data()[i]*scale);
        if(data[i] < 0) data[i] = 127;
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
void readFrame(const char *fileName) {
    static int loop = 3;
    VideoCapture video;
    string videoFile = fileName;
    start_time = chrono::system_clock::now();

    while (loop>0) {
        loop--;
        if (!video.open(videoFile)) {
            cout<<"Fail to open specified video file:" << videoFile << endl;
            exit(-1);
        }

        while (true) {
            usleep(20000);
            Mat img;
            if (queueInput.size() < 30) {
                if (!video.read(img) ) {
                    break;
                }

                mtxQueueInput.lock();
                queueInput.push(make_pair(idxInputImage++, img));
                mtxQueueInput.unlock();
            } else {
                usleep(10);
            }
        }

        video.release();
    }

    exit(0);
}

/**
 * @brief Thread entry for displaying image frames
 *
 * @param  none
 * @return none
 *
 */
void displayFrame() {
    Mat frame;

    while (true) {
        mtxQueueShow.lock();

        if (queueShow.empty()) {
            mtxQueueShow.unlock();
            usleep(10);
        } else if (idxShowImage == queueShow.top().first) {
            auto show_time = chrono::system_clock::now();
            stringstream buffer;
            frame = queueShow.top().second;
            auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
            buffer << fixed << setprecision(1)
                   << (float)queueShow.top().first / (dura / 1000000.f);
            string a = buffer.str() + " FPS";
            cv::putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{240, 240, 240},1);
            cv::imshow("ADAS Detection@Xilinx DPU", frame);

            idxShowImage++;
            queueShow.pop();
            mtxQueueShow.unlock();
            if (waitKey(1) == 'q') {
                bReading = false;
                exit(0);
            }
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
void postProcess(DPUTask* task, Mat& frame, int sWidth, int sHeight){
    const string classes[3] = {"car", "person", "cycle"};

    /* four output nodes of YOLO-v3 */
    const string outputs_node[4] = {"layer81_conv", "layer93_conv", "layer105_conv",  "layer117_conv"};

    vector<vector<float>> boxes;
    for(int i = 0; i < 4; i++){
        string output_node = outputs_node[i];
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());

        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());
        int8_t* dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
        get_output(dpuOut, sizeOut, scale, channel, height, width, result);

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
void runYOLO(DPUTask* task) {
    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);

    while (true) {
        pair<int, Mat> pairIndexImage;

        mtxQueueInput.lock();
        if (queueInput.empty()) {
            mtxQueueInput.unlock();
            if (bReading)
            {
                continue;
            } else {
                break;
            }
        } else {
            /* get an input frame from input frames queue */
            pairIndexImage = queueInput.front();
            queueInput.pop();
            mtxQueueInput.unlock();
        }
        vector<vector<float>> res;
        /* feed input frame into DPU Task with mean value */
        setInputImageForYOLO(task, pairIndexImage.second, mean);

        /* invoke the running of DPU for YOLO-v3 */
        dpuRunTask(task);

        postProcess(task, pairIndexImage.second, width, height);
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
    if (argc != 2) {
        cout << "Usage of ADAS detection: ./adas video-file" << endl;
        return -1;
    }

    /* Attach to DPU driver and prepare for running */
    dpuOpen();

    /* Load DPU Kernels for YOLO-v3 network model */
    DPUKernel *kernel = dpuLoadKernel("yolo");
    vector<DPUTask *> task(4);

    /* Create 4 DPU Tasks for YOLO-v3 network model */
    generate(task.begin(), task.end(),
    std::bind(dpuCreateTask, kernel, 0));

    /* Spawn 6 threads:
    - 1 thread for reading video frame
    - 4 identical threads for running YOLO-v3 network model
    - 1 thread for displaying frame in monitor
    */
    array<thread, 6> threadsList = {
    thread(readFrame, argv[1]),
    thread(displayFrame),
    thread(runYOLO, task[0]),
    thread(runYOLO, task[1]),
    thread(runYOLO, task[2]),
    thread(runYOLO, task[3]),

    };

    for (int i = 0; i < 6; i++) {
        threadsList[i].join();
    }

    /* Destroy DPU Tasks & free resources */
    for_each(task.begin(), task.end(), dpuDestroyTask);

    /* Destroy DPU Kernels & free resources */
    dpuDestroyKernel(kernel);

    /* Dettach from DPU driver & free resources */
    dpuClose();

    return 0;
}

