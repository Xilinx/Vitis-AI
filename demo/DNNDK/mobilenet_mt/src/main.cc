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

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

/* header file for Vitis AI advanced APIs */
#include <dnndk/dnndk.h>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for Caffe input images APIs */
#include "dputils.h"

using namespace cv;
using namespace std;
using namespace std::chrono;

int threadnum;

/* DPU Kernel name for MobileNet */
#define KRENEL_MOBILNET "mobilenet"
/* Input Node for Kernel MobileNet */
#define CONV_INPUT_NODE "263"
/* Output Node for Kernel MobileNet */
#define OUTPUT_NODE "417"

#define IMAGE_COUNT 1000

const string baseImagePath = "./image/";

/*#define SHOWTIME*/
#ifdef SHOWTIME
#define _T(func)                                                          \
{                                                                         \
    auto _start = system_clock::now();                                    \
    func;                                                                 \
    auto _end = system_clock::now();                                      \
    auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
    string tmp = #func;                                                   \
    tmp = tmp.substr(0, tmp.find('('));                                   \
    cout << "[TimeTest]" << left << setw(30) << tmp;                      \
    cout << left << setw(10) << duration << "us" << endl;                 \
}
#else
#define _T(func) func;
#endif

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
        string name = entry->d_name;
        string ext = name.substr(name.find_last_of(".") + 1);
        if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
            (ext == "PNG") || (ext == "png")) {
            images.push_back(name);
            }
        }
    }

    closedir(dir);
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const &path, vector<string> &kinds) {
    kinds.clear();
    fstream fkinds(path);

    if (fkinds.fail()) {
        fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
        exit(1);
    }

    string kind;
    while (getline(fkinds, kind)) {
        kinds.push_back(kind);
    }

    fkinds.close();
}

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float *d, int size, int k, vector<string> &vkinds) {
    assert(d && size > 0 && k > 0);
    priority_queue<pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(pair<float, int>(d[i], i));
    }

    for (auto i = 0; i < k; ++i) {
        pair<float, int> ki = q.top();
        printf("[Top]%d prob = %-8f  name = %s\n", i, d[ki.second], vkinds[ki.second].c_str());
        q.pop();
    }
}

/**
 * @brief Run DPU Task for MobileNet
 *
 * @param taskMobilenet - pointer to MobileNet Task
 * @param img - The mat to be process
 *
 * @return none
 */
void runMobilenet(DPUTask *taskMobilenet, Mat &img) {
    assert(taskMobilenet);

    /* Get the output Tensor for Mobilenet Task  */
    int8_t *outAddr = (int8_t *)dpuGetOutputTensorAddress(taskMobilenet, OUTPUT_NODE);
    /* Get size of the output Tensor for Mobilenet Task  */
    int size = dpuGetOutputTensorSize(taskMobilenet, OUTPUT_NODE);
    /* Get channel count of the output Tensor for MobileNet Task  */
    int channel = dpuGetOutputTensorChannel(taskMobilenet, OUTPUT_NODE);
    /* Get scale of the output Tensor for Mobilenet Task  */
    float out_scale = dpuGetOutputTensorScale(taskMobilenet, OUTPUT_NODE);

    float *softmax = new float[size];

    vector<float> mean{104, 117, 123};
    float scale = 0.00390625;
    dpuSetInputImageWithScale(taskMobilenet, CONV_INPUT_NODE, img, mean.data(), scale);

    /* Launch RetNet50 Task */
    _T(dpuRunTask(taskMobilenet));

    /* Calculate softmax on DPU and display TOP-5 classification results */
    _T(dpuRunSoftmax(outAddr, softmax, channel, size/channel, out_scale));

    delete[] softmax;
}

/*
 * @brief  - Entry of classify using Mobilenet
 *
 * @param kernelMobilenet - point to DPU Kernel of Mobilenet
 */
void classifyEntry(DPUKernel *kernelMobilenet) {
    vector<string> kinds, images;
    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: Not images exist in " << baseImagePath << endl;
        return;
    } else {
        cout << "total image : " << IMAGE_COUNT << endl;
    }

    /* Load all kinds words.*/
    LoadWords(baseImagePath + "words.txt", kinds);
    if (kinds.size() == 0) {
        cerr << "\nError: Not words exist in words.txt." << endl;
        return;
    }

    thread workers[threadnum];

    Mat img = imread(baseImagePath + images.at(0));
    auto _start = system_clock::now();

    for (auto i = 0; i < threadnum; i++) {
        workers[i] = thread([&,i]() {
            /* Create DPU Tasks from DPU Kernel */
            DPUTask *taskMobilenet = dpuCreateTask(kernelMobilenet, 0);

            for(unsigned int ind = i  ;ind < IMAGE_COUNT;ind+=threadnum) {
                /* Run MobileNet Task */
                runMobilenet(taskMobilenet, img);
            }

            /* Destroy DPU Tasks & free resources */
            dpuDestroyTask(taskMobilenet);
        });
    }

    /* Release thread resources. */
    for (auto &w : workers) {
        if (w.joinable()) w.join();
    }

    auto _end = system_clock::now();
    auto duration = (duration_cast<microseconds>(_end - _start)).count();

    cout << "[Time]" << duration << "us" << endl;
    cout << "[FPS]" << IMAGE_COUNT*1000000.0/duration  << endl;
}

/**
 * @brief Entry for runing MobileNet neural network
 *
 * @note Vitis AI advanced APIs prefixed with "dpu" are used to easily program &
 *       deploy MobileNet on DPU platform.
 *
 */
int main(int argc ,char** argv) {
    DPUKernel *kernelMobilenet;

    if(argc == 2)
        threadnum = stoi(argv[1]);
    else {
        cout << "please input thread number!" << endl;
        exit(-1);
    }

    /* Attach to DPU driver and prepare for running */
    dpuOpen();

    /* Create DPU Task for MobileNet */
    kernelMobilenet = dpuLoadKernel(KRENEL_MOBILNET);

    /* Entry of classify using Mobilenet */
    classifyEntry(kernelMobilenet);

    /* Destroy DPU Task & free resources */
    dpuDestroyKernel(kernelMobilenet);

    /* Dettach from DPU driver & free resources */
    dpuClose();

    return 0;
}
