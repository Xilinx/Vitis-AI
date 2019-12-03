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

/* 3.16GOP times calculation for Inception_V1 CONV */
#define INCEPTIONV1_WORKLOAD_CONV (3.16f)

#define KRENEL_Inception_V1 "inception_v1_0"

/* Input Node for Kernel Inception_V1 */
#define INPUT_NODE "conv1_7x7_s2"
/* Output Node for Inception_V1 */
#define OUTPUT_NODE "loss3_classifier"

#define IMAGE_COUNT 1000

const string baseImagePath = "./image/";

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
 * @brief Run Task for Inception_V1
 *
 * @param taskInception_V1 - pointer to GooLeNet Task
 *
 * @return none
 */
void runInception_V1(DPUTask *taskInception_V1, Mat img) {
    assert(taskInception_V1);

    int8_t *outAddr = (int8_t *)dpuGetOutputTensorAddress(taskInception_V1, OUTPUT_NODE);
    int size = dpuGetOutputTensorSize(taskInception_V1, OUTPUT_NODE);
    int channel = dpuGetOutputTensorChannel(taskInception_V1, OUTPUT_NODE);
    float scale = dpuGetOutputTensorScale(taskInception_V1, OUTPUT_NODE);
    float *softmax = new float[size];
    
    _T(dpuSetInputImage2(taskInception_V1, INPUT_NODE, img));
    _T(dpuRunTask(taskInception_V1));
 
    _T(dpuRunSoftmax(outAddr, softmax, channel, size/channel, scale));

    delete[] softmax;
}

/*
 * @brief  - Entry of classify using Inception_V1
 *
 * @param kernel - point to DPU Kernel
 */
void classifyEntry(DPUKernel *kernelInception_V1) {
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
            DPUTask *taskInception_V1 = dpuCreateTask(kernelInception_V1, 0);
            for(unsigned int ind = i  ;ind < IMAGE_COUNT;ind+=threadnum) {

                /* Process the image using Inception_V1 model*/
                runInception_V1(taskInception_V1, img);
            }

            /* Destroy DPU Tasks & free resources */
            dpuDestroyTask(taskInception_V1);
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
 * @brief Entry for runing Inception_V1 neural network
 *
 */
int main(int argc ,char** argv) {
    if(argc == 2)
    threadnum = stoi(argv[1]);
    else {
    cerr << "please input thread number!" <<  endl;
    exit(-1);
    }
    /* DPU Kernels for runing Inception_V1 */
    DPUKernel *kernelInception_V1;

    /* Attach to DPU driver and prepare for runing */
    dpuOpen();

    /* Create DPU Kernels for Inception_V1 */
    kernelInception_V1 = dpuLoadKernel(KRENEL_Inception_V1);

    /* Entry of classify Inception_V1 */
    classifyEntry(kernelInception_V1);

    /* Destroy DPU Tasks & free resources */
    dpuDestroyKernel(kernelInception_V1);

    /* Dettach from DPU driver & release resources */
    dpuClose();

    return 0;
}
