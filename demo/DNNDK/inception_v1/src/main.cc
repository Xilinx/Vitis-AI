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
#include <sys/stat.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for Vitis AI advanced APIs */
#include <dnndk/dnndk.h>

/* header file for Caffe input images APIs */
#include "dputils.h"

using namespace std;
using namespace cv;

/* 3.16GOP times calculation for Inception_V1 CONV */
#define INCEPTIONV1_WORKLOAD (3.16f)

#define KRENEL_Inception_V1 "inception_v1_0"
/* Input Node for Kernel Inception_V1 */
#define INPUT_NODE "conv1_7x7_s2"
/* Output Node for Inception_V1 */
#define OUTPUT_NODE "loss3_classifier"

const string baseImagePath = "../dataset/image500_640_480/";

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(std::string const &path, std::vector<std::string> &images) {
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
            std::string name = entry->d_name;
            std::string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
            (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
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
void LoadWords(std::string const &path, std::vector<std::string> &kinds) {
    kinds.clear();
    std::fstream fkinds(path);

    if (fkinds.fail()) {
        fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
        exit(1);
    }

    std::string kind;
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
void TopK(const float *d, int size, int k, std::vector<std::string> &vkinds) {
    assert(d && size > 0 && k > 0);
    std::priority_queue<std::pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(std::pair<float, int>(d[i], i));
    }

    for (auto i = 0; i < k; ++i) {
        std::pair<float, int> ki = q.top();
        fprintf(stdout, "top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
            vkinds[ki.second].c_str());
        q.pop();
    }
}

/**
 * @brief Run DPU Task for Inception_V1
 *
 * @param taskInception_V1 - pointer to Inception_V1 Task
 *
 * @return none
 */
void runInception_V1(DPUTask *taskInception_V1) {
    assert(taskInception_V1);

    /* Mean value for Inception_V1 specified in Caffe prototxt */
    vector<string> kinds, images;

    /*Load all image names */
    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: Not images existing under " << baseImagePath << endl;
        return;
    }

    /*Load all kinds words.*/
    LoadWords(baseImagePath + "words.txt", kinds);
    if (kinds.size() == 0) {
        cerr << "\nError: No words exist in file words.txt." << endl;
        return;
    }

    /* Get the output Tensor for Inception_V1 Task  */
    int8_t *outAddr = (int8_t *)dpuGetOutputTensorAddress(taskInception_V1, OUTPUT_NODE);
    /* Get size of the output Tensor for Inception_V1 Task  */
    int size = dpuGetOutputTensorSize(taskInception_V1, OUTPUT_NODE); 
    /* Get channel count of the output Tensor for Inception_V1 Task  */
    int channel = dpuGetOutputTensorChannel(taskInception_V1, OUTPUT_NODE);
    /* Get scale of the output Tensor for Inception_V1 Task  */
    float scale = dpuGetOutputTensorScale(taskInception_V1, OUTPUT_NODE);
    float *softmax = new float[size];

    for (auto &image_name : images) {
        cout << "\nLoad image : " << image_name << endl;
        /* Load image and Set image into DPU Task for Inception_V1 */
        Mat image = imread(baseImagePath + image_name);
        dpuSetInputImage2(taskInception_V1, INPUT_NODE, image);

        /* Run Inception_V1 Task */
        cout << "\nRun DPU Task for Inception_V1 ..." << endl;
        dpuRunTask(taskInception_V1);

        /* Get DPU execution time (in us) of DPU Task */
        long long timeProf = dpuGetTaskProfile(taskInception_V1);
        cout << "  DPU Task Execution time: " << (timeProf * 1.0f) << "us\n";
        float prof = (INCEPTIONV1_WORKLOAD / timeProf) * 1000000.0f;
        cout << "  DPU Task Performance: " << prof << "GOPS\n";

        /* Calculate softmax on dpu and display TOP-5 classification results */
        dpuRunSoftmax(outAddr, softmax, channel, size/channel, scale);
	TopK(softmax, channel, 5, kinds);

        /* Display the image */
        cv::imshow("Classification of inception_v1", image);
        cv::waitKey(1);
    }

    delete[] softmax;
}

/**
 * @brief Entry for runing Inception_V1 neural network

 *
 */
int main(int argc, char *argv[]) {
    /* DPU Kernels/Tasks for runing Inception_V1 */
    DPUKernel *kernelInception_V1;
    DPUTask *taskInception_V1;

    /* Attach to DPU driver and prepare for runing */
    dpuOpen();

    /* Create DPU Kernels for Inception_V1 */
    kernelInception_V1 = dpuLoadKernel(KRENEL_Inception_V1);

    /* Create DPU Tasks for Inception_V1 */
    taskInception_V1 = dpuCreateTask(kernelInception_V1, 0);

    /* Run Inception_V1 Task */
    runInception_V1(taskInception_V1);

    /* Destroy DPU Tasks & free resources */
    dpuDestroyTask(taskInception_V1);

    /* Destroy DPU Kernels & free resources */
    dpuDestroyKernel(kernelInception_V1);

    /* Dettach from DPU driver & release resources */
    dpuClose();

    return 0;
}
