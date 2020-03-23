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
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for Vitis AI advanced API */
#include <dnndk/dnndk.h>

/* header file for Caffe input images APIs */
#include "dputils.h"

using namespace std;
using namespace cv;

/* 7.71 GOP MAdds for ResNet50 */
#define RESNET50_WORKLOAD (7.71f)
/* DPU Kernel name for ResNet50 */
#define KRENEL_RESNET50 "resnet50_0"
/* Input Node for Kernel ResNet50 */
#define INPUT_NODE      "conv1"
/* Output Node for Kernel ResNet50 */
#define OUTPUT_NODE     "fc1000"

const string baseImagePath = "../dataset/image500_640_480/";

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
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
                (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
    sort(images.begin(), images.end());
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
        printf("top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
        vkinds[ki.second].c_str());
        q.pop();
    }
}

/**
 * @brief Run DPU Task for ResNet50
 *
 * @param taskResnet50 - pointer to ResNet50 Task
 *
 * @return none
 */
void runResnet50(DPUTask *taskResnet50) {
    assert(taskResnet50);

    /* Mean value for ResNet50 specified in Caffe prototxt */
    vector<string> kinds, images;

    /* Load all image names.*/
    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: No images existing under " << baseImagePath << endl;
        return;
    }

    /* Load all kinds words.*/
    LoadWords(baseImagePath + "words.txt", kinds);
    if (kinds.size() == 0) {
        cerr << "\nError: No words exist in file words.txt." << endl;
        return;
    }

    /* Get the output Tensor for Resnet50 Task  */
    int8_t *outAddr = (int8_t *)dpuGetOutputTensorAddress(taskResnet50, OUTPUT_NODE);
    /* Get size of the output Tensor for Resnet50 Task  */
    int size = dpuGetOutputTensorSize(taskResnet50, OUTPUT_NODE);
    /* Get channel count of the output Tensor for ResNet50 Task  */
    int channel = dpuGetOutputTensorChannel(taskResnet50, OUTPUT_NODE);
    /* Get scale of the output Tensor for Resnet50 Task  */
    float out_scale = dpuGetOutputTensorScale(taskResnet50, OUTPUT_NODE);
    float *softmax = new float[size];

    for (auto &imageName : images) {
        cout << "\nLoad image : " << imageName << endl;
        /* Load image and Set image into DPU Task for ResNet50 */
        Mat image = imread(baseImagePath + imageName);
        dpuSetInputImage2(taskResnet50, INPUT_NODE, image);

        /* Launch RetNet50 Task */
        cout << "\nRun DPU Task for ResNet50 ..." << endl;
        dpuRunTask(taskResnet50);

        /* Get DPU execution time (in us) of DPU Task */
        long long timeProf = dpuGetTaskProfile(taskResnet50);
        cout << "  DPU Task Execution time: " << (timeProf * 1.0f) << "us\n";
        float prof = (RESNET50_WORKLOAD / timeProf) * 1000000.0f;
        cout << "  DPU Task Performance: " << prof << "GOPS\n";

        /* Calculate softmax on DPU and display TOP-5 classification results */
        dpuRunSoftmax(outAddr, softmax, channel, size/channel, out_scale);
        TopK(softmax, channel, 5, kinds);

        /* Display the impage */
        cv::imshow("Classification of ResNet50", image);
        cv::waitKey(1);
    }

    delete[] softmax;
}

/**
 * @brief Entry for runing ResNet50 neural network
 *
 * @note Vitis AI advanced APIs prefixed with "dpu" are used to easily program &
 *       deploy ResNet50 on DPU platform.
 *
 */
int main(void) {
    /* DPU Kernel/Task for running ResNet50 */
    DPUKernel *kernelResnet50;
    DPUTask *taskResnet50;

    /* Attach to DPU driver and prepare for running */
    dpuOpen();

    /* Load DPU Kernel for ResNet50 */
    kernelResnet50 = dpuLoadKernel(KRENEL_RESNET50);

    /* Create DPU Task for ResNet50 */
    taskResnet50 = dpuCreateTask(kernelResnet50, 0);

    /* Run ResNet50 Task */
    runResnet50(taskResnet50);

    /* Destroy DPU Task & free resources */
    dpuDestroyTask(taskResnet50);

    /* Destroy DPU Kernel & free resources */
    dpuDestroyKernel(kernelResnet50);

    /* Dettach from DPU driver & free resources */
    dpuClose();

    return 0;
}
