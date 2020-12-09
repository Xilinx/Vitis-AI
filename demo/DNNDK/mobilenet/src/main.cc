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

/* header file for Vitis AI advanced APIs */
#include <dnndk/dnndk.h>

/* header file for Caffe input images APIs */
#include "dputils.h"

using namespace std;
using namespace cv;

/* 0.56 GOP MAdds for MobileNet */
#define MOBILENET_WORKLOAD (0.56f)

/* DPU Kernel name for MobileNet */
#define KRENEL_MOBILENET "mobilenet"
/* Input Node for Kernel MobileNet */
#define CONV_INPUT_NODE "263"
/* Output Node for Kernel MobileNet */
#define OUTPUT_NODE "417"

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
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kind file
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
 * @brief Run DPU Task for MobileNet
 *
 * @param taskMobilenet - pointer to MobileNet Task
 *
 * @return none
 */
void runMobilenet(DPUTask *taskMobilenet) {
    assert(taskMobilenet);

    /* Mean value for MobileNet specified in Caffe prototxt */
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

    /* Get the output Tensor for Mobilenet Task  */
    int8_t *outAddr = (int8_t *)dpuGetOutputTensorAddress(taskMobilenet, OUTPUT_NODE);
    /* Get size of the output Tensor for Mobilenet Task  */
    int size = dpuGetOutputTensorSize(taskMobilenet, OUTPUT_NODE);
    /* Get channel count of the output Tensor for MobileNet Task  */
    int channel = dpuGetOutputTensorChannel(taskMobilenet, OUTPUT_NODE);
    /* Get scale of the output Tensor for Mobilenet Task  */
    float out_scale = dpuGetOutputTensorScale(taskMobilenet, OUTPUT_NODE);

    float *softmax = new float[size];

    for (auto &imageName : images) {
        cout << "\nLoad image : " << imageName << endl;
        /* Load image and Set image into DPU Task for MobileNet */
        Mat image = imread(baseImagePath + imageName);
        vector<float> mean{104, 117, 123};
        float scale = 0.00390625;
        dpuSetInputImageWithScale(taskMobilenet, CONV_INPUT_NODE, image, mean.data(), scale);

        /* Launch RetNet50 Task */
        cout << "\nRun DPU Task for MobileNet ..." << endl;
        dpuRunTask(taskMobilenet);

        /* Get DPU execution time (in us) of DPU Task */
        long long timeProf = dpuGetTaskProfile(taskMobilenet);
        cout << "  DPU Task Execution time: " << (timeProf * 1.0f) << "us\n";
        float prof = (MOBILENET_WORKLOAD / timeProf) * 1000000.0f;
        cout << "  DPU Task Performance: " << prof << "GOPS\n";

        /* Calculate softmax on DPU and display TOP-5 classification results */
        dpuRunSoftmax(outAddr, softmax, channel, size/channel, out_scale);
        TopK(softmax, channel, 5, kinds);

        /* Display the impage */
        cv::imshow("Classification of MobileNet", image);
        cv::waitKey(1);
    }

    delete[] softmax;
}

/**
 * @brief Entry for runing MobileNet neural network
 *
 */
int main(void) {
    /* DPU Kernel/Task for running MobileNet */
    DPUKernel *kernelMobilenet;
    DPUTask *taskMobilenet;

    /* Attach to DPU driver and prepare for running */
    dpuOpen();

    /* Load DPU Kernel for MobileNet */
    kernelMobilenet = dpuLoadKernel(KRENEL_MOBILENET);

    /* Create DPU Task for MobileNet */
    taskMobilenet = dpuCreateTask(kernelMobilenet, 0);

    /* Run MobileNet Task */
    runMobilenet(taskMobilenet);

    /* Destroy DPU Task & free resources */
    dpuDestroyTask(taskMobilenet);

    /* Destroy DPU Kernel & free resources */
    dpuDestroyKernel(kernelMobilenet);

    /* Dettach from DPU driver & free resources */
    dpuClose();

    return 0;
}
