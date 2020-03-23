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

#include <dnndk/n2cube.h>
#include "dputils.h"

using namespace std;
using namespace cv;
#define N2CUBE_SUCCESS 0
#define USE_NEON_OPT
/**
 * @brief Set image into DPU Task's input tensor, multiple IO supported.
 *
 * @note source data must be in in Caffe order: channel, height, width;
 *       source data type must be int8_t;
 *       source data will be converted from Caffe order to DPU order
 *
 * @param task - pointer to DPU task
 * @param nodeName - pointer to DPU Node name
 * @param image - input image in OpenCV Mat format. Single channel and 3- channel input image are supported
 *                Note: Only support CV_8U, please modify the code for other types.
 * @param mean - pointer to mean value array which contains 1 member for single channel input image or 3 members for 3-channel input image
 *               Note: You can get the mean values from the input Caffe prototxt. At present, the format of mean value file is not yet supported.
 * @param scale - scale value of input image
 * @param idx - the index of a single input tensor for the Node, with default value as 0
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuSetInputImageWithScale(DPUTask *task, const char* nodeName, const cv::Mat &image, float *mean, float scale, int idx)
{
    int value;
    int8_t *inputAddr;
    unsigned char *resized_data;
    cv::Mat newImage;
    float scaleFix;
    int height, width, channel;

    height = dpuGetInputTensorHeight(task, nodeName, idx);
    width = dpuGetInputTensorWidth(task, nodeName, idx);
    channel = dpuGetInputTensorChannel(task, nodeName, idx);

    if (height == image.rows && width == image.cols) {
        newImage = image;
    } else {
        newImage = cv::Mat (height, width, CV_8SC3,
                    (void*)dpuGetInputTensorAddress(task, nodeName, idx));
        cv::resize(image, newImage, newImage.size(), 0, 0, cv::INTER_LINEAR);
    }
    resized_data = newImage.data;

    inputAddr = dpuGetInputTensorAddress(task, nodeName, idx);
    scaleFix = dpuGetInputTensorScale(task, nodeName, idx);

    scaleFix = scaleFix*scale;

    if (newImage.channels() == 1) {
        for (int idx_h=0; idx_h<height; idx_h++) {
            for (int idx_w=0; idx_w<width; idx_w++) {
                for (int idx_c=0; idx_c<channel; idx_c++) {
                    value = *(resized_data+idx_h*width*channel+idx_w*channel+idx_c);
                    value = (int)((value - *(mean+idx_c)) * scaleFix);
                    inputAddr[idx_h*newImage.cols+idx_w] = (char)value;
                }
            }
        }
    } else {
#ifdef USE_NEON_OPT
        dpuProcessNormalizion(inputAddr, newImage.data, newImage.rows, newImage.cols, mean, scaleFix, newImage.step1());
#else
        for (int idx_h=0; idx_h<newImage.rows; idx_h++) {
            for (int idx_w=0; idx_w<newImage.cols; idx_w++) {
                for (int idx_c=0; idx_c<3; idx_c++) {
                    value = (int)((newImage.at<Vec3b>(idx_h, idx_w)[idx_c] - mean[idx_c]) * scaleFix);
                    inputAddr[idx_h*newImage.cols*3+idx_w*3+idx_c] = (char)value;
                }
            }
        }
#endif
    }

    return N2CUBE_SUCCESS;
}


/**
 * @brief Set image into DPU Task's input tensor with mean values, multiple IO supported.
 *
 * @note source data must be in in Caffe order: channel, height, width;
 *       source data type must be int8_t;
 *       source data will be converted from Caffe order to DPU order
 *
 * @param task - pointer to DPU task
 * @param nodeName - pointer to DPU Node name
 * @param image - input image in OpenCV Mat format. Single channel and 3- channel input image are supported
 *                Note: Only support CV_8U, please modify the code for other types.
 * @param mean - pointer to mean value array which contains 1 member for single channel input image or 3 members for 3-channel input image
 *               Note: You can get the mean values from the input Caffe prototxt. At present, the format of mean value file is not yet supported
 * @param idx - the index of a single input tensor for the Node, with default value as 0
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuSetInputImage(DPUTask *task, const char* nodeName, const cv::Mat &image, float *mean, int idx)
{

    return dpuSetInputImageWithScale(task, nodeName, image, mean, 1.0f, idx);
}


/**
 * @brief Set image into DPU Task's input tensor without mean values, multiple IO supported.
 *
 * @note source data must be in in Caffe order: channel, height, width;
 *       source data type must be int8_t;
 *       source data will be converted from Caffe order to DPU order
 *
 * @param task - pointer to DPU task
 * @param nodeName - pointer to DPU Node name
 * @param image - input image in OpenCV Mat format. Single channel and 3- channel input image are supported
 *                Note: Only support CV_8U, please modify the code for other types.
 * @param idx - the index of a single input tensor for the Node, with default value as 0
 *
 * @return 0 on success, or negative error ID in case of failure.
 */
int dpuSetInputImage2(DPUTask *task, const char* nodeName, const cv::Mat &image, int idx)
{
    float mean[3];

    dpuGetKernelMean(task,mean,image.channels()); //This API is only available for Caffe model
    return dpuSetInputImageWithScale(task, nodeName, image, mean, 1.0f, idx);
}

