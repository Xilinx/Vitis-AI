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

#include "pose.h"  
#include "dputils.h"

/**
 * Draw line on an image
 */
void drawline(Mat& img,
    Point2f point1, Point2f point2,
    Scalar colour, int thickness,
    float scale_w, float scale_h) {
    if ((point1.x > scale_w || point1.y > scale_h) && (point2.x > scale_w || point2.y > scale_h))
    {
        line(img, point1, point2, colour, thickness);
    }
}

/**
 * Draw lines on the image 
 */
void draw_img(Mat& img, vector<float>& results, float scale_w, float scale_h) {
    float mark = 5.f;
    float mark_w = mark * scale_w;
    float mark_h = mark * scale_h;
    vector<Point2f> pois(14);

    for (size_t i = 0; i < pois.size(); ++i) {
        pois[i].x = results[i * 2] * scale_w;
        pois[i].y = results[i * 2 + 1] * scale_h;
    }

    for (size_t i = 0; i < pois.size(); ++i) {
        circle(img, pois[i], 3, Scalar::all(255));
    }

    drawline(img, pois[0], pois[1], Scalar(255, 0, 0), 2, mark_w, mark_h);
    drawline(img, pois[1], pois[2], Scalar(255, 0, 0), 2, mark_w, mark_h);
    drawline(img, pois[6], pois[7], Scalar(255, 0, 0), 2, mark_w, mark_h);
    drawline(img, pois[7], pois[8], Scalar(255, 0, 0), 2, mark_w, mark_h);
    drawline(img, pois[3], pois[4], Scalar(0, 0, 255), 2, mark_w, mark_h);
    drawline(img, pois[4], pois[5], Scalar(0, 0, 255), 2, mark_w, mark_h);
    drawline(img, pois[9], pois[10], Scalar(0, 0, 255), 2, mark_w, mark_h);
    drawline(img, pois[10], pois[11], Scalar(0, 0, 255), 2, mark_w, mark_h);
    drawline(img, pois[12], pois[13], Scalar(0, 255, 255), 2, mark_w, mark_h);
    drawline(img, pois[0], pois[13], Scalar(0, 255, 255), 2, mark_w, mark_h);
    drawline(img, pois[3], pois[13], Scalar(0, 255, 255), 2, mark_w, mark_h);
    drawline(img, pois[0], pois[6], Scalar(0, 255, 255), 2, mark_w, mark_h);
    drawline(img, pois[3], pois[9], Scalar(0, 255, 255), 2, mark_w, mark_h);
    drawline(img, pois[6], pois[9], Scalar(0, 255, 255), 2, mark_w, mark_h);
}

/**
 * convert output data format
 */
void dpuOutputIn2F32(DPUTask* task, const char* nodeName, float* buffer, int size) {
    int8_t* outputAddr = dpuGetOutputTensorAddress(task, nodeName);
    float scale = dpuGetOutputTensorScale(task, nodeName);

    for (int idx = 0; idx < size; idx++) { 
        buffer[idx] = outputAddr[idx] * scale;
    }
}

/**
 * do average pooling calculation
 */
void CPUCalcAvgPool(DPUTask* conv, DPUTask* fc) {
    assert(conv && fc);
    DPUTensor* outTensor = dpuGetOutputTensor(conv, PT_CONV_OUTPUT_NODE);
    int outHeight = dpuGetTensorHeight(outTensor);
    int outWidth = dpuGetTensorWidth(outTensor);
    int outChannel = dpuGetTensorChannel(outTensor);
    int tensorSize = dpuGetTensorSize(outTensor);

    float* outBuffer = new float[tensorSize]; 
    dpuGetOutputTensorInHWCFP32(conv, PT_CONV_OUTPUT_NODE, outBuffer, tensorSize); 

    int8_t* fcInput = dpuGetInputTensorAddress(fc, PT_FC_NODE);
    float scaleFC = dpuGetInputTensorScale(fc, PT_FC_NODE);
    int length = outHeight * outWidth;
    float avg = static_cast<float>(length);

    for (int i = 0; i < outChannel; i++) {
        float sum = 0.0f;
        for (int j = 0; j < length; j++) {
            sum += outBuffer[outChannel * j + i];
        } 
        fcInput[i] = static_cast<int8_t>(sum / avg * scaleFC);
    } 

    delete[] outBuffer;
}
 
/**
 * construction  of GestureDetect
 *      initialize the DPU Kernels
 */
GestureDetect::GestureDetect() {
}

/**
 * destroy the DPU Kernels and Tasks
 */
GestureDetect::~GestureDetect() {
}

/**
 * @brief Init - initialize the 14pt model
 */
void GestureDetect::Init() {
    kernel_conv_PT = dpuLoadKernel(PT_KRENEL_CONV);
    kernel_fc_PT = dpuLoadKernel(PT_KRENEL_FC);

    task_conv_PT = dpuCreateTask(kernel_conv_PT, 0);
    dpuSetTaskPriority(task_conv_PT, 2);
    task_fc_PT = dpuCreateTask(kernel_fc_PT, 0);
    dpuSetTaskPriority(task_fc_PT, 1);
}

/**
 * @brief Finalize - release resource
 */
void GestureDetect::Finalize() {
    if(task_conv_PT) {
        dpuDestroyTask(task_conv_PT);
    }

    if(kernel_conv_PT) {
        dpuDestroyKernel(kernel_conv_PT);
    }

    if(task_fc_PT) {
        dpuDestroyTask(task_fc_PT);
    }

    if(kernel_fc_PT) {
        dpuDestroyKernel(kernel_fc_PT);
    }
}

/**
 *  @brief Run - run detection algorithm 
 */
void GestureDetect::Run(cv::Mat& img) { 
    vector<float> results(28);
    float mean[3] = {104, 117, 123};
    int width = dpuGetInputTensorWidth(task_conv_PT, PT_CONV_INPUT_NODE);
    int height = dpuGetInputTensorHeight(task_conv_PT, PT_CONV_INPUT_NODE); 

    dpuSetInputImage(task_conv_PT,PT_CONV_INPUT_NODE,img,mean);

    dpuRunTask(task_conv_PT);
    CPUCalcAvgPool(task_conv_PT, task_fc_PT);
    dpuRunTask(task_fc_PT);

    int channel = dpuGetOutputTensorChannel(task_fc_PT, PT_FC_NODE);

    dpuOutputIn2F32(task_fc_PT, PT_FC_NODE, results.data(), channel);

    float scale_w = (float)img.cols / (float)width;
    float scale_h = (float)img.rows / (float)height;

    draw_img(img, results, scale_w, scale_h);
}


