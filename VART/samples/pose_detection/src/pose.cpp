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

namespace detect {
using namespace vitis;
using namespace ai;
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
void dpuOutputIn2FP32(float* outputAddr, float* buffer, int size) {
    for (int idx = 0; idx < size; idx++) {
        buffer[idx] = outputAddr[idx];
    }

}

/**
 * do average pooling calculation
 */
void CPUCalAvgPool(float* data1, float*data2, int outWidth, int outHeight, int outChannel) {
    int length = outHeight * outWidth;
    for (int i = 0; i < outChannel; i++) {
        float sum = 0.0f;
        for (int j = 0; j < length; j++) {
            sum += data1[outChannel * j + i];
        }
        data2[i] = sum / length;
    }

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
void GestureDetect::Init(string& path) {
    string pose0;
    string pose2;
    if (path.c_str()[path.length()-1] == '/') {
        pose0 = path  + "pose_0" + "/";
        pose2 = path  + "pose_2" + "/";
    }
    else {
        pose0 = path  + "/" + "pose_0" + "/";
        pose2 = path  + "/" + "pose_2" + "/";
    }
    pt_runners = vitis::ai::DpuRunner::create_dpu_runner(pose0.c_str());
    fc_pt_runners = vitis::ai::DpuRunner::create_dpu_runner(pose2.c_str());
    pt_runner = pt_runners[0].get();
    fc_pt_runner = fc_pt_runners[0].get();
    GraphInfo shapes;
    GraphInfo fc_shapes;
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes;
    getTensorShape(pt_runner, &shapes,1, 1);
    fc_shapes.inTensorList = fc_inshapes;
    fc_shapes.outTensorList = fc_outshapes;
    getTensorShape(fc_pt_runner, &fc_shapes,1, 1);
}

/**
 * @brief Finalize - release resource
 */
void GestureDetect::Finalize() {
}

/**
 *  @brief Run - run detection algorithm
 */
void GestureDetect::Run(cv::Mat& img) {
    float mean[3] = {104, 117, 123};
    auto inTensors = cloneTensorBuffer(pt_runner->get_input_tensors());
    auto outTensors = cloneTensorBuffer(pt_runner->get_output_tensors());
    int batchSize = inTensors[0]->get_dim_size(0);
    int width = inshapes[0].width;
    int height = inshapes[0].height;
    int inSize = inshapes[0].size;
    int outSize = outshapes[0].size;
    float *imageInputs = new float[inSize * batchSize];
    float *results1 = new float[outSize * batchSize];
    Mat img2 = cv::Mat(height, width, CV_8SC3);
    cv::resize(img,img2, Size(width,height), 0,0, cv::INTER_LINEAR);
    if (pt_runner->get_tensor_format() == vitis::ai::DpuRunner::TensorFormat::NHWC) {
        for (int h =0; h< height;h++)
            for (int w =0; w< width;w++)
                for (int c =0; c< 3; c++)
                    imageInputs[h*width*3+w*3+c] =img2.at<Vec3b>(h,w)[c]- mean[c];
    } else {
        for (int c =0; c< 3; c++)
            for (int h =0; h< height;h++)
                for (int w =0; w< width;w++)
                    imageInputs[c*width*height+h*width+w] =img2.at<Vec3b>(h,w)[c]- mean[c];
    }

    std::vector<vitis::ai::CpuFlatTensorBuffer>inputs2, outputs2;
    inputs2.push_back(vitis::ai::CpuFlatTensorBuffer(imageInputs,inTensors[0].get()));
    outputs2.push_back(vitis::ai::CpuFlatTensorBuffer(results1,outTensors[0].get()));
    std::vector<vitis::ai::TensorBuffer*> inputsPtr2, outputsPtr2;
    inputsPtr2.push_back(&inputs2[0]);
    outputsPtr2.push_back(&outputs2[0]);
    auto job_id = pt_runner->execute_async(inputsPtr2,outputsPtr2);
    pt_runner->wait(job_id.first, -1);


    inTensors = cloneTensorBuffer(fc_pt_runner->get_input_tensors());
    outTensors = cloneTensorBuffer(fc_pt_runner->get_output_tensors());

    inSize = fc_inshapes[0].size;
    outSize = fc_outshapes[0].size;
    int outSize2 = fc_outshapes[1].size;
    float* datain0 = new float[inSize * batchSize];
    float* dataresult = new float[outSize * batchSize];
    CPUCalAvgPool(results1,datain0,outshapes[0].width,outshapes[0].height,outshapes[0].channel);

    std::vector<vitis::ai::CpuFlatTensorBuffer>inputs, outputs;
    inputs.push_back(vitis::ai::CpuFlatTensorBuffer(datain0,inTensors[0].get()));
    outputs.push_back(vitis::ai::CpuFlatTensorBuffer(dataresult,outTensors[0].get()));
    std::vector<vitis::ai::TensorBuffer*> inputsPtr, outputsPtr;
    inputsPtr.push_back(&inputs[0]);
    outputsPtr.push_back(&outputs[0]);
    auto job = fc_pt_runner->execute_async(inputsPtr,outputsPtr);
    fc_pt_runner->wait(job.first,-1);
    vector<float> results(28);

    dpuOutputIn2FP32(dataresult,results.data(),outSize);

    float scale_w = (float)img.cols / (float)width;
    float scale_h = (float)img.rows / (float)height;

    draw_img(img, results, scale_w, scale_h);
    delete[] imageInputs;
    delete[] results1;
    delete[] datain0;
    delete[] dataresult;
}

}
