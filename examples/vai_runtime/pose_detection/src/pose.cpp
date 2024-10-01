/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

/**
 * Draw line on an image
 */
void drawline(Mat& img, Point2f point1, Point2f point2, Scalar colour,
              int thickness, float scale_w, float scale_h) {
  if ((point1.x > scale_w || point1.y > scale_h) &&
      (point2.x > scale_w || point2.y > scale_h)) {
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
void dpuOutputIn2FP32(int8_t* outputAddr, float* buffer, int size,
                      float output_scale) {
  for (int idx = 0; idx < size; idx++) {
    buffer[idx] = outputAddr[idx] * output_scale;
  }
}

/**
 * do average pooling calculation
 */
void CPUCalAvgPool(int8_t* data1, int8_t* data2, int outWidth, int outHeight,
                   int outChannel, float pt_output_scale,
                   float fc_pt_input_scale) {
  int length = outHeight * outWidth;
  for (int i = 0; i < outChannel; i++) {
    float sum = 0.0f;
    for (int j = 0; j < length; j++) {
      sum += data1[outChannel * j + i];
    }
    int temp = (int)((sum / length) * pt_output_scale * fc_pt_input_scale);
    data2[i] = (int8_t)std::min(temp, 127);
  }
}

/**
 * construction  of GestureDetect
 *      initialize the DPU Kernels
 */
GestureDetect::GestureDetect() {}

/**
 * destroy the DPU Kernels and Tasks
 */
GestureDetect::~GestureDetect() {}

static int get_batch_size_of_runner(vart::Runner* r) {
  CHECK(!r->get_input_tensors().empty());
  return r->get_input_tensors()[0]->get_shape().at(0);
}

/**
 * @brief Init - initialize the 14pt model
 */
void GestureDetect::Init(string& path) {
  //  string pose0;
  //  string pose2;
  //  if (path.c_str()[path.length() - 1] == '/') {
  //    pose0 = path + "pose_0" + "/";
  //    pose2 = path + "pose_2" + "/";
  //  } else {
  //    pose0 = path + "/" + "pose_0" + "/";
  //    pose2 = path + "/" + "pose_2" + "/";
  //  }
  graph = xir::Graph::deserialize(path);
  auto subgraph = get_dpu_subgraph(graph.get());
  pt_runner = vart::Runner::create_runner(subgraph[0], "run");
  auto pt_batch_size = get_batch_size_of_runner(pt_runner.get());
  // fc_pt_runner = vart::Runner::create_runner(subgraph[1], "run");
  auto is_same = false;
  int num_of_tries = 0;
  do {
    fc_pt_runner = vart::Runner::create_runner(subgraph[1], "run");
    auto fc_batch_size = get_batch_size_of_runner(fc_pt_runner.get());
    is_same = fc_batch_size == pt_batch_size;
    num_of_tries++;
  } while (!is_same && num_of_tries < 10);
  CHECK_LT(num_of_tries, 10) << "too many tries...";

  GraphInfo shapes;
  GraphInfo fc_shapes;
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(pt_runner.get(), &shapes, 1, 1);
  fc_shapes.inTensorList = fc_inshapes;
  fc_shapes.outTensorList = fc_outshapes;
  getTensorShape(fc_pt_runner.get(), &fc_shapes, 1, 1);
}

/**
 * @brief Finalize - release resource
 */
void GestureDetect::Finalize() {}

/**
 *  @brief Run - run detection algorithm
 */
void GestureDetect::Run(cv::Mat& img) {
  float mean[3] = {104, 117, 123};
  auto inTensors = cloneTensorBuffer(pt_runner->get_input_tensors());
  auto outTensors = cloneTensorBuffer(pt_runner->get_output_tensors());
  auto pt_input_scale = get_input_scale(pt_runner->get_input_tensors()[0]);
  auto pt_output_scale = get_output_scale(pt_runner->get_output_tensors()[0]);
  int batchSize = inTensors[0]->get_shape().at(0);
  int width = inshapes[0].width;
  int height = inshapes[0].height;
  int inSize = inshapes[0].size;
  int outSize = outshapes[0].size;
  int8_t* imageInputs = new int8_t[inSize * batchSize];
  int8_t* results1 = new int8_t[outSize * batchSize];
  Mat img2 = cv::Mat(height, width, CV_8SC3);
  cv::resize(img, img2, Size(width, height), 0, 0, cv::INTER_LINEAR);

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int c = 0; c < 3; c++) {
        imageInputs[h * width * 3 + w * 3 + c] =
            (int8_t)((img2.at<Vec3b>(h, w)[c] - mean[c]) * pt_input_scale);
      }
    }
  }

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs2, outputs2;
  inputs2.push_back(
      std::make_unique<CpuFlatTensorBuffer>(imageInputs, inTensors[0].get()));
  outputs2.push_back(
      std::make_unique<CpuFlatTensorBuffer>(results1, outTensors[0].get()));
  std::vector<vart::TensorBuffer*> inputsPtr2, outputsPtr2;
  inputsPtr2.push_back(inputs2[0].get());
  outputsPtr2.push_back(outputs2[0].get());

  auto job_id = pt_runner->execute_async(inputsPtr2, outputsPtr2);
  pt_runner->wait(job_id.first, -1);

  inTensors = cloneTensorBuffer(fc_pt_runner->get_input_tensors());
  outTensors = cloneTensorBuffer(fc_pt_runner->get_output_tensors());
  auto fc_pt_input_scale =
      get_input_scale(fc_pt_runner->get_input_tensors()[0]);
  auto fc_pt_output_scale =
      get_output_scale(fc_pt_runner->get_output_tensors()[0]);

  inSize = fc_inshapes[0].size;
  outSize = fc_outshapes[0].size;
  int8_t* datain0 = new int8_t[inSize * batchSize];
  int8_t* dataresult = new int8_t[outSize * batchSize];
  CPUCalAvgPool(results1, datain0, outshapes[0].width, outshapes[0].height,
                outshapes[0].channel, pt_output_scale, fc_pt_input_scale);

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  inputs.push_back(
      std::make_unique<CpuFlatTensorBuffer>(datain0, inTensors[0].get()));
  outputs.push_back(
      std::make_unique<CpuFlatTensorBuffer>(dataresult, outTensors[0].get()));
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  inputsPtr.push_back(inputs[0].get());
  outputsPtr.push_back(outputs[0].get());
  auto job = fc_pt_runner->execute_async(inputsPtr, outputsPtr);
  fc_pt_runner->wait(job.first, -1);
  vector<float> results(28);

  dpuOutputIn2FP32(dataresult, results.data(), outSize, fc_pt_output_scale);

  float scale_w = (float)img.cols / (float)width;
  float scale_h = (float)img.rows / (float)height;

  draw_img(img, results, scale_w, scale_h);
  delete[] imageInputs;
  delete[] results1;
  delete[] datain0;
  delete[] dataresult;
}

}  // namespace detect
