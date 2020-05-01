/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>

using namespace std;
using namespace cv;

// The parameters of yolov3_voc, each value could be set as actual needs.
// Such format could be refer to the prototxts in /etc/dpu_model_param.d.conf/.
const string yolov3_config = {
    "   name: \"yolov3_voc_416\" \n"
    "   model_type : YOLOv3 \n"
    "   yolo_v3_param { \n"
    "     num_classes: 20 \n"
    "     anchorCnt: 3 \n"
    "     conf_threshold: 0.3 \n"
    "     nms_threshold: 0.45 \n"
    "     layer_name: \"81\" \n"
    "     layer_name: \"93\" \n"
    "     layer_name: \"105\" \n"
    "     biases: 10 \n"
    "     biases: 13 \n"
    "     biases: 16 \n"
    "     biases: 30 \n"
    "     biases: 33 \n"
    "     biases: 23 \n"
    "     biases: 30 \n"
    "     biases: 61 \n"
    "     biases: 62 \n"
    "     biases: 45 \n"
    "     biases: 59 \n"
    "     biases: 119 \n"
    "     biases: 116 \n"
    "     biases: 90 \n"
    "     biases: 156 \n"
    "     biases: 198 \n"
    "     biases: 373 \n"
    "     biases: 326 \n"
    "     test_mAP: false \n"
    "   } \n"};

int main(int argc, char* argv[]) {
  // A kernel name, it should be samed as the dnnc result.
  auto kernel_name = "yolov3_voc";
  // A image file.
  auto image_file_name = argv[1];
  // Create a dpu task object.
  auto task = vitis::ai::DpuTask::create(kernel_name);

  /* Pre-process Part */
  // Read image from a path.
  auto input_image = cv::imread(image_file_name);
  if (input_image.empty()) {
    cerr << "cannot load " << image_file_name << endl;
    abort();
  }
  // Resize it if its size is not match.
  cv::Mat image;
  auto input_tensor = task->getInputTensor(0u);
  CHECK_EQ((int)input_tensor.size(), 1)
      << " the dpu model must have only one input";
  auto width = input_tensor[0].width;
  auto height = input_tensor[0].height;
  auto size = cv::Size(width, height);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  // Set the mean values and scale values.
  task->setMeanScaleBGR({0.0f, 0.0f, 0.0f},
                        {0.00390625f, 0.00390625f, 0.00390625f});
  // Set the input image into dpu.
  task->setImageRGB(image);

  /* DPU Runtime */
  // Run the dpu.
  task->run(0u);

  /* Post-process part */
  // Get output.
  auto output_tensor = task->getOutputTensor(0u);
  // Create a config and set the correlating data to control post-process.
  vitis::ai::proto::DpuModelParam config;
  // Fill all the parameters.
  auto ok =
      google::protobuf::TextFormat::ParseFromString(yolov3_config, &config);
  if (!ok) {
    cerr << "Set parameters failed!" << endl;
    abort();
  }
  // Execute the yolov3 post-processing.
  auto results = vitis::ai::yolov3_post_process(
      input_tensor, output_tensor, config, input_image.cols, input_image.rows);

  /* Print the results */
  // Convert coordinate and draw boxes at origin image.
  for (auto& box : results.bboxes) {
    int label = box.label;
    float xmin = box.x * input_image.cols + 1;
    float ymin = box.y * input_image.rows + 1;
    float xmax = xmin + box.width * input_image.cols;
    float ymax = ymin + box.height * input_image.rows;
    if (xmin < 0.) xmin = 1.;
    if (ymin < 0.) ymin = 1.;
    if (xmax > input_image.cols) xmax = input_image.cols;
    if (ymax > input_image.rows) ymax = input_image.rows;
    float confidence = box.score;

    cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t" << xmax
         << "\t" << ymax << "\t" << confidence << "\n";
    rectangle(input_image, Point(xmin, ymin), Point(xmax, ymax),
              Scalar(0, 255, 0), 1, 1, 0);
  }
  imshow("", input_image);
  waitKey(0);

  return 0;
}
