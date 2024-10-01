/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

// The parameters of yolov3_voc_tf, each value could be set as actual needs.
// Such format could be refer to the prototxts in /etc/dpu_model_param.d.conf/.
const string yolov3_config = {
    "   name: \"yolov3_voc_416x416_tf\"\n"
    "   kernel {\n"
    "      name: \"yolov3_voc_416x416_tf\"\n"
    "      mean: 0.0\n"
    "      mean: 0.0\n"
    "      mean: 0.0\n"
    "      scale: 0.00390625\n"
    "      scale: 0.00390625\n"
    "      scale: 0.00390625\n"
    "   }\n"
    "   model_type : YOLOv3\n"
    "   yolo_v3_param {\n"
    "     num_classes: 20\n"
    "     anchorCnt: 3\n"
    "     layer_name: \"59\"\n"
    "     layer_name: \"67\"\n"
    "     layer_name: \"75\"\n"
    "     conf_threshold: 0.3\n"
    "     nms_threshold: 0.45\n"
    "     biases: 10\n"
    "     biases: 13\n"
    "     biases: 16\n"
    "     biases: 30\n"
    "     biases: 33\n"
    "     biases: 23\n"
    "     biases: 30\n"
    "     biases: 61\n"
    "     biases: 62\n"
    "     biases: 45\n"
    "     biases: 59\n"
    "     biases: 119\n"
    "     biases: 116\n"
    "     biases: 90\n"
    "     biases: 156\n"
    "     biases: 198\n"
    "     biases: 373\n"
    "     biases: 326\n"
    "     test_mAP: false\n"
    "   }\n"
    "   is_tf : true\n"
};

// old model obsoleted.
#if 0
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
#endif

int main(int argc, char* argv[]) {
  // A kernel name, it should be samed as the dnnc result. e.g.
  // /usr/share/vitis_ai_library/models/yolov3_voc/yolov3_voc.elf
  auto kernel_name = argv[1];

  // Read image from a path.
  vector<Mat> imgs;
  vector<string> imgs_names;
  for (int i = 2; i < argc; i++) {
    // image file names.
    auto img = cv::imread(argv[i]);
    if (img.empty()) {
      std::cout << "Cannot load " << argv[i] << std::endl;
      continue;
    }
    imgs.push_back(img);
    imgs_names.push_back(argv[i]);
  }
  if (imgs.empty()) {
    std::cerr << "No image load success!" << std::endl;
    abort();
  }
  // Create a dpu task object.
  auto task = vitis::ai::DpuTask::create(kernel_name);
  if (!task) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
  }  
  auto batch = task->get_input_batch(0, 0);
  // Set the mean values and scale values.
  task->setMeanScaleBGR({0.0f, 0.0f, 0.0f},
                        {0.00390625f, 0.00390625f, 0.00390625f});
  auto input_tensor = task->getInputTensor(0u);
  CHECK_EQ((int)input_tensor.size(), 1)
      << " the dpu model must have only one input";
  auto width = input_tensor[0].width;
  auto height = input_tensor[0].height;
  auto size = cv::Size(width, height);
  // Create a config and set the correlating data to control post-process.
  vitis::ai::proto::DpuModelParam config;
  // Fill all the parameters.
  auto ok =
      google::protobuf::TextFormat::ParseFromString(yolov3_config, &config);
  if (!ok) {
    cerr << "Set parameters failed!" << endl;
    abort();
  }

  vector<Mat> inputs;
  vector<int> input_cols, input_rows;
  for (long unsigned int i = 0, j = -1; i < imgs.size(); i++) {
    /* Pre-process Part */
    // Resize it if its size is not match.
    cv::Mat image;
    input_cols.push_back(imgs[i].cols);
    input_rows.push_back(imgs[i].rows);
    if (size != imgs[i].size()) {
      cv::resize(imgs[i], image, size);
    } else {
      image = imgs[i];
    }
    inputs.push_back(image);
    j++;
    if (j < batch - 1 && i < imgs.size() - 1) {
      continue;
    }

    // Set the input images into dpu.
    task->setImageRGB(inputs);

    /* DPU Runtime */
    // Run the dpu.
    task->run(0u);

    /* Post-process part */
    // Get output.
    auto output_tensor = task->getOutputTensor(0u);
    // Execute the yolov3 post-processing.
    auto results = vitis::ai::yolov3_post_process(
        input_tensor, output_tensor, config, input_cols, input_rows);

    /* Print the results */
    // Convert coordinate and draw boxes at origin image.
    for (int k = 0; k < static_cast<int>(inputs.size()); k++) {
      cout << "batch_index " << k << " "  //
           << "image_name " << imgs_names[i - j + k] << endl;
      for (auto& box : results[k].bboxes) {
        int label = box.label;
        float xmin = box.x * input_cols[k] + 1;
        float ymin = box.y * input_rows[k] + 1;
        float xmax = xmin + box.width * input_cols[k];
        float ymax = ymin + box.height * input_rows[k];
        if (xmin < 0.) xmin = 1.;
        if (ymin < 0.) ymin = 1.;
        if (xmax > input_cols[k]) xmax = input_cols[k];
        if (ymax > input_rows[k]) ymax = input_rows[k];
        float confidence = box.score;

        cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t"
             << xmax << "\t" << ymax << "\t" << confidence << "\n";
        rectangle(imgs[i - j + k], Point(xmin, ymin), Point(xmax, ymax),
                  Scalar(0, 255, 0), 1, 1, 0);
      }
      imwrite(imgs_names[i - j + k] + "_result.jpg", imgs[i - j + k]);
    }
    inputs.clear();
    input_cols.clear();
    input_rows.clear();
    j = -1;
  }
  return 0;
}
