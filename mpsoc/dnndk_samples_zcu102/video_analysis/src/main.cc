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

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Header file OpenCV for image processing
#include <opencv2/opencv.hpp>

// Header files for Vitis AI advanced APIs
#include <dnndk/dnndk.h>

/* header file for Caffe input images APIs */
#include "dputils.h"

using namespace std;
using namespace cv;

// DPU Kernel name for SSD Convolution layers
#define KERNEL_CONV "ssd"
// DPU node name for input and output
#define CONV_INPUT_NODE "conv1_1"
#define CONV_OUTPUT_NODE_LOC "mbox_loc"
#define CONV_OUTPUT_NODE_CONF "mbox_conf"

// 5.5GOP computation for SSD Convolution layers
const float SSD_WORKLOAD_CONV = 5.5f;

// detection params
const float NMS_THRESHOLD = 0.45;
const float CONF_THRESHOLD = 0.3;
const float CAR_THRESHOLD = 0.6;
const float BICYCLE_THRESHOLD = 0.5;
const float PERSON_THRESHOLD = 0.35;
const int TOP_K = 200;
const int KEEP_TOP_K = 200;

// input video
VideoCapture video;

// flags for each thread
bool is_reading = true;
bool is_running_1 = true;
bool is_running_2 = true;
bool is_running_3 = true;
bool is_running_4 = true;
bool is_running_5 = true;
bool is_running_6 = true;
bool is_displaying = true;

// comparison algorithm for priority_queue
class Compare {
    public:
    bool operator()(const pair<int, Mat> &n1, const pair<int, Mat> &n2) const {
        return n1.first > n2.first;
    }
};

queue<pair<int, Mat>> read_queue;                                               // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare> display_queue;  // display queue
mutex mtx_read_queue;                                                           // mutex of read queue
mutex mtx_display_queue;                                                        // mutex of display queue
int read_index = 0;                                                             // frame index of input video
int display_index = 0;                                                          // frame index to display

/**
 * @brief Create prior boxes for feature maps of one scale
 *
 * @note Each prior box is represented as a vector: c-x, c-y, width, height,
 *       variances.
 *
 * @param image_width - input width of CONV Task
 * @param image_height - input height of CONV Task
 * @param layer_width - width of a feature map
 * @param layer_height - height of a feature map
 * @param variances
 * @param min_sizes
 * @param max_sizes
 * @param aspect_ratios
 * @param offset
 * @param step_width
 * @param step_height
 *
 * @return the created priors
 */
vector<vector<float>> CreateOneScalePriors(int image_width, int image_height, int layer_width,
                                           int layer_height, vector<float> &variances,
                                           vector<float> &min_sizes, vector<float> &max_sizes,
                                           vector<float> &aspect_ratios, float offset,
                                           float step_width, float step_height) {
    // Generate boxes' dimensions
    vector<pair<float, float>> boxes_dims;
    for (size_t i = 0; i < min_sizes.size(); ++i) {
        // first prior: aspect_ratio = 1, size = min_size
        boxes_dims.emplace_back(min_sizes[i], min_sizes[i]);
        // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
        if (!max_sizes.empty()) {
            boxes_dims.emplace_back(sqrt(min_sizes[i] * max_sizes[i]),
                                    sqrt(min_sizes[i] * max_sizes[i]));
        }
        // rest of priors
        for (auto ar : aspect_ratios) {
            float w = min_sizes[i] * sqrt(ar);
            float h = min_sizes[i] / sqrt(ar);
            boxes_dims.emplace_back(w, h);
            boxes_dims.emplace_back(h, w);
        }
    }

    vector<vector<float>> priors;
    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            float center_x = (w + offset) * step_width;
            float center_y = (h + offset) * step_height;
            for (auto &dims : boxes_dims) {
                vector<float> box(8);
                // c-x, c-y, width, height
                box[0] = center_x / image_width;
                box[1] = center_y / image_height;
                box[2] = dims.first / image_width;
                box[3] = dims.second / image_height;
                // variances
                for (int i = 4; i < 8; ++i) {
                    box[i] = variances[i - 4];
                }
                priors.emplace_back(box);
            }
        }
    }

    return priors;
}

/**
 * @brief Create prior boxes for feature maps of all scales
 *
 * @note Each prior box is represented as a vector: c-x, c-y, width, height,
 *       variences.
 *
 * @param none
 *
 * @return all prior boxes arranged from large scales to small scales
 */
vector<vector<float>> CreatePriors() {
    int image_width = 480, image_height = 360;
    int layer_width, layer_height;
    vector<float> variances = {0.1, 0.1, 0.2, 0.2};
    vector<float> min_sizes;
    vector<float> max_sizes;
    vector<float> aspect_ratios;
    float offset = 0.5;
    float step_width, step_height;

    vector<vector<vector<float>>> prior_boxes;
    // conv4_3_norm_mbox_priorbox
    layer_width = 60;
    layer_height = 45;
    min_sizes = {15.0, 30.0};
    max_sizes = {33.0, 66.0};
    aspect_ratios = {2};
    step_width = step_height = 8;
    prior_boxes.emplace_back(CreateOneScalePriors(image_width, image_height, layer_width,
                                                  layer_height, variances, min_sizes, max_sizes,
                                                  aspect_ratios, offset, step_width, step_height));
    // fc7_mbox_priorbox
    layer_width = 30;
    layer_height = 23;
    min_sizes = {66.0};
    max_sizes = {127.0};
    aspect_ratios = {2, 3};
    step_width = step_height = 16;
    prior_boxes.emplace_back(CreateOneScalePriors(image_width, image_height, layer_width,
                                                  layer_height, variances, min_sizes, max_sizes,
                                                  aspect_ratios, offset, step_width, step_height));
    // conv6_2_mbox_priorbox
    layer_width = 15;
    layer_height = 12;
    min_sizes = {127.0};
    max_sizes = {188.0};
    aspect_ratios = {2, 3};
    step_width = step_height = 32;
    prior_boxes.emplace_back(CreateOneScalePriors(image_width, image_height, layer_width,
                                                  layer_height, variances, min_sizes, max_sizes,
                                                  aspect_ratios, offset, step_width, step_height));
    // conv7_2_mbox_priorbox
    layer_width = 8;
    layer_height = 6;
    min_sizes = {188.0};
    max_sizes = {249.0};
    aspect_ratios = {2, 3};
    step_width = step_height = 64;
    prior_boxes.emplace_back(CreateOneScalePriors(image_width, image_height, layer_width,
                                                  layer_height, variances, min_sizes, max_sizes,
                                                  aspect_ratios, offset, step_width, step_height));
    // conv8_2_mbox_priorbox
    layer_width = 6;
    layer_height = 4;
    min_sizes = {249.0};
    max_sizes = {310.0};
    aspect_ratios = {2};
    step_width = step_height = 100;
    prior_boxes.emplace_back(CreateOneScalePriors(image_width, image_height, layer_width,
                                                  layer_height, variances, min_sizes, max_sizes,
                                                  aspect_ratios, offset, step_width, step_height));
    // conv9_2_mbox_priorbox
    layer_width = 4;
    layer_height = 2;
    min_sizes = {310.0};
    max_sizes = {372.0};
    aspect_ratios = {2};
    step_width = step_height = 300;
    prior_boxes.emplace_back(CreateOneScalePriors(image_width, image_height, layer_width,
                                                  layer_height, variances, min_sizes, max_sizes,
                                                  aspect_ratios, offset, step_width, step_height));

    // prior boxes
    vector<vector<float>> priors;
    int num_priors = 0;
    for (auto &p : prior_boxes) {
        num_priors += p.size();
    }
    priors.clear();
    priors.reserve(num_priors);
    for (size_t i = 0; i < prior_boxes.size(); ++i) {
        priors.insert(priors.end(), prior_boxes[i].begin(), prior_boxes[i].end());
    }

    return priors;
}

/**
 * @brief Calculate overlap ratio of two boxes
 *
 * @note Each box is represented as a vector: left-x, top-y, right-x, bottom-y,
 *       label, confidence.
 *
 * @param box1 - reference to a box
 * @param box2 - reference to another box
 *
 * @return ratio of intersection area to union area
 */
float IOU(vector<float> &box1, vector<float> &box2) {
    float left, top, right, bottom;
    float intersection_width, intersection_height, intersection_area, union_area;

    // box type: left-x, top-y, right-x, bottom-y, label, confidence
    left = max(box1[0], box2[0]);
    top = max(box1[1], box2[1]);
    right = min(box1[2], box2[2]);
    bottom = min(box1[3], box2[3]);

    intersection_width = right - left;
    intersection_height = bottom - top;
    if (intersection_width <= 0 || intersection_height <= 0) {
        return 0;
    }

    intersection_area = intersection_width * intersection_height;
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) +
                 (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection_area;

    return intersection_area / union_area;
}

/**
 * @brief Non-Maximum Supression
 *
 * @note Each box is represented as a vector: left-x, top-y, right-x, bottom-y,
 *       label, confidence.
 *
 * @param boxes - reference to vector of all boxes
 * @param nms_threshold - maximum overlap ratio of two boxes
 *
 * @return vector of kept boxes
 */
vector<vector<float>> NMS(vector<vector<float>> &boxes, float nms_threshold) {
    vector<vector<float>> kept_boxes;
    vector<pair<int, float>> sorted_boxes(boxes.size());
    vector<bool> box_processed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); i++) {
        sorted_boxes[i].first = i;
        sorted_boxes[i].second = boxes[i][5];
    }
    sort(sorted_boxes.begin(), sorted_boxes.end(),
         [](const pair<int, float> &ls, const pair<int, float> &rs) {
             return ls.second > rs.second;
         });

    for (size_t pair_i = 0; pair_i < boxes.size(); pair_i++) {
        size_t i = sorted_boxes[pair_i].first;
        if (box_processed[i]) {
            continue;
        }
        kept_boxes.emplace_back(boxes[i]);
        for (size_t pair_j = pair_i + 1; pair_j < boxes.size(); pair_j++) {
            size_t j = sorted_boxes[pair_j].first;
            if (box_processed[j]) {
                continue;
            }
            if (IOU(boxes[i], boxes[j]) >= nms_threshold) {
                box_processed[j] = true;
            }
        }
    }

    return kept_boxes;
}

/**
 * @brief Get final boxes according to location, confidence and prior boxes
 *
 * @note Each box is represented as a vector: left-x, top-y, right-x, bottom-y,
 *       label, confidence.
 *       Location, confidence and prior boxes are arranged in the same order to
 *       match correspondingly.
 *
 * @param loc - pointer to localization result of all boxes
 * @param loc_scale - scale to miltiply to transform loc from int8_t to float
 * @param conf_softmax - pointer to softmax result of conf
 * @param priors - reference to vector of prior boxes created in initialization
 * @param top_k - maximum number of boxes(with large confidence) to keep
 * @param nms_threshold - maximum overlap ratio of two boxes
 * @param conf_threshold - minimum confidence of kept boxes
 *
 * @return vector of final boxes
 */
vector<vector<float>> Detect(int8_t *loc, float loc_scale, float *conf_softmax,
                             vector<vector<float>> &priors, size_t top_k, float nms_threshold,
                             float conf_threshold) {
    vector<vector<float>> results;
    vector<vector<vector<float>>> detections(4);

    for (size_t i = 0; i < priors.size(); ++i) {
        vector<float> box(6);
        // Compute c-x, c-y, width, height
        float center_x = priors[i][2] * loc[i * 4] * loc_scale * priors[i][4] + priors[i][0];
        float center_y = priors[i][3] * loc[i * 4 + 1] * loc_scale * priors[i][5] + priors[i][1];
        float width = priors[i][2] * exp(loc[i * 4 + 2] * loc_scale * priors[i][6]);
        float height = priors[i][3] * exp(loc[i * 4 + 3] * loc_scale * priors[i][7]);
        // Transform box to left-x, top-y, right-x, bottom-y, label, confidence
        box[0] = center_x - width / 2.0;   // left-x
        box[1] = center_y - height / 2.0;  // top-y
        box[2] = center_x + width / 2.0;   // right-x
        box[3] = center_y + height / 2.0;  // bottom-y
        for (int j = 0; j < 4; ++j) {
            box[j] = min(max(box[j], 0.f), 1.f);
        }
        // label and confidence
        for (int label_index = 0; label_index < 4; ++label_index) {
            float confidence = conf_softmax[i * 4 + label_index];
            if (confidence > conf_threshold) {
                box[4] = label_index;
                box[5] = confidence;
                detections[label_index].emplace_back(box);
            }
        }
    }

    // Apply NMS and get top_k boxes for each class individually
    for (size_t i = 1; i < detections.size(); ++i) {
        vector<vector<float>> one_class_nms = NMS(detections[i], nms_threshold);
        if (top_k > one_class_nms.size()) {
            top_k = one_class_nms.size();
        }
        for (size_t j = 0; j < top_k; ++j) {
            results.emplace_back(one_class_nms[j]);
        }
    }

    // Keep keep_top_k boxes per image
    sort(results.begin(), results.end(),
         [](const vector<float> &ls, const vector<float> &rs) { return ls[5] > rs[5]; });
    if (results.size() > KEEP_TOP_K) {
        results.resize(KEEP_TOP_K);
    }

    return results;
}

/**
 * @brief Run DPU and ARM Tasks for SSD, and put image into display queue
 *
 * @param task_conv - pointer to SSD CONV Task
 * @param is_running - status flag of RunSSD thread
 *
 * @return none
 */
void RunSSD(DPUTask *task_conv, bool &is_running) {
    // Initializations
    int8_t *loc = (int8_t *)dpuGetOutputTensorAddress(task_conv, CONV_OUTPUT_NODE_LOC);
    int8_t *conf = (int8_t *)dpuGetOutputTensorAddress(task_conv, CONV_OUTPUT_NODE_CONF);
    float loc_scale = dpuGetOutputTensorScale(task_conv, CONV_OUTPUT_NODE_LOC);
    float conf_scale = dpuGetOutputTensorScale(task_conv, CONV_OUTPUT_NODE_CONF);
    int size = dpuGetOutputTensorSize(task_conv, CONV_OUTPUT_NODE_CONF);
    vector<string> label = {"background", "car", "bicycle", "person"};
    vector<vector<float>> priors = CreatePriors();

    float *conf_softmax = new float[size];

    // Run detection for images in read queue
    while (is_running) {
        // Get an image from read queue
        int index;
        Mat img;
        mtx_read_queue.lock();
        if (read_queue.empty()) {
            mtx_read_queue.unlock();
            if (is_reading) {
                continue;
            } else {
                is_running = false;
                break;
            }
        } else {
            index = read_queue.front().first;
            img = read_queue.front().second;
            read_queue.pop();
            mtx_read_queue.unlock();
        }

        // Set image into CONV Task with mean value
        dpuSetInputImage2(task_conv, (char *)CONV_INPUT_NODE, img);

        // Run CONV Task on DPU
        dpuRunTask(task_conv);

        // Show DPU performance every 50 frames
        if (display_index % 50 == 0) {
            cout << "Run SSD CONV ..." << endl;
            // Get DPU execution time (in us) of CONV Task
            uint64_t timeProf = dpuGetTaskProfile(task_conv);
            cout << "  DPU CONV Execution time: " << timeProf << "us" << endl;
            // Get DPU performance for CONV Task
            cout << "  DPU CONV Performance: " << (SSD_WORKLOAD_CONV * 1.0 / timeProf) * 1000000.0f
                 << "GOPS" << endl;
        }

        // Run Softmax 
        dpuRunSoftmax(conf,conf_softmax,4,size/4,conf_scale);
        
        // Post-process
        vector<vector<float>> results =
            Detect(loc, loc_scale, conf_softmax, priors, TOP_K, NMS_THRESHOLD, CONF_THRESHOLD);

        // Modify image to display
        for (size_t i = 0; i < results.size(); ++i) {
            int label_index = results[i][4];
            float confidence = results[i][5];
            float x_min = results[i][0] * img.cols;
            float y_min = results[i][1] * img.rows;
            float x_max = results[i][2] * img.cols;
            float y_max = results[i][3] * img.rows;
            if (label_index == 1 && confidence >= CAR_THRESHOLD) {  // car
                rectangle(img, cvPoint(x_min, y_min), cvPoint(x_max, y_max), Scalar(0, 0, 255), 1,
                          1, 0); 
            } else if (label_index == 2 && confidence >= BICYCLE_THRESHOLD) {  // bicycle
                rectangle(img, cvPoint(x_min, y_min), cvPoint(x_max, y_max), Scalar(0, 255, 0), 1,
                          1, 0); 
            } else if (label_index == 3 && confidence >= PERSON_THRESHOLD) {  // person
                rectangle(img, cvPoint(x_min, y_min), cvPoint(x_max, y_max), Scalar(255, 0, 0), 1,
                          1, 0); 
            } else {  // background
            }
        }

        // Put image into display queue
        mtx_display_queue.lock();
        display_queue.push(make_pair(index, img));
        mtx_display_queue.unlock();
    }

    delete[] conf_softmax;
}

/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(bool &is_reading) {
    while (is_reading) {
        Mat img;
        if (read_queue.size() < 30) {
            if (!video.read(img)) {
                cout << "Video end." << endl;
                is_reading = false;
                break;
            }
            mtx_read_queue.lock();
            read_queue.push(make_pair(read_index++, img));
            mtx_read_queue.unlock();
        } else {
            usleep(20);
        }
    }
}

/**
 * @brief Display frames in display queue
 *
 * @param is_displaying - status flag of Display thread
 *
 * @return none
 */
void Display(bool &is_displaying) {
    Mat image(360, 480, CV_8UC3);
    imshow("Video Analysis @Xilinx DPU", image);
    //cvMoveWindow("Video A @Xilinx DPU", 700, 200);
    while (is_displaying) {
        mtx_display_queue.lock();
        if (display_queue.empty()) {
            if (is_running_1 || is_running_2 || is_running_3 || is_running_4 || is_running_5 || is_running_6) {
                mtx_display_queue.unlock();
                usleep(20);
            } else {
                is_displaying = false;
                break;
            }
        } else if (display_index == display_queue.top().first) {
            // Display image
            imshow("Video Analysis @Xilinx DPU", display_queue.top().second);
            display_index++;
            display_queue.pop();
            mtx_display_queue.unlock();
            if (waitKey(1) == 'q') {
                is_reading = false;
                is_running_1 = false;
                is_running_2 = false;
                is_running_3 = false;
                is_running_4 = false;
                is_running_5 = false;
                is_running_6 = false;
                is_displaying = false;
                break;
            }
        } else {
            mtx_display_queue.unlock();
        }
    }
}

/**
 * @brief Entry for runing SSD neural network 
 *
 * @arg file_name[string] - path to file for detection
 * 
 */
int main(int argc, char **argv) {
    // Check args
    if (argc != 2) {
        cout << "Usage of video analysis demo: ./videoAnalysis file_name[string]" << endl;
        cout << "\tfile_name: path to your file for detection" << endl;
        return -1;
    }

    // DPU Kernels/Tasks for runing SSD
    DPUKernel *kernel_conv;
    DPUTask *task_conv_1, *task_conv_2, *task_conv_3, *task_conv_4, *task_conv_5, *task_conv_6;

    // Attach to DPU driver and prepare for runing
    dpuOpen();

    // Create DPU Kernels and Tasks for CONV Nodes in SSD
    kernel_conv = dpuLoadKernel(KERNEL_CONV);
    task_conv_1 = dpuCreateTask(kernel_conv, 0);
    task_conv_2 = dpuCreateTask(kernel_conv, 0);
    task_conv_3 = dpuCreateTask(kernel_conv, 0);
    task_conv_4 = dpuCreateTask(kernel_conv, 0);
    task_conv_5 = dpuCreateTask(kernel_conv, 0);
    task_conv_6 = dpuCreateTask(kernel_conv, 0);

    // Initializations
    string file_name = argv[1];
    cout << "Detect video: " << file_name << endl;
    video.open(file_name);
    if (!video.isOpened()) {
        cout << "Failed to open video: " << file_name;
        return -1;
    }

    // Run tasks for SSD
    array<thread, 8> threads = {
        thread(Read, ref(is_reading)),
        thread(RunSSD, task_conv_1, ref(is_running_1)),
        thread(RunSSD, task_conv_2, ref(is_running_2)),
        thread(RunSSD, task_conv_3, ref(is_running_3)),
        thread(RunSSD, task_conv_4, ref(is_running_4)),
        thread(RunSSD, task_conv_5, ref(is_running_5)),
        thread(RunSSD, task_conv_6, ref(is_running_6)),
        thread(Display, ref(is_displaying))};
    for (int i = 0; i < 8; ++i) {
        threads[i].join();
    }

    // Destroy DPU Tasks and Kernels and free resources
    dpuDestroyTask(task_conv_1);
    dpuDestroyTask(task_conv_2);
    dpuDestroyTask(task_conv_3);
    dpuDestroyTask(task_conv_4);
    dpuDestroyTask(task_conv_5);
    dpuDestroyTask(task_conv_6);
    dpuDestroyKernel(kernel_conv);

    // Detach from DPU driver and release resources
    dpuClose();

    video.release();
    return 0;
}
