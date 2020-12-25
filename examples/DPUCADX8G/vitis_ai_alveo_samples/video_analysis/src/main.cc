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
//#include "common.h"
// Header files for APIs
#include <dpu/dpu_runner.hpp>

using namespace std;
using namespace cv;
using namespace vitis;
using namespace ai;

// 5.5GOP computation for SSD Convolution layers
const float SSD_WORKLOAD_CONV = 5.5f;

// detection params
const float NMS_THRESHOLD = 0.35;
const float CONF_THRESHOLD = 0.3;
const float CAR_THRESHOLD = 0.6;
const float BICYCLE_THRESHOLD = 0.5;
const float PERSON_THRESHOLD = 0.35;
const int TOP_K = 200;
const int KEEP_TOP_K = 200;
const int TNUM = 1;

// input video
VideoCapture video;

// flags for each thread
bool is_reading = true;
array<bool, TNUM> is_running;
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

//GraphInfo shapes;

/**
 *  @brief calculate softmax
 *  @param data - pointer to input buffer
 *  @param size - size of input buffer
 *  @param result - calculation result
 *  
 *  @return none
 */
void CPUCalcSoftmax(const float *data, size_t size, float *result) {
    assert(data && result);
    double sum = 0.0f;

    for (size_t i = 0; i < size; i++) {
        result[i] = exp(data[i]);
        sum += result[i];
    }

    for (size_t i = 0; i < size; i++) {
        result[i] /= sum;
    }
}


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
vector<vector<float>> Detect(float *loc, float loc_scale, float *conf_softmax,
                             vector<vector<float>> &priors, size_t top_k, float nms_threshold,
                             float conf_threshold) {
    vector<vector<float>> results;
    vector<vector<vector<float>>> detections(4);
    //size_t max_top_k = top_k;

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
        //top_k = max_top_k;
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

void PermFlatConcat(vector<float *> &outdata, float * loc, float * conf, vector<vitis::ai::Tensor*> &outputTensors) {
    float * ploc = loc;
    float * pconf = conf;

    for (int i = 0; i < outputTensors.size(); i++) {
        int c = outputTensors[i]->get_dim_size(1);
        int h = outputTensors[i]->get_dim_size(2);
        int w = outputTensors[i]->get_dim_size(3);
        for (int hh=0;hh<h;hh++)
            for (int ww=0;ww<w;ww++)
                for (int cc=0;cc<c;cc++) {
                    if (i % 2 ==0) {
                        *ploc = outdata[i][cc*h*w+hh*w+ww];
                        ploc++;
                    } else {
                        *pconf = outdata[i][cc*h*w+hh*w+ww];
                        pconf++;
                    }
                }
    }
}


/**
 * @brief Run DPU and ARM Tasks for SSD, and put image into display queue
 *
 * @param task_conv - pointer to SSD CONV Task
 * @param is_running - status flag of RunSSD thread
 *
 * @return none
 */
void RunSSD(vitis::ai::DpuRunner *runner, bool &is_running) {
    float loc_scale = 1.0;
    // get out tensors and shapes
    auto outputTensors = runner->get_output_tensors();
    auto inputTensors = runner->get_input_tensors();
	
	int inHeight;
    int inWidth;
	int inChannel;
    int inSize = inputTensors[0]->get_element_num() 
      / inputTensors[0]->get_dim_size(0);
    int *outSize = new int[outputTensors.size()];
    int size[2];
    for (int i = 0; i < outputTensors.size(); i++) {
        outSize[i] = outputTensors[i]->get_element_num()
            / outputTensors[i]->get_dim_size(0);
        size[i%2] += outSize[i];
    }
	
	if (runner->get_tensor_format() == DpuRunner::TensorFormat::NCHW) {
      inHeight = inputTensors[0]->get_dim_size(2);
      inWidth = inputTensors[0]->get_dim_size(3);
	  inChannel = inputTensors[0]->get_dim_size(1);
    } else {
      inHeight = inputTensors[0]->get_dim_size(1);
      inWidth = inputTensors[0]->get_dim_size(2);
	  inChannel=inputTensors[0]->get_dim_size(3);
    }
	
    vector<float*> outdata;
    for (int i = 0; i < outputTensors.size(); i++) {
        outdata.push_back(new float[outSize[i]]);
    }
    float *imageInputs = new float[inSize];

    float * loc = 0;
    float * conf = 0;
    if (outputTensors.size() == 2) {
        loc = outdata[0];
        conf = outdata[1];
    } else {
        loc = new float[size[0]];
        conf = new float[size[1]];
    }

    vector<string> label = {"background", "car", "bicycle", "person"};
    vector<vector<float>> priors = CreatePriors();

    float *conf_softmax = new float[size[1]];

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
        // image pre-process
        float mean[3] = {104,117,123};
        Mat image2 = cv::Mat(inHeight,inWidth,CV_8SC3);
        resize(img, image2, Size(inWidth, inHeight), 0, 0, INTER_LINEAR);
	if (runner->get_tensor_format() == vitis::ai::DpuRunner::TensorFormat::NHWC) {
            for (int h =0; h< inHeight;h++)
                for (int w =0; w< inWidth;w++)
                    for (int c =0; c< 3; c++)
                        imageInputs[h*inWidth*3+w*3+c] =image2.at<Vec3b>(h,w)[c]- mean[c];
	} else {
            for (int c =0; c< 3; c++)
                for (int h =0; h< inHeight;h++)
                    for (int w =0; w< inWidth;w++)
                        imageInputs[c*inWidth*inHeight+h*inWidth+w] =image2.at<Vec3b>(h,w)[c]- mean[c];
	}
        // input/output prepare
        std::vector<vitis::ai::CpuFlatTensorBuffer>inputs, outputs;
        inputs.push_back(vitis::ai::CpuFlatTensorBuffer(imageInputs,inputTensors[0]));
        for (int i = 0; i < outputTensors.size(); i++)
            outputs.push_back(vitis::ai::CpuFlatTensorBuffer(outdata[i],outputTensors[i]));

        std::vector<vitis::ai::TensorBuffer*> inputsPtr, outputsPtr;
        inputsPtr.push_back(&inputs[0]);

        for (int i = 0; i < outputTensors.size(); i++)
            outputsPtr.push_back(&outputs[i]);
        // execute 
        auto job_id = runner->execute_async(inputsPtr,outputsPtr);
        runner->wait(job_id.first,-1);
        if (outputTensors.size() > 2) {
            PermFlatConcat(outdata, loc, conf, outputTensors);
        }

        // Run Softmax 
        for (int i = 0; i < size[1]/4; i++) {
        CPUCalcSoftmax(&conf[i*4], 4, &conf_softmax[i*4]);
	}
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
	static int fm_cnt=0;
    Mat image(360, 480, CV_8UC3);
    //imshow("Video Analysis @Xilinx DPU", image);
    while (is_displaying) {
        mtx_display_queue.lock();
        if (display_queue.empty()) {
            if (any_of(is_running.begin(), is_running.end(), [](bool cond){return cond;})) {

                mtx_display_queue.unlock();
                usleep(20);
            } else {
                is_displaying = false;
                break;
            }
        } else if (display_index == display_queue.top().first) {
            // Display image
			Mat frame=display_queue.top().second;
			char img_name[20];
            sprintf(img_name,"output/image%d.jpg",fm_cnt);
            cv::imwrite(img_name,frame);
			fm_cnt++;
			
            //imshow("Video Analysis @Xilinx DPU", display_queue.top().second);
            display_index++;
            display_queue.pop();
            mtx_display_queue.unlock();
            /* if (waitKey(1) == 'q') {
                is_reading = false;
                for(int i = 0; i < TNUM; ++i) {
                    is_running[i] = false;
                }

                is_displaying = false;
                break;
            } */
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
    if (argc != 3) {
        cout << "Usage of video analysis demo: ./videoAnalysis file_name[string] path(for json file)" << endl;
        cout << "\tfile_name: path to your file for detection" << endl;
        return -1;
    }

    // Initializations
    string file_name = argv[1];
    cout << "Detect video: " << file_name << endl;
    video.open(file_name);
    if (!video.isOpened()) {
        cout << "Failed to open video: " << file_name;
        return -1;
    }

    //create runner
    auto runners = vitis::ai::DpuRunner::create_dpu_runner(argv[2]);
    auto runner = runners[0].get();

    // Run tasks for SSD
    vector<thread> threads(TNUM);
    is_running.fill(true);
       for(int i = 0; i < TNUM; ++i) {
        threads[i] = thread(RunSSD, ref(runner), ref(is_running[i]));
    }
    threads.push_back(thread(Read, ref(is_reading)));
    threads.push_back(thread(Display, ref(is_displaying)));

    for (int i = 0; i < 2+TNUM; ++i) {
        threads[i].join();
    }

    video.release();
    return 0;
}
