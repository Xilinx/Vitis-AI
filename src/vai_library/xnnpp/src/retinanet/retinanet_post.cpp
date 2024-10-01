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
#include "./retinanet_post.hpp"

#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>
// #include <vitis/ai/env_config.hpp>
//#include "./prior_boxes.hpp"
#include "./retinanet_detector.hpp"
#include <algorithm>
#include <cmath>

using namespace std;

const size_t CLASS_NUMS = 264;

namespace vitis {
namespace ai {

float BASE_ANCHOR_SIZE[5] = {32, 64, 128, 256, 512};
size_t IMAGE_SIZE[2] = {800, 800};
size_t FEATURE_MAP[5][2] = {
    {100, 100},
    {50, 50},
    {25, 25},
    {13, 13},
    {7, 7}
};

bool comp_level_info(const RetinaNetLevelInfo& a, const RetinaNetLevelInfo& b) {
    return a.score > b.score;
}

RetinaNetPost::~RetinaNetPost() {
};

RetinaNetPost::RetinaNetPost(const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config) :
    conf_threshold_(config.retinanet_param().conf_threshold()),
    nms_threshold_(config.retinanet_param().nms_threshold()),
    top_k_(config.retinanet_param().top_k()),
    box_threshold_(config.retinanet_param().box_threshold()),
    input_tensors_(input_tensors),
    output_tensors_(output_tensors) {

    detector_ = vitis::ai::CreateRetinaNetUniform(nms_threshold_, conf_threshold_);
}

std::vector<RetinaNetBoundingBox> RetinaNetPost::retinanet_post_process_internal_uniform(
    unsigned int batch_id) {

    /*for (size_t i = 0; i < output_tensors_.size(); ++i) {
        std::cout << "tensor id : " << i
            << " name : " << output_tensors_[i].name
            << " size : " << output_tensors_[i].size
            << " height : " << output_tensors_[i].height
            << " width : " << output_tensors_[i].width
            << " batch : " << output_tensors_[i].batch
            << " channel : " << output_tensors_[i].channel
            << std::endl;
    }*/

    // for each leve
    __TIC__(RetinaNet_logits)
    std::vector<RetinaNetLevelInfo> level_infos;
    for (size_t tensor_id = 0; tensor_id < 5; ++tensor_id) {
        size_t height = output_tensors_[tensor_id].height;
        size_t width = output_tensors_[tensor_id].width;
        size_t level_id = tensor_id;

        /*std::string file_name = std::string("data/") + output_tensors_[tensor_id].name + ".bin";
        float* conf_buf = new float[height * width * 9 * CLASS_NUMS];
        FILE* fp = fopen(file_name.c_str(), "rb");
        memset(conf_buf, 0, height * width * 9 * CLASS_NUMS * sizeof(float));
        fread(conf_buf, 1, height * width * 9 * CLASS_NUMS * sizeof(float), fp);
        fclose(fp);*/

        /*do {
            std::cout << "reg output" << std::endl;
            float scale = library::tensor_scale(output_tensors_[tensor_id + 5]);
            for (size_t i = 0; i < 10; ++i) {
                int8_t* buf = (int8_t*)output_tensors_[tensor_id + 5].get_data(0);
                std::cout << scale * buf[i] << " ";
            }
            std::cout << std::endl;
        } while (0);*/
        /*do {
            std::cout << "cls output" << std::endl;
            float scale = library::tensor_scale(output_tensors_[tensor_id]);
            for (size_t i = 0; i < 100; ++i) {
                int8_t* buf = (int8_t*)output_tensors_[tensor_id].get_data(0);
                float x = buf[i] * scale;
                float sigmoid_value = 1 / (1 + expf(-1.0f *  x));
                std::cout << sigmoid_value << "\t";
            }
            std::cout << std::endl;
        } while(0);*/
        std::vector<RetinaNetLevelInfo> temp_level_infos;
        float scale = library::tensor_scale(output_tensors_[tensor_id]);
        __TIC__(RetinaNet_sigmoid)
        float sigmoid_threshold = -1.0 * log(1.0 / conf_threshold_ - 1);
        for (size_t anchor_id = 0; anchor_id < height * width * 9; ++anchor_id) {
            int8_t* label_buf = (int8_t*)output_tensors_[tensor_id].get_data(0) + CLASS_NUMS * anchor_id;
            //float* label_buf = conf_buf + CLASS_NUMS * anchor_id;
            for (size_t c = 0; c < CLASS_NUMS; ++c) {
                float x = label_buf[c] * scale;
                if (x < sigmoid_threshold) {
                    continue;
                }
                //float x = label_buf[c];
                float sigmoid_value = 1 / (1 + expf(-1.0 * x));
                //if (sigmoid_value < conf_threshold_) {
                //    continue;
                //}
                temp_level_infos.push_back(RetinaNetLevelInfo{level_id, anchor_id, sigmoid_value, c});
            }
        }
        __TOC__(RetinaNet_sigmoid)
        std::sort(temp_level_infos.begin(), temp_level_infos.end(), comp_level_info);
        if (temp_level_infos.size() > top_k_) {
            temp_level_infos.resize(top_k_);
        }
        level_infos.insert(level_infos.end(), temp_level_infos.begin(), temp_level_infos.end());
    }
    __TOC__(RetinaNet_logits)
    //std::sort(level_infos.begin(), level_infos.end(), comp_level_info);

    /*std::vector<std::vector<float>> dump_data_bufs;
    for (size_t level_id = 0; level_id < 5; ++level_id) {
        size_t height = output_tensors_[level_id + 5].height;
        size_t width = output_tensors_[level_id + 5].width;
        std::vector<float> level_bufs(height * width * 9 * 4);
        {
            std::string box_file_name = std::string("data/") + output_tensors_[level_id + 5].name + ".bin";
            FILE* fp = fopen(box_file_name.c_str(), "rb");
            fread((char*)level_bufs.data(), 1, height * width * 9 * 4 * sizeof(float), fp);
            fclose(fp);
        }
        dump_data_bufs.push_back(level_bufs);
    }*/

    std::vector<std::vector<float>> boxes;
    std::vector<float> scores;
    std::vector<size_t> labels;

    __TIC__(RetinaNet_regression)
    for (size_t i = 0; i < level_infos.size(); ++i) {

        size_t anchor_id = level_infos[i].anchor_id;
        size_t level_id = level_infos[i].level_id;

        size_t tensor_id = level_id + 5;
        float scale = library::tensor_scale(output_tensors_[tensor_id]);
        int8_t* box_buf = (int8_t*)output_tensors_[tensor_id].get_data(0);
        float conf_box[4] = {
            //dump_data_bufs[level_id][anchor_id * 4 + 0],
            //dump_data_bufs[level_id][anchor_id * 4 + 1],
            //dump_data_bufs[level_id][anchor_id * 4 + 2],
            //dump_data_bufs[level_id][anchor_id * 4 + 3]
            box_buf[anchor_id * 4 + 0] * scale,
            box_buf[anchor_id * 4 + 1] * scale,
            box_buf[anchor_id * 4 + 2] * scale,
            box_buf[anchor_id * 4 + 3] * scale
        };

        float anchor_box[4];
        do {
            size_t stride_x = (anchor_id / 9) % FEATURE_MAP[level_id][1];
            size_t stride_y = (anchor_id / 9) / FEATURE_MAP[level_id][1];
            float stride_width = IMAGE_SIZE[1] * 1.0 / FEATURE_MAP[level_id][1];
            float stride_height = IMAGE_SIZE[0] * 1.0 / FEATURE_MAP[level_id][0];
            float stride[4] = {
                stride_width * stride_x,
                stride_height * stride_y,
                stride_width * stride_x,
                stride_height * stride_y
            };
            size_t ratio_id = (anchor_id % 9) / 3;
            size_t scale_id = (anchor_id % 9) % 3;
            float ratios[3] = {0.5, 1.0, 2.0};
            float scales[3] = {pow(2, 0.0f / 3), pow(2, 1.0f / 3), pow(2, 2.0f/ 3)};
            float base_anchor[4] = {
                -1.0f * BASE_ANCHOR_SIZE[level_id] * scales[scale_id] / pow(ratios[ratio_id], 0.5f) / 2,
                -1.0f * BASE_ANCHOR_SIZE[level_id] * scales[scale_id] * pow(ratios[ratio_id], 0.5f) / 2,
                BASE_ANCHOR_SIZE[level_id] * scales[scale_id] / pow(ratios[ratio_id], 0.5f) / 2,
                BASE_ANCHOR_SIZE[level_id] * scales[scale_id] * pow(ratios[ratio_id], 0.5f) / 2
            };

            anchor_box[0] = stride[0] + base_anchor[0];
            anchor_box[1] = stride[1] + base_anchor[1];
            anchor_box[2] = stride[2] + base_anchor[2];
            anchor_box[3] = stride[3] + base_anchor[3];
        } while (0);

        //
        float pre_box[4];
        do {
            float anchor_width = anchor_box[2] - anchor_box[0];
            float anchor_height = anchor_box[3] - anchor_box[1];
            float ctr_x = anchor_box[0] + anchor_width * 0.5;
            float ctr_y = anchor_box[1] + anchor_height * 0.5;

            float pre_ctr_x = conf_box[0] * anchor_width + ctr_x;
            float pre_ctr_y = conf_box[1] * anchor_height + ctr_y;
            float pre_w = expf(conf_box[2]) * anchor_width;
            float pre_h = expf(conf_box[3]) * anchor_height;

            pre_box[0] = pre_ctr_x - 0.5 * pre_w;
            pre_box[1] = pre_ctr_y - 0.5 * pre_h;
            pre_box[2] = pre_ctr_x + 0.5 * pre_w;
            pre_box[3] = pre_ctr_y + 0.5 * pre_h;

        } while (0);
        std::vector<float> pred_box = {
            std::max(0.0f, std::min(IMAGE_SIZE[1] * 1.0f, pre_box[0])),
            std::max(0.0f, std::min(IMAGE_SIZE[0] * 1.0f, pre_box[1])),
            std::max(0.0f, std::min(IMAGE_SIZE[1] * 1.0f, pre_box[2])),
            std::max(0.0f, std::min(IMAGE_SIZE[0] * 1.0f, pre_box[3]))
        };
        /*std::cout << "box info, "
            << " level_id : " << level_id
            << " anchorid : " << anchor_id
            //<< " anchor_box"
            //<< " " << anchor_box[0]
            //<< " " << anchor_box[1]
            //<< " " << anchor_box[2]
            //<< " " << anchor_box[3]
            //<< " regression"
            //<< " " << conf_box[0]
            //<< " " << conf_box[1]
            //<< " " << conf_box[2]
            //<< " " << conf_box[3]
            << " pred_box "
            << "\t" << (int)pred_box[0]
            << "\t" << (int)pred_box[1]
            << "\t" << (int)pred_box[2]
            << "\t" << (int)pred_box[3]
            //<< " stride_x : " << stride_x
            //<< " stride_y : " << stride_y
            //<< " stride_w : " << stride_width
            //<< " stride_h : " << stride_height
            //<< " IMAGE_WIDTH : " << IMAGE_SIZE[1]
            //<< " IMAGE_HEIGHT : " << IMAGE_SIZE[0]
            //<< " FEATURE_MAP_W : " << FEATURE_MAP[level_id][1]
            //<< " FEATURE_MAP_H : " << FEATURE_MAP[level_id][0]
            << std::endl;*/

        if (pred_box[3] - pred_box[1] < box_threshold_ || pred_box[2] - pred_box[0] < box_threshold_) {
            continue;
        }
        scores.push_back(level_infos[i].score);
        boxes.push_back(pred_box);
        labels.push_back(level_infos[i].class_id);
    }
    __TOC__(RetinaNet_regression)

    std::vector<RetinaNetBoundingBox> results;

    std::vector<size_t> box_pos = detector_->detect(scores, boxes, labels);
    for (size_t i = 0; i < box_pos.size(); ++i) {
        size_t pos = box_pos[i];
        //float sigmoid_value = 1 / (1 + expf(-1.0 * scores[pos]));
        results.push_back(RetinaNetBoundingBox{boxes[pos][0], boxes[pos][1], boxes[pos][2],
            boxes[pos][3], scores[pos], labels[pos]});

    }
    std::sort(results.begin(), results.end(), [](const RetinaNetBoundingBox& b1, const RetinaNetBoundingBox& b2) {
            return b1.score > b2.score;
        });
    return results;
}

std::vector<std::vector<RetinaNetBoundingBox>> RetinaNetPost::retinanet_post_process(size_t batch_size) {
    __TIC__(RetinaNet_total_batch)
    auto ret = std::vector<std::vector<RetinaNetBoundingBox>>{};
    ret.reserve(batch_size);
    for (auto i = 0u; i < batch_size; ++i) {
        ret.emplace_back(retinanet_post_process_internal_uniform(i));
    }
    __TOC__(RetinaNet_total_batch)
    return ret;
}

}  // namespace ai
}  // namespace vitis
