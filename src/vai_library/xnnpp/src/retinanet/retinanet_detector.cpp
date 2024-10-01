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

#include "./retinanet_detector.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <thread>
#include <tuple>

#include <vitis/ai/profiling.hpp>
#include "vitis/ai/nnpp/apply_nms.hpp"

using namespace cv;
using namespace std;

namespace vitis {
namespace ai {

RetinaNetDetector::RetinaNetDetector(float nms_threshold, float conf_threshold) :
    nms_threshold_(nms_threshold),
    confidence_threshold_(conf_threshold) {

}

std::vector<size_t> RetinaNetDetector::detect(const std::vector<float>& scores,
    const std::vector<std::vector<float>>& boxes, const std::vector<size_t>& labels) {

    std::map<size_t, RetinaNetNMSInfo> nms_infos;
    for (size_t i = 0; i < scores.size(); ++i) {
        size_t class_id = labels[i];
        if (nms_infos.find(class_id) == nms_infos.end()) {
            nms_infos[class_id] = RetinaNetNMSInfo{};
        }
        nms_infos[class_id].scores.push_back(scores[i]);
        nms_infos[class_id].boxes.push_back(boxes[i]);
        nms_infos[class_id].pos.push_back(i);
    }

    std::vector<size_t> results;
    __TIC__(RetinaNet_nms_total)
    for (auto iter : nms_infos) {

        size_t class_id = iter.first;
        //std::cout << "class_id : " << class_id << std::endl;
        std::vector<size_t> temp_ret;
        /*for (size_t i = 0; i < nms_infos[class_id].boxes.size(); ++i) {
            std::cout << "box, "
                << "\t" << (nms_infos[class_id].boxes)[i][0]
                << "\t" << (nms_infos[class_id].boxes)[i][1]
                << "\t" << (nms_infos[class_id].boxes)[i][2]
                << "\t" << (nms_infos[class_id].boxes)[i][3]
                << ", score : " << (nms_infos[class_id].scores)[i]
                << std::endl;
        }*/
        applyNMS(nms_infos[class_id].boxes, nms_infos[class_id].scores, nms_threshold_, confidence_threshold_, temp_ret);
        for (size_t i = 0; i < temp_ret.size(); ++i) {
            temp_ret[i] = nms_infos[class_id].pos[temp_ret[i]];
        }
        results.insert(results.end(), temp_ret.begin(), temp_ret.end());
    }
    __TOC__(RetinaNet_nms_total)

    return results;
}

std::unique_ptr<RetinaNetDetector> CreateRetinaNetUniform(float nms_thres, float conf_thres) {
    return std::unique_ptr<RetinaNetDetector>(new RetinaNetDetector(nms_thres, conf_thres));
}

}  // namespace ai
}  // namespace vitis
