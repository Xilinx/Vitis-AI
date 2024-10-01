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
#include "vitis/ai/nnpp/centerpoint.hpp"
#include "./centerpointutil.hpp"


const int out_size_factor = 4;
std::vector<float> voxel_size{0.2 , 0.2};
std::vector<int> pc_range{0, -40}; 
float score_threshold = 0.1;
//std::vector<float> post_center_range{0, -40, -5.5, 80, 40, 2.5};
std::vector<float> post_center_range{0, -40, -6, 80, 40, 8};
const int K = 500;
std::vector<int> bev{0, 1, 3, 4, 6};
int post_max_size = 83;
float nms_thr = 0.2;
const std::vector<std::string> heads{"reg", "height", "dim", "rot", "heatmap"}; 

namespace vitis {
namespace ai {
using namespace std;

template<typename T>
void writefile(string& filename, vector<T>& data) {
    ofstream output_file(filename);
      for(size_t i = 0; i < data.size(); i++)
            output_file << data[i] << endl;
}


void format_separate(std::vector<T3D<float>>& input, std::map<string, T3D<float>>& format) {
  for (auto i = 0u; i < input.size(); i++) {
    format.insert(pair<string, T3D<float>>(heads[i], input[i])); 
  }

}


void format_flatten(std::vector<T3D<float>>& input, std::vector<std::map<string, T3D<float>>>& formats) {
  for (auto i = 0u; i < input.size(); i = i + heads.size()) {
    auto start = input.begin() + i;
    auto end = input.begin() + i + heads.size();
    std::vector<T3D<float>> sep(start, end);
    std::map<string, T3D<float>> format;
    format_separate(sep, format);
    formats.push_back(format);
  }
}

template<typename T>
void pt(T3D<T> ele,string name, int a,int b) {
  for(auto i = a; i < b; i++) 
    cout << name << " "<< i << " " << ele.data[i] << endl;
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<float>>
decode(T3D<float> heatmap, T3D<float> reg, T3D<float> hei, T3D<float> rot, T3D<float> dim) { 
//process heat
  auto topk_result = topK(heatmap, K);  //topK return std::make_tuple(topk_score, topk_inds, topk_clses, topk_ys, topk_xs); 
  auto topk_scores =  std::get<0>(topk_result);
  auto topk_inds =  std::get<1>(topk_result);
  auto topk_clses =  std::get<2>(topk_result);
  auto topk_ys =  std::get<3>(topk_result);
  auto topk_xs =  std::get<4>(topk_result);
//process reg
  transpose_and_gather_feat(reg, topk_inds);
  reg.reshape(1, K, 2);
  topk_xs.reshape(1, K, 1);
  topk_ys.reshape(1, K, 1);
  T3D<float> xs;
  T3D<float> ys;
  xs.resize(1, K, 1);
  ys.resize(1, K, 1);
  for(auto i = 0u; i < topk_xs.data.size(); i++) {
    xs.data[i] = topk_xs.data[i] + reg.data[i * 2];
  }
  for(auto i = 0u; i < topk_ys.data.size(); i++) {
    ys.data[i] = topk_ys.data[i] + reg.data[i*2 + 1];
  }
//process rotsine and rotcosine
  T3D<float> rot_sine;
  T3D<float> rot_cosine;
  auto rs_len = rot.data.size() / rot.channel;
  rot_sine.resize(1, rs_len, 1);
  rot_cosine.resize(1, rs_len, 1);
  copy_n(rot.data.begin(), rs_len, rot_sine.data.begin());
  copy_n(rot.data.begin() + rs_len, rs_len, rot_cosine.data.begin());
  transpose_and_gather_feat(rot_sine, topk_inds);
  rot_sine.reshape(1,K,1);
  transpose_and_gather_feat(rot_cosine, topk_inds);
  rot_cosine.reshape(1,K,1);
  rot = atan2_(rot_sine, rot_cosine);
  rot.reshape(1, K, 1);
//process hei, dim
  transpose_and_gather_feat(hei, topk_inds);
  hei.reshape(1, K, 1);
  transpose_and_gather_feat(dim, topk_inds);
  dim.reshape(1, K, 3);
  topk_scores.reshape(1, K, 1); // 1 K
  topk_clses.reshape(1, K, 1); // 1 K
  for (auto i = 0u; i < xs.data.size(); i++) {
    xs.data[i] = xs.data[i] * out_size_factor * voxel_size[0] + pc_range[0];
  }
  for (auto i = 0u; i < ys.data.size(); i++) {
    ys.data[i] = ys.data[i] * out_size_factor * voxel_size[1] + pc_range[1];
  }
// no vel, get final boxes
  std::vector<std::vector<float>> final_box_preds(K);
  for(auto i = 0u; i < final_box_preds.size(); i++) {
    final_box_preds[i] = {xs.at(0, i, 0), ys.at(0, i, 0), hei.at(0, i, 0), 
            dim.at(0, i, 0), dim.at(0, i, 1), dim.at(0, i, 2), rot.at(0, i, 0)
    };
  }
  std::vector<bool> thresh_mask(topk_scores.data.size());
  for(auto i = 0u; i < thresh_mask.size(); i++) {
    topk_scores.data[i] > score_threshold ? thresh_mask[i] = true : thresh_mask[i] = false;
  }
  std::vector<bool> mask(final_box_preds.size());
  for (auto i = 0u; i < mask.size(); i++) {
    if(final_box_preds[i][0] >= post_center_range[0] && final_box_preds[i][1] >= post_center_range[1]
      && final_box_preds[i][2] >= post_center_range[2] && final_box_preds[i][0] <= post_center_range[3]
      && final_box_preds[i][1] <= post_center_range[4] && final_box_preds[i][2] <= post_center_range[5])
    {
      mask[i] = true;
    }
  }
  std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<float>>  predictions_dicts;
  for(auto i = 0u; i < thresh_mask.size(); i++) {
    if(thresh_mask[i] == true && mask[i] == true) {
      std::get<0>(predictions_dicts).push_back(final_box_preds[i]);
      std::get<1>(predictions_dicts).push_back(topk_scores.data[i]);
      std::get<2>(predictions_dicts).push_back(topk_clses.data[i]);
    }
  }
  return predictions_dicts;
}

std::vector<CenterPointResult> postproess_api(std::vector<T3D<float>> points) {
  std::vector<std::map<string, T3D<float>>> formats;
  format_flatten(points, formats);
  // for (auto i = 0u; i < formats.size(); i++) {
  std::vector<CenterPointResult> bbox_final;
  for (auto i = 0u; i < 1; i++) {
    auto heatmap = formats[i]["heatmap"];
    std::string  name = "heatmap.txt";
    //writefile(name, heatmap.data);
    sigmoid_n(heatmap.data);
    auto reg = formats[i]["reg"];
    auto hei = formats[i]["height"];
    auto rot = formats[i]["rot"];
    // auto vel = formats[i]["vel"];
    auto dim = formats[i]["dim"];
    exp_n(dim.data);
    //decode
    auto predictions_dicts = decode(heatmap, reg, hei, rot, dim);
    auto bbox_preds = std::get<0>(predictions_dicts);
    auto top_scores = std::get<1>(predictions_dicts);
    auto top_labels = std::get<2>(predictions_dicts);
    //
    auto boxes_for_nms = xywhr2xyxyr(bbox_preds, bev);
    //# the nms in 3d detection just remove overlap boxes
    std::vector<std::vector<float>> _top_scores(1);
    _top_scores[0] = top_scores;
    auto selected = nms_3d_multiclasses(boxes_for_nms, _top_scores, 1, score_threshold, nms_thr, post_max_size);
    //select predictions
    //std::vector<std::vector<float>> selected_boxes;
    //std::vector<float> selected_labels;
    //std::vector<float> selected_scores;
    //std::vector<CenterPointResult> bbox_final(selected.size());
    bbox_final.resize(selected.size());
    for (auto i = 0u; i < selected.size(); i++) {
      auto bbox_temp = bbox_preds[selected[i].first];
      bbox_temp[2] = bbox_temp[2] - bbox_temp[5] * 0.5;
      bbox_final[i].bbox = bbox_temp;
      bbox_final[i].label = top_labels[selected[i].first];
      bbox_final[i].score = top_scores[selected[i].first];
      for (auto j = 0u; j < bbox_final[i].bbox.size(); j++) {
        //cout << bbox_final[i].bbox[j] << " ";
      }
      //cout << "   "   << bbox_final[i].score << endl;
    }
  }
  return bbox_final;
   

}


std::vector<CenterPointResult> post_process(std::vector<std::vector<vitis::ai::library::OutputTensor>> output_tensors, size_t batch_ind) {
  std::vector<vitis::ai::library::OutputTensor> out_tensor;
  for (auto i = 0u; i < heads.size(); ++i) {
    for (auto j = 0u; j < output_tensors[0].size(); ++j) {
      if (output_tensors[0][j].name.find(heads[i]) != std::string::npos) {
        out_tensor.emplace_back(output_tensors[0][j]);
      }
    }
  }
  // cout << out_tensor.size() << endl;
  std::vector<T3D<float>> points;
	std::vector<size_t> channels{2, 1, 3, 2, 1};
  for (auto i = 0u; i < out_tensor.size(); i++) {
    auto outscale = vitis::ai::library::tensor_scale(out_tensor[i]);
    auto H = out_tensor[i].height;
    auto W = out_tensor[i].width;
    auto C = out_tensor[i].channel;
    T3D<float> point; 
		point.resize(C, H, W);
    // cout << "HWC " << H << "/" << W << "/" << C << "/"  << endl;
    for (auto h = 0u; h < H; h++) {
      for (auto w = 0u; w < W; w++) {
        for (auto c = 0u; c < C; c++) {
          point.data[c*H*W + h*W + w] = ((int8_t*)out_tensor[i].get_data(batch_ind))[h*W*C +w*C + c] * outscale;
        }
      }  
    }
    //std::string name = to_string(i) +"_out.txt";
    //writefile(name, point.data);
		point.reshape(channels[i], 100, 100);
    points.push_back(point);
  }
  return postproess_api(points); 
}

}
}
