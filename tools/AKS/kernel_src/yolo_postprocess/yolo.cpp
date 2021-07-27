// Copyright 2019 Xilinx Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

#include <iostream>
#include "yolo.h"

using namespace std;

static float sigmoid(float p) { 
    return 1.0f / (1.0f + exp(p * -1.0f)); 
}

static float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0f, x2 - w2 / 2.0f);
  float right = min(x1 + w1 / 2.0f, x2 + w2 / 2.0f);
  return right - left;
}

static float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0)
    return 0;

  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0f / union_area;
}

static void correct_region_boxes(vector<vector<float>> &boxes, 
                                int w, int h, int netw, int neth) {
  int new_w = 0;
  int new_h = 0;
  int n = boxes.size();

  if (((float)netw / w) < ((float)neth / h)) {
    new_w = netw;
    new_h = (h * netw) / w;
  } else {
    new_h = neth;
    new_w = (w * neth) / h;
  }
  for (int i = 0; i < n; ++i) {
    boxes[i][0] = (boxes[i][0] - (netw - new_w) / 2. / netw) / ((float)new_w / (float)netw);
    boxes[i][1] = (boxes[i][1] - (neth - new_h) / 2. / neth) / ((float)new_h / (float)neth);
    boxes[i][2] *= (float)netw / new_w;
    boxes[i][3] *= (float)neth / new_h;
  }
}

static vector<vector<float>> applyNMS(vector<vector<float>> &boxes, int classes,
                                      const float nms, const float conf) {
  vector<vector<float>> result;

  for (int k = 0; k < classes; k++) {
    vector<pair<int, float>> order;
    order.reserve(boxes.size()/8); 	// arbitrarily, allocate only 1/8th of boxes
    for (size_t i = 0; i < boxes.size(); ++i) {
	  auto prob = boxes[i][6+k];
      if(prob > conf) {
		order.emplace_back(i, prob);
        boxes[i][4] = k;
        boxes[i][5] = prob;
      }
    }
    sort(order.begin(), order.end(),
         [](const pair<int, float> &ls, const pair<int, float> &rs) {
           return ls.second > rs.second;
         });

    vector<bool> exist_box(boxes.size(), true);

    for (size_t _i = 0; _i < order.size(); ++_i) {
      size_t i = order[_i].first;
      if (!exist_box[i])
        continue;
      if (order[_i].second < conf) {
        exist_box[i] = false;
        continue;
      }
      /* add a box as result */
      result.push_back(boxes[i]);

      for (size_t _j = _i + 1; _j < order.size(); ++_j) {
        size_t j = order[_j].first;
        if (!exist_box[j])
          continue;
        float ovr = cal_iou(boxes[j], boxes[i]);
        if (ovr >= nms) {
          exist_box[j] = false;
        }
      }
    }
  }
  return result;
}

static void detect(vector<vector<float>> &boxes, float *result, const float* biases, int height,
                   int width, int num, int sHeight, int sWidth, float scale,
                   int num_classes, int anchor_cnt, int conf_thresh) {
  int classes = num_classes;
  int anchorCnt = anchor_cnt;
  int conf_box = 5 + num_classes;
  int channels = conf_box * anchorCnt;

  // float swap[width*height][anchorCnt][5+classes];
  // float* swap_ptr = &swap[0][0][0];
  float* swap_ptr = new float[(width * height) * (anchorCnt) * (conf_box)];

  int idx = 0;
  for(int h =0; h < height; h++)
     for(int w =0; w < width; w++)
         for(int c =0 ; c < channels; c++)
             swap_ptr[idx++] = result[c*height*width+h*width+w];
             // swap[h*width+w][c/(5+classes)][c%(5+classes)] = result[c*height*width+h*width+w];
  // memcpy(swap_ptr, result, sizeof(float) * width*height*anchorCnt*(5+classes));

  idx = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < anchor_cnt; ++c) {
        // int idx = ((h * width + w) * anchor_cnt + c) * conf_box;
        float obj_score = sigmoid(swap_ptr[idx + 4] * scale);
        // float obj_score = sigmoid(swap[h*width+w][c][4]);
        if (obj_score <= conf_thresh) {
          idx += conf_box;
          continue;
        }

        // Apply softmax on class scores
        vector<float> cls;
        float s = 0.0;
        for (int i = 0; i < num_classes; ++i)
          // cls.push_back(swap[h*width+w][c][5 + i]);
          cls.push_back(swap_ptr[idx+5+i]);
        float large = *max_element(cls.begin(), cls.end());
        for (size_t i = 0; i < cls.size(); ++i) {
          cls[i] = exp(cls[i] - large);
          s += cls[i];
        }
        // vector<float>::iterator biggest = max_element(cls.begin(), cls.end());
        // large = *biggest;
        for (size_t i = 0; i < cls.size(); ++i)
          cls[i] = cls[i] * obj_score / s;
        vector<float> box; box.reserve(conf_box);

        box.push_back((w + sigmoid(swap_ptr[idx+0])) / float(width));
        box.push_back((h + sigmoid(swap_ptr[idx+1])) / float(height));
        box.push_back(exp(swap_ptr[idx+2]) * biases[2 * c + 2 * anchor_cnt * num] / float(width));
        box.push_back(exp(swap_ptr[idx+3]) * biases[2 * c + 2 * anchor_cnt * num + 1] / float(height));
        box.push_back(-1);
        box.push_back(obj_score);
		box.insert(box.end(), cls.begin(), cls.end());
        boxes.push_back(box);
        idx += conf_box;
      }
    }
  }
  delete[] swap_ptr;
}

extern "C"
int yolov2_postproc(Array* output_tensors, int nArrays, const float* biases, int net_h, int net_w, int classes, int anchor_cnt, 
                                        int img_height, int img_width, float conf_thresh, float iou_thresh, int batch_idx, float** retboxes) {
  int sWidth = net_w;
  int sHeight = net_h;

  auto num_classes = classes;
  auto nms_thresh = iou_thresh;
  auto mAP = true;

  /* four output nodes of YOLO-v2 */
  vector<vector<float>> boxes;
  int out_num = nArrays;
  for (int i = 0; i < out_num; i++) {
    int width = output_tensors[i].width;
    int height = output_tensors[i].height;
    int channels = output_tensors[i].channels;
	int size = width * height * channels;
    float *data = output_tensors[i].data;
    float scale = 1.0;
    size_t sizeOut = height * width * anchor_cnt;
    boxes.reserve(sizeOut);
    /* Store the object detection frames as coordinate information  */
    detect(boxes, data, biases, height, width, i, sHeight, sWidth, scale,
				num_classes, anchor_cnt, conf_thresh);
  }  
  /* Restore the correct coordinate frame of the original image */
  if (mAP) {
    correct_region_boxes(boxes, img_width, img_height, sWidth, sHeight);
  }

  /* Apply the computation for NMS */
  vector<vector<float>> res = applyNMS(boxes, num_classes, nms_thresh, conf_thresh);

  // Copy the output to external float array
  int nboxes = res.size();
  float* ret_iter = new float [nboxes * 7 * sizeof(float)];

  int iter = 0;
  for(int i=0; i<nboxes; ++i, iter+=7) {
    int label = res[i][4];
    if (res[i][label + 6] > conf_thresh) {
      float score = res[i][label + 6];
      float label = res[i][4];
      float x = res[i][0]; // - res[i][2] / 2.0;
      float y = res[i][1]; // - res[i][3] / 2.0;
      float w = res[i][2];
      float h = res[i][3];

      float llx = std::max(0.0f, (x - w/2.0f) * img_width);     // left
      float urx = std::min((float)(img_width - 1), (x + w/2.0f) * img_width);    // right
      float lly = std::max(0.0f, (y + h/2.0f) * img_height);    // bottom
      float ury = std::min((float)(img_height-1), (y - h/2.0f) * img_height);    // top

      ret_iter[iter + 0] = batch_idx;
      ret_iter[iter + 1] = llx;
      ret_iter[iter + 2] = lly;
      ret_iter[iter + 3] = urx;
      ret_iter[iter + 4] = ury;
      ret_iter[iter + 5] = label;
      ret_iter[iter + 6] = score;
    }
  }
  *retboxes = ret_iter;

  return nboxes;
}

extern "C"
int yolov3_postproc(Array* outputs, int nArrays, const float* biases, int net_h, int net_w, int classes, int anchorCnt, 
                                        int img_height, int img_width, float conf_thresh, float iou_thresh, int batch_idx, float** retboxes) {
  vector<vector<float>> boxes;
  vector<int> scale_feature; scale_feature.reserve(nArrays);
  for (int iter = 0; iter < nArrays; iter++){
    int width = outputs[iter].width;
    scale_feature.push_back(width);
  }
  sort(scale_feature.begin(), scale_feature.end(), [](int a, int b){return a > b;});
  for (int iter = 0; iter < nArrays; iter++) {
    const float* result = outputs[iter].data;
    int height = outputs[iter].height;
    int width = outputs[iter].width;
    int channels = outputs[iter].channels;
    int index = 0;
    for (int i = 0; i < nArrays; i++) {
      if(width == scale_feature[i]) {
        index = i;
        break;
      }
    }
    // swap the network's result to a new sequence
    float swap[width*height][anchorCnt][5+classes];
    for(int h =0; h < height; h++) {
      for(int w =0; w < width; w++) {
        for(int c =0 ; c < channels; c++) {
          //std::cout << "swap[" << h*width+w << "][" << c/(5+classes) << "][" << c%(5+classes) << "] = ";
          //std::cout << "result[" << c*height*width+h*width+w << "]" << std::endl;
          swap[h*width+w][c/(5+classes)][c%(5+classes)] = result[c*height*width+h*width+w];
        }
      }
    }

    // compute the coordinate of each primary box
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int n = 0 ; n < anchorCnt; n++) {
          float obj_score = sigmoid(swap[h*width+w][n][4]);
          if(obj_score < conf_thresh)
            continue;
          vector<float>box,cls;
          float x = (w + sigmoid(swap[h*width+w][n][0])) / width;
          float y = (h + sigmoid(swap[h*width+w][n][1])) / height;
          float ww = exp(swap[h*width+w][n][2]) * biases[2*n+2*anchorCnt*index] / net_w;
          float hh = exp(swap[h*width+w][n][3]) * biases[2*n+2*anchorCnt*index+1] / net_h;
          box.push_back(x);
          box.push_back(y);
          box.push_back(ww);
          box.push_back(hh);
          box.push_back(-1);
          box.push_back(obj_score);
          for(int p =0; p < classes; p++)
            box.push_back(obj_score * sigmoid(swap[h*width+w][n][5+p]));
          boxes.push_back(box);
        }
      }
    }
  }

  correct_region_boxes(boxes, img_width, img_height, net_w, net_h);
  vector<vector<float>> res = applyNMS(boxes, classes, iou_thresh, conf_thresh);

  // Copy the output to external float array
  int nboxes = res.size();
  float* ret_iter = new float [nboxes * 7 * sizeof(float)];

  int iter = 0;
  for(int i=0; i<nboxes; ++i) {
    auto &tmpBox = res[i];
    float llx = std::max(0.0f, (tmpBox[0] - tmpBox[2]/2.0f) * img_width);     // left
    float urx = std::min((float)(img_width - 1), (tmpBox[0] + tmpBox[2]/2.0f) * img_width);    // right
    float lly = std::min((float)(img_height-1), (tmpBox[1] + tmpBox[3]/2.0f) * img_height);    // bottom
    float ury = std::max(0.0f, (tmpBox[1] - tmpBox[3]/2.0f) * img_height);    // top

    ret_iter[iter + 0] = batch_idx;
    ret_iter[iter + 1] = llx;
    ret_iter[iter + 2] = lly;
    ret_iter[iter + 3] = urx;
    ret_iter[iter + 4] = ury;
    ret_iter[iter + 5] = tmpBox[4];
    ret_iter[iter + 6] = tmpBox[5];
    iter += 7;
  }
  *retboxes = ret_iter;
  return nboxes;
}

// Clear buffer
extern "C" 
void clearBuffer(float* buf) {
    delete[] buf;
}
