#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/densebox_map_evaluation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DenseboxMapEvaluationLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  channel_axes_ = this->layer_param_.densebox_map_evaluation_param().channel_axes();
  nms_threshold_ = this->layer_param_.densebox_map_evaluation_param().nms_threshold();
  conf_threshold_ = this->layer_param_.densebox_map_evaluation_param().conf_threshold();
  iou_threshold_ = this->layer_param_.densebox_map_evaluation_param().iou_threshold();
}

template <typename Dtype>
void DenseboxMapEvaluationLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 3) << "The bottom number of this layer must be 3!";
  CHECK_EQ(bottom[0]->channels(), 4) << "The first bottom should be the bb predict layer!";
  //CHECK_EQ(bottom[0]->width(), 80)  << "The width is 80";
  CHECK_EQ(bottom[1]->channels(), 2) << "The second bottom should be the class predict layer!";
  CHECK_EQ(bottom[2]->channels(), 1) << "The third bottom should be the bb ground truth layer!";
  CHECK_GT(bottom[2]->height(), 100) << "The height of bb ground truth is 100!";
  outer_num_ = bottom[0]->count(0, channel_axes_);
  inner_num_ = bottom[0]->count(channel_axes_ + 1);
  vector<int> top_shape(2, 1);  // 4 dimension, 1, 1, vail_detect_num, 5;
  top_shape.push_back(1); // set a fake number firstly
  top_shape.push_back(5);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DenseboxMapEvaluationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int dim = bottom[0]->count() / outer_num_;
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int height_gt = bottom[2]->height();
  int dim_gt = bottom[2]->count() / outer_num_;
  Dtype score;
  int xmin, ymin, xmax, ymax;
  int xmin_sec, ymin_sec, xmax_sec, ymax_sec;
  int xmin_iou, ymin_iou, xmax_iou, ymax_iou;
  std::vector<std::pair<BBox, float>> pred_bbox_conf;
  std::vector<std::pair<BBox, float>> nms_bbox_conf_val;
  std::vector<std::pair<BBox, float>>* nms_bbox_conf = &nms_bbox_conf_val;
  std::vector<std::vector<std::pair<BBox, bool>>> gt_bboxs_all;
  int num_bb_gt = 0;
  int id = 0;
  int size = 0;
  int area_iou, area_1, area_2;
  const Dtype* pred_bb = bottom[0]->cpu_data();
  const Dtype* pred_conf = bottom[1]->cpu_data();
  const Dtype* gt_bb = bottom[2]->cpu_data();
  int pred_num_count = 0;
  for (int m = 0; m < outer_num_; m++) {
    pred_bbox_conf.clear();
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
         // get the positive probility score
         score = static_cast<float>(pred_conf[m * 2 * inner_num_ + inner_num_ + h * width + w]);
         CHECK_GE(score, 0.0);
         CHECK_LE(score, 1.0);
         if (score < conf_threshold_) {
           continue;
         }
         xmin = static_cast<int>(pred_bb[m * dim + h * width + w] + 4 * w);
         ymin = static_cast<int>(pred_bb[m * dim + inner_num_ + h * width + w] + 4 * h);
         xmax = static_cast<int>(pred_bb[m * dim + 2 * inner_num_ + h * width + w] + 4 * w);
         ymax = static_cast<int>(pred_bb[m * dim + 3 * inner_num_ + h * width + w] + 4 * h);
         BBox bb_pred{xmin, ymin, xmax, ymax, m};
         pred_bbox_conf.push_back(std::make_pair(bb_pred, score));
      }
    }
    int before_size = nms_bbox_conf->size(); 
    pred_num_count += pred_bbox_conf.size();
    std::sort(pred_bbox_conf.begin(), pred_bbox_conf.end(), cmp);
    ApplyNms(pred_bbox_conf, nms_bbox_conf);
    int after_size = nms_bbox_conf->size();
    // LOG(ERROR) << "detect images: " << (after_size - before_size);
    for (int j = before_size; j < after_size; j++) {
      xmin = static_cast<int>((*nms_bbox_conf)[j].first.xmin);
      ymin = static_cast<int>((*nms_bbox_conf)[j].first.ymin);   
      xmax = static_cast<int>((*nms_bbox_conf)[j].first.xmax);
      ymax = static_cast<int>((*nms_bbox_conf)[j].first.ymax);
      score = static_cast<float>((*nms_bbox_conf)[j].second);
      // LOG(ERROR) << "dxmin: " << xmin << " dymin: " << ymin << " dxmax: " <<  xmax << " dymax: " << ymax << " score: " << score;
    }
    std::vector<std::pair<BBox, bool>> gt_bboxs_one;
    for (int h = 0; h < height_gt; h++) {
      xmin = static_cast<int>(gt_bb[m * dim_gt + h * 4]);
      ymin = static_cast<int>(gt_bb[m * dim_gt + h * 4 + 1]);
      xmax = static_cast<int>(gt_bb[m * dim_gt + h * 4 + 2]);
      ymax = static_cast<int>(gt_bb[m * dim_gt + h * 4 + 3]);
      if (xmin == 0 && ymin == 0 && xmax == 0 && ymax == 0) {
        break;
      }
      // LOG(ERROR) << "xmin: " << xmin << " ymin: " << ymin << " xmax: " << xmax << " ymax: " << ymax;
      BBox bb_gt{xmin, ymin, xmax, ymax, m};
      gt_bboxs_one.push_back(std::make_pair(bb_gt, false));
    }
    num_bb_gt += gt_bboxs_one.size();
    gt_bboxs_all.push_back(gt_bboxs_one);
  }
  // LOG(ERROR) << "num_bb_gt: " << num_bb_gt;
  CHECK_EQ(gt_bboxs_all.size(), outer_num_) << "The gt_bboxes_all size must be same with the outer_num!";
  const int num_gt = num_bb_gt;
  int index_detect = 0;
  vector<int> top_shape(2, 1);
  top_shape.push_back(nms_bbox_conf->size() + 1);
  top_shape.push_back(5);
  top[0]->Reshape(top_shape);
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.0), top_data);
  top_data[5 * index_detect] = -1;
  top_data[5 * index_detect + 1] = 1;
  top_data[5 * index_detect + 2] = num_gt;
  top_data[5 * index_detect + 3] = -1;
  top_data[5 * index_detect + 4] = -1;
  index_detect++;

  bool is_match;
  float fscore = 0;
  for (auto iter = nms_bbox_conf->begin(); iter != nms_bbox_conf->end(); iter++) {
    is_match = false;
    id = (*iter).first.id;
    xmin = (*iter).first.xmin;
    ymin = (*iter).first.ymin;
    xmax = (*iter).first.xmax;
    ymax = (*iter).first.ymax;
    fscore = (*iter).second;
    size = gt_bboxs_all[id].size();
    area_1 = (xmax - xmin + 1) * (ymax - ymin + 1);
    float overlap_max = 0.0;
    float overlap = 0.0;
    int overlap_max_index = -1;
    for (int i = 0; i < size; i++) {
      xmin_sec = gt_bboxs_all[id][i].first.xmin;
      ymin_sec = gt_bboxs_all[id][i].first.ymin;
      xmax_sec = gt_bboxs_all[id][i].first.xmax;
      ymax_sec = gt_bboxs_all[id][i].first.ymax;
      xmin_iou = std::max(xmin, xmin_sec);
      xmax_iou = std::min(xmax, xmax_sec);
      ymin_iou = std::max(ymin, ymin_sec);
      ymax_iou = std::min(ymax, ymax_sec);
      area_2 = (xmax_sec - xmin_sec + 1) * (ymax_sec - ymin_sec + 1);
      if (xmin_iou >= xmax_iou || ymin_iou >= ymax_iou) {
        continue;
      }
      area_iou = (xmax_iou - xmin_iou + 1) * (ymax_iou - ymin_iou + 1);
      overlap = static_cast<float>(area_iou) / (area_1 + area_2 -area_iou);
      if (overlap >= iou_threshold_) {
        if (overlap > overlap_max) {
          overlap_max = overlap;
          overlap_max_index = i;
          is_match = true;
        }  
      }
    }
    if (is_match && overlap_max_index != -1 && !gt_bboxs_all[id][overlap_max_index].second) {
      gt_bboxs_all[id][overlap_max_index].second = true;
      top_data[5 * index_detect] = id;
      top_data[5 * index_detect + 1] = 1;
      top_data[5 * index_detect + 2] = fscore;     
      top_data[5 * index_detect + 3] = 1;
      top_data[5 * index_detect + 4] = 0;
    } else {
      top_data[5 * index_detect] = id;
      top_data[5 * index_detect + 1] = 1;
      top_data[5 * index_detect + 2] = fscore;     
      top_data[5 * index_detect + 3] = 0;
      top_data[5 * index_detect + 4] = 1;
    }
    index_detect++;    
  }
}

template <typename Dtype>
void DenseboxMapEvaluationLayer<Dtype>::ApplyNms(
  const std::vector<std::pair<BBox, float>> & pred_bbox_conf, std::vector<std::pair<BBox, float>>* nms_bbox_conf) {
  vector<int> areas_bb_vec;
  vector<bool> is_overlap(pred_bbox_conf.size(), false);
  int xmin, ymin, xmax, ymax;
  int xmin_sec, ymin_sec, xmax_sec, ymax_sec;
  int xmin_iou, ymin_iou, xmax_iou, ymax_iou;
  int area_iou, area_1, area_2;
  float iou;
  for (auto iter = pred_bbox_conf.begin(); iter != pred_bbox_conf.end(); iter++) {
    xmin = (*iter).first.xmin;
    ymin = (*iter).first.ymin;
    xmax = (*iter).first.xmax;
    ymax = (*iter).first.ymax;
    areas_bb_vec.push_back((xmax - xmin + 1) * (ymax - ymin + 1));
  }
  int size = pred_bbox_conf.size();
  for (int i = 0; i < size; i++) {
    if (is_overlap[i]) {
      continue;
    }
    nms_bbox_conf->push_back(pred_bbox_conf[i]);
    xmin = pred_bbox_conf[i].first.xmin;
    xmax = pred_bbox_conf[i].first.xmax;
    ymin = pred_bbox_conf[i].first.ymin;
    ymax = pred_bbox_conf[i].first.ymax;
    area_1 = areas_bb_vec[i];
    for (int j = i + 1; j < size; j++) {
      if (is_overlap[j]) {
        continue;
      }
      xmin_sec = pred_bbox_conf[j].first.xmin;
      xmax_sec = pred_bbox_conf[j].first.xmax;
      ymin_sec = pred_bbox_conf[j].first.ymin;
      ymax_sec = pred_bbox_conf[j].first.ymax;
      xmin_iou = std::max(xmin, xmin_sec);
      xmax_iou = std::min(xmax, xmax_sec);
      ymin_iou = std::max(ymin, ymin_sec);
      ymax_iou = std::min(ymax, ymax_sec);
      if (xmin_sec == xmin_iou && xmax_sec == xmax_iou && ymin_sec == ymin_iou && ymax_sec == ymax_iou) {
        is_overlap[j] = true;
      } else if (xmin == xmin_iou && xmax == xmax_iou && ymin == ymin_iou && ymax == ymax_iou) {
        is_overlap[j] = true;
      } else if (xmin_iou < xmax_iou && ymin_iou < ymax_iou) {
        area_2 = areas_bb_vec[j];
        area_iou = (ymax_iou - ymin_iou + 1) * (xmax_iou - xmin_iou + 1);
        iou = static_cast<float>(area_iou) / (area_1 + area_2 - area_iou);
        if (iou >= nms_threshold_) {
          is_overlap[j] = true;
        }
      }
    }
  }
}


INSTANTIATE_CLASS(DenseboxMapEvaluationLayer);
REGISTER_LAYER_CLASS(DenseboxMapEvaluation);

}  // namespace caffe
