#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/yolo_eval_detection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

class BoxData {
 public:
  int label_;
  float score_;
  vector<float> box_;
};

template <typename Dtype>
Dtype Overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  Dtype left = std::max(x1 - w1 / 2, x2 - w2 / 2);
  Dtype right = std::min(x1 + w1 / 2, x2 + w2 / 2);
  return right - left;
}

template <typename Dtype>
Dtype Calc_iou(const vector<Dtype>& box, const vector<Dtype>& truth) {
  Dtype w = Overlap(box[0], box[2], truth[0], truth[2]);
  Dtype h = Overlap(box[1], box[3], truth[1], truth[3]);
  if (w < 0 || h < 0) return 0;
  Dtype inter_area = w * h;
  Dtype union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area / union_area;
}

inline float sigmoid(float x)
{
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
float softmax_region(Dtype* input, int classes)
{
  Dtype sum = 0;
  Dtype large = input[0];

  for (int i = 0; i < classes; ++i){
    if (input[i] > large)
      large = input[i];
  }
  for (int i = 0; i < classes; ++i){
    Dtype e = exp(input[i] - large);
    sum += e;
    input[i] = e;
  }
  for (int i = 0; i < classes; ++i){
    input[i] = input[i] / sum;
  }
  return 0;
}
vector<BoxData> correct_region_boxes( vector<BoxData> tmp_boxes, int n, int w, int h, int netw, int neth)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        tmp_boxes[i].box_[0] =  (tmp_boxes[i].box_[0] - (netw - new_w)/2./netw) / ((float)new_w/netw);
        tmp_boxes[i].box_[1] =  (tmp_boxes[i].box_[1] - (neth - new_h)/2./neth) / ((float)new_h/neth);
        tmp_boxes[i].box_[2] *= (float)netw/new_w;
        tmp_boxes[i].box_[3] *= (float)neth/new_h;
        tmp_boxes[i].box_[0] *= w;
        tmp_boxes[i].box_[2] *= w;
        tmp_boxes[i].box_[1] *= h;
        tmp_boxes[i].box_[3] *= h;
    }
  return tmp_boxes;
}

bool BoxSortDecendScore(const BoxData& box1, const BoxData& box2) {
  return box1.score_ > box2.score_;
}

void ApplyNms(const vector<BoxData>& boxes, vector<int>* idxes, float threshold) {
  map<int, int> idx_map;
  for (int i = 0; i < boxes.size() - 1; ++i) {
    if (idx_map.find(i) != idx_map.end()) {
      continue;
    }
    vector<float> box1 = boxes[i].box_;
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (idx_map.find(j) != idx_map.end()) {
        continue;
      }
      vector<float> box2 = boxes[j].box_;
      float iou = Calc_iou(box1, box2);
      if (iou >= threshold) {
        idx_map[j] = 1;
      }
    }
  }
  for (int i = 0; i < boxes.size(); ++i) {
    if (idx_map.find(i) == idx_map.end()) {
      idxes->push_back(i);
    }
  }
}

template <typename Dtype>
void GetGTBox(int side_height, int side_width, const Dtype* label_data, map<int, vector<BoxData> >* gt_boxes) {
  for (int i = 0; i < 30; ++ i) {
    if (label_data[i * 5 + 1] == 0) {	
	break; 
    }
    BoxData gt_box;
    int label = label_data[i * 5 + 0];
    gt_box.label_ = label;
    gt_box.score_ = i; 
    int box_index = i * 5 + 1;
    for (int j = 0; j < 4; ++j) {
      gt_box.box_.push_back(label_data[box_index + j]);
    }
    if (gt_boxes->find(label) == gt_boxes->end()) {
      (*gt_boxes)[label] = vector<BoxData>(1, gt_box);
    } else {
      (*gt_boxes)[label].push_back(gt_box);
    }
  }
}

template <typename Dtype>
void GetPredBox(int side_height ,int side_width, int num_object, int num_class, Dtype* input_data, map<int, vector<BoxData> >* pred_boxes, int score_type, float nms_threshold, vector<Dtype> biases) {
  vector<BoxData> tmp_boxes;
  //int pred_example_count = 0; 
  for (int j = 0; j < side_height; ++j)
    for (int i = 0; i < side_width; ++i)
      for (int n = 0; n < 5; ++n)
      {
	int index = (j * side_width + i) * num_object * (num_class + 1 + 4) + n * (num_class + 1 + 4);
	float x = (i + sigmoid(input_data[index + 0])) / side_width;
	float y = (j + sigmoid(input_data[index + 1])) / side_height;
 	float w = (exp(input_data[index + 2]) * biases[2 * n]) / side_width;
	float h = (exp(input_data[index + 3]) * biases[2 * n + 1]) / side_height;
	softmax_region(input_data + index + 5, num_class);
	
	int pred_label = 0;
	float max_prob = input_data[index + 5];
	for (int c = 0; c < num_class; ++c)
	{
	  if (max_prob < input_data[index + 5 + c])
	  {
	    max_prob = input_data[index + 5 + c];
	    pred_label = c; 
	  }	
	}
	BoxData pred_box;
	pred_box.label_ = pred_label;
	
        float obj_score = sigmoid(input_data[index + 4]);
	if (score_type == 0) {
	  pred_box.score_ = obj_score;
	} else if (score_type == 1) {
	  pred_box.score_ = max_prob;
	} else {
	  pred_box.score_ = obj_score * max_prob;
	}

	pred_box.box_.push_back(x);
	pred_box.box_.push_back(y);
   	pred_box.box_.push_back(w);
	pred_box.box_.push_back(h);
      //  if(pred_box.score_ > 0.005)	
	tmp_boxes.push_back(pred_box);
      }  
  tmp_boxes = correct_region_boxes(tmp_boxes , tmp_boxes.size(), 1296 , 784 , 1280 , 768); 
  if (nms_threshold >= 0) {
    std::sort(tmp_boxes.begin(), tmp_boxes.end(), BoxSortDecendScore);
    vector<int> idxes;
    ApplyNms(tmp_boxes, &idxes, nms_threshold);
    for (int i = 0; i < idxes.size(); ++i) {
      BoxData box_data = tmp_boxes[idxes[i]];
      if (box_data.score_ < 0.005) // from darknet
	continue;
      if (pred_boxes->find(box_data.label_) == pred_boxes->end()) {
        (*pred_boxes)[box_data.label_] = vector<BoxData>();
      }
     //  box_data.box_[0] = box_data.box_[0] - box_data.box_[2]/2.0 + 1;
     //  box_data.box_[1] = box_data.box_[1] - box_data.box_[3]/2.0 + 1;
     //  box_data.box_[2] = box_data.box_[0] + box_data.box_[2] ;
     //  box_data.box_[3] = box_data.box_[1] + box_data.box_[3] ;
       //++pred_example_count; 
      //LOG(INFO)<<"box:"<<box_data.box_[0]<<" "<<box_data.box_[1]<<" "<<box_data.box_[2]<<" "<<box_data.box_[3];
      (*pred_boxes)[box_data.label_].push_back(box_data);
    }
  } else {
    for (std::map<int, vector<BoxData> >::iterator it = pred_boxes->begin(); it != pred_boxes->end(); ++it) {
      std::sort(it->second.begin(), it->second.end(), BoxSortDecendScore);
    }
  }
}
template <typename Dtype>
void YoloEvalDetectionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  YoloEvalDetectionParameter param = this->layer_param_.yolo_eval_detection_param();
  side_width = bottom[0]->width();
  side_height = bottom[0]->height();
  num_class_ = param.num_class();
  num_object_ = param.num_object();
  threshold_ = param.threshold();
  
  nms_ = param.nms();
  
  for (int c = 0; c < param.biases_size(); ++c){
    biases_.push_back(param.biases(c));
  }

  switch (param.score_type()) {
    case YoloEvalDetectionParameter_ScoreType_OBJ:
      score_type_ = 0;
      break;
    case YoloEvalDetectionParameter_ScoreType_PROB:
      score_type_ = 1;
      break;
    case YoloEvalDetectionParameter_ScoreType_MULTIPLY:
      score_type_ = 2;
      break;
    default:
      LOG(FATAL) << "Unknow score type.";
  }
}

template <typename Dtype>
void YoloEvalDetectionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int input_count = bottom[0]->count(1); 
  int tmp_input_count = side_width * side_height * num_object_ *( num_class_ + 4 + 1 ); //13*13*5*25
  // label: isobj, class_label, coordinates
  
  CHECK_EQ(input_count, tmp_input_count);
  //CHECK_EQ(label_count, tmp_label_count);

  vector<int> top_shape(2, 1);
  //top_shape[0] = bottom[0]->num();
  top_shape.push_back(1);
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void YoloEvalDetectionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Blob<Dtype> swap;
  swap.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_object_, bottom[0]->channels()/num_object_);  
  
  Dtype* swap_data = swap.mutable_cpu_data();
  int index = 0;
  for (int b = 0; b < bottom[0]->num(); ++b)
    for (int h = 0; h < bottom[0]->height(); ++h)
      for (int w = 0; w < bottom[0]->width(); ++w)
        for (int c = 0; c < bottom[0]->channels(); ++c)
	{
	  swap_data[index++] = bottom[0]->data_at(b,c,h,w);
	}  
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  
  for (int i = 0; i < bottom[0]->num(); ++i) {
    int input_index = i * bottom[0]->count(1);
    int true_index = i * bottom[1]->count(1);
    int top_index = i * top[0]->count(1);
    map<int, vector<BoxData> > gt_boxes;
    GetGTBox(side_height,side_width , label_data + true_index, &gt_boxes);

     //int class_count[num_class_]={0};
     int *class_count = new int[num_class_]();
     int gt_count = 0;
    for (std::map<int, vector<BoxData > >::iterator it = gt_boxes.begin(); it != gt_boxes.end(); ++it) {
      vector<BoxData>& g_boxes = it->second;
      for (int j = 0; j < g_boxes.size(); ++j) {
          int xx = g_boxes[j].label_;
          class_count[xx] += 1;
          ++gt_count;
      }
      /*for (int c = 0; c < num_class_; c++){
	top_data[num_det * 5] = -1;
	top_data[num_det * 5 + 1] = c;
	top_data[num_det * 5 + 2] = class_count[c];
	top_data[num_det * 5 + 3] = -1;
	top_data[num_det * 5 + 4] = -1;
        ++num_det;
        }
        */
    }
     
    map<int, vector<BoxData> > pred_boxes;
    GetPredBox(side_height, side_width, num_object_, num_class_, swap_data + input_index, &pred_boxes, score_type_, nms_, biases_);

    int index = top_index + num_class_ * 5;
    int pred_count(0);
    int num_det =0;
    for (std::map<int, vector<BoxData> >::iterator it = pred_boxes.begin(); it != pred_boxes.end(); ++it) {
      int label = it->first;
      vector<BoxData>& p_boxes = it->second;
      vector<int> top_shape(2,1);
      top_shape.push_back(num_class_ + p_boxes.size() );
      top_shape.push_back(5);
      top[0]->Reshape(top_shape);
      top_data = top[0]->mutable_cpu_data();
      for (int c = 0; c < num_class_; c++){
	top_data[num_det * 5] = -1;
	top_data[num_det * 5 + 1] = c;
	top_data[num_det * 5 + 2] = class_count[c];
	top_data[num_det * 5 + 3] = -1;
	top_data[num_det * 5 + 4] = -1;
        ++num_det;
        }
      if (gt_boxes.find(label) == gt_boxes.end()) {
        for (int b = 0; b < p_boxes.size(); ++b) {
          top_data[index + pred_count * 5 + 0] = p_boxes[b].label_;
          top_data[index + pred_count * 5 + 1] = p_boxes[b].label_;
          top_data[index + pred_count * 5 + 2] = p_boxes[b].score_;
          top_data[index + pred_count * 5 + 3] = 0; //tp
          top_data[index + pred_count * 5 + 4] = 1; //fp
          ++pred_count;
        }
        continue;
      } 
      
      vector<BoxData>& g_boxes = gt_boxes[label];
      vector<bool> records(g_boxes.size(), false);
      for (int k = 0; k < p_boxes.size(); ++k) {
        top_data[index + pred_count * 5 + 0] = p_boxes[k].label_;
        top_data[index + pred_count * 5 + 1] = p_boxes[k].label_;
        top_data[index + pred_count * 5 + 2] = p_boxes[k].score_;
        float max_iou(-1);
        int idx(-1);
        for (int g = 0; g < g_boxes.size(); ++g) {
          //LOG(INFO)<<"box:"<<p_boxes[k].box_[0]<<" "<<p_boxes[k].box_[1]<<" "<<p_boxes[k].box_[2]<<" "<<p_boxes[k].box_[3];
          float iou = Calc_iou(p_boxes[k].box_, g_boxes[g].box_);
          if (iou > max_iou) {
            max_iou = iou;
            idx = g;
          }
        }
        if (max_iou >= threshold_) { 
            if (!records[idx]) {
              records[idx] = true;
              top_data[index + pred_count * 5 + 3] = 1;
              top_data[index + pred_count * 5 + 4] = 0;
            } else {
              top_data[index + pred_count * 5 + 3] = 0;
              top_data[index + pred_count * 5 + 4] = 1;
            }
        }
        else {
	      top_data[index + pred_count * 5 + 3] = 0;
	      top_data[index + pred_count * 5 + 4] = 1;
	}
        ++pred_count;
      }
    }
    delete [] class_count;
  }
  //for (int b = 0; b < top[0]->num(); ++b)
  //  for (int h = 0; h < top[0]->height(); ++h)
  //    for (int w = 0; w < top[0]->width(); ++w)
  //      for (int c = 0; c < top[0]->channels(); ++c)
  //{
  //	  LOG(INFO)<<"top-data:"<<top[0]->data_at(b,c,h,w);
  //	}  
}

INSTANTIATE_CLASS(YoloEvalDetectionLayer);
REGISTER_LAYER_CLASS(YoloEvalDetection);

}  // namespace caffe
