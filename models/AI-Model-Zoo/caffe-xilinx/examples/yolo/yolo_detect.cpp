// This is a demo code for using a Yolo model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include "caffe/transformers/yolo_transformer.hpp"
#include <string>
#include <iostream>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

float sigmoid(float p){
  return 1.0 / (1 + exp(-p * 1.0));
}

float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0,x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0,x2 + w2 / 2.0);
  return right - left;
}

float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w<0 || h<0)
    return 0;
  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

vector<string> InitLabelfile(const string& labels) {
  vector<string> new_labels;
  if(labels != "") {
    size_t start = 0;
    size_t index = labels.find(",", start);
    while (index != std::string::npos) {
      new_labels.push_back(labels.substr(start, index - start));
      start = index + 1;
      index = labels.find(",", start); 
     }
      new_labels.push_back(labels.substr(start));
  }
  else {
    new_labels = {"aeroplane","bicycle","bird","boat","bottle",
                   "bus","car","cat","chair","cow","diningtable",
                   "dog","horse","motorbike","person","pottedplant",
                   "sheep","sofa","train","tvmonitor"};
  }
  //for (auto it = 0; it < new_labels.size(); it ++) {
  //   LOG(INFO) << "new_labels List:" << new_labels[it] ;
  //}
  return new_labels;
}


vector<float> InitBiases(const string& biases) {
  vector<float> new_biases;
  if(biases != "") {
    size_t start = 0;
    size_t index = biases.find(",", start);
    while (index != std::string::npos) {
      new_biases.push_back(atof(biases.substr(start, index - start).c_str()));
      start = index + 1;
      index = biases.find(",", start); 
     }
      new_biases.push_back(atof(biases.substr(start).c_str()));
  }
  else {
    new_biases = {10,13,16,30,33,23, 30,61,62,45,59,119, 116,90,156,198,373,326};
  }
  //for (auto it = 0; it < new_biases.size(); it ++) {
  //   LOG(INFO) << "new_biases List:" << new_biases[it] ;
  //}
  return new_biases;
}

vector<vector<float>> apply_nms(vector<vector<float>>boxes , int classes ,float thres) {
  vector<pair<int, float>> order(boxes.size());
  vector<vector<float>> result;
  for( int k = 0; k <classes; k++){
    for(size_t i = 0; i < boxes.size(); ++i){
      order[i].first = i;
      boxes[i][4] = k;
      order[i].second = boxes[i][6+k];
    }
    sort(order.begin(), order.end(),[](const pair<int, float>& ls, const pair<int, float>& rs) {return ls.second > rs.second;});
    vector<bool> exist_box(boxes.size(), true);
    for(size_t _i = 0; _i < boxes.size(); ++_i){
      size_t i = order[_i].first;
      if(!exist_box[i]) continue;
      if(boxes[i][6 + k] < 0.005) {
        exist_box[i] = false;
        continue;
      }
      result.push_back(boxes[i]);
      for(size_t _j = _i+1; _j < boxes.size(); ++_j){
        size_t j = order[_j].first;
        if(!exist_box[j]) continue;
        float ovr = cal_iou(boxes[j], boxes[i]);
        if(ovr >= thres) exist_box[j] = false;
      }
    }
  }
  return result;
}

vector<vector<float>> correct_region_boxes(
    vector<vector<float>> boxes, int n, int w, int h, int netw, int neth) {
  int new_w = 0;
  int new_h = 0;
  if (((float)netw / w) < ((float)neth / h)){
    new_w = netw;
    new_h = (h * netw) / w;
  } else {
    new_w = (w * neth) / h;
    new_h = neth;
  }
  for(int i =0; i < boxes.size() ; i++) {
    boxes[i][0] = (boxes[i][0] - (netw - new_w) / 2.0 / netw) / ((float)new_w / netw);
    boxes[i][1] = (boxes[i][1] - (neth - new_h) / 2.0 / neth) / ((float)new_h / neth);
    boxes[i][2] *= (float)netw / new_w;
    boxes[i][3] *= (float)neth / new_h;
  }
  return boxes;
}

class Detector {
public:
  Detector(const string& model_file, const string& weights_file);

  vector<vector<float>> Detectv3(string file, int classes, vector<float> biases ,int anchorCnt, int w, int h);
  vector<vector<float>> Detectv2(string file, int classes, vector<float> biases ,int anchorCnt, int w, int h);
private:
  boost::shared_ptr<Net<float>> net_;
  cv::Size input_geometry_;
  int num_channels_;
};

Detector::Detector(const string& model_file, const string& weights_file){
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

vector<vector<float> > Detector::Detectv2(string file ,int classes ,vector<float>biases, int anchorCnt, int img_width , int img_height) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  image img = load_image_yolo(file.c_str(), input_layer->width(), input_layer->height(), input_layer->channels());
  input_layer->set_cpu_data(img.data);
  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  int height = net_->output_blobs()[0]->height();
  int width = net_->output_blobs()[0]->width();
  float swap[height*width][anchorCnt][classes + 5];
  for(int h =0; h < height; h++)
   for(int w =0; w < width; w++)
    for(int c =0 ; c < anchorCnt*(classes + 5);c++)
     swap[h * width + w][c / (classes + 5)][c % (classes + 5)] = result[c * height * width + h * width + w];
   vector<vector<float> > boxes;
   for(int h =0; h < height; h++){
     for(int w =0; w < width; w++){
      for(int n =0 ; n < anchorCnt; n++){
       vector<float>box , cls;
       float s = 0.0;
       float x = (w + sigmoid(swap[h * width + w][n][0])) / static_cast<float>(width);
       float y = (h + sigmoid(swap[h * width + w][n][1])) / static_cast<float>(height);
       float ww = (exp(swap[h * width + w][n][2])* biases[2 * n]) / static_cast<float>(width);
       float hh = (exp(swap[h * width + w][n][3])* biases[2 * n+1]) / static_cast<float>(height);
       float obj_score = sigmoid(swap[h * width + w][n][4]);
       for(int p =0; p < classes; p++)
        cls.push_back(swap[h * width + w][n][5 + p]);
      float large = *max_element(cls.begin(),cls.end());
      for(int p =0; p <cls.size() ;p++){
        cls[p] = exp(cls[p] -large);
        s += cls[p];
      }
      vector<float>::iterator biggest = max_element(cls.begin(),cls.end());
      large = * biggest;
      for(int p =0; p < cls.size() ; p++)
        cls[p] = cls[p] / s;
      box.push_back(x);
      box.push_back(y);
      box.push_back(ww);
      box.push_back(hh);
      box.push_back(-1);
      box.push_back(obj_score);
      for (int p =0; p < cls.size(); p ++)
        {box.push_back(obj_score * cls[p]);
      }
      boxes.push_back(box);
    }
  }
}
boxes = correct_region_boxes(boxes, boxes.size() ,img_width ,img_height ,input_geometry_.width ,input_geometry_.height);
vector<vector<float> > res = apply_nms(boxes ,classes ,0.45);
return res;
}

vector<vector<float>> Detector::Detectv3(string file, int classes, vector<float> biases ,int anchorCnt, int img_width, int img_height) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  image img = load_image_yolo(file.c_str(), input_layer->width(), input_layer->height(), input_layer->channels());
  input_layer->set_cpu_data(img.data);
  net_->Forward();
  /* Copy the output layer to a std::vector */
  vector<vector<float>> boxes;
  vector<int> scale_feature;
  for (int iter = 0; iter < net_->num_outputs(); iter++){
    int width = net_->output_blobs()[iter]->width();
    scale_feature.push_back(width);
  }
  sort(scale_feature.begin(), scale_feature.end(), [](int a, int b){return a > b;});
  for (int iter = 0; iter < net_->num_outputs(); iter++) {
    Blob<float>* result_blob = net_->output_blobs()[iter];
    const float* result = result_blob->cpu_data();
    int height = net_->output_blobs()[iter]->height();
    int width = net_->output_blobs()[iter]->width();
    int channles = net_->output_blobs()[iter]->channels();
    int index = 0;
    for (int i = 0; i < net_->num_outputs(); i++) {
      if(width == scale_feature[i]) {
        index = i;
        break;
      }
    }
    // swap the network's result to a new sequence
    float swap[width*height][anchorCnt][5+classes];
    for(int h =0; h < height; h++)
      for(int w =0; w < width; w++)
        for(int c =0 ; c < channles; c++)
          swap[h * width + w][c / (5 + classes)][c % (5 + classes)] = result[c * height * width + h * width + w];

    // compute the coordinate of each primary box
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int n = 0 ; n < anchorCnt; n++) {
          float obj_score = sigmoid(swap[h * width + w][n][4]);
          if(obj_score < 0.005)
            continue;
          vector<float>box,cls;
          float x = (w + sigmoid(swap[h * width + w][n][0])) / width;
          float y = (h + sigmoid(swap[h * width + w][n][1])) / height;
          float ww = exp(swap[h * width + w][n][2]) * biases[2 * n + 2 * anchorCnt * index] / input_geometry_.width;
          float hh = exp(swap[h * width + w][n][3]) * biases[2 * n + 2 * anchorCnt * index + 1] / input_geometry_.height;
          box.push_back(x);
          box.push_back(y);
          box.push_back(ww);
          box.push_back(hh);
          box.push_back(-1);
          box.push_back(obj_score);
          for(int p =0; p < classes; p++)
            box.push_back(obj_score * sigmoid(swap[h * width + w][n][5 + p]));
          boxes.push_back(box);
        }
      }
    }
  }
  boxes = correct_region_boxes(boxes,boxes.size(), img_width,img_height, input_geometry_.width, input_geometry_.height);
  vector<vector<float>> res = apply_nms(boxes, classes, 0.45);
  return res;
}


DEFINE_string(file_type, "image",
  "Default support for image");
DEFINE_string(out_file, "",
  "If provided, store the detection results in the out_file.");
DEFINE_string(labels, "",
  "If not provided, the defult support for VOC.");
DEFINE_string(biases, "",
  "If not provided, the defult support for VOC.");
DEFINE_string(model_type, "yolov3",
  "If not provided, the defult support yolov3 model.model-type for detection {yolov2,yolov3}.");
DEFINE_double(confidence_threshold, 0.005,
  "If not provided, the threshold for testing mAP is used by default.");
DEFINE_int32(classes, 80,
  "If not provided, the category of the COCO dataset is used by default.");
DEFINE_int32(anchorCnt,3,
  "If not provided, the number of anchors using yolov3 is used by default.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using Yolo mode.\n"
    "Usage:\n"
    "   yolo_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/yolo/yolo_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& file_type = FLAGS_file_type;
  const string& output_filename = FLAGS_out_file;
  const string& model_type = FLAGS_model_type;
  const float confidence  = FLAGS_confidence_threshold;
  const int classes  = FLAGS_classes;
  const int anchorCnt  = FLAGS_anchorCnt;
  ofstream fout;
  fout.open(output_filename.c_str());
  // Initialize the network.
  Detector detector(model_file, weights_file);

  // Initialize the label,biases.
  vector<string> labels = InitLabelfile(FLAGS_labels); 
  vector<float> biases = InitBiases(FLAGS_biases); 
  // Process image one by one.
  ifstream infile(argv[3]);
  string file;
  while (infile >> file) {
    if (file_type == "image") {
      // CHECK(img) << "Unable to decode image " << file;
      cv::Mat img = cv::imread(file, -1);
      int w = img.cols;
      int h = img.rows;
      vector<vector<float>> results;
      if(model_type == "yolov3")
        results = detector.Detectv3(file, classes,biases, anchorCnt, w, h);
      else
        results = detector.Detectv2(file, classes,biases, anchorCnt, w, h);
      for(int i = 0; i < results.size(); i++) {
        float xmin = (results[i][0] - results[i][2] / 2.0) * w + 1;
        float ymin = (results[i][1] - results[i][3] / 2.0) * h + 1;
        float xmax = (results[i][0] + results[i][2] / 2.0) * w + 1;
        float ymax = (results[i][1] + results[i][3] / 2.0) * h + 1;
        if (xmin < 0) xmin = 1;
        if (ymin < 0) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;
        if (results[i][results[i][4] + 6] > confidence) {
          fout << file <<" "<<labels[results[i][4]]<<" " << results[i][6 + results[i][4]] << " " << xmin << " "
               << ymin << " " << xmax << " " << ymax << endl;
          cv::rectangle(img,cvPoint(xmin,ymin), cvPoint(xmax,ymax), cv::Scalar(255,0,255), 2, 1, 0);
          LOG(INFO) << file << " " << labels[results[i][4]] << " " << results[i][6 + results[i][4]]
                    << " " << xmin << " " << ymin << " " << xmax << " " << ymax << endl;
         }
      }
      cv::imwrite("detection.jpg", img);
    } else {
      LOG(FATAL) << "Unknown file_type: " << file_type;
    }
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV

