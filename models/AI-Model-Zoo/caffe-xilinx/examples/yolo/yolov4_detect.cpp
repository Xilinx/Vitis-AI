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
#include<unistd.h>
#include <sys/stat.h>
#include<sstream>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

string convert_str(float Num)
{
  ostringstream oss;
  oss<<Num;
  string str(oss.str());
  return str;
}

float sigmoid(float p){
  return 1.0 / (1 + exp(-p * 1.0));
}

float overlap(float x1, float w1, float x2, float w2) {
  float left = max(x1 - w1 / 2.0,x2 - w2 / 2.0);
  float right = min(x1 + w1 / 2.0,x2 + w2 / 2.0);
  return right - left;
}

float box_c(vector<float> a, vector<float> b) {
  float top = min(a[1] - a[3] / 2.0, b[1] - b[3] / 2.0);
  float bot = max(a[1] + a[3] / 2.0, b[1] + b[3] / 2.0);
  float left = min(a[0] - a[2] / 2.0, b[0] - b[2] / 2.0);
  float right = max(a[0] + a[2] / 2.0, b[0] + b[2] / 2.0);
 
  float res = (bot-top)*(bot-top) + (right-left)*(right-left);
  return res;
}

/*
float cal_iou(vector<float> box, vector<float> truth) {
  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w<0 || h<0)
    return 0;
  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}
*/
float cal_iou(vector<float> box, vector<float> truth) {
  float c = box_c(box, truth);

  float w = overlap(box[0], box[2], truth[0], truth[2]);
  float h = overlap(box[1], box[3], truth[1], truth[3]);
  if (w<0 || h<0)
    return 0;
  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  float iou = inter_area * 1.0 / union_area;
  if (c==0)
    return iou;
  
  float d = (box[0] - truth[0]) * (box[0] - truth[0]) + (box[1] - truth[1]) * (box[1] - truth[1]);
  float u = pow(d / c, 0.6);
  return iou - u;
}

vector<string> InitLabelfile(const string& labels, string dataset="COCO") {
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
  else if(dataset == "VOC") {
    new_labels = {"aeroplane","bicycle","bird","boat","bottle",
                   "bus","car","cat","chair","cow","diningtable",
                   "dog","horse","motorbike","person","pottedplant",
                   "sheep","sofa","train","tvmonitor"};
  }
  else if(dataset.find("COCO") != string::npos) {
    new_labels = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",  
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",  
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",  
                  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",  
                  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
                  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",  
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",  
                  "sofa", "pottedplant","bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",  
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",  
                  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}; 
  } 
  else {
    LOG(FATAL) << "Unknown dataset: " << dataset;
  } 
  //for (auto it = 0; it < new_labels.size(); it ++) {
  //   LOG(INFO) << "new_labels List:" << new_labels[it] ;
  //}
  return new_labels;
}

int imagename_to_id(string imagename) {
  int idx1 = imagename.find_last_of('.'); 
  int idx2 = imagename.find_last_of('_'); 
  string id = imagename.substr(idx2+1, idx1-idx2-1); 
  int image_id = atoi(id.c_str()); 
  return image_id;
}

string get_imagename(string imagename) {
  int idx1 = imagename.find_last_of('.'); 
  string image_name = imagename.substr(idx1+1); 
  return image_name;
}

int imagename_to_id_2017(string cfgfile)
{
    int idx1 = cfgfile.find_last_of('.');
    int idx2 = cfgfile.find_last_of('/');
    
    string id = cfgfile.substr(idx2+1, idx1-idx2-1); 
    int image_id = atoi(id.c_str());
    return image_id; 
}

vector<int> coco_id_map_dict() {
  vector<int> category_ids;
  category_ids = {1,2,3,4,5,6,7,8,9,10, 
                  11,13,14,15,16,17,18,19,20,21, 
                  22,23,24,25,27,28,31,32,33,34, 
                  35,36,37,38,39,40,41,42,43,44, 
                  46,47,48,49,50,51,52,53,54,55, 
                  56,57,58,59,60,61,62,63,64,65, 
                  67,70,72,73,74,75,76,77,78,79, 
                  80,81,82,84,85,86,87,88,89,90}; 
  return category_ids;
}

vector<float> InitBiases(const string& biases, string dataset="COCO", string model_type="yolov4") {
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
  else if(dataset.find("COCO") != string::npos) {
    if (model_type == "yolov4-tiny")
        new_biases = {23,27,  37,58,  81,82, 81, 82, 135,169, 334, 319};
    else
        new_biases = {12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401};
  }
  else {
    LOG(FATAL) << "Unknown datatset: " << dataset;
  } 
  //for (auto it = 0; it < new_biases.size(); it ++) {
  //  cout << " " << new_biases[it] ;
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
      if(boxes[i][6 + k] <= 0.001) { //<
        exist_box[i] = false;
        continue;
      }
      result.push_back(boxes[i]);
      for(size_t _j = _i+1; _j < boxes.size(); ++_j){
        size_t j = order[_j].first;
        if(!exist_box[j]) continue;
        float ovr = cal_iou(boxes[j], boxes[i]);
        if(ovr > thres) exist_box[j] = false; //>=
      }
    }
  }
  return result;
}



vector<vector<float>> correct_region_boxes(
    vector<vector<float>> boxes, int n, int w, int h, int netw, int neth, int letter) {
  int new_w = 0;
  int new_h = 0;
  if (letter) {
    if (((float)netw / w) < ((float)neth / h)){
      new_w = netw;
      new_h = (h * netw) / w;
    } else {
      new_w = (w * neth) / h;
      new_h = neth;
    }
  } else {
    new_w = netw; 
    new_h = neth;
  }
  for(int i =0; i < boxes.size() ; i++) {
    boxes[i][0] = (boxes[i][0] - (netw - new_w) / 2.0 / netw) / ((float)new_w / netw);
    boxes[i][1] = (boxes[i][1] - (neth - new_h) / 2.0 / neth) / ((float)new_h / neth);
    boxes[i][2] *= (float)netw / new_w;
    boxes[i][3] *= (float)neth / new_h;
   
    // NOTE: !relative
    boxes[i][0] *= w;
    boxes[i][1] *= h;
    boxes[i][2] *= w;
    boxes[i][3] *= h;
  }
 
 
  return boxes;
}

vector<vector<float>> correct_region_boxes(
    vector<vector<float>> boxes, int n, int w, int h, int netw, int neth) {
    return correct_region_boxes(boxes, n, w, h, netw, neth, 1);
}

class Detector {
public:
  Detector(const string& model_file, const string& weights_file);

  vector<vector<float>> Detectv4(string file, int classes, vector<float> biases ,int anchorCnt, int w, int h, int letterbox, float conf_threshold, float nms_threshold);
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
  Caffe::SetDevice(1);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

vector<vector<float>> Detector::Detectv4(string file, int classes, vector<float> biases ,int anchorCnt, int img_width, int img_height, int letterbox, float conf_threshold, float nms_threshold) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  image img = load_image_yolov4(file.c_str(), input_layer->width(), input_layer->height(), input_layer->channels(), letterbox);

  input_layer->set_cpu_data(img.data);
  net_->Forward();
  free_image(img);
  /* Copy the output layer to a std::vector */
  vector<vector<float>> boxes;
  vector<int> scale_feature;
  for (int iter = 0; iter < net_->num_outputs(); iter++){
    int width = net_->output_blobs()[iter]->width();
    
    int channels = net_->output_blobs()[iter]->channels();
    if (channels != 255)
        continue;

    scale_feature.push_back(width);
  }
  sort(scale_feature.begin(), scale_feature.end(), [](int a, int b){return a > b;});
  
  
  for (int iter = 0; iter < net_->num_outputs(); iter++) {
    Blob<float>* result_blob = net_->output_blobs()[iter];
    const float* result = result_blob->cpu_data();
    int height = net_->output_blobs()[iter]->height();
    int width = net_->output_blobs()[iter]->width();
    int channels = net_->output_blobs()[iter]->channels();
    
    if (channels != 255)
        continue;
   
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
        for(int c =0 ; c < channels; c++)
          swap[h * width + w][c / (5 + classes)][c % (5 + classes)] = result[c * height * width + h * width + w];
    
    // compute the coordinate of each primary box
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int n = 0 ; n < anchorCnt; n++) {
          float obj_score = sigmoid(swap[h * width + w][n][4]);
          if(obj_score > conf_threshold) {
            vector<float> box,cls;
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
  } 
  boxes = correct_region_boxes(boxes,boxes.size(), img_width,img_height, input_geometry_.width, input_geometry_.height, letterbox);
  vector<vector<float>> res = apply_nms(boxes, classes, nms_threshold);
  return res;
}



DEFINE_string(mode, "eval",
  "Default support for eval mode {eval, demo}");
DEFINE_string(file_type, "image",
  "Default support for image");
DEFINE_string(out_file, "out_file.txt",
  "If provided, store the detection results in the out_file.");
DEFINE_string(labels, "",
  "If not provided, the defult support for COCO.");
DEFINE_string(dataset, "COCO_2014",
  "If not provided, the defult support for COCO {COCO_2014, COCO_2017, VOC}.");
DEFINE_string(biases, "",
  "If not provided, the defult support for COCO.");
DEFINE_string(model_type, "yolov4",
  "If not provided, the defult support yolov4 model.model-type for detection {yolov4}.");
DEFINE_double(confidence_threshold, 0.001,
  "If not provided, the threshold for testing mAP is used by default.");
DEFINE_double(nms_threshold, 0.45,
  "If not provided, the nms threshold for testing mAP is used by default.");
DEFINE_int32(classes, 80,
  "If not provided, the category of the COCO dataset is used by default.");
DEFINE_int32(anchorCnt,3,
  "If not provided, the number of anchors using yolov4 is used by default.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using Yolo mode.\n"
    "Usage:\n"
    "   yolo_detect [FLAGS] model_file weights_file image_root list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 5) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/yolo/yolo_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mode = FLAGS_mode;
  const string& file_type = FLAGS_file_type;
  const string& output_filename = FLAGS_out_file;
  const string& model_type = FLAGS_model_type;
  const float conf_threshold  = FLAGS_confidence_threshold;
  const float nms_threshold  = FLAGS_nms_threshold;
  const int classes  = FLAGS_classes;
  const int anchorCnt  = FLAGS_anchorCnt;
  vector<int> ccoco_id_map_dict;
  ofstream fout;
  fout.open(output_filename.c_str());
  // Initialize the network.
  Detector detector(model_file, weights_file);
   
  // Initialize the label,biases.
  vector<string> labels = InitLabelfile(FLAGS_labels, FLAGS_dataset); 
  vector<float> biases = InitBiases(FLAGS_biases, FLAGS_dataset, FLAGS_model_type); 
  // Process image one by one.
  const string& image_root = argv[3];
  ifstream infile(argv[4]);
  string file;
  if (FLAGS_dataset.find("COCO") != string::npos && mode == "eval") {
    fout << "[" << endl;
    ccoco_id_map_dict = coco_id_map_dict();
  }
  if (mode == "demo") {
    if (0 != access("demo", 0))
    {
      if (0 != mkdir("demo", S_IRWXU | S_IRWXG))
      {
         
         LOG(FATAL) << "Fail to create demo dir!";
      }
    }
  } 

  while (infile >> file) {
    if (file_type == "image") {
      // CHECK(img) << "Unable to decode image " << file;
      cv::Mat img = cv::imread(image_root + "/" + file, -1);
      LOG(INFO) << "inference " << file << endl;
      int w = img.cols;
      int h = img.rows;
      vector<vector<float>> results;
      if (model_type == "yolov4" || model_type == "yolov4-tiny")
        results = detector.Detectv4(image_root + '/' + file, classes,biases, anchorCnt, w, h, 0, conf_threshold, nms_threshold);
      else
        LOG(FATAL) << "Unknown model_type: " << model_type;
        
      for(int i = 0; i < results.size(); i++) {
        // NOTE: !relative
        float xmin = results[i][0] - results[i][2] / 2.0;
        float ymin = results[i][1] - results[i][3] / 2.0;
        float xmax = results[i][0] + results[i][2] / 2.0;
        float ymax = results[i][1] + results[i][3] / 2.0;
 
        /*
        float xmin = (results[i][0] - results[i][2] / 2.0) * w;
        float ymin = (results[i][1] - results[i][3] / 2.0) * h;
        float xmax = (results[i][0] + results[i][2] / 2.0) * w;
        float ymax = (results[i][1] + results[i][3] / 2.0) * h;
        */
        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;
        
        if (results[i][results[i][4] + 6] > conf_threshold) {
        // if (results[i][results[i][4] + 6] > 0) {
          if (mode == "eval") {
            if (FLAGS_dataset.find("COCO") != string::npos) {
              int image_id;
              if (FLAGS_dataset == "COCO_2014")
                // for val2014
                image_id = imagename_to_id(file);
              else if (FLAGS_dataset == "COCO_2017")
                // for val2017
                image_id = imagename_to_id_2017(file);
              else
                LOG(FATAL) << "Unknown dataset: " << FLAGS_dataset;

              // vector<int> ccoco_id_map_dict = coco_id_map_dict();
              // LOG(INFO) << "class: " << labels[results[i][4]] << " id: " << results[i][4] << " coco_id: " << ccoco_id_map_dict[int(results[i][4])]  <<endl;
              fout << fixed << setprecision(0) <<"{\"image_id\":" << image_id << ", \"category_id\":" << ccoco_id_map_dict[int(results[i][4])]<< ", \"bbox\":[" << fixed << 
              setprecision(6) << xmin << ", " << ymin << ", " << xmax-xmin << ", " << ymax-ymin << "], \"score\":" << results[i][6+results[i][4]]  
              << "}," << endl; 
            }   
            else {
              fout << file <<" "<<labels[results[i][4]]<<" " << results[i][6 + results[i][4]] << " " << xmin << " "
                   << ymin << " " << xmax << " " << ymax << endl;
            }
          }
          else if (mode == "demo") { 
            
            cv::rectangle(img,cvPoint(xmin,ymin), cvPoint(xmax,ymax), cv::Scalar(255,0,255), 2, 1, 0);
            cv::putText(img, labels[int(results[i][4])], cvPoint(xmin,ymin), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 0.5, 4, 0);
            cv::putText(img, convert_str(results[i][6+results[i][4]]), cvPoint(xmin,ymin+15), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 0.5, 4, 0);
          }
          else {
            LOG(FATAL) << "Unknown mode: " << mode;
          }  
          // LOG(INFO) << file << " " << labels[results[i][4]] << " " << results[i][6 + results[i][4]]
          //          << " " << xmin << " " << ymin << " " << xmax << " " << ymax << endl;
        }
      }
      if (mode == "demo") {        
        cv::imwrite("demo/" + file, img);
      }

      results.clear();
    } else {
      LOG(FATAL) << "Unknown file_type: " << file_type;
    }
  }
  if (FLAGS_dataset.find("COCO") != string::npos) {
    fout.seekp(-2L, ios::end); 
    fout << endl << "]" << endl; 
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV

