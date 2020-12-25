// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define pi 3.1415926

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

class Seg_Detector {
 public:
  Seg_Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Seg_Detect(const cv::Mat& img, cv::Mat& segmap);
 
  std::vector<vector<float> > NMS(const vector<vector<float>> detections ,float nms_thrshold);
 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

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
    new_labels = {"background","aeroplane","bicycle","bird","boat","bottle",
                   "bus","car","cat","chair","cow","diningtable",
                   "dog","horse","motorbike","person","pottedplant",
                   "sheep","sofa","train","tvmonitor"};
  }
  //for (auto it = 0; it < new_labels.size(); it ++) {
  //   LOG(INFO) << "new_labels List:" << new_labels[it] ;
  //}
  return new_labels;
}

Seg_Detector::Seg_Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 2) << "Network should have exactly two outputs.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Seg_Detector::Seg_Detect(const cv::Mat& img, cv::Mat& segmap) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  Blob<float>* seg_blob = net_->output_blobs()[1];
  
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  
  cv::resize(segmap, segmap, cv::Size(input_geometry_.width, input_geometry_.height), 0, 0, cv::INTER_NEAREST); 
  
  for (int h = 0; h < seg_blob->shape(2); h++) {
      for (int w = 0; w < seg_blob->shape(3); w++) {
          float max = seg_blob->data_at(0, 0, h, w);
          int idx = 0;
          for (int c = 1; c < seg_blob->shape(1); c++) {
              if (seg_blob->data_at(0, c, h, w) > max) {
                  max = seg_blob->data_at(0, c, h, w);
                  idx = c;
              }
          }
          segmap.at<int>(h, w) = idx;
          //LOG(INFO) << idx;

      }
  }

  cv::resize(segmap, segmap, cv::Size(img.cols, img.rows), 0, 0, cv::INTER_NEAREST); 
  
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 9;
      continue;
    }
    vector<float> detection(result, result + 9);
    detections.push_back(detection);
    result += 9;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Seg_Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Seg_Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Seg_Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(labels, "",
  "If not provided, the defult support for VOC.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& file_type = FLAGS_file_type;
  const string& out_file = FLAGS_out_file;
  const float confidence_threshold = FLAGS_confidence_threshold;
  vector<string> labels = InitLabelfile(FLAGS_labels);
  int count = 1;
  int colors[19][3]= {{128,  64, 128},{244,  35, 232},{ 70,  70,  70},{102, 102, 156},
                      {190, 153, 153},{153, 153, 153},{250, 170,  30},{220, 220,   0},
                      {107, 142,  35},{152, 251, 152},{  0, 130, 180},{220,  20,  60},
                      {255,   0,   0},{  0,   0, 142},{  0,   0,  70},{  0,  60, 100},
                      {  0,  80, 100},{  0,   0, 230},{119,  11,  32}};
                      
  // Initialize the network.
  Seg_Detector detector(model_file, weights_file, mean_file, mean_value);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);

  // Process image one by one.
  std::ifstream infile(argv[3]);
  std::string file;
  while (infile >> file) {
    if (file_type == "image") {
      cv::Mat img = cv::imread(file, -1);
      cv::Mat segmap(img.rows, img.cols, CV_32S);
      CHECK(!img.empty()) << "Unable to decode image " << file;
      std::vector<vector<float> > detections = detector.Seg_Detect(img, segmap);
      cv::imwrite("./seg_result/"+file,segmap);
      /* Print the detection results. */
      
      for (int i = 0; i < img.cols; i++ ) {
        for (int j = 0; j < img.rows; j++ ) {
          for (int c = 0; c < 3; c++ )
            img.at<cv::Vec3b>(j,i)[c] = int(img.at<cv::Vec3b>(j,i)[c] * 0.4) + int(colors[segmap.at<int>(j,i)][c] * 0.6); 
        }
      }
      
      for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax , angle_cos , angle_sin].
        CHECK_EQ(d.size(), 9);
        const float score = d[2];
        if (score >= confidence_threshold && d[7] < 1.5 && d[8] < 1.5) {
          out << file << " ";
          out << labels[static_cast<int>(d[1])] << " ";
          out << score << " ";
          out << static_cast<int>(d[3] * img.cols) << " ";
          out << static_cast<int>(d[4] * img.rows) << " ";
          out << static_cast<int>(d[5] * img.cols) << " ";
          out << static_cast<int>(d[6] * img.rows) << " ";
          out << static_cast<int>(atan2(d[8],d[7]) * 180.0 /pi) << std::endl;
	  cv::rectangle(img,cvPoint(d[3]*img.cols,d[4]*img.rows),cvPoint(d[5]*img.cols,d[6]*img.rows),cv::Scalar(255,0,255),2,1,0);
	  cv::putText(img,labels[d[1]],cvPoint(d[3]*img.cols,d[4]*img.rows+4),CV_FONT_HERSHEY_DUPLEX,0.4,cv::Scalar(255,255,255),1,1);
          cv::putText(img,std::to_string(score),cvPoint(d[3]*img.cols+8,d[4]*img.rows-10),CV_FONT_HERSHEY_DUPLEX,0.4,cv::Scalar(255,255,255),1,1);
	  }
       }
     cv::imwrite(std::to_string(count)+".jpg",img);
    } else if (file_type == "video") {
        cv::VideoCapture cap(file);
	double fps = cap.get(CV_CAP_PROP_FPS);
	int width1=static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
	int height1=static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	cv::VideoWriter wrVideo;
        wrVideo.open("output.avi",CV_FOURCC('D','I','V','X'),fps,cv::Size(width1,height1),true);
	  if (!cap.isOpened()) {
             LOG(FATAL) << "Failed to open video: " << file;
          }
        cv::Mat img;
        int frame_count = 0;
        while (true) {
           bool success = cap.read(img);
           if (!success) {
               LOG(INFO) << "Process " << frame_count << " frames from " << file;
             break;
           }
           CHECK(!img.empty()) << "Error when read frame";
           cv::Mat segmap(img.rows, img.cols, CV_32S);
           std::vector<vector<float> > detections = detector.Seg_Detect(img, segmap);
           /* Print the detection results. */
        
           for (int i = 0; i < img.cols; i++ ) {
             for (int j = 0; j < img.rows; j++ ) {
               for (int c = 0; c < 3; c++ )
                  img.at<cv::Vec3b>(j,i)[c] = int(img.at<cv::Vec3b>(j,i)[c] * 0.4) + int(colors[segmap.at<int>(j,i)][c] * 0.6); 
               }  
             }
           for (int i = 0; i < detections.size(); ++i) {
             const vector<float>& d = detections[i];
             // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax , angle_cos , angle_sin].
             CHECK_EQ(d.size(), 9);
             const float score = d[2];
          
           if (score >= confidence_threshold) {
             out << file << "_";
             out << std::setfill('0') << std::setw(6) << frame_count << " ";
             out << labels[static_cast<int>(d[1])] << " ";
             out << score << " ";
             out << static_cast<int>(d[3] * img.cols) << " ";
             out << static_cast<int>(d[4] * img.rows) << " ";
             out << static_cast<int>(d[5] * img.cols) << " ";
             out << static_cast<int>(d[6] * img.rows) << " ";
             out << static_cast<int>(atan2(d[8],d[7]) * 180.0 /pi) << std::endl;
  	     cv::rectangle(img,cvPoint(d[3]*img.cols,d[4]*img.rows),cvPoint(d[5]*img.cols,d[6]*img.rows),cv::Scalar(255,0,255),2,1,0);
  	     cv::putText(img,labels[d[1]],cvPoint(d[3]*img.cols,d[4]*img.rows+4),CV_FONT_HERSHEY_DUPLEX,0.4,cv::Scalar(255,255,255),1,1);
  	     cv::putText(img,std::to_string(score),cvPoint(d[3]*img.cols+8,d[4]*img.rows-10),CV_FONT_HERSHEY_DUPLEX,0.4,cv::Scalar(255,255,255),1,1);
              }
          }
        wrVideo << img;
        ++frame_count;
      }
      if (cap.isOpened()) {
        cap.release();
      }
	  wrVideo.release();
    } else {
      LOG(FATAL) << "Unknown file_type: " << file_type;
    }
  count++;
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
