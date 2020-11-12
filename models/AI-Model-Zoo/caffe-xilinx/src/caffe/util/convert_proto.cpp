#include "caffe/util/convert_proto.hpp"
#include "caffe/util/insert_splits.hpp"
#include <tuple>

using namespace std;

namespace caffe {

NetParameter RemoveLayerByType(const NetParameter &model_in,
                               const string type) {
  NetParameter model_out;
  map<string, string> type_blobs;

  for (size_t i = 0; i < model_in.layer_size(); i++) {
    auto &cur_layer = model_in.layer(i);
    if (cur_layer.type() == type) {
      CHECK(!(cur_layer.bottom_size() > 1 && cur_layer.top_size() > 1))
          << "Remove Layer Error: Cannot remove layers with multi-bottom and "
             "multi-top: "
          << cur_layer.name();
      DLOG(INFO) << "Remove " << type << " Type Layer: " << cur_layer.name();
      if (cur_layer.bottom_size() == 0 || cur_layer.top_size() == 0) {
        continue;
      }
      for (size_t j = 0; j < model_in.layer(i).top_size(); j++) {
        type_blobs.insert(
            make_pair(model_in.layer(i).top(j), model_in.layer(i).bottom(0)));
      }
    } else {
      LayerParameter *layer = model_out.add_layer();
      *layer = model_in.layer(i);
      for (size_t i = 0; i < layer->bottom_size(); i++) {
        auto res = type_blobs.find(layer->bottom(i));
        if (res != type_blobs.end()) {
          layer->set_bottom(i, res->second);
        }
      }
      for (size_t i = 0; i < layer->top_size(); i++) {
        auto res = type_blobs.find(layer->top(i));
        if (res != type_blobs.end()) {
          layer->set_top(i, res->second);
        }
      }
    }
  }

  return model_out;
}

NetParameter RemoveLayerByPhase(const NetParameter &model_in,
                                caffe::Phase phase) {
  NetParameter model_out;
  string phase_string = phase == caffe::TRAIN ? "TRAIN" : "TEST";
  map<string, string> phase_blobs;

  for (size_t i = 0; i < model_in.layer_size(); i++) {
    auto &cur_layer = model_in.layer(i);

    if (cur_layer.include_size() == 1 &&
        cur_layer.include(0).phase() == phase) {
      CHECK(!(cur_layer.bottom_size() > 1 && cur_layer.top_size() > 1))
          << "Remove Layer Error: Cannot remove layers with multi-bottom and "
             "multi-top: "
          << cur_layer.name();
      DLOG(INFO) << "Remove " << phase_string
                 << " Phase Layer: " << cur_layer.name();
      if (cur_layer.bottom_size() == 0 || cur_layer.top_size() == 0) {
        continue;
      }
      if (cur_layer.bottom_size() > 1) {
        for (size_t j = 0; j < model_in.layer(i).bottom_size(); j++) {
          phase_blobs.insert(
              make_pair(model_in.layer(i).bottom(j), model_in.layer(i).top(0)));
        }
      } else if (cur_layer.top_size() > 1) {
        for (size_t j = 0; j < model_in.layer(i).top_size(); j++) {
          phase_blobs.insert(
              make_pair(model_in.layer(i).top(j), model_in.layer(i).bottom(0)));
        }
      }
    } else {
      LayerParameter *layer = model_out.add_layer();
      *layer = model_in.layer(i);
      for (size_t i = 0; i < layer->bottom_size(); i++) {
        auto res = phase_blobs.find(layer->bottom(i));
        if (res != phase_blobs.end()) {
          layer->set_bottom(i, res->second);
        }
      }
      for (size_t i = 0; i < layer->top_size(); i++) {
        auto res = phase_blobs.find(layer->top(i));
        if (res != phase_blobs.end()) {
          layer->set_top(i, res->second);
        }
      }
    }
  }

  return model_out;
}

NetParameter RemoveLayerByName(const NetParameter &model_in,
                               const string name) {
  NetParameter model_out;
  map<string, string> name_blobs;

  for (size_t i = 0; i < model_in.layer_size(); i++) {
    auto &cur_layer = model_in.layer(i);
    if (cur_layer.name() == name) {
      CHECK(!(cur_layer.bottom_size() > 1 && cur_layer.top_size() > 1))
          << "Remove Layer Error: Cannot remove layers with multi-bottom and "
             "multi-top: "
          << cur_layer.name();
      DLOG(INFO) << "Remove " << name << " Name Layer: " << cur_layer.name();
      if (cur_layer.bottom_size() > 1) {
        for (size_t j = 0; j < model_in.layer(i).bottom_size(); j++) {
          name_blobs.insert(
              make_pair(model_in.layer(i).bottom(j), model_in.layer(i).top(0)));
        }
      } else if (cur_layer.top_size() > 1) {
        for (size_t j = 0; j < model_in.layer(i).top_size(); j++) {
          name_blobs.insert(
              make_pair(model_in.layer(i).top(j), model_in.layer(i).bottom(0)));
        }
      }
    } else {
      LayerParameter *layer = model_out.add_layer();
      *layer = model_in.layer(i);
      for (size_t i = 0; i < layer->bottom_size(); i++) {
        auto res = name_blobs.find(layer->bottom(i));
        if (res != name_blobs.end()) {
          layer->set_bottom(i, res->second);
        }
      }
      for (size_t i = 0; i < layer->top_size(); i++) {
        auto res = name_blobs.find(layer->top(i));
        if (res != name_blobs.end()) {
          layer->set_top(i, res->second);
        }
      }
    }
  }

  return model_out;
}

NetParameter RemoveDuplicatedLayerByType(const NetParameter &model_in,
                                         const string type) {
  NetParameter model_out;
  std::map<string, LayerParameter> layer_map;

  for (size_t i = 0; i < model_in.layer_size(); i++) {
    LayerParameter cur_layer = model_in.layer(i);
    if (cur_layer.type() == type) {
      if (!layer_map.count(cur_layer.name())) {
        cur_layer.clear_include();
        *(model_out.add_layer()) = cur_layer;
        layer_map.insert(make_pair(cur_layer.name(), cur_layer));
      } else {
        LOG(INFO) << "Remove duplicated layer: " << cur_layer.name() << "("
                  << cur_layer.type() << ")";
      }
    } else {
      *(model_out.add_layer()) = cur_layer;
    }
  }

  return model_out;
}

bool CheckImageDataLayer(const LayerParameter *data_layer) {
  CHECK_EQ(data_layer->type(), "ImageData");
  const int new_height = data_layer->image_data_param().new_height();
  const int new_width = data_layer->image_data_param().new_width();
  const bool is_color = data_layer->image_data_param().is_color();
  string root_folder = data_layer->image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
        (new_height > 0 && new_width > 0))
      << " Please set new_height and new_width at the same time in "
         "ImageDataLayer.";
  // Read the file with filenames and labels
  const string &source = data_layer->image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  CHECK(infile.is_open())
      << " Could not open file: " << source
      << ", please check the root_folder and source in ImageDataLayer.";
  vector<std::pair<std::string, int>> lines_;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  int lines_id_ = 0;
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << " Could not load " << lines_[lines_id_].first
                     << ", please check the image file.";

  return true;
}

bool CheckModel(const NetParameter &net_param) {
  bool model_check = true;
  ostringstream ss;
  ss << "\n";
  for (auto i = 0; i < net_param.layer_size(); i++) {
    if (net_param.layer(i).type() == "ImageData") {
      CheckImageDataLayer(&net_param.layer(i));
    } else if (net_param.layer(i).type() == "LRN") {
      ss << "*** Deephi DNNDK could not support LRN Layer, ("
         << net_param.layer(i).name()
         << "), please delete it or replace it with BatchNorm + Scale Layer "
            "and "
            "retrain.\n";
      model_check = false;
    }
  }
  CHECK(model_check) << ss.str();
  return model_check;
}

/*
 *  white_list_entry:
 *  <support, comment>
 *  support:
 *     0. do not support
 *     1. do support
 *     2. will be processed by tool-chain
 *     3. ignore
*/
typedef std::tuple<int, string> white_list_entry;
typedef std::map<string, white_list_entry> white_list;

bool CheckModelAndSave(const NetParameter &net_param, const string filename) {
  bool pass = true;

  white_list wl;
  // DPU do not support
  wl.emplace("LRN",
             make_tuple(0, "Suggest: Replace it with BatchNorm+Scale Layer"));
  wl.emplace("Normalize",
             make_tuple(0, "Suggest: Replace it with BatchNorm+Scale Layer"));
  wl.emplace("Tiling", make_tuple(0, ""));
  wl.emplace("Softmax", make_tuple(0, ""));
  wl.emplace("Upsample", make_tuple(0, ""));
  // DPU do support
  wl.emplace("Convolution", make_tuple(1, ""));
  wl.emplace("Deconvolution", make_tuple(1, ""));
  wl.emplace("InnerProduct", make_tuple(1, ""));
  wl.emplace("Eltwise", make_tuple(1, ""));
  wl.emplace("Pooling", make_tuple(1, ""));
  wl.emplace("ReLU", make_tuple(1, ""));
  wl.emplace("PReLU", make_tuple(1, ""));
  wl.emplace("Concat", make_tuple(1, ""));
  // Will be processed by toolchain
  wl.emplace("BatchNorm", make_tuple(2, "Will be merged into Convolution"));
  wl.emplace("Scale", make_tuple(2, "Will be merged into Convolution"));
  wl.emplace("SoftmaxWithLoss", make_tuple(2, "Will be convert to Softmax"));
  wl.emplace("ImageData", make_tuple(2, "Will be convert to Input"));
  wl.emplace("Data", make_tuple(2, "Will be convert to Input"));
  // Ignore
  wl.emplace("Flatten", make_tuple(3, ""));
  wl.emplace("Permute", make_tuple(3, ""));
  wl.emplace("PriorBox", make_tuple(3, ""));
  wl.emplace("Loss", make_tuple(3, ""));
  wl.emplace("Sigmoid", make_tuple(3, ""));
  wl.emplace("Slice", make_tuple(3, ""));
  wl.emplace("Dropout", make_tuple(3, ""));
  wl.emplace("Accuracy", make_tuple(3, ""));

  ofstream fs;
  fs.open(filename);
  for (auto i = 0; i < net_param.layer_size(); i++) {
    auto it = wl.find(net_param.layer(i).type());
    if (it != wl.end()) {
      auto &entry = it->second;
      auto &support = get<0>(entry);
      auto &comment = get<1>(entry);
      if (support == 0) {
        cout << i << ". " << net_param.layer(i).name() << " ("
             << net_param.layer(i).type() << ") do not support by Deephi DPU. "
             << comment << "\n";
        fs << i << " " << net_param.layer(i).name() << " "
           << net_param.layer(i).type() << " " << support << " \"" << comment
           << "\"\n";
        pass = false;
      } else if (support == 2) {
        // cout << i << ". " << net_param.layer(i).name() << " (" <<
        // net_param.layer(i).type()
        // << ") will be processed by toolchain. "
        // << comment << "\n";
        fs << i << " " << net_param.layer(i).name() << " "
           << net_param.layer(i).type() << " " << support << " \"" << comment
           << "\"\n";
      } else if (support == 3) {
        // cout << i << ". " << net_param.layer(i).name() << " (" <<
        // net_param.layer(i).type()
        // << ") can be ignored in forward. "
        // << comment << "\n";
        fs << i << " " << net_param.layer(i).name() << " "
           << net_param.layer(i).type() << " " << support << " \"" << comment
           << "\"\n";
      }
    } else {
      cout << i << ". " << net_param.layer(i).name() << " ("
           << net_param.layer(i).type()
           << ") is not known by DNNDK and may raise some error, please "
              "concat Deephi.\n";
    }
  }

  fs.close();
  return pass;
}

NetParameter RemoveBlobs(const NetParameter &model_in) {
  NetParameter model_out = model_in;
  for (size_t i = 0; i < model_out.layer_size(); i++) {
    model_out.mutable_layer(i)->clear_blobs();
  }
  return model_out;
}

template <typename Dtype>
void DataStat(const int n, const Dtype *data, const char *title) {
  double max = FLT_MIN, min = FLT_MAX;
  vector<int> count(11, 0);
  for (int i = 0; i < n; ++i) {
    double val = fabs(data[i]);
    min = fabs(val) < fabs(min) ? fabs(val) : min;
    max = fabs(val) > fabs(max) ? fabs(val) : max;
    if (val < 0.03125)
      ++count[0];
    else if (val < 0.0625)
      ++count[1];
    else if (val < 0.125)
      ++count[2];
    else if (val < 0.25)
      ++count[3];
    else if (val < 0.5)
      ++count[4];
    else if (val < 1)
      ++count[5];
    else if (val < 2)
      ++count[6];
    else if (val < 4)
      ++count[7];
    else if (val < 8)
      ++count[8];
    else if (val < 16)
      ++count[9];
    else
      ++count[10];
  }
  printf("%s max/min: %-9g %-9g\n", title, max, min);
  printf("Range    <2^-5    <2^-4    <2^-3    <0.25     <0.5       <1       <2 "
         "      <4       <8      <16     <max\n");
  printf("Count");
  for (int j = 0; j < count.size(); ++j)
    printf(" %8d", count[j]);
  printf("\n\n");
  fflush(stdout);
}

template <typename Dtype>
void BindWeightWithProfile(Blob<float> &new_weight, const Dtype *weight,
                           const Dtype *scale, const Dtype *var, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif
  const int n = new_weight.count();
  const int kernel_dim = new_weight.count(1);
  auto bind_weight = new_weight.mutable_cpu_data();

//#define DEBUG_BN_MERGE
#ifdef DEBUG_BN_MERGE
  DataStat(n, weight, "Weights before merge BN");
#endif // DEBUG_BN_MERGE

  int m = n / kernel_dim;
  vector<double> factor(m, 1.0);
#ifdef DEBUG_BN_MERGE
  vector<double> inv_var(m, 1.0);
#endif // DEBUG_BN_MERGE
  for (int i = 0; i < m; i++) {
#ifdef DEBUG_BN_MERGE
    inv_var[i] = 1.0 / sqrt(var[i] + eps);
#endif // DEBUG_BN_MERGE
    factor[i] = double(scale[i]) / sqrt(double(var[i]) + eps);
  }

#ifdef DEBUG_BN_MERGE
  DataStat(m, scale, "Scale from BN+scale");
  DataStat(m, var, "Variance from BN+scale");
  DataStat<double>(m, &(inv_var[0]), "1/sqrt(var) from BN+scale");
  DataStat<double>(m, &(factor[0]), "Scale/sqrt(var) from BN+scale");
#endif // DEBUG_BN_MERGE

  for (int i = 0; i < n; i++) {
    int c = i / kernel_dim;
    bind_weight[i] = weight[i] * factor[c];
  }

#ifdef DEBUG_BN_MERGE
  DataStat(n, bind_weight, "Weights after merge BN");
#endif // DEBUG_BN_MERGE
}

template <typename Dtype>
void BindWeight(const int n, const int kernel_dim, const Dtype *weight,
                const Dtype *scale, const Dtype *var, Dtype *bind_weight,
                float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif

  int m = n / kernel_dim;
  vector<Dtype> factor(m, 1.0);
  for (int i = 0; i < m; i++)
    factor[i] = scale[i] / sqrt(var[i] + eps);

  for (int i = 0; i < n; i++) {
    int c = i / kernel_dim;
    bind_weight[i] = weight[i] * factor[c];
  }
}

template <typename Dtype>
void BindDeconvWeight(const int n, const int kernel_dim, const int ch_in,
                      const Dtype *weight, const Dtype *scale, const Dtype *var,
                      Dtype *bind_weight, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif

  int m = n / kernel_dim;
  vector<Dtype> factor(m, 1.0);
  for (int i = 0; i < m; i++)
    factor[i] = scale[i] / sqrt(var[i] + eps);

  int ch_out = n / kernel_dim;
  for (int i = 0; i < n; i++) {
    int c = (i / (kernel_dim / ch_in)) % ch_out;
    bind_weight[i] = weight[i] * factor[c];
  }
}

template <typename Dtype>
void BindBias(const int n, const Dtype *mean, const Dtype *scale,
              const Dtype *var, const Dtype *bias_conv, const Dtype *bias_bn,
              Dtype *bind_bias, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif

  if (bias_conv != nullptr) {
#ifdef DEBUG_BN_MERGE
    DataStat(n, bias_conv, "Bias before merge BN");
#endif // DEBUG_BN_MERGE
  }

  for (int i = 0; i < n; i++) {
    if (bias_conv != nullptr) {
      bind_bias[i] =
          Dtype(double(bias_bn[i]) +
                (double(bias_conv[i]) - double(mean[i])) * double(scale[i]) *
                    (1 / sqrt(double(var[i]) + eps)));
    } else {
      bind_bias[i] = Dtype(double(bias_bn[i]) -
                           double(mean[i]) * double(scale[i]) *
                               (1 / sqrt(double(var[i]) + eps)));
    }
  }

#ifdef DEBUG_BN_MERGE
  DataStat(n, bind_bias, "Bias after merge BN");
#endif // DEBUG_BN_MERGE
}

// BindBNConvWeight: Bind weight in merging BatchNorm + Conv to Conv
template <typename Dtype>
void BindBNConvWeight(const int num, const int ch, const int kernel_dim,
                      const Dtype *weight, const Dtype *scale, const Dtype *var,
                      Dtype *bind_weight, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif
  int inner_kernel_dim = kernel_dim / ch;
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < kernel_dim; j++) {
      int c = j / inner_kernel_dim;
      bind_weight[i * kernel_dim + j] =
          weight[i * kernel_dim + j] * scale[c] * (1 / sqrt(var[c] + eps));
    }
  }
}

// BindBNConvBias: Bind new bias in merging BatchNorm + Conv to Conv
template <typename Dtype>
void BindBNConvBias(const int num, const int ch, const int kernel_dim,
                    const Dtype *mean, const Dtype *scale, const Dtype *var,
                    const Dtype *weight, const Dtype *bias_conv,
                    const Dtype *bias_bn, Dtype *bind_bias, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif
  for (int i = 0; i < num; i++) {
    bind_bias[i] = 0;
    for (int j = 0; j < ch; j++) {
      Dtype weight_sum = 0;
      for (int k = 0; k < kernel_dim / ch; k++) {
        weight_sum += weight[i * kernel_dim + j * kernel_dim / ch + k];
      }
      bind_bias[i] +=
          weight_sum *
          (bias_bn[j] - scale[j] * mean[j] * (1 / sqrt(var[j] + eps)));
    }
    if (bias_conv != nullptr) {
      bind_bias[i] += bias_conv[i];
    }
  }
}

template <typename Dtype>
void ScaleInvVar(const int n, const Dtype *scale, const Dtype *var,
                 Dtype *scale_inv_var, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif
  for (int i = 0; i < n; i++) {
    scale_inv_var[i] = scale[i] * (1 / sqrt(var[i] + eps));
  }
}

vector<vector<bool>> BuildConnections(const NetParameter &net) {
  vector<vector<bool>> connect(net.layer_size(),
                               vector<bool>(net.layer_size(), false));

  for (size_t source_layer_index = 0; source_layer_index < net.layer_size() - 1;
       source_layer_index++) {
    for (size_t top_index = 0;
         top_index < net.layer(source_layer_index).top_size(); top_index++) {
      for (size_t target_layer_index = source_layer_index + 1;
           target_layer_index < net.layer_size(); target_layer_index++) {
        for (size_t bottom_index = 0;
             bottom_index < net.layer(target_layer_index).bottom_size();
             bottom_index++) {
          if (net.layer(source_layer_index).top(top_index) ==
              net.layer(target_layer_index).bottom(bottom_index)) {
            connect[source_layer_index][target_layer_index] = true;
            connect[target_layer_index][source_layer_index] = true;
          }
        }
      }
    }
  }
  return connect;
}

void ShowConnections(NetParameter net, vector<vector<bool>> connect) {
  for (int i = 0; i < connect.size(); i++) {
    for (int j = 0; j < connect[i].size(); j++) {
      if (connect[i][j] == true) {
        std::cout << "layer[" << i << "](" << net.layer(i).name() << ")("
                  << net.layer(i).type() << ")"
                  << " <----> "
                  << "layer[" << j << "](" << net.layer(j).name() << ")("
                  << net.layer(j).type() << ")" << std::endl;
      }
    }
    std::cout << std::endl;
  }
}

// Infer net type from layer types, will be override by FLAGS_net_type. Mainly
// used for net testing.
// Options: detection, segmentation, other
const string InferNetType(const NetParameter &net_param) {
  for (auto i = 0; i < net_param.layer_size(); i++) {
    if (net_param.layer(i).type() == "AnnotatedData") {
      return "detection";
    } else if (net_param.layer(i).type() == "SegmentPixelIOU") {
      return "segmentation";
    } else if (net_param.layer(i).type() == "YoloEvalDetection") {
      return "detection";
    }
  }
  return "other";
}

const vector<int> GetInputShape(NetParameter net_param,
                                const string &input_blob) {
  net_param.mutable_state()->set_phase(TRAIN);
  Net<float> net(net_param);
  CHECK(net.has_blob(input_blob))
      << " No blob named " << input_blob
      << " found, Please rename model's input blob name as " << input_blob;
  return net.blob_by_name(input_blob)->shape();
}

TransformationParameter GetTransformParam(const NetParameter &net) {
  TransformationParameter transform_param;
  for (int i = 0; i < net.layer_size(); i++) {
    const LayerParameter &cur_layer = net.layer(i);
    if (cur_layer.has_transform_param()) {
      transform_param = cur_layer.transform_param();
    }
  }
  return transform_param;
}

int GetNextLayer(const NetParameter &net, const int index, const string type,
                 const vector<vector<bool>> &connect) {
  for (int i = index + 1; i < net.layer_size(); i++) {
    if (connect[i][index] && net.layer(i).type() == type) {
      return i;
    }
  }
  return -1;
}

int GetPreviousLayer(const NetParameter &net, const int index,
                     const string type, const vector<vector<bool>> &connect) {
  for (int i = index - 1; i >= 0; i--) {
    if (connect[i][index] && net.layer(i).type() == type) {
      return i;
    }
  }
  return -1;
}

bool IsInplace(const LayerParameter *layer) {
  if (layer->bottom_size() == 1 && layer->top_size() == 1) {
    if (layer->bottom(0) == layer->top(0)) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool isSingleOutputLayer(const NetParameter &net, const int index,
                         const vector<vector<bool>> &connect) {
  int count = 0;
  for (int i = index + 1; i < net.layer_size(); i++) {
    if (connect[index][i]) {
      if (IsInplace(&net.layer(i)))
        return true;
      ++count;
      if (count > 1)
        return false;
    }
  }
  return true;
}

int GetLastInputLayer(const NetParameter &net) {
  // find last input layer
  for (int i = net.layer_size() - 1; i >= 0; i--) {
    for (int j = 0; j < net.layer(i).top_size(); j++) {
      if (net.layer(i).top(j) == "data" &&
          net.layer(i).type() != "FixedNeuron") {
        return i;
      }
    }
  }
  for (int i = net.layer_size() - 1; i >= 0; i--) {
    if (net.layer(i).type() == "Data" || net.layer(i).type() == "ImageData" ||
        net.layer(i).type() == "BoxData" ||
        net.layer(i).type() == "EnhancedImageData" ||
        net.layer(i).type() == "Input" || net.layer(i).type() == "Power" ||
        net.layer(i).type() == "DetectNetTransformation" ||
        net.layer(i).type() == "AnnotatedData" ||
        net.layer(i).type() == "HDF5Data") {
      return i;
    }
  }
  return -1;
}

typedef map<string, vector<int>> FIX_INFO;
typedef pair<string, vector<int>> FIX_INFO_ITEM;

bool IsFixSkipLayer(LayerParameter layer) {
  vector<string> fix_skip_types = {"ReLU",    "Pooling", "Permute", "Reshape",
                                   "Flatten", "Concat",  "Split"};
  if (find(fix_skip_types.begin(), fix_skip_types.end(), layer.type()) !=
      fix_skip_types.end()) {
    if (layer.type() == "Pooling" && (layer.pooling_param().pool() !=
                                      caffe::PoolingParameter_PoolMethod_MAX)) {
      return false;
    } else {
      return true;
    }
  }
  return false;
}

LayerParameter TransformConvFixed(const LayerParameter &conv_fixed,
                                  int &weight_dpos, int &bias_dpos) {
  LayerParameter conv;
  conv.CopyFrom(conv_fixed);
  conv.set_type("Convolution");
  conv.clear_fixed_param();
  conv.clear_blobs();

  // const FixedParameter& fixed_param = conv_fixed.fixed_param();
  const ConvolutionParameter &conv_param = conv.convolution_param();
  const int bit_width = conv_fixed.fixed_param().bit_width();

  // Use Blob<float> to fix weigths and bias, because fix operations are only
  // supported in GPU mode.
  // Copy weights
  Blob<float> weight;
  weight.FromProto(conv_fixed.blobs(0), 1);

  // Copy bias
  Blob<float> bias;
  if (conv_param.bias_term()) {
    bias.FromProto(conv_fixed.blobs(1), 1);
  }
  if (conv_fixed.fixed_param().fixed_method() ==
      FixedParameter_FixedMethod_OVER_FLOW) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_overflow(weight, bit_width));
    // Fix bias
    if (conv_param.bias_term()) {
      bias_dpos = (int)std::floor(caffe_fix_pos_overflow(bias, bit_width));
    } else {
      bias_dpos = 0;
    }
  } else if (conv_fixed.fixed_param().fixed_method() ==
             FixedParameter_FixedMethod_DIFF_S) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_diffs(weight, bit_width));
    // Fix bias
    if (conv_param.bias_term()) {
      bias_dpos = (int)std::floor(caffe_fix_pos_diffs(bias, bit_width));
    } else {
      bias_dpos = 0;
    }
  } else {
    LOG(FATAL) << "Unknown fixed method: "
               << conv_fixed.fixed_param().fixed_method();
  }

  caffe_fix(weight, weight, bit_width, weight_dpos);
  if (conv_param.bias_term()) {
    caffe_fix(bias, bias, bit_width, bias_dpos);
  }

  // Convert blobs to BlobProto
  weight.ToProto(conv.add_blobs());
  if (conv_param.bias_term()) {
    bias.ToProto(conv.add_blobs());
  }
  return conv;
}

LayerParameter TransformConvBNFixed(const LayerParameter &conv_bn_fixed,
                                    int &weight_dpos, int &bias_dpos) {
  LayerParameter conv;
  conv.CopyFrom(conv_bn_fixed);
  conv.set_type("Convolution");
  conv.clear_fixed_param();
  conv.clear_blobs();
  conv.clear_batch_norm_param();
  conv.clear_param();

  // const FixedParameter& fixed_param = conv_fixed.fixed_param();
  ConvolutionParameter &conv_param = *conv.mutable_convolution_param();
  conv_param.set_bias_term(true);
  const int bit_width = conv_bn_fixed.fixed_param().bit_width();

  // Copy weights & bias
  Blob<float> weight;
  Blob<float> bias;
  vector<int> shape(1, conv_bn_fixed.blobs(2).data_size());
  bias.Reshape(shape);

  weight.FromProto(conv_bn_fixed.blobs(0), 1);

  for (int i = 0; i < conv_bn_fixed.blobs(2).data_size(); i++) {
    bias.mutable_cpu_data()[i] = conv_bn_fixed.blobs(2).data(i);
  }

  auto scale = conv_bn_fixed.blobs(1).data().data();
  auto mean = conv_bn_fixed.blobs(3).data().data();
  auto var = conv_bn_fixed.blobs(4).data().data();

  float eps = conv_bn_fixed.batch_norm_param().eps();
  BindWeight(weight.count(), weight.count(1), weight.cpu_data(), scale, var,
             weight.mutable_cpu_data(), eps);
  BindBias(bias.count(), mean, scale, var, (float *)nullptr, bias.cpu_data(),
           bias.mutable_cpu_data(), eps);

  if (conv_bn_fixed.fixed_param().fixed_method() ==
      FixedParameter_FixedMethod_OVER_FLOW) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_overflow(weight, bit_width));

    // Fix bias
    if (conv_param.bias_term()) {
      bias_dpos = (int)std::floor(caffe_fix_pos_overflow(bias, bit_width));
    } else {
      bias_dpos = 0;
    }
  } else if (conv_bn_fixed.fixed_param().fixed_method() ==
             FixedParameter_FixedMethod_DIFF_S) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_diffs(weight, bit_width));

    // Fix bias
    if (conv_param.bias_term()) {
      bias_dpos = (int)std::floor(caffe_fix_pos_diffs(bias, bit_width));
    } else {
      bias_dpos = 0;
    }
  } else {
    LOG(FATAL) << "Unknown fixed method: "
               << conv_bn_fixed.fixed_param().fixed_method();
  }

  caffe_fix(weight, weight, bit_width, weight_dpos);
  if (conv_param.bias_term()) {
    caffe_fix(bias, bias, bit_width, bias_dpos);
  }

  // Convert blobs to BlobProto
  weight.ToProto(conv.add_blobs());
  bias.ToProto(conv.add_blobs());

  return conv;
}

LayerParameter TransformFCFixed(const LayerParameter &fc_fixed,
                                int &weight_dpos, int &bias_dpos) {
  LayerParameter fc;
  fc.CopyFrom(fc_fixed);
  fc.set_type("InnerProduct");
  fc.clear_fixed_param();
  fc.clear_blobs();

  // const FixedParameter& fixed_param = conv_fixed.fixed_param();
  const InnerProductParameter &fc_param = fc.inner_product_param();
  const int bit_width = fc_fixed.fixed_param().bit_width();

  // Copy weights
  Blob<float> weight;
  weight.FromProto(fc_fixed.blobs(0), 1);

  // Copy bias
  Blob<float> bias;
  if (fc_param.bias_term()) {
    bias.FromProto(fc_fixed.blobs(1), 1);
  }

  if (fc_fixed.fixed_param().fixed_method() ==
      FixedParameter_FixedMethod_OVER_FLOW) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_overflow(weight, bit_width));
    // Fix bias
    if (fc_param.bias_term()) {
      bias_dpos = (int)std::floor(caffe_fix_pos_overflow(bias, bit_width));
    } else {
      bias_dpos = 0;
    }
  } else if (fc_fixed.fixed_param().fixed_method() ==
             FixedParameter_FixedMethod_DIFF_S) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_diffs(weight, bit_width));
    // Fix bias
    if (fc_param.bias_term()) {
      bias_dpos = (int)std::floor(caffe_fix_pos_diffs(bias, bit_width));
    } else {
      bias_dpos = 0;
    }
  } else {
    LOG(FATAL) << "Unknown fixed method: "
               << fc_fixed.fixed_param().fixed_method();
  }
  caffe_fix(weight, weight, bit_width, weight_dpos);
  if (fc_param.bias_term()) {
    caffe_fix(bias, bias, bit_width, bias_dpos);
  }

  // Convert blobs to BlobProto
  weight.ToProto(fc.add_blobs());
  if (fc_param.bias_term()) {
    bias.ToProto(fc.add_blobs());
  }
  return fc;
}

LayerParameter TransformBatchNormFixed(const LayerParameter &bn_fixed,
                                       int &weight_dpos, int &bias_dpos) {
  LayerParameter bn;
  bn.CopyFrom(bn_fixed);
  bn.set_type("BatchNorm");
  bn.clear_fixed_param();

  // const FixedParameter& fixed_param = conv_fixed.fixed_param();
  const int bit_width = bn_fixed.fixed_param().bit_width();

  // Copy weights
  Blob<float> weight;
  weight.FromProto(bn_fixed.blobs(0), 1);

  // Copy bias
  Blob<float> bias;
  bias.FromProto(bn_fixed.blobs(1), 1);

  // Bind scale and bias
  auto scale = bn_fixed.blobs(0).data().data();
  auto bn_bias = bn_fixed.blobs(1).data().data();
  auto mean = bn_fixed.blobs(2).data().data();
  auto var = bn_fixed.blobs(3).data().data();
  float eps = bn_fixed.batch_norm_param().eps();
  ScaleInvVar(weight.channels(), scale, var, weight.mutable_cpu_data(), eps);
  BindBias(bias.channels(), mean, scale, var, (float *)nullptr, bn_bias,
           bias.mutable_cpu_data(), eps);
  for (int i = 0; i < bn_fixed.blobs(2).data_size(); i++) {
    bn.mutable_blobs(2)->set_data(i, 0);
    bn.mutable_blobs(3)->set_data(i, 1);
  }

  if (bn_fixed.fixed_param().fixed_method() ==
      FixedParameter_FixedMethod_OVER_FLOW) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_overflow(weight, bit_width));
    // Fix bias
    bias_dpos = (int)std::floor(caffe_fix_pos_overflow(bias, bit_width));
  } else if (bn_fixed.fixed_param().fixed_method() ==
             FixedParameter_FixedMethod_DIFF_S) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_diffs(weight, bit_width));
    // Fix bias
    bias_dpos = (int)std::floor(caffe_fix_pos_diffs(bias, bit_width));
  } else {
    LOG(FATAL) << "Unknown fixed method: "
               << bn_fixed.fixed_param().fixed_method();
  }
  caffe_fix(weight, weight, bit_width, weight_dpos);
  caffe_fix(bias, bias, bit_width, bias_dpos);

  // Convert blobs to BlobProto
  weight.ToProto(bn.mutable_blobs(0));
  bias.ToProto(bn.mutable_blobs(1));
  return bn;
}

LayerParameter TransformPReLUFixed(const LayerParameter &prelu_fixed,
                                   int &weight_dpos, int &bias_dpos) {
  LayerParameter prelu;
  prelu.CopyFrom(prelu_fixed);
  prelu.set_type("PReLU");
  prelu.clear_fixed_param();
  prelu.clear_blobs();

  // const FixedParameter& fixed_param = conv_fixed.fixed_param();
  // const PReLUParameter &prelu_param = prelu.prelu_param();
  const int bit_width = prelu_fixed.fixed_param().bit_width();

  // Copy weights
  Blob<float> weight;
  weight.FromProto(prelu_fixed.blobs(0), 1);

  if (prelu_fixed.fixed_param().fixed_method() ==
      FixedParameter_FixedMethod_OVER_FLOW) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_overflow(weight, bit_width));
    bias_dpos = 0; // no use
  } else if (prelu_fixed.fixed_param().fixed_method() ==
             FixedParameter_FixedMethod_DIFF_S) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_diffs(weight, bit_width));
    bias_dpos = 0; // no use
  } else {
    LOG(FATAL) << "Unknown fixed method: "
               << prelu_fixed.fixed_param().fixed_method();
  }
  caffe_fix(weight, weight, bit_width, weight_dpos);

  // Convert blobs to BlobProto
  weight.ToProto(prelu.add_blobs());
  return prelu;
}

LayerParameter TransformDeconvFixed(const LayerParameter &deconv_fixed,
                                    int &weight_dpos, int &bias_dpos) {
  LayerParameter deconv;
  deconv.CopyFrom(deconv_fixed);
  deconv.set_type("Deconvolution");
  deconv.clear_fixed_param();
  deconv.clear_blobs();

  const ConvolutionParameter &deconv_param = deconv.convolution_param();
  const int bit_width = deconv_fixed.fixed_param().bit_width();

  // Copy weights
  Blob<float> weight;
  weight.FromProto(deconv_fixed.blobs(0), 1);

  // Copy bias
  Blob<float> bias;
  if (deconv_param.bias_term()) {
    bias.FromProto(deconv_fixed.blobs(1), 1);
  }

  if (deconv_fixed.fixed_param().fixed_method() ==
      FixedParameter_FixedMethod_OVER_FLOW) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_overflow(weight, bit_width));
    // Fix bias
    if (deconv_param.bias_term()) {
      bias_dpos = (int)std::floor(caffe_fix_pos_overflow(bias, bit_width));
    } else {
      bias_dpos = 0;
    }
  } else if (deconv_fixed.fixed_param().fixed_method() ==
             FixedParameter_FixedMethod_DIFF_S) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_diffs(weight, bit_width));
    // Fix bias
    if (deconv_param.bias_term()) {
      bias_dpos = (int)std::floor(caffe_fix_pos_diffs(bias, bit_width));
    } else {
      bias_dpos = 0;
    }
  } else {
    LOG(FATAL) << "Unknown fixed method: "
               << deconv_fixed.fixed_param().fixed_method();
  }
  caffe_fix(weight, weight, bit_width, weight_dpos);
  if (deconv_param.bias_term()) {
    caffe_fix(bias, bias, bit_width, bias_dpos);
  }

  // Convert blobs to BlobProto
  weight.ToProto(deconv.add_blobs());
  if (deconv_param.bias_term()) {
    bias.ToProto(deconv.add_blobs());
  }
  return deconv;
}

LayerParameter TransformScaleFixed(const LayerParameter &scale_fixed,
                                   int &weight_dpos, int &bias_dpos) {
  LayerParameter scale;
  scale.CopyFrom(scale_fixed);
  scale.set_type("Scale");
  scale.clear_fixed_param();
  scale.clear_blobs();

  if (scale_fixed.blobs_size() == 0)
    return scale;

  const ScaleParameter &scale_param = scale.scale_param();
  const int bit_width = scale_fixed.fixed_param().bit_width();

  // Copy weights
  Blob<float> weight;
  weight.FromProto(scale_fixed.blobs(0), 1);

  // Copy bias
  Blob<float> bias;
  if (scale_param.bias_term()) {
    bias.FromProto(scale_fixed.blobs(1), 1);
  }

  if (scale_fixed.fixed_param().fixed_method() ==
      FixedParameter_FixedMethod_OVER_FLOW) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_overflow(weight, bit_width));
    // Fix bias
    if (scale_param.bias_term()) {
      bias_dpos = (int)std::floor(caffe_fix_pos_overflow(bias, bit_width));
    } else {
      bias_dpos = 0;
    }
  } else if (scale_fixed.fixed_param().fixed_method() ==
             FixedParameter_FixedMethod_DIFF_S) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_diffs(weight, bit_width));
    // Fix bias
    if (scale_param.bias_term()) {
      bias_dpos = (int)std::floor(caffe_fix_pos_diffs(bias, bit_width));
    } else {
      bias_dpos = 0;
    }
  } else {
    LOG(FATAL) << "Unknown fixed method: "
               << scale_fixed.fixed_param().fixed_method();
  }
  caffe_fix(weight, weight, bit_width, weight_dpos);
  if (scale_param.bias_term()) {
    caffe_fix(bias, bias, bit_width, bias_dpos);
  }

  // Convert blobs to BlobProto
  weight.ToProto(scale.add_blobs());
  if (scale_param.bias_term()) {
    bias.ToProto(scale.add_blobs());
  }
  return scale;
}

LayerParameter TransformNormalizeFixed(const LayerParameter &normalize_fixed,
                                       int &weight_dpos, int &bias_dpos) {
  LayerParameter normalize;
  normalize.CopyFrom(normalize_fixed);
  normalize.set_type("Normalize");
  normalize.clear_fixed_param();
  normalize.clear_blobs();

  const int bit_width = normalize_fixed.fixed_param().bit_width();

  // Copy weights
  Blob<float> weight;
  weight.FromProto(normalize_fixed.blobs(0), 1);

  if (normalize_fixed.fixed_param().fixed_method() ==
      FixedParameter_FixedMethod_OVER_FLOW) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_overflow(weight, bit_width));
    // No bias
    bias_dpos = 0;
  } else if (normalize_fixed.fixed_param().fixed_method() ==
             FixedParameter_FixedMethod_DIFF_S) {
    // Fix weights
    weight_dpos = (int)std::floor(caffe_fix_pos_diffs(weight, bit_width));
    // No bias
    bias_dpos = 0;
  } else {
    LOG(FATAL) << "Unknown fixed method: "
               << normalize_fixed.fixed_param().fixed_method();
  }
  caffe_fix(weight, weight, bit_width, weight_dpos);

  // Convert blobs to BlobProto
  weight.ToProto(normalize.add_blobs());
  return normalize;
}

// Update data fix_info for relu and max pooling layers
void UpdateDataFixInfo(const NetParameter &net_in, FIX_INFO &data_infos) {
  bool finish = false;
  while (!finish) {
    finish = true;
    DLOG(INFO) << "--------------do update--------------";
    for (size_t i = 0; i < net_in.layer_size(); i++) {
      const LayerParameter *cur_layer = &net_in.layer(i);
      if (IsFixSkipLayer(*cur_layer)) {

        if (cur_layer->bottom_size() == 1 &&
            (data_infos.find(cur_layer->bottom(0)) != data_infos.end())) {
          for (size_t j = 0; j < cur_layer->top_size(); j++) {
            if (data_infos.find(cur_layer->top(j)) == data_infos.end()) {
              data_infos.insert(FIX_INFO_ITEM(
                  cur_layer->top(j), data_infos.at(cur_layer->bottom(0))));
              finish = false;
              DLOG(INFO) << "update: " << cur_layer->top(j) << " as "
                         << cur_layer->bottom(0);
            }
          }
        }
        if (cur_layer->top_size() == 1 &&
            (data_infos.find(cur_layer->top(0)) != data_infos.end())) {
          for (size_t j = 0; j < cur_layer->bottom_size(); j++) {
            if (data_infos.find(cur_layer->bottom(j)) == data_infos.end()) {
              data_infos.insert(FIX_INFO_ITEM(
                  cur_layer->bottom(j), data_infos.at(cur_layer->top(0))));
              finish = false;
              DLOG(INFO) << "update: " << cur_layer->bottom(j) << " as "
                         << cur_layer->top(0);
            }
          }
        }
      }
    }
  }
}

void TransformFix2FloatWithFixInfo(NetParameter &net_in, NetParameter &net_out,
                                   FIX_INFO &weight_infos, FIX_INFO &bias_infos,
                                   FIX_INFO &data_infos,
                                   bool keep_fixed_neuron) {
  for (size_t i = 0; i < net_in.layer_size(); i++) {
    const LayerParameter *cur_layer = &net_in.layer(i);
    DLOG(INFO) << "transforming: " << cur_layer->name();
    if (cur_layer->type() == "ConvolutionFixed" ||
        cur_layer->type() == "ConvolutionFixedData") {
      // ConvolutionFixed --> Convolution
      int weight_dpos, bias_dpos;

      *net_out.add_layer() =
          TransformConvFixed(*cur_layer, weight_dpos, bias_dpos);

      vector<int> weight_fix_info, bias_fix_info;
      weight_fix_info.push_back(cur_layer->fixed_param().bit_width());
      weight_fix_info.push_back(weight_dpos);
      bias_fix_info.push_back(cur_layer->fixed_param().bit_width());
      bias_fix_info.push_back(bias_dpos);

      weight_infos.insert(FIX_INFO_ITEM(cur_layer->name(), weight_fix_info));
      bias_infos.insert(FIX_INFO_ITEM(cur_layer->name(), bias_fix_info));

    } else if (cur_layer->type() == "ConvolutionBNFixed" ||
               cur_layer->type() == "ConvolutionBNFixedData") {
      // ConvolutionBNFixed --> Convolution
      int weight_dpos;
      int bias_dpos;
      *net_out.add_layer() =
          TransformConvBNFixed(*cur_layer, weight_dpos, bias_dpos);

      vector<int> weight_fix_info, bias_fix_info;
      weight_fix_info.push_back(cur_layer->fixed_param().bit_width());
      weight_fix_info.push_back(weight_dpos);
      bias_fix_info.push_back(cur_layer->fixed_param().bit_width());
      bias_fix_info.push_back(bias_dpos);

      weight_infos.insert(FIX_INFO_ITEM(cur_layer->name(), weight_fix_info));
      bias_infos.insert(FIX_INFO_ITEM(cur_layer->name(), bias_fix_info));

    } else if (cur_layer->type() == "InnerProductFixed" ||
               cur_layer->type() == "InnerProductFixedData") {
      // InnerProductFixed --> InnerProduct
      int weight_dpos;
      int bias_dpos;
      *net_out.add_layer() =
          TransformFCFixed(*cur_layer, weight_dpos, bias_dpos);
      vector<int> weight_fix_info, bias_fix_info;
      weight_fix_info.push_back(cur_layer->fixed_param().bit_width());
      weight_fix_info.push_back(weight_dpos);
      bias_fix_info.push_back(cur_layer->fixed_param().bit_width());
      bias_fix_info.push_back(bias_dpos);

      weight_infos.insert(FIX_INFO_ITEM(cur_layer->name(), weight_fix_info));
      bias_infos.insert(FIX_INFO_ITEM(cur_layer->name(), bias_fix_info));

    } else if (cur_layer->type() == "PReLUFixed") {
      // PReLUFixed --> PReLU
      int weight_dpos;
      int bias_dpos;
      *net_out.add_layer() =
          TransformPReLUFixed(*cur_layer, weight_dpos, bias_dpos);
      vector<int> weight_fix_info, bias_fix_info;
      weight_fix_info.push_back(cur_layer->fixed_param().bit_width());
      weight_fix_info.push_back(weight_dpos);
      bias_fix_info.push_back(cur_layer->fixed_param().bit_width());
      bias_fix_info.push_back(bias_dpos);

      weight_infos.insert(FIX_INFO_ITEM(cur_layer->name(), weight_fix_info));
      bias_infos.insert(FIX_INFO_ITEM(cur_layer->name(), bias_fix_info));

    } else if (cur_layer->type() == "BatchNormFixed") {
      // BatchNormFixed --> BatchNorm
      int weight_dpos;
      int bias_dpos;
      *net_out.add_layer() =
          TransformBatchNormFixed(*cur_layer, weight_dpos, bias_dpos);
      vector<int> weight_fix_info, bias_fix_info;
      weight_fix_info.push_back(cur_layer->fixed_param().bit_width());
      weight_fix_info.push_back(weight_dpos);
      bias_fix_info.push_back(cur_layer->fixed_param().bit_width());
      bias_fix_info.push_back(bias_dpos);

      weight_infos.insert(FIX_INFO_ITEM(cur_layer->name(), weight_fix_info));
      bias_infos.insert(FIX_INFO_ITEM(cur_layer->name(), bias_fix_info));

    } else if (cur_layer->type() == "DeconvolutionFixed") {
      // DeconvolutionFixed --> Deconvolution
      int weight_dpos;
      int bias_dpos;
      *net_out.add_layer() =
          TransformDeconvFixed(*cur_layer, weight_dpos, bias_dpos);

      vector<int> weight_fix_info, bias_fix_info;
      weight_fix_info.push_back(cur_layer->fixed_param().bit_width());
      weight_fix_info.push_back(weight_dpos);
      bias_fix_info.push_back(cur_layer->fixed_param().bit_width());
      bias_fix_info.push_back(bias_dpos);

      weight_infos.insert(FIX_INFO_ITEM(cur_layer->name(), weight_fix_info));
      bias_infos.insert(FIX_INFO_ITEM(cur_layer->name(), bias_fix_info));

    } else if (cur_layer->type() == "ScaleFixed") {
      // ScaleFixed --> Scale
      int weight_dpos;
      int bias_dpos;
      *net_out.add_layer() =
          TransformScaleFixed(*cur_layer, weight_dpos, bias_dpos);

      vector<int> weight_fix_info, bias_fix_info;
      weight_fix_info.push_back(cur_layer->fixed_param().bit_width());
      weight_fix_info.push_back(weight_dpos);
      bias_fix_info.push_back(cur_layer->fixed_param().bit_width());
      bias_fix_info.push_back(bias_dpos);

      weight_infos.insert(FIX_INFO_ITEM(cur_layer->name(), weight_fix_info));
      bias_infos.insert(FIX_INFO_ITEM(cur_layer->name(), bias_fix_info));

    } else if (cur_layer->type() == "NormalizeFixed") {
      // ScaleFixed --> Scale
      int weight_dpos;
      int bias_dpos;
      *net_out.add_layer() =
          TransformNormalizeFixed(*cur_layer, weight_dpos, bias_dpos);

      vector<int> weight_fix_info, bias_fix_info;
      weight_fix_info.push_back(cur_layer->fixed_param().bit_width());
      weight_fix_info.push_back(weight_dpos);
      bias_fix_info.push_back(cur_layer->fixed_param().bit_width());
      bias_fix_info.push_back(bias_dpos);

      weight_infos.insert(FIX_INFO_ITEM(cur_layer->name(), weight_fix_info));
      bias_infos.insert(FIX_INFO_ITEM(cur_layer->name(), bias_fix_info));

    } else if (cur_layer->type() == "FixedNeuron") {
      // FixedNeuron
      DLOG(INFO) << "add data_fix_info: " << cur_layer->name();
      // CHECK_EQ(cur_layer->bottom(0), cur_layer->top(0))
      // << "FixedNeuron layer " << cur_layer->name() << " should be inplace";
      vector<int> data_fix_info;
      data_fix_info.push_back(cur_layer->fixed_param().bit_width());
      data_fix_info.push_back((int)std::floor(cur_layer->blobs(0).data(0)));

      data_infos.insert(FIX_INFO_ITEM(cur_layer->bottom(0), data_fix_info));
      data_infos.insert(FIX_INFO_ITEM(cur_layer->top(0), data_fix_info));
      for (auto it : data_infos) {
        DLOG(INFO) << "layer: " << i << " data: " << it.first << " "
                   << it.second[0] << " " << it.second[1];
      }
      if (keep_fixed_neuron) {
        *net_out.add_layer() = *cur_layer;
      }
    } else {
      // Copy Other Layers
      *net_out.add_layer() = *cur_layer;
    }
  }

  UpdateDataFixInfo(net_in, data_infos);
  for (auto it : data_infos) {
    DLOG(INFO) << "data: " << it.first << " " << it.second[0] << " "
               << it.second[1];
    if (it.second[1] == SHRT_MAX) {
      LOG(WARNING) << "[DEPLOY WARNING] Layer " << it.first
                   << "'s output blob is all zero, this may cause error for "
                      "DNNC compiler. Please check the float model.";
    } else if (it.second[1] == SHRT_MIN) {
      LOG(WARNING) << "[DEPLOY WARNING] Layer " << it.first
                   << "'s output blob is Nan, this may cause error for "
                      "DNNC compiler. Please check the float model.";
    }
  }
  for (auto it : weight_infos) {
    DLOG(INFO) << "weight: " << it.first << " " << it.second[0] << " "
               << it.second[1];
    if (it.second[1] == SHRT_MAX) {
      weight_infos[it.first][1] = 0;
      LOG(WARNING) << "[DEPLOY WARNING] Layer " << it.first
                   << "'s weight blob is all zero, this may cause error for "
                      "the following step of deployment. Please check the float model. "
                      "Change quantization step to 1 for going through.";
    } else if (it.second[1] == SHRT_MIN) {
      LOG(FATAL) << "[DEPLOY ERROR] Layer " << it.first
                 << "'s weight blob is Nan, this may cause error for "
                    "DNNC compiler. Please check the float model.";
    }
  }
  for (auto it : bias_infos) {
    DLOG(INFO) << "bias: " << it.first << " " << it.second[0] << " "
               << it.second[1];
    if (it.second[1] == SHRT_MAX) {
      bias_infos[it.first][1] = 0;
      LOG(WARNING) << "[DEPLOY WARNING] Layer " << it.first
                   << "'s bias blob is all zero, this may cause error for "
                      "the following step of deployment. Please check the float model. "
                      "Change quantization step to 1 for going through.";
    } else if (it.second[1] == SHRT_MIN) {
      LOG(FATAL) << "[DEPLOY ERROR] Layer " << it.first
                   << "'s bias blob is Nan, this means some error happen. "
                      "Please check the float model.";
    }
  }
}

vector<int> GetLayerFixInfo(const string &layer_name, FIX_INFO &fix_infos) {
  auto it = fix_infos.find(layer_name);
  if (it != fix_infos.end()) {
    return it->second;
  } else {
    DLOG(WARNING) << "Warning: can't find layer " << layer_name << "'s dpos!";
    return vector<int>();
  }
}

vector<int> GetDataFixInfo(const string &blob_name, FIX_INFO &data_infos) {
  auto it = data_infos.find(blob_name);
  if (it != data_infos.end()) {
    return it->second;
  } else {
    DLOG(WARNING) << "Warning: Can't find blob " << blob_name << "'s dpos!";
    return vector<int>();
  }
}

void ConcatVector(vector<int> &vec1, vector<int> &vec2, int times) {
  if (!vec2.empty()) {
    for (size_t i = 0; i < times; i++) {
      vec1.insert(vec1.end(), vec2.begin(), vec2.end());
    }
  }
  return;
}

vector<vector<int>> GetFixInfo(NetParameter &net, FIX_INFO &weight_infos,
                               FIX_INFO &bias_infos, FIX_INFO &data_infos) {
  vector<vector<int>> fix_infos; // dpos[i] = [index_i, bit_width_0, dpos_0,
                                 // bit_width_1, dpos_1, ... ]
  int debugMode = 0;
  if (getenv("DECENT_DEBUG"))
    debugMode = stoi(getenv("DECENT_DEBUG"));
  // debugMode = 2 needs match Caffe Deploy accuracy, where allignment of
  // concat is not needed here. Because Caffe deploy concat layer has
  // self-allignment.
  if (0 && debugMode < 2) {
    // pre-processing concat layer bottom data fixed positions
    // to allign them, because there is no allignment concat DPU forwarding
    // To do the allignment here is to avoid the bottom fixed postion
    // NOT ALLIGNED dumped to fixed info file.
    while (1) {
      bool posChanged = false;
      for (int i = net.layer_size() - 1; i >= 0; --i) {
        const LayerParameter *cur_layer = &net.layer(i);
        if (cur_layer->type() == "Concat") {
          // get minimum fix position and update top data position
          int minPos = data_infos[cur_layer->top(0)][1];
          for (unsigned j = 0; j < cur_layer->bottom_size(); j++) {
            if (minPos > data_infos[cur_layer->bottom(j)][1]) {
              minPos = data_infos[cur_layer->bottom(j)][1];
              data_infos[cur_layer->top(0)][1] = minPos;
              posChanged = true;
            }
          }
          // overwrite all bottom postions with the top position
          for (unsigned j = 0; j < cur_layer->bottom_size(); j++) {
            vector<int> input_info =
                GetDataFixInfo(cur_layer->bottom(j), data_infos);
            if (minPos != input_info[1]) {
              LOG(WARNING) << "Adjust Concat fix_info: output("
                           << cur_layer->top(0) << ", " << minPos
                           << ") --> input(" << cur_layer->bottom(j) << ", "
                           << input_info[1] << ").";
              data_infos[cur_layer->bottom(j)][1] = minPos;
              posChanged = true;
            }
          }
        }
      }
      if (!posChanged)
        break;
    }
  }

  for (size_t i = 0; i < net.layer_size(); i++) {
    const LayerParameter *cur_layer = &net.layer(i);
    DLOG(INFO) << "Processing Layer: " << i << "/" << net.layer_size() - 1
               << ": " << cur_layer->name() << " " << cur_layer->type();
    vector<int> cur_fix_info;
    if (cur_layer->type() == "Convolution" ||
        cur_layer->type() == "InnerProduct" ||
        cur_layer->type() == "BatchNorm" ||
        cur_layer->type() == "Deconvolution" || cur_layer->type() == "Scale") {
      // fix info conv = [index_i, input_info, output_info,
      // weight_info, bias_info]
      cur_fix_info.push_back(i);
      vector<int> input_info, output_info, weight_info, bias_info;
      input_info = GetDataFixInfo(cur_layer->bottom(0), data_infos);
      output_info = GetDataFixInfo(cur_layer->top(0), data_infos);
      weight_info = GetLayerFixInfo(cur_layer->name(), weight_infos);
      bias_info = GetLayerFixInfo(cur_layer->name(), bias_infos);

      if (debugMode == 1) {
        if (input_info[1] + weight_info[1] < bias_info[1]) 
          DLOG(INFO) << "Find bias right shifting in layer: " << cur_layer->name()
                     << " [i o w b] = [" 
                     << input_info[1] << " "
                     << output_info[1] << " "
                     << weight_info[1] << " "
                     << bias_info[1] << "] "
                     << "shift to input*w: " << bias_info[1] - input_info[1] - weight_info[1];
      }

      ConcatVector(cur_fix_info, input_info);
      ConcatVector(cur_fix_info, output_info);
      ConcatVector(cur_fix_info, weight_info);
      ConcatVector(cur_fix_info, bias_info);

    } else if (cur_layer->type() == "PReLU" ||
               cur_layer->type() == "Normalize") {
      // fix info bn = [index_i, input_info, output_info,
      // weight_info, bias_info]
      cur_fix_info.push_back(i);
      vector<int> input_info, output_info, weight_info, bias_info;
      input_info = GetDataFixInfo(cur_layer->bottom(0), data_infos);
      output_info = GetDataFixInfo(cur_layer->top(0), data_infos);
      weight_info = GetLayerFixInfo(cur_layer->name(), weight_infos);

      ConcatVector(cur_fix_info, input_info);
      ConcatVector(cur_fix_info, output_info);
      ConcatVector(cur_fix_info, weight_info);

    } else if (cur_layer->type() == "Eltwise") {
      cur_fix_info.push_back(i);
      vector<int> output_info = GetDataFixInfo(cur_layer->top(0), data_infos);
      for (size_t j = 0; j < cur_layer->bottom_size(); j++) {
        vector<int> input_info =
            GetDataFixInfo(cur_layer->bottom(j), data_infos);
        ConcatVector(cur_fix_info, input_info);
// DPU Eltwise forwarding will set top data fixed postion to the minimum
// fixed position of all bottoms, so here the action is not needed anymore.
#if 0
        if ( debugMode < 2 && input_info.size() > 0 && output_info.size() > 0 &&
            output_info[1] > input_info[1]) {
          LOG(WARNING) << "Adjust Eltwise fix_info: output("
                       << cur_layer->top(0) << ", " << output_info[1]
                       << ") <-- input(" << cur_layer->bottom(j) << ", "
                       << input_info[1] << ").";
          data_infos[cur_layer->top(0)][1] = input_info[1];
          output_info[1] = input_info[1];
        }
#endif
      }
      ConcatVector(cur_fix_info, output_info);

    }
#if 0
    else if (cur_layer->type() == "Concat") {
      cur_fix_info.push_back(i);
      vector<int> output_info = GetDataFixInfo(cur_layer->top(0), data_infos);
      for (size_t j = 0; j < cur_layer->bottom_size(); j++) {
        vector<int> input_info =
            GetDataFixInfo(cur_layer->bottom(j), data_infos);
        if (output_info[1] != input_info[1]) {
          LOG(WARNING) << "Adjust Concat fix_info: output(" << cur_layer->top(0)
                       << ", " << output_info[1] << ") --> input("
                       << cur_layer->bottom(j) << ", " << input_info[1] << ").";
          data_infos[cur_layer->bottom(j)][1] = output_info[1];
        }
        ConcatVector(cur_fix_info, output_info);
      }
      ConcatVector(cur_fix_info, output_info);

    }
#endif
    else if (IsFixSkipLayer(*cur_layer)) {
      // update fix skip layers input and output dpos
      // info pooling = [index_i, input_info, output_info]

      cur_fix_info.push_back(i);
      for (size_t j = 0; j < cur_layer->bottom_size(); j++) {
        vector<int> input_info =
            GetDataFixInfo(cur_layer->bottom(j), data_infos);
        ConcatVector(cur_fix_info, input_info);
      }
      for (size_t k = 0; k < cur_layer->top_size(); k++) {
        vector<int> output_info = GetDataFixInfo(cur_layer->top(k), data_infos);
        ConcatVector(cur_fix_info, output_info);
      }
    } else {
      cur_fix_info.push_back(i);
      for (size_t j = 0; j < cur_layer->bottom_size(); j++) {
        vector<int> input_info =
            GetDataFixInfo(cur_layer->bottom(j), data_infos);
        ConcatVector(cur_fix_info, input_info);
      }
      for (size_t k = 0; k < cur_layer->top_size(); k++) {
        vector<int> output_info = GetDataFixInfo(cur_layer->top(k), data_infos);
        ConcatVector(cur_fix_info, output_info);
      }
    }

    if (!cur_fix_info.empty()) {
      fix_infos.push_back(cur_fix_info);
    }
  }
  return fix_infos;
}

void WriteFixInfoToFile(NetParameter &net, vector<vector<int>> fix_infos,
                        string file_name) {
  std::ofstream ofile;
  ofile.open(file_name);
  for (auto it = fix_infos.begin(); it != fix_infos.end(); it++) {
    ofile << (*it)[0] << " " << net.layer((*it)[0]).name();
    for (int i = 1; i < it->size(); i++) {
      ofile << " " << (*it)[i];
    }
    ofile << std::endl;
  }
  ofile.close();
  return;
}

void WriteFixInfoToModel(NetParameter &model, vector<vector<int>> fix_infos) {
  for (auto it = fix_infos.begin(); it != fix_infos.end(); it++) {
    DLOG(INFO) << "Writing fix info to model: layer " << (*it)[0] << "/"
               << model.layer_size()
               << " Name: " << model.layer((*it)[0]).name();
    auto layer_param = model.mutable_layer((*it)[0]);
    for (int i = 1; i < it->size(); i++) {
      layer_param->mutable_fixed_param()->add_fix_info((*it)[i]);
    }
  }
  return;
}

// Merge model
void MergeConvBatchNorm2Conv(const NetParameter &net_in, NetParameter &net_out,
                             const int conv_index, const int bn_index,
                             vector<bool> &processed, bool binary) {
  const LayerParameter &conv_param = net_in.layer(conv_index);
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(conv_param);
  layer_param->set_top(0, bn_param.top(0));
  layer_param->mutable_convolution_param()->set_bias_term(true);
  // Set Param
  layer_param->clear_param();
  float lr_mult[2] = {1, 1};
  float decay_mult[2] = {1, 0};
  for (int j = 0; j < 2; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  layer_param->clear_batch_norm_param();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 5)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    auto weight = conv_param.blobs(0).data().data();
    const float *bias_conv = nullptr;
    if (conv_param.convolution_param().bias_term()) {
      bias_conv = conv_param.blobs(1).data().data();
    }
    auto scale = bn_param.blobs(0).data().data();
    auto bias_bn = bn_param.blobs(1).data().data();
    auto mean = bn_param.blobs(2).data().data();
    auto var = bn_param.blobs(3).data().data();

    Blob<float> new_weight;
    Blob<float> new_bias;
    new_weight.FromProto(conv_param.blobs(0), 1);
    new_bias.FromProto(bn_param.blobs(1), 1);

    // BindWeight(new_weight.count(), new_weight.count(1), weight, scale, var,
    //           new_weight.mutable_cpu_data());
    float eps = bn_param.batch_norm_param().eps();
    BindWeightWithProfile(new_weight, weight, scale, var, eps);

    if (conv_param.convolution_param().bias_term()) {
      BindBias(new_bias.count(), mean, scale, var, bias_conv, bias_bn,
               new_bias.mutable_cpu_data(), eps);
    } else {
      BindBias(new_bias.count(), mean, scale, var, (float *)nullptr, bias_bn,
               new_bias.mutable_cpu_data(), eps);
    }
    new_weight.ToProto(layer_param->add_blobs());
    new_bias.Reshape({new_bias.count()});
    new_bias.ToProto(layer_param->add_blobs());
  }
  processed[conv_index] = true;
  processed[bn_index] = true;
}

void MergeDeconvBatchNorm2Deconv(const NetParameter &net_in,
                                 NetParameter &net_out, const int conv_index,
                                 const int bn_index, vector<bool> &processed,
                                 bool binary) {
  const LayerParameter &conv_param = net_in.layer(conv_index);
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(conv_param);
  layer_param->set_top(0, bn_param.top(0));
  layer_param->mutable_convolution_param()->set_bias_term(true);
  // Set Param
  layer_param->clear_param();
  float lr_mult[2] = {1, 1};
  float decay_mult[2] = {1, 0};
  for (int j = 0; j < 2; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  layer_param->clear_batch_norm_param();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 5)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    auto weight = conv_param.blobs(0).data().data();
    const float *bias_conv = nullptr;
    if (conv_param.convolution_param().bias_term()) {
      bias_conv = conv_param.blobs(1).data().data();
    }
    auto scale = bn_param.blobs(0).data().data();
    auto bias_bn = bn_param.blobs(1).data().data();
    auto mean = bn_param.blobs(2).data().data();
    auto var = bn_param.blobs(3).data().data();

    Blob<float> new_weight;
    Blob<float> new_bias;
    new_weight.FromProto(conv_param.blobs(0), 1);
    new_bias.FromProto(bn_param.blobs(1), 1);

    float eps = bn_param.batch_norm_param().eps();
    int ch_in = new_weight.shape()[0];
    int kernel_dim = ch_in * new_weight.count(2);
    BindDeconvWeight(new_weight.count(), kernel_dim, ch_in, weight, scale, var,
                     new_weight.mutable_cpu_data(), eps);

    if (conv_param.convolution_param().bias_term()) {
      BindBias(new_bias.count(), mean, scale, var, bias_conv, bias_bn,
               new_bias.mutable_cpu_data(), eps);
    } else {
      BindBias(new_bias.count(), mean, scale, var, (float *)nullptr, bias_bn,
               new_bias.mutable_cpu_data(), eps);
    }
    new_weight.ToProto(layer_param->add_blobs());
    new_bias.Reshape({new_bias.count()});
    new_bias.ToProto(layer_param->add_blobs());
  }
  processed[conv_index] = true;
  processed[bn_index] = true;
}

void MergeFCBatchNorm2FC(const NetParameter &net_in, NetParameter &net_out,
                         const int fc_index, const int bn_index,
                         vector<bool> &processed, bool binary) {
  const LayerParameter &fc_param = net_in.layer(fc_index);
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(fc_param);
  layer_param->set_top(0, bn_param.top(0));
  layer_param->mutable_inner_product_param()->set_bias_term(true);
  // Set Param
  layer_param->clear_param();
  float lr_mult[2] = {1, 1};
  float decay_mult[2] = {1, 0};
  for (int j = 0; j < 2; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  layer_param->clear_batch_norm_param();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 5)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    auto weight = fc_param.blobs(0).data().data();
    const float *bias_fc = nullptr;
    if (fc_param.inner_product_param().bias_term()) {
      bias_fc = fc_param.blobs(1).data().data();
    }
    auto scale = bn_param.blobs(0).data().data();
    auto bias_bn = bn_param.blobs(1).data().data();
    auto mean = bn_param.blobs(2).data().data();
    auto var = bn_param.blobs(3).data().data();

    Blob<float> new_weight;
    Blob<float> new_bias;
    new_weight.FromProto(fc_param.blobs(0), 1);
    new_bias.FromProto(bn_param.blobs(1), 1);

    float eps = bn_param.batch_norm_param().eps();
    BindWeight(new_weight.count(), new_weight.count(1), weight, scale, var,
               new_weight.mutable_cpu_data(), eps);
    if (fc_param.inner_product_param().bias_term()) {
      BindBias(new_bias.count(), mean, scale, var, bias_fc, bias_bn,
               new_bias.mutable_cpu_data(), eps);
    } else {
      BindBias(new_bias.count(), mean, scale, var, (float *)nullptr, bias_bn,
               new_bias.mutable_cpu_data(), eps);
    }
    new_weight.ToProto(layer_param->add_blobs());
    new_bias.Reshape({new_bias.count()});
    new_bias.ToProto(layer_param->add_blobs());
  }
  processed[fc_index] = true;
  processed[bn_index] = true;
}

// Merge model
void MergeBatchNormConv2Conv(const NetParameter &net_in, NetParameter &net_out,
                             const int bn_index, const int conv_index,
                             vector<bool> &processed, bool binary) {
  const LayerParameter &conv_param = net_in.layer(conv_index);
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(conv_param);
  layer_param->set_bottom(0, bn_param.bottom(0));
  layer_param->mutable_convolution_param()->set_bias_term(true);
  // Set Param
  layer_param->clear_param();
  float lr_mult[2] = {1, 1};
  float decay_mult[2] = {1, 0};
  for (int j = 0; j < 2; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  layer_param->clear_batch_norm_param();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 5)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    auto weight = conv_param.blobs(0).data().data();
    const float *bias_conv = nullptr;
    if (conv_param.convolution_param().bias_term()) {
      bias_conv = conv_param.blobs(1).data().data();
    }
    auto scale = bn_param.blobs(0).data().data();
    auto bias_bn = bn_param.blobs(1).data().data();
    auto mean = bn_param.blobs(2).data().data();
    auto var = bn_param.blobs(3).data().data();

    Blob<float> new_weight;
    Blob<float> new_bias;
    new_weight.FromProto(conv_param.blobs(0), 1);
    if (conv_param.convolution_param().bias_term()) {
      new_bias.FromProto(conv_param.blobs(1), 1);
    } else {
      new_bias.Reshape({1, new_weight.shape(1), 1, 1});
      LOG(INFO) << "new_bias.shape(): " << new_bias.shape_string();
    }

    float eps = bn_param.batch_norm_param().eps();
    BindBNConvWeight(new_weight.num(), new_weight.channels(),
                     new_weight.count(1), weight, scale, var,
                     new_weight.mutable_cpu_data(), eps);
    if (conv_param.convolution_param().bias_term()) {
      BindBNConvBias(new_weight.num(), new_weight.channels(),
                     new_weight.count(1), mean, scale, var, weight, bias_conv,
                     bias_bn, new_bias.mutable_cpu_data(), eps);
    } else {
      BindBNConvBias(new_weight.num(), new_weight.channels(),
                     new_weight.count(1), mean, scale, var, weight,
                     (float *)nullptr, bias_bn, new_bias.mutable_cpu_data(),
                     eps);
    }
    new_weight.ToProto(layer_param->add_blobs());
    new_bias.Reshape({new_bias.count()});
    new_bias.ToProto(layer_param->add_blobs());
  }
  processed[conv_index] = true;
  processed[bn_index] = true;
}

void MergeBatchNorm2Scale(const NetParameter &net_in, NetParameter &net_out,
                          const int bn_index, vector<bool> &processed,
                          bool binary) {
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(bn_param);
  layer_param->set_type("Scale");
  layer_param->mutable_scale_param()->set_bias_term(true);
  // Set Param
  layer_param->clear_param();
  float lr_mult[2] = {1, 1};
  float decay_mult[2] = {1, 0};
  for (int j = 0; j < 2; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  layer_param->clear_batch_norm_param();

  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 5)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();

    Blob<float> new_weight;
    Blob<float> new_bias;
    new_weight.FromProto(bn_param.blobs(0), 1);
    new_bias.FromProto(bn_param.blobs(1), 1);

    auto scale = bn_param.blobs(0).data().data();
    auto bn_bias = bn_param.blobs(1).data().data();
    auto mean = bn_param.blobs(2).data().data();
    auto var = bn_param.blobs(3).data().data();

    // Bind scale and bias
    float eps = bn_param.batch_norm_param().eps();
    ScaleInvVar(new_weight.channels(), scale, var,
                new_weight.mutable_cpu_data(), eps);
    BindBias(new_bias.channels(), mean, scale, var, (float *)nullptr, bn_bias,
             new_bias.mutable_cpu_data(), eps);

    new_weight.Reshape({new_weight.count()});
    new_weight.ToProto(layer_param->add_blobs());
    new_bias.Reshape({new_bias.count()});
    new_bias.ToProto(layer_param->add_blobs());
  }
  processed[bn_index] = true;
}

void MergeConvBatchNorm2ConvBNFixed(const NetParameter &net_in,
                                    NetParameter &net_out, const int conv_index,
                                    const int bn_index, vector<bool> &processed,
                                    bool binary) {
  const LayerParameter &conv_param = net_in.layer(conv_index);
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(conv_param);
  layer_param->set_top(0, bn_param.top(0));
  layer_param->mutable_convolution_param()->set_bias_term(false);
  // Set Param
  layer_param->clear_param();
  float lr_mult[5] = {1, 1, 1, 0, 0};
  float decay_mult[5] = {1, 0, 0, 0, 0};
  for (int j = 0; j < 5; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  // Set batch_norm_param
  if (bn_param.batch_norm_param().has_scale_filler()) {
    layer_param->mutable_batch_norm_param()->mutable_scale_filler()->CopyFrom(
        bn_param.batch_norm_param().scale_filler());
  } else {
    layer_param->mutable_batch_norm_param()->mutable_scale_filler()->set_type(
        "constant");
    layer_param->mutable_batch_norm_param()->mutable_scale_filler()->set_value(
        1.);
  }
  if (bn_param.batch_norm_param().has_bias_filler()) {
    layer_param->mutable_batch_norm_param()->mutable_bias_filler()->CopyFrom(
        bn_param.batch_norm_param().bias_filler());
  } else {
    layer_param->mutable_batch_norm_param()->mutable_bias_filler()->set_type(
        "constant");
    layer_param->mutable_batch_norm_param()->mutable_bias_filler()->set_value(
        0.);
  }
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 5)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    *(layer_param->add_blobs()) = conv_param.blobs(0); // weight
    *(layer_param->add_blobs()) = bn_param.blobs(0);   // scale
    *(layer_param->add_blobs()) = bn_param.blobs(1);   // bias
    *(layer_param->add_blobs()) = bn_param.blobs(2);   // mean
    *(layer_param->add_blobs()) = bn_param.blobs(3);   // variance
    *(layer_param->add_blobs()) = bn_param.blobs(4);   // redundant_blob
    if (conv_param.blobs_size() == 2) {                // convert bias
      // new_bias = bn_bias + conv_bias * scale / sqrt(variance)
      for (size_t i = 0; i < conv_param.blobs(1).data_size(); i++) {
        float new_bias = bn_param.blobs(0).data().data()[i] *
                         conv_param.blobs(1).data().data()[i];
        new_bias /= sqrt(bn_param.blobs(3).data().data()[i] + 0.00001);
        new_bias += bn_param.blobs(1).data().data()[i];
        layer_param->mutable_blobs(2)->set_data(i, new_bias);
      }
    }
  }
  processed[conv_index] = true;
  processed[bn_index] = true;
}

void MergeBvlcBatchNormScale2BatchNorm(const NetParameter &net_in,
                                       NetParameter &net_out,
                                       const int bn_index, const int sc_index,
                                       vector<bool> &processed, bool binary) {
  const LayerParameter &bn_param = net_in.layer(bn_index);
  const LayerParameter &sc_param = net_in.layer(sc_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(bn_param);
  layer_param->set_top(0, sc_param.top(0));
  // Set Param
  layer_param->clear_param();
  float lr_mult[4] = {1, 1, 0, 0};
  float decay_mult[4] = {0, 0, 0, 0};
  for (int j = 0; j < 4; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  // Set batch_norm_param
  if (sc_param.scale_param().has_filler()) {
    layer_param->mutable_batch_norm_param()->mutable_scale_filler()->CopyFrom(
        sc_param.scale_param().filler());
  } else {
    layer_param->mutable_batch_norm_param()->mutable_scale_filler()->set_type(
        "constant");
    layer_param->mutable_batch_norm_param()->mutable_scale_filler()->set_value(
        1.);
  }
  if (sc_param.scale_param().has_bias_filler()) {
    layer_param->mutable_batch_norm_param()->mutable_bias_filler()->CopyFrom(
        sc_param.scale_param().bias_filler());
  } else {
    layer_param->mutable_batch_norm_param()->mutable_bias_filler()->set_type(
        "constant");
    layer_param->mutable_batch_norm_param()->mutable_bias_filler()->set_value(
        0.);
  }
  // Copy Blobs
  if (binary) {
    CHECK_EQ(sc_param.blobs_size(), 2)
        << " Wrong ScaleLayer blob size of layer: " << sc_param.name();
    CHECK_EQ(bn_param.blobs_size(), 3)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    *(layer_param->add_blobs()) = sc_param.blobs(0); // scale
    *(layer_param->add_blobs()) = sc_param.blobs(1); // bias
    *(layer_param->add_blobs()) = bn_param.blobs(0); // mean
    *(layer_param->add_blobs()) = bn_param.blobs(1); // variance
    *(layer_param->add_blobs()) = bn_param.blobs(2); // redundant_blob
    caffe_scal<float>(
        layer_param->blobs(0).data_size(),
        1 / layer_param->blobs(4).data().data()[0],
        layer_param->mutable_blobs(2)->mutable_data()->mutable_data());
    caffe_scal<float>(
        layer_param->blobs(0).data_size(),
        1 / layer_param->blobs(4).data().data()[0],
        layer_param->mutable_blobs(3)->mutable_data()->mutable_data());
    layer_param->mutable_blobs(4)->set_data(0, 1);
    // Change blob dim
    for (size_t i = 0; i < layer_param->blobs_size() - 1; i++) {
      if (layer_param->blobs(i).shape().dim_size() == 1) {
        DLOG(INFO) << "Update blob shape dim: " << layer_param->name()
                   << " blob[" << i << "]";
        layer_param->mutable_blobs(i)->mutable_shape()->clear_dim();
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(
            bn_param.blobs(0).shape().dim(0));
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
      } else if (layer_param->blobs(i).shape().dim_size() != 4) {
        DLOG(WARNING) << "Warning: wrong dim_size for layer: "
                      << layer_param->name() << " blob[" << i << "]";
      }
    }
  }
  processed[sc_index] = true;
  processed[bn_index] = true;
}

void ConvertBvlcBatchNorm(const NetParameter &net_in, NetParameter &net_out,
                          const int bn_index, vector<bool> &processed,
                          bool binary) {
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(bn_param);
  layer_param->set_type("BatchNorm");
  DLOG(INFO) << "Update BVLC BatchNorm: " << layer_param->name();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 3)
        << "Wrong Bvlc BatchNormLayer blobs_size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    // Make up scale and bias blobs
    *(layer_param->add_blobs()) = bn_param.blobs(0); // scale
    *(layer_param->add_blobs()) = bn_param.blobs(0); // bias
    for (size_t i = 0; i < bn_param.blobs(0).shape().dim(0); i++) {
      layer_param->mutable_blobs(0)->set_data(i, 1);
      layer_param->mutable_blobs(1)->set_data(i, 0);
    }
    // Copy mean and variance blobs, take care of scale_factor
    *(layer_param->add_blobs()) = bn_param.blobs(0); // mean
    *(layer_param->add_blobs()) = bn_param.blobs(1); // variance
    float scale_factor =
        bn_param.blobs(2).data(0) == 0 ? 0 : 1 / bn_param.blobs(2).data(0);
    for (size_t i = 0; i < bn_param.blobs(0).shape().dim(0); i++) {
      layer_param->mutable_blobs(2)->set_data(i, bn_param.blobs(0).data(i) *
                                                     scale_factor);
      layer_param->mutable_blobs(3)->set_data(i, bn_param.blobs(1).data(i) *
                                                     scale_factor);
    }
    BlobProto *redundant_blob = layer_param->add_blobs(); // redundant_blob
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->add_data(1);
    // Change blob dim
    for (size_t i = 0; i < layer_param->blobs_size() - 1; i++) {
      if (layer_param->blobs(i).shape().dim_size() == 1) {
        DLOG(INFO) << "Update blob shape dim: " << layer_param->name()
                   << " blob[" << i << "]";
        layer_param->mutable_blobs(i)->mutable_shape()->clear_dim();
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(
            bn_param.blobs(0).shape().dim(0));
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
      } else if (layer_param->blobs(i).shape().dim_size() != 4) {
        DLOG(WARNING) << "Warning: wrong dim_size for layer: "
                      << layer_param->name() << " blob[" << i << "]";
      }
    }
  }
  processed[bn_index] = true;
}

void ConvertBN2BatchNorm(const NetParameter &net_in, NetParameter &net_out,
                         const int bn_index, vector<bool> &processed,
                         bool binary) {
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(bn_param);
  layer_param->set_type("BatchNorm");
  DLOG(INFO) << "Change BN to BatchNorm: " << layer_param->name();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 4) << "Wrong BNLayer blobs_size of layer: "
                                       << bn_param.name();
    layer_param->clear_blobs();
    *(layer_param->add_blobs()) = bn_param.blobs(0);      // scale
    *(layer_param->add_blobs()) = bn_param.blobs(1);      // bias
    *(layer_param->add_blobs()) = bn_param.blobs(2);      // mean
    *(layer_param->add_blobs()) = bn_param.blobs(3);      // variance
    BlobProto *redundant_blob = layer_param->add_blobs(); // redundant_blob
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->add_data(1);
    // Change blob dim
    for (size_t i = 0; i < layer_param->blobs_size() - 1; i++) {
      if (layer_param->blobs(i).shape().dim_size() == 1) {
        DLOG(INFO) << "Update blob shape dim: " << layer_param->name()
                   << " blob[" << i << "]";
        layer_param->mutable_blobs(i)->mutable_shape()->clear_dim();
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(
            bn_param.blobs(0).shape().dim(0));
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
      } else if (layer_param->blobs(i).shape().dim_size() != 4) {
        DLOG(WARNING) << "Warning: wrong dim_size for layer: "
                      << layer_param->name() << " blob[" << i << "]";
      }
    }
  }
  processed[bn_index] = true;
}

// Convert float2fix
BatchNormParameter BuildBatchNormParam() {
  BatchNormParameter batch_norm_param;
  batch_norm_param.mutable_scale_filler()->set_type("constant");
  batch_norm_param.mutable_scale_filler()->set_value(1.);
  batch_norm_param.mutable_bias_filler()->set_type("constant");
  batch_norm_param.mutable_bias_filler()->set_value(0.);
  return batch_norm_param;
}

FixedParameter BuildFixedParam(const int method, const int bit_width) {
  FixedParameter fixed_param;
  if (method == 0) {
    fixed_param.set_fixed_method(FixedParameter_FixedMethod_OVER_FLOW);
  } else if (method == 1) {
    fixed_param.set_fixed_method(FixedParameter_FixedMethod_DIFF_S);
  } else if (method == 2) {
    fixed_param.set_fixed_method(FixedParameter_FixedMethod_DIFF_A);
  } else if (method == 3) {
    fixed_param.set_fixed_method(FixedParameter_FixedMethod_DIFF_S_SIGMOID);
  } else {
    LOG(FATAL) << "Unsupported Fix Method: " << method;
  }
  fixed_param.set_bit_width(bit_width);
  return fixed_param;
}

void InsertFixedNeuron(std::unordered_set<std::string> &fixed_blobs,
                       const std::string &name, NetParameter &net,
                       const int method, const int data_bit,
                       bool isLastDataLayer) {
  if (data_bit <= 0 || data_bit >= 64)
    return;
  if (fixed_blobs.find(name) == fixed_blobs.end()) {
    fixed_blobs.insert(name);
    LayerParameter *layer_param = net.add_layer();
    layer_param->set_name(name + "_fixed");
    layer_param->set_type("FixedNeuron");
    layer_param->add_bottom(name);
    layer_param->add_top(name);

    // add fixed_param
    FixedParameter data_fixed_param = BuildFixedParam(method, data_bit);
    if (isLastDataLayer)
      data_fixed_param.set_follow_data_layer(true);
    *(layer_param->mutable_fixed_param()) = data_fixed_param;

    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(0);
    param->set_decay_mult(0);
  }
}

void SearchInsertNeuronLayer(const NetParameter &net_in, NetParameter &net_out,
                             const int index, vector<bool> &need_insert_fix,
                             std::unordered_set<std::string> &fixed_blobs,
                             const vector<vector<bool>> &connect,
                             const int method, const int data_bit, 
                             map<string, pair<int,int> > &special_layers) {
  // It is possible that current layer top is not only connected to a pooling
  // layer, but also other types layers at the same time in a model. So it is
  // safe to insert the fixed neuron layer before pooling layer.
  //
  // Inception_V1 concat layers bottoms are the only one tops of their previous
  // layers. In this situation, only insert fixed neuron after concat.
  //

  int notAlignConcat = 0;
  //if (getenv("NOT_ALIGN_CONCAT"))
  //  notAlignConcat = stoi(getenv("NOT_ALIGN_CONCAT"));

  int concat_index = GetNextLayer(net_in, index, "Concat", connect);
  // handle conv + permute + flatten in ssd
  if (concat_index == -1) {
    int permute_index = GetNextLayer(net_in, index, "Permute", connect);
    if (permute_index != -1) {
      int flatten_index =
          GetNextLayer(net_in, permute_index, "Flatten", connect);
      if (flatten_index != -1) {
        concat_index = GetNextLayer(net_in, flatten_index, "Concat", connect);
      }
    }
  }
  if (!isSingleOutputLayer(net_in, index, connect)) {
    for (auto i = 0; i < net_in.layer(index).top_size(); ++i) {
      InsertFixedNeuron(fixed_blobs, net_in.layer(index).top(i), net_out,
                        method, data_bit);
      need_insert_fix[index] = true;
    }
  } else {
    int relu_index = GetNextLayer(net_in, index, "ReLU", connect);
    if (relu_index != -1) {
      concat_index = GetNextLayer(net_in, relu_index, "Concat", connect);
      if (!isSingleOutputLayer(net_in, relu_index, connect)){
        need_insert_fix[relu_index] = true;
        if (special_layers.count(net_in.layer(index).name()) > 0){
            pair<int,int> tmp_pair(special_layers[net_in.layer(index).name()].first, special_layers[net_in.layer(index).name()].second);
            special_layers[net_in.layer(relu_index).name()] = tmp_pair;
        }
      }
      else {
        int pooling_index =
            GetNextLayer(net_in, relu_index, "Pooling", connect);
        if (pooling_index != -1 &&
            net_in.layer(pooling_index).pooling_param().pool() !=
                caffe::PoolingParameter_PoolMethod_AVE) {
          concat_index = GetNextLayer(net_in, pooling_index, "Concat", connect);
          if (!isSingleOutputLayer(net_in, pooling_index, connect)){
            need_insert_fix[pooling_index] = true;
            if (special_layers.count(net_in.layer(index).name()) > 0){
                pair<int,int> tmp_pair(special_layers[net_in.layer(index).name()].first, special_layers[net_in.layer(index).name()].second);
                special_layers[net_in.layer(pooling_index).name()] = tmp_pair;
            }
          }
          else if (concat_index == -1 || notAlignConcat){
            // ReLU + MAX Pooling
            need_insert_fix[pooling_index] = true;
            if (special_layers.count(net_in.layer(index).name()) > 0){
                pair<int,int> tmp_pair(special_layers[net_in.layer(index).name()].first, special_layers[net_in.layer(index).name()].second);
                special_layers[net_in.layer(pooling_index).name()] = tmp_pair;      
            }
          }
        } else {
          // ReLU
          if (concat_index == -1 || notAlignConcat){
            need_insert_fix[relu_index] = true;
            if (special_layers.count(net_in.layer(index).name()) > 0){
                pair<int,int> tmp_pair(special_layers[net_in.layer(index).name()].first, special_layers[net_in.layer(index).name()].second);
                special_layers[net_in.layer(relu_index).name()] = tmp_pair;            
            }
          }
        }
      }
    } else {
      // MAX Pooling
      int pooling_index = GetNextLayer(net_in, index, "Pooling", connect);
      if (pooling_index != -1 &&
          net_in.layer(pooling_index).pooling_param().pool() !=
              caffe::PoolingParameter_PoolMethod_AVE) {
        concat_index = GetNextLayer(net_in, pooling_index, "Concat", connect);
        if (!isSingleOutputLayer(net_in, pooling_index, connect)){
          need_insert_fix[pooling_index] = true;
          if (special_layers.count(net_in.layer(index).name()) > 0){
              pair<int,int> tmp_pair(special_layers[net_in.layer(index).name()].first, special_layers[net_in.layer(index).name()].second);
              special_layers[net_in.layer(pooling_index).name()] = tmp_pair;
          }
        }
        else if (concat_index == -1 || notAlignConcat){
          need_insert_fix[pooling_index] = true;
          if (special_layers.count(net_in.layer(index).name()) > 0){
              pair<int,int> tmp_pair(special_layers[net_in.layer(index).name()].first, special_layers[net_in.layer(index).name()].second);
              special_layers[net_in.layer(pooling_index).name()] = tmp_pair;
          }
        }
      } else {
        if (concat_index == -1 || notAlignConcat) {
          InsertFixedNeuron(fixed_blobs, net_in.layer(index).top(0), net_out,
                            method, data_bit);
          need_insert_fix[index] = true;
        }
      }
    }
  }
}

void ConvertConvolution2Fixed(const NetParameter &net_in, NetParameter &net_out,
                              const int index, vector<bool> &need_insert_fix,
                              const vector<vector<bool>> &connect,
                              std::unordered_set<std::string> &fixed_blobs,
                              vector<bool> &processed, const int method,
                              const int weight_bit, const int data_bit,
                              const bool add_bn, map<string, pair<int,int> > &special_layers) {
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(net_in.layer(index));

  if (add_bn) {
    layer_param->set_type("ConvolutionBNFixed");
    layer_param->mutable_convolution_param()->set_bias_term(false);
    *(layer_param->mutable_batch_norm_param()) = BuildBatchNormParam();
    *(layer_param->mutable_fixed_param()) = BuildFixedParam(method, weight_bit);
    // Set param
    layer_param->clear_param();
    float lr_mult[5] = {1., 1., 1., 0., 0.};
    float decay_mult[5] = {1., 0., 0., 0., 0.};
    for (int j = 0; j < 5; j++) {
      ParamSpec *param = layer_param->add_param();
      param->set_lr_mult(lr_mult[j]);
      param->set_decay_mult(decay_mult[j]);
    }

    // Conv + BatchNorm -> ConvBNFix
    SearchInsertNeuronLayer(net_in, net_out, index, need_insert_fix,
                            fixed_blobs, connect, method, data_bit, special_layers);
    processed[index] = true;

  } else {
    // Conv -> ConvFix
    layer_param->set_type("ConvolutionFixed");
    *(layer_param->mutable_fixed_param()) =
        BuildFixedParam(method == 3 ? 1 : method, weight_bit);
    SearchInsertNeuronLayer(net_in, net_out, index, need_insert_fix,
                            fixed_blobs, connect, method, data_bit, special_layers);
    processed[index] = true;
  }
  return;
}

void ConvertBatchNorm2Fixed(const NetParameter &net_in, NetParameter &net_out,
                            const int index, vector<bool> &need_insert_fix,
                            const vector<vector<bool>> &connect,
                            std::unordered_set<std::string> &fixed_blobs,
                            vector<bool> &processed, const int method,
                            const int weight_bit, const int data_bit, 
                            map<string, pair<int,int> > &special_layers) {
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(net_in.layer(index));
  layer_param->set_type("BatchNormFixed");
  *(layer_param->mutable_fixed_param()) = BuildFixedParam(method, weight_bit);
  layer_param->set_phase(TEST);
  // Set param
  layer_param->clear_param();
  float lr_mult[5] = {1., 1., 1., 0., 0.};
  float decay_mult[5] = {1., 0., 0., 0., 0.};
  for (int j = 0; j < 5; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  SearchInsertNeuronLayer(net_in, net_out, index, need_insert_fix, fixed_blobs,
                          connect, method, data_bit, special_layers);
  processed[index] = true;
}

void ConvertInnerProduct2Fixed(const NetParameter &net_in,
                               NetParameter &net_out, const int index,
                               vector<bool> &need_insert_fix,
                               const vector<vector<bool>> &connect,
                               std::unordered_set<std::string> &fixed_blobs,
                               vector<bool> &processed, const int method,
                               const int weight_bit, const int data_bit, 
                               map<string, pair<int,int> > &special_layers) {
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(net_in.layer(index));
  layer_param->set_type("InnerProductFixed");
  *(layer_param->mutable_fixed_param()) = BuildFixedParam(method, weight_bit);
  SearchInsertNeuronLayer(net_in, net_out, index, need_insert_fix, fixed_blobs,
                          connect, method, data_bit, special_layers);
  processed[index] = true;
}

void ConvertPReLU2Fixed(const NetParameter &net_in, NetParameter &net_out,
                        const int index, vector<bool> &need_insert_fix,
                        const vector<vector<bool>> &connect,
                        std::unordered_set<std::string> &fixed_blobs,
                        vector<bool> &processed, const int method,
                        const int weight_bit, const int data_bit, 
                        map<string, pair<int,int> > &special_layers) {
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(net_in.layer(index));
  layer_param->set_type("PReLUFixed");
  *(layer_param->mutable_fixed_param()) = BuildFixedParam(method, weight_bit);
  SearchInsertNeuronLayer(net_in, net_out, index, need_insert_fix, fixed_blobs,
                          connect, method, data_bit, special_layers);
  processed[index] = true;
}

void ConvertDeconvolution2Fixed(const NetParameter &net_in,
                                NetParameter &net_out, const int index,
                                vector<bool> &need_insert_fix,
                                const vector<vector<bool>> &connect,
                                std::unordered_set<std::string> &fixed_blobs,
                                vector<bool> &processed, const int method,
                                const int weight_bit, const int data_bit, 
                                map<string, pair<int,int> > &special_layers) {
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(net_in.layer(index));
  if (layer_param->convolution_param().group() == 1 ||
      layer_param->convolution_param().group() ==
          layer_param->convolution_param().num_output()) {
    LOG(WARNING) << "1-Group|Fully-Group Deconvolution Layer ("
                 << layer_param->name() << ") detected.";
    layer_param->set_type("DeconvolutionFixed");
    *(layer_param->mutable_fixed_param()) = BuildFixedParam(method, weight_bit);
  } else {
    LOG(FATAL) << "Deconvolution Layer (" << layer_param->name()
               << ") is not supported by Deephi DNNDK, please change it to "
                  "either fully-group or 1-group and retrain.";
  }
  SearchInsertNeuronLayer(net_in, net_out, index, need_insert_fix, fixed_blobs,
                          connect, method, data_bit, special_layers);
  processed[index] = true;
}

void ConvertScale2Fixed(const NetParameter &net_in, NetParameter &net_out,
                        const int index, vector<bool> &need_insert_fix,
                        const vector<vector<bool>> &connect,
                        std::unordered_set<std::string> &fixed_blobs,
                        vector<bool> &processed, const int method,
                        const int weight_bit, const int data_bit, 
                        map<string, pair<int,int> > &special_layers) {
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(net_in.layer(index));
  layer_param->set_type("ScaleFixed");
  *(layer_param->mutable_fixed_param()) = BuildFixedParam(method, weight_bit);
  SearchInsertNeuronLayer(net_in, net_out, index, need_insert_fix, fixed_blobs,
                          connect, method, data_bit, special_layers);
  processed[index] = true;
}

void ConvertNormalize2Fixed(const NetParameter &net_in, NetParameter &net_out,
                            const int index, vector<bool> &need_insert_fix,
                            const vector<vector<bool>> &connect,
                            std::unordered_set<std::string> &fixed_blobs,
                            vector<bool> &processed, const int method,
                            const int weight_bit, const int data_bit, 
                            map<string, pair<int,int> > &special_layers) {
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(net_in.layer(index));
  layer_param->set_type("NormalizeFixed");
  *(layer_param->mutable_fixed_param()) = BuildFixedParam(method, weight_bit);
  SearchInsertNeuronLayer(net_in, net_out, index, need_insert_fix, fixed_blobs,
                          connect, method, data_bit, special_layers);
  processed[index] = true;
}

bool NeedIgnore(const string &layer_name,
                std::unordered_map<std::string, bool> &ignore_layers) {
  bool need_ignore = false;
  if (ignore_layers.count(layer_name)) {
    need_ignore = true;
    ignore_layers.at(layer_name) = true;
    DLOG(INFO) << "Ignore layers found: " << layer_name;
  }
  return need_ignore;
}

void ConvertSoftLoss2Soft(NetParameter &net, const int soft_index) {
  CHECK_EQ(net.layer(soft_index).type(), "SoftmaxWithLoss");
  LayerParameter *soft_param = net.mutable_layer(soft_index);
  soft_param->set_type("Softmax");
  soft_param->clear_include();
  CHECK_GE(soft_param->bottom_size(), 2);
  soft_param->mutable_bottom()->DeleteSubrange(1,
                                               soft_param->bottom_size() - 1);
  soft_param->mutable_top()->DeleteSubrange(1, soft_param->top_size() - 1);
}

// Convert train_val.prototxt to deploy.prototxt
// Function:
//      1) Remove Split
//      2) Remove Train Phase Layer
//      3) Remove Test Phase Layer
//      4) Remove Dropout Layer
//      5) Convert SoftmaxWithLoss to Softmax
NetParameter ConvertTrain2Deploy(const NetParameter &train_net) {
  int debugMode = 0;
  if (getenv("DECENT_DEBUG"))
    debugMode = stoi(getenv("DECENT_DEBUG"));

  NetParameter deploy_net = RemoveLayerByPhase(train_net, TRAIN);
  deploy_net = RemoveLayerByPhase(deploy_net, TEST);
  deploy_net = RemoveLayerByType(deploy_net, "Split");
  deploy_net = RemoveLayerByType(deploy_net, "Dropout");
  if (debugMode == 6)
    deploy_net = RemoveLayerByType(deploy_net, "ImageData");
  if (debugMode > 1)
    // debug mode to match Caffe deploy dev branch accuracy
    deploy_net = RemoveLayerByType(deploy_net, "SoftmaxWithLoss");
  else {
    for (auto i = 0; i < deploy_net.layer_size(); i++) {
      auto &cur_layer = deploy_net.layer(i);
      if (cur_layer.type() == "SoftmaxWithLoss") {
        ConvertSoftLoss2Soft(deploy_net, i);
      }
    }
  }
  return deploy_net;
}

NetParameter MergeBvlcBatchNormInNet(const NetParameter &net_in, bool binary) {
  vector<vector<bool>> connect = BuildConnections(net_in);
  vector<bool> processed(net_in.layer_size(), false);
  NetParameter net_out;

  for (int i = 0; i < net_in.layer_size(); i++) {
    if (processed[i]) {
      continue;
    }
    const LayerParameter *cur_layer = &net_in.layer(i);
    if (cur_layer->type() == "BatchNorm") {
      int scale_index = GetNextLayer(net_in, i, "Scale", connect);
      int b_size = cur_layer->blobs_size();
      if (binary)
        CHECK(b_size == 3 || b_size == 5)
            << "Wrong Blobs for BatchNorm: " << b_size
            << ", Only support BatchNorm with 3 blobs or 5 blobs";
      if (!binary) {
        // For prototxt
        if (scale_index != -1) {
          DLOG(INFO) << " Merge BvlcBatchNorm + Scale -> BatchNorm: "
                     << net_in.layer(i).name() << " + "
                     << net_in.layer(scale_index).name();
          MergeBvlcBatchNormScale2BatchNorm(net_in, net_out, i, scale_index,
                                            processed, binary);
        } else {
          *(net_out.add_layer()) = *cur_layer;
          processed[i] = true;
        }
      } else {
        // For Caffemodel
        if (scale_index != -1) {
          CHECK(b_size == 3)
              << "Wrong Blobs for BatchNorm Layer " << net_in.layer(i).name()
              << ", BatchNorm with 5 blobs should not followed by Scale layer.";
          DLOG(INFO) << " Merge BvlcBatchNorm + Scale -> BatchNorm: "
                     << net_in.layer(i).name() << " + "
                     << net_in.layer(scale_index).name();
          MergeBvlcBatchNormScale2BatchNorm(net_in, net_out, i, scale_index,
                                            processed, binary);
        } else {
          if (b_size == 3) {
            LOG(WARNING) << "BVLC BatchNormLayer without ScaleLayer detected: "
                         << i << ", it will be converted to NVCaffe Format.";
            ConvertBvlcBatchNorm(net_in, net_out, i, processed, binary);
          } else {
            *(net_out.add_layer()) = *cur_layer;
            processed[i] = true;
          }
        }
      }

    } else if (cur_layer->type() == "BN") {
      ConvertBN2BatchNorm(net_in, net_out, i, processed, binary);

    } else {
      *(net_out.add_layer()) = *cur_layer;
      processed[i] = true;
    }
  }
  return net_out;
}

NetParameter MergePostBatchNormInNet(const NetParameter &net_in, bool binary,
                                     bool keep_convbn) {
  vector<vector<bool>> connect = BuildConnections(net_in);
  vector<bool> processed(net_in.layer_size(), false);
  NetParameter net_out;

  for (int i = 0; i < net_in.layer_size(); i++) {
    if (processed[i]) {
      continue;
    }
    const LayerParameter *cur_layer = &net_in.layer(i);
    // Merge Convolution + BatchNorm
    if (cur_layer->type() == "Convolution") {
      int bn_index = GetNextLayer(net_in, i, "BatchNorm", connect);
      int relu_index = GetNextLayer(net_in, i, "ReLU", connect);
      // stop merge for Convolution + Relu + Batchnorm
      if (bn_index != -1 && (relu_index == -1 || relu_index > bn_index)) {
        if (!keep_convbn) {
          // Merge Conv + BatchNorm -> Conv
          DLOG(INFO) << " Merge ConvBatchNorm -> Conv: "
                     << net_in.layer(i).name() << " + "
                     << net_in.layer(bn_index).name();
          MergeConvBatchNorm2Conv(net_in, net_out, i, bn_index, processed,
                                  binary);
        } else {
          // Merge Conv + BatchNorm -> ConvBNFix
          DLOG(INFO) << " Merge ConvBatchNorm -> ConvBNFixed: "
                     << net_in.layer(i).name() << " + "
                     << net_in.layer(bn_index).name();
          MergeConvBatchNorm2ConvBNFixed(net_in, net_out, i, bn_index,
                                         processed, binary);
        }
      } else {
        *(net_out.add_layer()) = *cur_layer;
        processed[i] = true;
      }

    } else if (cur_layer->type() == "Deconvolution") {
      int bn_index = GetNextLayer(net_in, i, "BatchNorm", connect);
      int relu_index = GetNextLayer(net_in, i, "ReLU", connect);
      // stop merge for Convolution + Relu + Batchnorm
      if (bn_index != -1 && (relu_index == -1 || relu_index > bn_index)) {
        // Merge Deconv + BatchNorm -> Deconv
        DLOG(INFO) << " Merge DeconvBatchNorm -> Deconv: "
                   << net_in.layer(i).name() << " + "
                   << net_in.layer(bn_index).name();
        MergeDeconvBatchNorm2Deconv(net_in, net_out, i, bn_index, processed,
                                    binary);
      } else {
        *(net_out.add_layer()) = *cur_layer;
        processed[i] = true;
      }

    } else if (cur_layer->type() == "InnerProduct") {
      int bn_index = GetNextLayer(net_in, i, "BatchNorm", connect);
      if (bn_index != -1) {
        // Merge InnerProduct + BatchNorm -> InnerProduct
        LOG(INFO) << " Merge InnerProductBatchNorm -> InnerProduct: "
                  << net_in.layer(i).name() << " + "
                  << net_in.layer(bn_index).name();
        MergeFCBatchNorm2FC(net_in, net_out, i, bn_index, processed, binary);
      } else {
        *(net_out.add_layer()) = *cur_layer;
        processed[i] = true;
      }

    } else {
      *(net_out.add_layer()) = *cur_layer;
      processed[i] = true;
    }
  }
  return net_out;
}

NetParameter MergePreBatchNormInNet(const NetParameter &net_in, bool binary,
                                    bool keep_convbn) {
  vector<vector<bool>> connect = BuildConnections(net_in);
  vector<bool> processed(net_in.layer_size(), false);
  NetParameter net_out;

  for (int i = 0; i < net_in.layer_size(); i++) {
    if (processed[i]) {
      continue;
    }
    const LayerParameter *cur_layer = &net_in.layer(i);
    // Merge BatchNorm + Convolution
    if (cur_layer->type() == "BatchNorm") {
      bool need_merge = false;
      int conv_index = GetNextLayer(net_in, i, "Convolution", connect);
      int relu_index = GetNextLayer(net_in, i, "ReLU", connect);
      // stop merge for batchnorm + relu + convolution
      // stop merge for batchnorm + convolution_with_pad
      if (conv_index != -1 && (relu_index == -1 || relu_index > conv_index)) {
        auto &conv_param = net_in.layer(conv_index).convolution_param();
        if (conv_param.pad_size()) {
          for (int j = 0; j < conv_param.pad_size(); j++) {
            if (conv_param.pad(j) != 0) {
              need_merge = false;
              break;
            } else {
              need_merge = true;
            }
          }
        } else if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
          need_merge = false;
        } else {
          need_merge = true;
        }
      }

      if (need_merge) {
        if (!keep_convbn) {
          // Merge Conv + BatchNorm -> Conv
          DLOG(INFO) << " Merge BatchNormConv -> Conv: "
                     << net_in.layer(i).name() << " + "
                     << net_in.layer(conv_index).name();
          MergeBatchNormConv2Conv(net_in, net_out, i, conv_index, processed,
                                  binary);
        } else {
          // Merge BatchNorm + Conv -> ConvBNFix
          DLOG(INFO) << " Merge BatchNormConv -> ConvBNFixed: "
                     << net_in.layer(i).name() << " + "
                     << net_in.layer(conv_index).name();
          MergeConvBatchNorm2ConvBNFixed(net_in, net_out, conv_index, i,
                                         processed, binary);
        }

      } else {
        // Merge BatchNorm -> Scale
        DLOG(INFO) << " Merge BatchNorm -> Scale: " << net_in.layer(i).name();
        MergeBatchNorm2Scale(net_in, net_out, i, processed, binary);
      }

    } else {
      *(net_out.add_layer()) = *cur_layer;
      processed[i] = true;
    }
  }
  return net_out;
}

// This is a function to merge Scale layer into BatchNorm Layer or BatchNorm
// layer into Convolution Layer for prototxt or caffemodel, support caffe1,
// nvcaffe, caffe_parallel's float model.
// Function:
//      1) Convolution + BatchNorm + (Scale) -> Convolution
//      2) (If keep_bn) BatchNorm + Scale -> BatchNorm
//      3) (If keep_convbn) Convolution + BatchNorm + (Scale) ->
//      ConvolutionBNFixed
int MergeLayers(const NetParameter &net_in, NetParameter &net_out, bool binary,
                bool keep_bn, bool keep_convbn) {
  // 0. InsertSplits
  NetParameter net_split;
  // ignore bottom err in InsertSplits if binary == true
  InsertSplits(net_in, &net_split, binary);

  // 1. Merge BatchNorm + Scale and deal with BVLCBatchNorm/BN
  NetParameter net_1 = MergeBvlcBatchNormInNet(net_split, binary);

  // 2. Merge Convolution + BatchNorm && InnerProduce + BatchNorm
  if (!keep_bn) {
    NetParameter net_2 = MergePostBatchNormInNet(net_1, binary, keep_convbn);
    net_out = MergePreBatchNormInNet(net_2, binary, keep_convbn);
  } else {
    net_out = net_1;
  }
  // 3. RemoveSplits
  net_out = RemoveLayerByType(net_out, "Split");
  return 0;
}

// This is a script to convert float prototxt to the fix prototxt.
// Usage:
//    convert_float2fix float_prototxt_in fix_prototxt_out
// Function:
//    1) Convolution -> ConvolutionFixed or ConvoluitonBNFixed
//    2) Convolution + BatchNorm -> ConvolutionBNFixed
//    3) Convolution + BatchNorm + Scale -> ConvolutionBNFixed
//    4) BatchNorm -> BatchNormFixed
//    5) BatchNorm + Scale -> BatchNormFixed
//    6) InnerProduct -> InnerProductFixed
//    7) Insert FixedNeuron:
//       a) After Last Input/ImageData/Data layer
//       b) After ConvolutionFixed + (ReLu) + (Max Pooling)
//       c) After ConvolutionBNFixed + (ReLu) + (Max Pooling)
//       d) After BatchNormFixed
//       e) After InnerProductFixed
//       f) After Eltwise
//       g) After Concat, Note: FixedNeuron right before Concat will be delete
int ConvertFloat2Fix(const NetParameter &net_in, NetParameter &net_out,
                     const string &FLAGS_ignore_layers,
                     const string &FLAGS_ignore_layers_file, const int method,
                     const int origin_weight_bit, const int origin_data_bit,
                     const bool add_bn, const string &FLAGS_sigmoided_layers, 
                     const string &FLAGS_special_layers) {

  vector<vector<bool>> connect = BuildConnections(net_in);
  vector<bool> need_insert_fix(net_in.layer_size(), false);
  vector<bool> processed(net_in.layer_size(), false);
  std::unordered_set<std::string> fixed_blobs;

  map<string, pair<int,int> > special_layers;
  if (FLAGS_special_layers != "")
    handleParam2map(FLAGS_special_layers, special_layers, origin_data_bit);

  std::unordered_map<std::string, bool> ignore_layers;
  InitIgnoreFile(ignore_layers, FLAGS_ignore_layers_file, FLAGS_ignore_layers);

  vector<string> sigmoided_layers;
  InitSigmoidedLayers(sigmoided_layers, FLAGS_sigmoided_layers);

  int last_input_layer = GetLastInputLayer(net_in);

  for (int i = 0; i < net_in.layer_size(); i++) {
    if (processed[i]) {
      continue;
    }
    const LayerParameter *cur_layer = &net_in.layer(i);

    int data_bit = origin_data_bit;
    int weight_bit = origin_weight_bit;
    if (special_layers.count(cur_layer->name()) > 0)
    {
      data_bit = special_layers[cur_layer->name()].first;
      weight_bit = special_layers[cur_layer->name()].second;
    }

    if (NeedIgnore(cur_layer->name(), ignore_layers)) {
      *(net_out.add_layer()) = *cur_layer;
      processed[i] = true;
      continue;
    }

    if (i == last_input_layer) {
      *(net_out.add_layer()) = *cur_layer;
      // Use Overflow method for last input layer
      InsertFixedNeuron(fixed_blobs, net_in.layer(i).top(0), net_out, 0,
                        data_bit, true);
      processed[i] = true;

    } else if (cur_layer->type() == "Convolution") {
      if (method == 1 &&
          find(sigmoided_layers.begin(), sigmoided_layers.end(),
               cur_layer->name()) != sigmoided_layers.end()) {
        ConvertConvolution2Fixed(net_in, net_out, i, need_insert_fix, connect,
                                 fixed_blobs, processed, 3, weight_bit,
                                 data_bit, add_bn, special_layers);
      } else {
        ConvertConvolution2Fixed(net_in, net_out, i, need_insert_fix, connect,
                                 fixed_blobs, processed, method, weight_bit,
                                 data_bit, add_bn, special_layers);
      }

    } else if (cur_layer->type() == "Pooling") {
      *(net_out.add_layer()) = *cur_layer;
      if (need_insert_fix[i]) {
        InsertFixedNeuron(fixed_blobs, cur_layer->top(0), net_out, method,
                          data_bit);
      } else if (net_in.layer(i).pooling_param().pool() ==
                 caffe::PoolingParameter_PoolMethod_AVE) {
        int notAlignConcat = 0;
        if (getenv("NOT_ALIGN_CONCAT"))
          notAlignConcat = stoi(getenv("NOT_ALIGN_CONCAT"));

        int concat_index = GetNextLayer(net_in, i, "Concat", connect);
        if (!isSingleOutputLayer(net_in, i, connect)
            || concat_index == -1 
            || notAlignConcat ) {
          InsertFixedNeuron(fixed_blobs, cur_layer->top(0), net_out, method,
                            data_bit);
        }
      }
      processed[i] = true;

    } else if (cur_layer->type() == "ReLU") {
      *(net_out.add_layer()) = *cur_layer;
      if (need_insert_fix[i]) {
        InsertFixedNeuron(fixed_blobs, cur_layer->top(0), net_out, method,
                          data_bit);
      }
      processed[i] = true;

    } else if (cur_layer->type() == "ReLU6") {
      *(net_out.add_layer()) = *cur_layer;
      InsertFixedNeuron(fixed_blobs, cur_layer->top(0), net_out, method,
                        data_bit);
      processed[i] = true;

    } else if (cur_layer->type() == "Slice") {
      *(net_out.add_layer()) = *cur_layer;
      SearchInsertNeuronLayer(net_in, net_out, i, need_insert_fix, fixed_blobs,
                              connect, method, data_bit, special_layers);
      processed[i] = true;

    } else if (cur_layer->type() == "GSTiling" ||
               cur_layer->type() == "Tiling") {
      *(net_out.add_layer()) = *cur_layer;
      SearchInsertNeuronLayer(net_in, net_out, i, need_insert_fix, fixed_blobs,
                              connect, method, data_bit, special_layers);
      processed[i] = true;

    } else if (cur_layer->type() == "Eltwise") {
      *(net_out.add_layer()) = *cur_layer;
      SearchInsertNeuronLayer(net_in, net_out, i, need_insert_fix, fixed_blobs,
                              connect, method, data_bit, special_layers);
      processed[i] = true;

    } else if (cur_layer->type() == "Concat" ||
               cur_layer->type() == "DeephiResize") {
      *(net_out.add_layer()) = *cur_layer;
      if (cur_layer->type() == "Concat") {
        InsertFixedNeuron(fixed_blobs, cur_layer->top(0), net_out, method,
                          data_bit);
      } else
        SearchInsertNeuronLayer(net_in, net_out, i, need_insert_fix,
                                fixed_blobs, connect, method, data_bit, special_layers);
      processed[i] = true;

    } else if (cur_layer->type() == "InnerProduct") {
      ConvertInnerProduct2Fixed(net_in, net_out, i, need_insert_fix, connect,
                                fixed_blobs, processed, method, weight_bit,
                                data_bit, special_layers);

    } else if (cur_layer->type() == "PReLU") {
      ConvertPReLU2Fixed(net_in, net_out, i, need_insert_fix, connect,
                         fixed_blobs, processed, method, weight_bit, data_bit, special_layers);

    } else if (cur_layer->type() == "BatchNorm") {
      ConvertBatchNorm2Fixed(net_in, net_out, i, need_insert_fix, connect,
                             fixed_blobs, processed, method, weight_bit,
                             data_bit, special_layers);

    } else if (cur_layer->type() == "Deconvolution") {
      ConvertDeconvolution2Fixed(net_in, net_out, i, need_insert_fix, connect,
                                 fixed_blobs, processed, method, weight_bit,
                                 data_bit, special_layers);

    } else if (cur_layer->type() == "Scale") {
      ConvertScale2Fixed(net_in, net_out, i, need_insert_fix, connect,
                         fixed_blobs, processed, method, weight_bit, data_bit, special_layers);

    } else if (cur_layer->type() == "Normalize") {
      ConvertNormalize2Fixed(net_in, net_out, i, need_insert_fix, connect,
                             fixed_blobs, processed, method, weight_bit,
                             data_bit, special_layers);

    } else {
      *(net_out.add_layer()) = *cur_layer;
      processed[i] = true;
    }
  }

  // Output Ignore Layers not found
  for (auto iter = ignore_layers.begin(); iter != ignore_layers.end(); iter++) {
    if (iter->second == 0) {
      LOG(WARNING) << "Warning: Ignore layer not found: " << iter->first
                   << endl;
    }
  }
  return 0;
}

void InitIgnoreFile(std::unordered_map<std::string, bool> &ignore_layers,
                    const string ignore_layers_file_in,
                    const string ignore_layers_in) {
  // Read from ignore_layers_file_in
  deephi::QuantizeParameter quantize_param;
  if (ignore_layers_file_in != "") {
    caffe::ReadProtoFromTextFileOrDie(ignore_layers_file_in, &quantize_param);
    if (quantize_param.ignore_layer_size() == 0) {
      LOG(WARNING) << "Warning: Empty Ingnore Layers File!";
    }
    for (int i = 0; i < quantize_param.ignore_layer_size(); i++) {
      ignore_layers[quantize_param.ignore_layer(i)] = 0;
    }
  }

  // Read from ignore_layers_in
  if (ignore_layers_in != "") {
    size_t start = 0;
    size_t index = ignore_layers_in.find(",", start);
    while (index != std::string::npos) {
      ignore_layers[ignore_layers_in.substr(start, index - start)] = 0;
      start = index + 1;
      index = ignore_layers_in.find(",", start);
    }
    ignore_layers[ignore_layers_in.substr(start)] = 0;
  }
  for (auto it = ignore_layers.begin(); it != ignore_layers.end(); it++) {
    LOG(INFO) << "Ignore Layers List: " << it->first;
  }
}

void InitSigmoidedLayers(vector<string> &sigmoided_layers,
                         const string &sigmoided_layers_in) {
  if (sigmoided_layers_in != "") {
    size_t start = 0;
    size_t index = sigmoided_layers_in.find(",", start);
    while (index != std::string::npos) {
      sigmoided_layers.push_back(
          sigmoided_layers_in.substr(start, index - start));
      start = index + 1;
      index = sigmoided_layers_in.find(",", start);
    }
    sigmoided_layers.push_back(sigmoided_layers_in.substr(start));
  }
  for (int i = 0; i < sigmoided_layers.size(); i++) {
    DLOG(INFO) << "Sigmoided Layers List [" << i << "]:" << sigmoided_layers[i];
  }
}

void MergeModelAndWeights( NetParameter& model, 
                           NetParameter& weights ) {
  // Create a Net<float> and copy trained layers from weights to deal with
  // mismatch of model and weights.
  model.mutable_state()->set_phase(TRAIN);
  Net<float> caffe_net(model);
  caffe_net.CopyTrainedLayersFrom(weights);
  caffe_net.ToProto(&weights);
  CheckModel(weights);
  DLOG(INFO) << "net loaded. ";
} 

bool isNum(const string &str)  {  
    for (int i = 0; i < str.size(); i++){
        if (str[i] >= '0' && str[i] <= '9')
            continue;
        else
            return false;
    } 
    return true;
}

void handleParam2map(const string str, map<string, pair<int,int> > &amap, const int data_bit)
{
  vector<string> ret;
  size_t start = 0, index = str.find_first_of(',', 0);
  while (index != str.npos)
  {
    if (start != index)
      ret.push_back(str.substr(start, index - start));
    start = index + 1;
    index = str.find_first_of(',', start);
  }
  if (!str.substr(start).empty())
    ret.push_back(str.substr(start));

  for (size_t i = 0; i < ret.size(); i++){
    if (!isNum(ret[i])){
        if((i+1 < ret.size() && !isNum(ret[i+1])) || (i+1 >= ret.size()))
            CHECK_GT(0, 0) << "Wrong special_layers parameter format !";
        if (i+1 < ret.size() && i+2 < ret.size() && isNum(ret[i+1]) && isNum(ret[i+2])){
            pair<int,int> tmp_pair(atoi(ret[i+1].c_str()),atoi(ret[i+2].c_str()));
            amap.insert(make_pair(ret[i],tmp_pair));
            i = i + 2;
        } else {
            pair<int,int> tmp_pair(atoi(ret[i+1].c_str()), data_bit);
            amap.insert(make_pair(ret[i],tmp_pair));
            i++;
        }    
    } else
        CHECK_GT(0, 0) << "Wrong special_layers parameter format !";
  }
}

} // namespace caffe
