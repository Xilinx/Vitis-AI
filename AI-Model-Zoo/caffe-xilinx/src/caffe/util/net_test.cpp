#include <cstring>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include "caffe/util/bbox_util.hpp"
#include "caffe/util/convert_proto.hpp"
#include "caffe/util/net_test.hpp"
using namespace std;

namespace caffe {

void print_file_as_DPU(const vector<int> &shape, const float *input, int length,
                       int fixed_pos, string filename, bool dumpTxt) {
  if (shape.empty())
    return;
  int n = shape[0];
  int c = shape.size() < 2 ? 1 : shape[1];
  int h = 1, w = 1;
  if (shape.size() == 4) {
    h = shape[2];
    w = shape[3];
  }
  float scale = pow(2.0, fixed_pos);

  if (dumpTxt) {
    ofstream write_out(filename + ".txt");
    for (int i1 = 0; i1 < n; ++i1) {
      for (int i2 = 0; i2 < h; ++i2) {
        for (int i3 = 0; i3 < w; ++i3) {
          for (int i4 = 0; i4 < c; ++i4) {
            int offset = i1 * c * h * w + i4 * h * w + i2 * w + i3;
            int num = int(input[offset] * scale);
            write_out << num << endl;
          }
        }
      }
    }
    write_out.close();
  }
  ofstream write_out(filename + ".bin", ios::binary);
  for (int i1 = 0; i1 < n; ++i1) {
    for (int i2 = 0; i2 < h; ++i2) {
      for (int i3 = 0; i3 < w; ++i3) {
        for (int i4 = 0; i4 < c; ++i4) {
          int offset = i1 * c * h * w + i4 * h * w + i2 * w + i3;
          int num = int(input[offset] * scale);
          write_out.write((char *)&num, sizeof(char));
        }
      }
    }
  }
  write_out.close();
}

void print_file(const vector<int> &shape, const float *input, int length,
                string filename, float scale = 1.0) {
  ofstream write_out(filename);
  write_out << "shape:";
  for (unsigned s = 0; s < shape.size(); ++s)
    write_out << " " << shape[s];
  write_out << endl;
  for (int i = 0; i < length; ++i) {
    float num = input[i] * scale;
    if (fabsf(num) < std::numeric_limits<float>::min())
      num = 0;
    write_out << num << endl;
  }
  write_out.close();
}

void dump_params(Net<float> &net) {
  int debugMode = 0;
  if (getenv("DECENT_DEBUG"))
    debugMode = stoi(getenv("DECENT_DEBUG"));

  if (debugMode < 2)
    return;

  LOG(INFO) << "Dumping convolution layers parameters ...";
  string dump_path = "dump_gpu";
  dump_path += '/';
  mkdir(dump_path.c_str(), 0777);

  bool dumpAsDPU = (debugMode > 3);
  bool dumpTxt = (debugMode > 4);

  for (size_t i = 0; i < net.layers().size(); i++) {
    const LayerParameter& cur_layer = net.layers()[i]->layer_param();
    if (cur_layer.type() == "ConvolutionFixed" ||
        cur_layer.type() == "ConvolutionFixedData") {
      LayerParameter tmpLayer;
      tmpLayer.CopyFrom(cur_layer);
      const ConvolutionParameter &conv_param = tmpLayer.convolution_param();
      if (tmpLayer.blobs().empty()){
        net.layers()[i]->blobs()[0]->ToProto(tmpLayer.add_blobs());
        if (conv_param.bias_term()) {
          net.layers()[i]->blobs()[1]->ToProto(tmpLayer.add_blobs());
        }
      }
      int weight_dpos, bias_dpos;
      LayerParameter fixed_layer =
          TransformConvFixed(tmpLayer, weight_dpos, bias_dpos);
      string layer_name = fixed_layer.name();
      string name = layer_name; 
      replace(name.begin(), name.end(), '/', '_');
      Blob<float> weight;
      weight.FromProto(fixed_layer.blobs(0), 1);
      const vector<int> &shape = weight.shape();
      if (dumpAsDPU) {
        LOG(INFO) << "Dumping layer " << layer_name << " weights:  fix position is "
                  << weight_dpos;
        print_file_as_DPU(shape, weight.cpu_data(), weight.count(), 
                          weight_dpos, dump_path + name +"_weights", dumpTxt);
      } else
        print_file(shape, weight.cpu_data(), weight.count(),
                   dump_path + name + "_weight.txt", 1.0);
      // Copy bias
      Blob<float> bias;
      if (conv_param.bias_term()) {
        bias.FromProto(fixed_layer.blobs(1), 1);
        const vector<int> &shape = bias.shape();
        if (dumpAsDPU) {
          LOG(INFO) << "Dumping layer " << layer_name << " bias:  fix position is "
                    << bias_dpos;
          print_file_as_DPU(shape, bias.cpu_data(), bias.count(), 
                            bias_dpos, dump_path + name + "_bias", dumpTxt);
        } else
          print_file(shape, bias.cpu_data(), bias.count(),
                     dump_path + name + "_bias.txt", 1.0);
      }
    }
  }
  LOG(INFO) << "Dumping convolution layers parameters done!";
}

void dump_tops(Net<float> &net, int iter) {
  int debugMode = 0;
  if (getenv("DECENT_DEBUG"))
    debugMode = stoi(getenv("DECENT_DEBUG"));

  if (debugMode < 2)
    return;

  LOG(INFO) << "Dumping layers output ...";
  assert(iter >= 0);
  string dump_path = "dump_gpu";
  dump_path += '/';
  mkdir(dump_path.c_str(), 0777);
  // dump_path += iter;

  vector<string> layer_names = net.layer_names();
  vector<vector<Blob<float> *>> all_tops = net.top_vecs();
  bool dumpAllLayers = (debugMode > 2);
  bool dumpAsDPU = (debugMode > 3);
  bool dumpTxt = (debugMode > 4);

  // Sync fixed positions for all layers
  const vector<vector<Blob<float> *>> &bottom_vecs = net.bottom_vecs();
  const vector<vector<Blob<float> *>> &top_vecs = net.top_vecs();
  bool finish = false;
  while (!finish) {
    finish = true;
    for (size_t i = 0; i < net.layers().size(); i++) {
      Layer<float> *layer = net.layers()[i].get();
      if (IsFixSkipLayer(layer->layer_param()) 
          || layer->layer_param().type() == "GSTiling") {
        if (bottom_vecs[i].size() == 1 &&
            bottom_vecs[i][0]->fixed_pos() != 999) {
          for (size_t j = 0; j < top_vecs[i].size(); j++) {
            if (top_vecs[i][j]->fixed_pos() == 999) {
              top_vecs[i][j]->set_fixed_pos(bottom_vecs[i][0]->fixed_pos());
              finish = false;
            }
          }
        }
        if (top_vecs[i].size() == 1 && top_vecs[i][0]->fixed_pos() != 999) {
          for (size_t j = 0; j < bottom_vecs[i].size(); j++) {
            if (bottom_vecs[i][j]->fixed_pos() == 999) {
              bottom_vecs[i][j]->set_fixed_pos(top_vecs[i][0]->fixed_pos());
              finish = false;
            }
          }
        }
      }
    }
  }

  for (int i = 0; i < layer_names.size(); ++i) {
    // only dump the last 5 layers
    if (!dumpAllLayers && i < layer_names.size() - 5)
      continue;

    // some layer names includes char ‘/’, replace them with underline
    string name = layer_names[i];
    replace(name.begin(), name.end(), '/', '_');
    float fix_scale = 1.0;
    const vector<int> &shape = all_tops[i][0]->shape();
    if (dumpAsDPU) {
      // ignore name include "_fixed" and "0_split" layers
      if (name.find("_fixed") < name.length() ||
          name.find("0_split") < name.length())
        continue;
      int scale = all_tops[i][0]->fixed_pos();
      LOG(INFO) << "Dumping layer " << layer_names[i] << ":  fix position is "
                << scale;
      print_file_as_DPU(shape, all_tops[i][0]->cpu_data(),
                        all_tops[i][0]->count(), scale, dump_path + name,
                        dumpTxt);
    } else
      print_file(shape, all_tops[i][0]->cpu_data(), all_tops[i][0]->count(),
                 dump_path + name + ".txt", fix_scale);
  }
  LOG(INFO) << "Dumping layers output done!";
}

void test_detection(Net<float> &caffe_net, const string &ap_style,
                    const int &test_iter) {
  LOG(INFO) << "Testing Detection Net with AP_STYLE: " << ap_style << " ...";
  std::map<int, std::map<int, vector<std::pair<float, int>>>> all_true_pos;
  std::map<int, std::map<int, vector<std::pair<float, int>>>> all_false_pos;
  std::map<int, std::map<int, int>> all_num_pos;
  // float loss = 0;
  for (int i = 0; i < test_iter; ++i) {
    // float iter_loss;
    const vector<Blob<float> *> &result = caffe_net.Forward(nullptr);
    // loss += iter_loss;
    for (int j = 0; j < result.size(); ++j) {
      CHECK_EQ(result[j]->width(), 5);
      const float *result_vec = result[j]->cpu_data();
      int num_det = result[j]->height();
      for (int k = 0; k < num_det; ++k) {
        int item_id = static_cast<int>(result_vec[k * 5]);
        int label = static_cast<int>(result_vec[k * 5 + 1]);
        if (item_id == -1) {
          // Special row of storing number of positives for a label.
          if (all_num_pos[j].find(label) == all_num_pos[j].end()) {
            all_num_pos[j][label] = static_cast<int>(result_vec[k * 5 + 2]);
          } else {
            all_num_pos[j][label] += static_cast<int>(result_vec[k * 5 + 2]);
          }
        } else {
          // Normal row storing detection status.
          float score = result_vec[k * 5 + 2];
          int tp = static_cast<int>(result_vec[k * 5 + 3]);
          int fp = static_cast<int>(result_vec[k * 5 + 4]);
          if (tp == 0 && fp == 0) {
            // Ignore such case. It happens when a detection bbox is matched to
            // a difficult gt bbox and we don't evaluate on difficult gt bbox.
            continue;
          }
          all_true_pos[j][label].push_back(std::make_pair(score, tp));
          all_false_pos[j][label].push_back(std::make_pair(score, fp));
        }
      }
    }
    LOG(INFO) << "Test iter: " << i + 1 << "/" << test_iter;
    // dump output of every layers
    dump_tops(caffe_net, i);
  }

  for (int i = 0; i < all_true_pos.size(); ++i) {
    if (all_true_pos.find(i) == all_true_pos.end()) {
      LOG(FATAL) << "Missing output_blob true_pos: " << i;
    }
    const std::map<int, vector<std::pair<float, int>>> &true_pos =
        all_true_pos.find(i)->second;
    if (all_false_pos.find(i) == all_false_pos.end()) {
      LOG(FATAL) << "Missing output_blob false_pos: " << i;
    }
    const std::map<int, vector<std::pair<float, int>>> &false_pos =
        all_false_pos.find(i)->second;
    if (all_num_pos.find(i) == all_num_pos.end()) {
      LOG(FATAL) << "Missing output_blob num_pos: " << i;
    }
    const std::map<int, int> &num_pos = all_num_pos.find(i)->second;
    std::map<int, float> APs;
    float mAP = 0.;
    // Sort true_pos and false_pos with descend scores.
    for (std::map<int, int>::const_iterator it = num_pos.begin();
         it != num_pos.end(); ++it) {
      int label = it->first;
      int label_num_pos = it->second;
      if (true_pos.find(label) == true_pos.end()) {
        LOG(WARNING) << "Missing true_pos for label: " << label;
        continue;
      }
      const vector<std::pair<float, int>> &label_true_pos =
          true_pos.find(label)->second;
      if (false_pos.find(label) == false_pos.end()) {
        LOG(WARNING) << "Missing false_pos for label: " << label;
        continue;
      }
      const vector<std::pair<float, int>> &label_false_pos =
          false_pos.find(label)->second;
      vector<float> prec, rec;
      caffe::ComputeAP(label_true_pos, label_num_pos, label_false_pos, ap_style,
                       &prec, &rec, &(APs[label]));
      mAP += APs[label];
    }
    mAP /= num_pos.size();
    const int output_blob_index = caffe_net.output_blob_indices()[i];
    const string &output_name = caffe_net.blob_names()[output_blob_index];
    LOG(INFO) << "Test Results:";
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mAP;
  }
}

void test_segmentation(Net<float> &caffe_net, const int &class_num,
                       const int &test_iter) {
  LOG(INFO) << "Testing Segmentation Net ...";
  std::vector<u_int64_t> tp_;
  std::vector<u_int64_t> fp_;
  std::vector<u_int64_t> fn_;
  tp_.resize(class_num, 0);
  fp_.resize(class_num, 0);
  fn_.resize(class_num, 0);
  vector<float> iou(class_num);

  // float loss = 0;
  for (int i = 0; i < test_iter; ++i) {
    // float iter_loss;
    const vector<Blob<float> *> &result = caffe_net.Forward(nullptr);
    for (int j = 0; j < result.size(); j++) {
      if (result[j]->num() == 3 && result[j]->channels() == class_num) {
        const float *result_vec = result[j]->cpu_data();
        for (int c = 0; c < class_num; ++c) {
          tp_[c] += static_cast<int>(result_vec[c]);
          fp_[c] += static_cast<int>(result_vec[class_num + c]);
          fn_[c] += static_cast<int>(result_vec[class_num * 2 + c]);
        }
      }
    }
    LOG(INFO) << "Test iter: " << i + 1 << "/" << test_iter;
    // dump output of every layers
    dump_tops(caffe_net, i);
  }

  for (int c = 0; c < class_num; ++c) {
    u_int64_t denom = tp_[c] + fp_[c] + fn_[c];
    if (denom == 0) {
      iou[c] = NAN;
    } else {
      iou[c] = tp_[c] * 1. / denom;
    }
  }

  // average IOU on C classes
  float mIOU = 0;
  int count = 0;
  for (int c = 0; c < class_num; ++c) {
    if (!std::isnan(iou[c])) {
      mIOU += iou[c];
      count++;
    }
  }
  if (count == 0) {
    mIOU = NAN;
  } else {
    mIOU = mIOU / count;
  }
  LOG(INFO) << "    Test net output #" << 0
            << ": segmentation_eval_classIOU = " << mIOU;
}

void test_general(Net<float> &caffe_net, const int &test_iter) {
  LOG(INFO) << "Testing ...";
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < test_iter; ++i) {
    float iter_loss;
    const vector<Blob<float> *> &result = caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float *result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        if (k < 10) {
          const std::string &output_name =
              caffe_net.blob_names()[caffe_net.output_blob_indices()[j]];
          LOG(INFO) << "Test iter: " << i + 1 << "/" << test_iter << ", "
                    << output_name << " = " << score;
          if (k == 9)
            LOG(INFO) << "Compressed the following log of this type!";
        }
      }
    }
    // dump output of every layers
    dump_tops(caffe_net, i);
  }
  loss /= test_iter;
  LOG(INFO) << "Test Results: ";
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string &output_name =
        caffe_net.blob_names()
            [caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight =
        caffe_net.blob_loss_weights()
            [caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / test_iter;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight << " = "
                      << loss_weight * mean_score << " loss)";
    }
    if (i < 100) {
      LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
      if (i == 99)
        LOG(INFO) << "Compressed the following log of this type!";
    }
  }
}

void test_net_by_type(Net<float> &test_net, const string &net_type,
                      const int &test_iter, const string &ap_style,
                      const string &seg_metric, const int &seg_class_num) {
  LOG(INFO) << "Net type: " << net_type;
  if (net_type == "detection") {
    LOG(INFO) << "AP computation style for detection net: " << ap_style;
  } else if (net_type == "segmentation") {
    LOG(INFO) << "Testing metric for segmentation net: " << seg_metric;
    LOG(INFO) << "Number of classes for segmentation net: " << seg_class_num;
  }

  LOG(INFO) << "Test Start, total iterations: " << test_iter;

  // dump convolution layers parameters
  dump_params(test_net);

  if (net_type == "detection")
    test_detection(test_net, ap_style, test_iter);
  else if (net_type == "segmentation")
    test_segmentation(test_net, seg_class_num, test_iter);
  else
    test_general(test_net, test_iter);
  LOG(INFO) << "Test Done!";
}

} // namespace_caffe
