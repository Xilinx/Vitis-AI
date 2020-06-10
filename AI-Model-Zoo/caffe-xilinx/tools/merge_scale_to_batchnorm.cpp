// This is a script to merge Scale layer into BatchNorm Layer for prototxt or
// Caffemodel
// Usage:
//    merge_scale_to_batchnorm origin_in merged_out
// Function:
//    1) BatchNorm + Scale -> BatchNorm

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream> // NOLINT(readability/streams)
#include <string>
#include <unordered_set>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/convert_proto.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

DEFINE_bool(binary, false, "Set to true for caffemodel converting");

using namespace caffe; // NOLINT(build/namespaces)
using namespace std;

void MergeBatchNormScale(const NetParameter &net_in, NetParameter &net_out,
                         const int bn_index, const int sc_index,
                         vector<bool> &processed, bool binary = false) {
  const LayerParameter &bn_param = net_in.layer(bn_index);
  const LayerParameter &sc_param = net_in.layer(sc_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(bn_param);
  layer_param->set_top(0, sc_param.top(0));
  layer_param->clear_param();
  // Set Param
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
    layer_param->clear_blobs();
    CHECK_EQ(sc_param.blobs_size(), 2)
        << " Wrong ScaleLayer blob size of layer: " << sc_param.name();
    CHECK_EQ(bn_param.blobs_size(), 3)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
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
    for (size_t i = 0; i<layer_param->blobs_size()-1; i++) {
      if (layer_param->blobs(i).shape().dim_size()==1) {
        DLOG(INFO) << "Update blob shape dim: " << layer_param->name() << " blob[" << i << "]";
        layer_param->mutable_blobs(i)->mutable_shape()->clear_dim();
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(bn_param.blobs(0).shape().dim(0));
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
      } else if (layer_param->blobs(i).shape().dim_size()!=4) {
        LOG(WARNING) << "Warning: wrong dim_size for layer: " << layer_param->name() << " blob[" << i << "]";
      }
    }
  }

  processed[sc_index] = true;
  processed[bn_index] = true;
}

void MergeScale2BatchNorm(const NetParameter &net_in, NetParameter &net_out,
                          bool binary = false) {
  vector<vector<bool>> connect = BuildConnections(net_in);
  vector<bool> processed(net_in.layer_size(), false);

  for (int i = 0; i < net_in.layer_size(); i++) {
    if (processed[i]) {
      continue;
    }
    const LayerParameter *cur_layer = &net_in.layer(i);
    if (cur_layer->type() == "BatchNorm") {
      int scale_index = GetNextLayer(net_in, i, "Scale", connect);
      if (scale_index != -1) {
        MergeBatchNormScale(net_in, net_out, i, scale_index, processed, binary);
      } else {
        *(net_out.add_layer()) = *cur_layer;
        processed[i] = true;
      }
    } else if (cur_layer->type() == "BN") {
      LayerParameter *layer_param = net_out.add_layer();
      layer_param->CopyFrom(*cur_layer);
      layer_param->set_type("BatchNorm");
      processed[i] = true;
    } else {
      *(net_out.add_layer()) = *cur_layer;
      processed[i] = true;
    }
  }
}

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = 1;
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetUsageMessage("\n  Usage: merge_scale_to_batchnorm [Options] "
                            "origin_in merged_out ");

  ::google::ParseCommandLineFlags(&argc, &argv, true);

  if (argc == 3) {
    NetParameter net_in, net_out;
    if (!FLAGS_binary) {
      ReadProtoFromTextFileOrDie(string(argv[1]), &net_in);
    } else {
      ReadProtoFromBinaryFileOrDie(string(argv[1]), &net_in);
    }

    MergeScale2BatchNorm(net_in, net_out, FLAGS_binary);

    if (!FLAGS_binary) {
      WriteProtoToTextFile(net_out, string(argv[2]));
    } else {
      WriteProtoToBinaryFile(net_out, string(argv[2]));
    }

    return 0;
  } else {
    ::google::ShowUsageWithFlagsRestrict(argv[0], "tools/merge_scale_to_batchnorm");
    return -1;
  }
}
