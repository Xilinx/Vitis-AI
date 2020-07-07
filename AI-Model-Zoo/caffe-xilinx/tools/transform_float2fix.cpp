/*
* Copyright 2019 Xilinx Inc.
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

// This is a script to transform float caffemodel to fix caffemodel for direct fix or fix finetune
// Function:
// 1) Convolution -> ConvolutionFix
// 2) Convolution + BatchNorm + Scale -> ConvolutionBNFix
// 3) Convolution + BatchNorm -> ConvolutionBNFix
// 4) Add FixedNeuron
// Usage:
//    transform_float2fix float_caffemodel_in fix_caffemodel_out [OPTIONS]

#include <vector>
#include <string>
#include <map>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/quantize.hpp"
#include "caffe/util/convert_proto.hpp"

using namespace caffe;
using namespace std;

DEFINE_int32(gpu, 0, "GPU device id");

LayerParameter TransformConvBNScale2ConvBNFix(const LayerParameter& conv, const LayerParameter& bn, const LayerParameter& scale = LayerParameter() ) {
  LayerParameter conv_bn_fix;
  conv_bn_fix.CopyFrom(conv);
  conv_bn_fix.set_type("ConvolutionBNFix");
  conv_bn_fix.clear_top();
  conv_bn_fix.clear_blobs();
  conv_bn_fix.add_top(bn.top(0));
  *conv_bn_fix.mutable_convolution_param() = conv.convolution_param();
  conv_bn_fix.mutable_convolution_param()->set_bias_term(false);
  *conv_bn_fix.mutable_batch_norm_param() = bn.batch_norm_param();

  *conv_bn_fix.add_blobs() = conv.blobs(0);
  for (size_t i = 0; i<bn.blobs_size(); i++) {
    *conv_bn_fix.add_blobs() = bn.blobs(i);
  }
  if (conv.convolution_param().bias_term() ) {
    for (size_t i = 0; i<conv.blobs(1).data_size(); i++ ) {
      float new_bias = bn.blobs(0).data().data()[i] * conv.blobs(1).data().data()[i];
      new_bias /= sqrt(bn.blobs(3).data().data()[i] + 0.00001);
      new_bias += bn.blobs(1).data().data()[i];
      conv_bn_fix.mutable_blobs(2)->set_data(i,new_bias);
    }
  }
  return conv_bn_fix;
}

void TransformFloat2Fix(NetParameter& net_in, NetParameter& net_out) {
  vector<vector<bool>> connection = BuildConnections(net_in);
  vector<bool> added(net_in.layer_size(), false);
  for (size_t i = 0; i < net_in.layer_size(); i++) {
    if(added[i]) continue;
    const LayerParameter* cur_layer = &net_in.layer(i);
	if (cur_layer->type() == "Convolution" ) {
      int next_bn_id = GetNextLayer(net_in, i, "BatchNorm", connection);
      if (next_bn_id != -1 ) {
        if ( net_in.layer(next_bn_id).blobs_size() == 5 ) {
        // Convolution + Batchnorm --> Convolution
            *net_out.add_layer() = TransformConvBNScale2ConvBNFix(*cur_layer, net_in.layer(next_bn_id));
            added[next_bn_id] = true;

        } else if (net_in.layer(next_bn_id).blobs_size() == 2) {
        // Convolution + Batchnorm + Scale --> Convolution
            LOG(FATAL) << "Convolution + Batchnorm + Scale not implemented, use merge_batchnorm2bn first!";
        } else {
            LOG(FATAL) << "Wrong Batchnorm Layer!";
        }
      }

    } else {
      // Copy Other Layers
      *net_out.add_layer() = *cur_layer;
    }
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetUsageMessage(
      "\n Brief: This tool transforms float caffemodel to fix caffemodel for direct fix or fix finetune "
      "\n Usage: transform_float2fix [Options] fixed_model_in "
      "float_model_out");
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  if (argc == 3) {
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);

    NetParameter net_in, net_out;
    ReadProtoFromBinaryFileOrDie(argv[1], &net_in);

    net_in = RemoveLayerByType(net_in, "Split");

    TransformFloat2Fix(net_in, net_out);

    WriteProtoToBinaryFile(net_out, argv[2]);
    return 0;
  } else {
    ::google::ShowUsageWithFlagsRestrict(argv[0], "tools/transform_float2fix");
    return -1;
  }
}
