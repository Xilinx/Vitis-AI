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

// This is a script to transform fix caffemodel to float caffemodel and output
// fix_info for DPU deployment.
// Usage:
//    convert_fix2float fix_caffemodel_in float_caffemodel_out fix_info_out

#include <map>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/convert_proto.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/quantize.hpp"

DEFINE_int32(gpu, 0,
             "GPU device id, note: this tool can only run in GPU mode ");
DEFINE_bool(keep_fixed_neuron, false,
            "Remain FixedNeuron layers in the output float caffemodel ");
DEFINE_string(fix_info, "", "Filename for output fix_info ");
DEFINE_string(prototxt, "", "Filename for output prototxt");

using namespace caffe;
using namespace std;

#if defined(USE_CUDNN) and !defined(CPU_ONLY)

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = 1;
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetUsageMessage(
      "\n Brief: This tool transforms fix caffemodel to float caffemodel and "
      "can"
      "generate fix_info for DPU deployment. "
      "\n Usage: transform_fixed2float [Options] fixed_model_in "
      "float_model_out");
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  if (argc == 3) {
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);

    NetParameter net_in, net_out;
    ReadProtoFromBinaryFileOrDie(argv[1], &net_in);

    net_in = RemoveLayerByType(net_in, "Split");

    FIX_INFO weight_infos;
    FIX_INFO bias_infos;
    FIX_INFO data_infos;

    TransformFix2FloatWithFixInfo(net_in, net_out, weight_infos, bias_infos,
                                  data_infos, FLAGS_keep_fixed_neuron);

    // Output fix_info for DPU deployment
    if (FLAGS_fix_info != "") {
      vector<vector<int>> fix_infos =
          GetFixInfo(net_out, weight_infos, bias_infos, data_infos);
      WriteFixInfoToFile(net_out, fix_infos, FLAGS_fix_info);
    }

    WriteProtoToBinaryFile(net_out, argv[2]);
    if (FLAGS_prototxt != "") {
      net_out = RemoveBlobs(net_out);
      WriteProtoToTextFile(net_out, FLAGS_prototxt);
    }
    return 0;
  } else {
    ::google::ShowUsageWithFlagsRestrict(argv[0], "tools/transform_fix2float");
    return -1;
  }
}
#else
int main(int argc, char **argv) {
  FLAGS_alsologtostderr = 1;
  ::google::InitGoogleLogging(argv[0]);
  LOG(ERROR) << "transform_fix2float need GPU and cuDNN.";
  return -1;
}
#endif
