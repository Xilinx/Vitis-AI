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

#include <cstring>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream> // NOLINT(readability/streams)
#include <string>
#include <unordered_set>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/convert_proto.hpp"
#include "caffe/util/io.hpp"
#include <iostream>

DEFINE_bool(add_bn, false,
            "Add BatchNorm layers after every Convolution layer");
DEFINE_string(ignore_layers, "",
              "list of layers to be ignore during conversion, comma-delimited");
DEFINE_string(special_layers, "", "Optinal: List of layers which has special data_bit or weight_bit for "
                           "quantization, comma-delimited, e.g. 'data_in,16,conv2,8,16', "
                           "note:(1) the first number after layer_name represent data_bit, and the second "
                           "number represent weight_bit(optional, can be omitted). (2) We "
                           "have take some necessary measures to check the input parameters"); 
DEFINE_string(
    ignore_layers_file, "",
    "Optional: Prototxt file which defines the layers to be ignore during "
    "quantization, each line formated as 'ignore_layer: \"layer_name\"'");
DEFINE_int32(
    method, 0,
    "fix method for weights and bias. 0: overflow, 1: diffs, 2: diffa");
DEFINE_int32(data_bit, 8, "bit_width for fixed_neuron layers");
DEFINE_int32(weights_bit, 8,
             "bit_width for weights and bias of convolution and fc layers");
DEFINE_string(sigmoided_layers, "", "Optinal: list of layers before sigmoid "
                                    "operation, to be fixed with optimization "
                                    "for sigmoid accuracy");

using std::ofstream;

using namespace caffe; // NOLINT(build/namespaces)
using namespace std;

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = 1;
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetUsageMessage("\n  Usage: convert_float2fix [Options] "
                            "float_prototxt_in fix_prototxt_out ");

  ::google::ParseCommandLineFlags(&argc, &argv, true);

  if (argc == 3) {
    NetParameter net_in, net_out;
    ReadProtoFromTextFileOrDie(string(argv[1]), &net_in);

    ConvertFloat2Fix(net_in, net_out, FLAGS_ignore_layers,
                     FLAGS_ignore_layers_file, FLAGS_method, FLAGS_weights_bit,
                     FLAGS_data_bit, FLAGS_add_bn, FLAGS_sigmoided_layers, FLAGS_special_layers);

    WriteProtoToTextFile(net_out, string(argv[2]));

    return 0;
  } else {
    ::google::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_float2fix");
    return -1;
  }
}
