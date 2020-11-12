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

using namespace caffe; // NOLINT(build/namespaces)
using namespace std;

DEFINE_bool(keep_bn, false, "False: BatchNorm will be merged into Convolution");
DEFINE_bool(keep_convbn, false, "False: Merge Conv+BatchNorm -> Conv; True: "
                                "Merge Conv+BatchNorm -> ConvBNFixed");
DEFINE_string(model_in, "", "Input prototxt ");
DEFINE_string(model_out, "", "Output prototxt ");
DEFINE_string(weights_in, "", "Input caffemodel ");
DEFINE_string(weights_out, "", "Output caffemodel ");
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
DEFINE_int32(weight_bit, 8,
             "bit_width for weights and bias of convolution and fc layers");
DEFINE_string(sigmoided_layers, "", "Optinal: list of layers before sigmoid "
                                    "operation, to be fixed with optimization "
                                    "for sigmoid accuracy");

// Global
NetParameter model_in, model_out;
bool binary = false;

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func)                                             \
  namespace {                                                                  \
  class __Registerer_##func {                                                  \
  public: /* NOLINT */                                                         \
    __Registerer_##func() { g_brew_map[#func] = &func; }                       \
  };                                                                           \
  __Registerer_##func g_registerer_##func;                                     \
  }

static BrewFunction GetBrewFunction(const caffe::string &name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin(); it != g_brew_map.end();
         ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL; // not reachable, just to suppress old compiler warnings.
  }
}

int fix() {
  CheckModel(model_in);
  NetParameter merged_model;
  MergeLayers(model_in, merged_model, binary, FLAGS_keep_bn, FLAGS_keep_convbn);
  ConvertFloat2Fix(merged_model, model_out, FLAGS_ignore_layers,
                   FLAGS_ignore_layers_file, FLAGS_method, FLAGS_weight_bit,
                   FLAGS_data_bit, FLAGS_add_bn, FLAGS_sigmoided_layers, FLAGS_special_layers);
  return 0;
}
RegisterBrewFunction(fix);

int merge() {
  MergeLayers(model_in, model_out, binary, FLAGS_keep_bn, FLAGS_keep_convbn);
  return 0;
}
RegisterBrewFunction(merge);

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = 1;
  gflags::SetUsageMessage(
      "command line brew\n"
      "usage: convert_model <command> <args>\n\n"
      "commands:\n"
      "  1.merge           merge Convolution + Batchnorm + "
      "Scale into Convolution or ConvolutionBNFixed\n"
      "  2.fix       convert float prototxt to fixed for fix train\n\n"
      "examples:\n"
      "  1.merge Convolution + BatchNorm + (Scale) --> Convolution for "
      "prototxt: "
      "./convert_model merge -model_in origin.prototxt -model_out "
      "merged.prototxt\n"
      "  2.merge Convolution + BatchNorm + (Scale) --> Convolution for "
      "caffemodel: ./convert_model merge -weights_in origin.caffemodel "
      "-model_out merged.prototxt\n"
      "  3.merge Bvlc-BatchNorm+Scale --> Nv-BatchNorm: ./convert_model merge "
      "-model_in origin.prototxt -model_out merged.prototxt -keep_bn\n"
      "  4.convert float prototxt to fixed: ./convert_model fix -model_in "
      "origin.prototxt -model_out fixed.prototxt\n");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);

  if (argc == 2) {
    // Input
    CHECK((FLAGS_model_in.size() && FLAGS_model_out.size()) ||
          (FLAGS_weights_in.size() && FLAGS_weights_out.size()))
        << "Wrong arguments assigned, please see the examples by "
           "./convert_model";
    if (FLAGS_model_in != "") {
      ReadNetParamsFromTextFileOrDie(string(FLAGS_model_in), &model_in);
    } else if (FLAGS_weights_in != "") {
      binary = true;
      ReadNetParamsFromBinaryFileOrDie(string(FLAGS_weights_in), &model_in);
      LOG(INFO) << "net loaded: " << model_in.layer_size();
    } else {
      LOG(FATAL) << "No model_in or weights_in assigned!";
      ::google::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_model");
    }

    GetBrewFunction(caffe::string(argv[1]))();

    // Output
    if (FLAGS_model_out != "") {
      WriteProtoToTextFile(model_out, FLAGS_model_out);
    } else if (FLAGS_weights_out != "") {
      WriteProtoToBinaryFile(model_out, FLAGS_weights_out);
    } else {
      LOG(FATAL) << "No model_out or weights_out assigned!";
      ::google::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_model");
    }

    return 0;
  } else {
    ::google::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_model");
    return -1;
  }
}

