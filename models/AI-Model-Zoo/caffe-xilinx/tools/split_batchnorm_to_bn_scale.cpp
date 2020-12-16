#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/convert_proto.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::LayerParameter;
using caffe::Net;
using caffe::NetParameter;
using caffe::Solver;
using caffe::shared_ptr;
using std::map;
using std::ostringstream;
using std::string;
using std::vector;

DEFINE_string(in_weights, "",
    "The input weights need to be transformed.");
DEFINE_string(out_weights, "transformed.caffemodel",
    "The output transformed weights.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

// Hack to convert macro to string
#define STRINGIZE(m) #m
#define STRINGIZE2(m) STRINGIZE(m)

void Split(const NetParameter& net_in, NetParameter* net_out) {
  for (int i = 0; i < net_in.layer_size(); ++i) {
    const LayerParameter& cur_layer = net_in.layer(i);
    if (cur_layer.type() != "BatchNorm") {
      net_out->add_layer()->CopyFrom(cur_layer);
      continue;
    }
    LayerParameter* bn_layer = net_out->add_layer();
    bn_layer->CopyFrom(cur_layer);
    bn_layer->mutable_batch_norm_param()->clear_scale_filler();
    bn_layer->mutable_batch_norm_param()->clear_bias_filler();
    bn_layer->clear_param();

    LayerParameter* scale_layer = net_out->add_layer();
    scale_layer->set_name(cur_layer.name() + "_sc");
    scale_layer->set_type("Scale");
    scale_layer->add_bottom(cur_layer.top(0));
    scale_layer->add_top(cur_layer.top(0)); // in-place
    scale_layer->mutable_scale_param()->set_bias_term(true);
    //auto* scale_param = scale_layer->mutable_scale_param();
    //scale_param->mutable_filler->CopyFrom(cur_layer->scale_filler());
    //if (cur_layer->has_bias_filler()) {
    //  scale_param->mutable_bias_term() = true;
    //  scale_param->mutable_bias_filler->CopyFrom(cur_layer->bias_filler());
    //}
  }
}

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  //FLAGS_colorlogtostderr = 1;
  // Set version
  //gflags::SetVersionString(version_str());
  // Usage message.
  gflags::SetUsageMessage("\nUsage:\n"
      "split_batchnrom_to_bn_scale in.prototxt out.prototxt\n");

  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);

  if (argc == 3) {
    NetParameter net_in;
    NetParameter net_out;
    ReadNetParamsFromTextFileOrDie(argv[1], &net_in);
    Split(net_in, &net_out);
    WriteProtoToTextFile(net_out, argv[2]);
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
