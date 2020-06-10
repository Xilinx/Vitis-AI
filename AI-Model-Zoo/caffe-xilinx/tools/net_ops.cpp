#include <vector>
#include <string>
#include <fstream>

#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"


DEFINE_string(output, "", "Output results to file");
DEFINE_int32(gpu, 0, "gpu device");

using namespace std;
using namespace caffe;

size_t GetLayerOps(const Net<float>& net, int layer) { //, size_t &param) {

  const auto& layer_param = net.layers()[layer]->layer_param();
  const auto& blobs = net.layers()[layer]->blobs();
  const auto& top = net.top_vecs()[layer];

  if (layer_param.type().compare(0, 11, "Convolution") == 0 ||
      layer_param.type().compare(0, 12, "InnerProduct") == 0) {
    CHECK_GE(blobs.size(), 1);
    CHECK_EQ(top.size(), 1);
    size_t each_output_ops = blobs[0]->count(1)*2; // Mul + Add
    if (layer_param.has_convolution_param()) {
      auto conv_param = layer_param.convolution_param();
      each_output_ops *= conv_param.has_group() ? conv_param.group() : 1;
    }
    return each_output_ops * top[0]->count(1);
  }
  return 0;

/*
	if(layer_proto.type()==string("Convolution")||layer_proto.type()==string("ConvolutionData")
		||layer_proto.type()==string("ConvolutionFixed")||layer_proto.type()==string("ConvolutionBNFixed")
		||layer_proto.type()==string("ConvolutionFixedData")||layer_proto.type()==string("ConvolutionBNFixedData")) {
		const vector<Blob<float>*>& top_vecs=net.top_vecs()[layer_index];
		CHECK_EQ(top_vecs.size(), 1);
		const vector<caffe::shared_ptr<Blob<float>>>& blobs=net.layers()[layer_index]->blobs();
		CHECK_GE(blobs.size(), 1);
		CHECK_LE(blobs.size(), 5);
		size_t op_each_output=blobs[0]->count(1)*2;
		for(size_t blob_num=0; blob_num<blobs.size(); blob_num++) {
			param += blobs[blob_num]->count();
		}
		if(blobs.size()==2) {
			op_each_output+=1;
		}
                ConvolutionParameter conv_param = layer_proto.convolution_param();
                if (conv_param.has_group() && conv_param.group() > 0)
                    op_each_output *= conv_param.group();
                LOG(INFO) << op_each_output << ", " << top_vecs[0]->count(1);
		return op_each_output*top_vecs[0]->count(1);
	}
	else if(layer_proto.type()==string("BN")||layer_proto.type()==string("BNData")) {
		const vector<caffe::shared_ptr<Blob<float>>>& blobs=net.layers()[layer_index]->blobs();
		for(size_t blob_num=0; blob_num<blobs.size(); blob_num++) {
			param += blobs[blob_num]->count();
		}
		return 0;
	}
	else if(layer_proto.type()==string("InnerProduct")) {
		const vector<Blob<float>*>& top_vecs=net.top_vecs()[layer_index];
		CHECK_EQ(top_vecs.size(), 1);
		const vector<caffe::shared_ptr<Blob<float>>>& blobs=net.layers()[layer_index]->blobs();
		CHECK_GE(blobs.size(), 1);
		CHECK_LE(blobs.size(), 2);
		size_t op_each_output=blobs[0]->count(1)*2;
		if(blobs.size()==2) {
			op_each_output+=1;
		}
		for(size_t blob_num=0; blob_num<blobs.size(); blob_num++) {
			param += blobs[blob_num]->count();
		}
		return op_each_output*top_vecs[0]->count(1);
	}
	else {
		return 0;
	}
*/
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr=1;
  ::google::InitGoogleLogging(argv[0]);
  ::google::SetUsageMessage("\nusage: net_ops prototxt");
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 2) {
    ::google::ShowUsageWithFlagsRestrict(argv[0], "tools/net_ops");
    return -1;
  }

  Caffe::set_mode(Caffe::CPU);
  Caffe::SetDevice(0);

  ofstream ofile(FLAGS_output);
  Net<float> net(argv[1], TEST);
  size_t total_ops = 0; // , total_param=0;

  for (auto i = 0; i < net.layers().size(); ++i) {
    // size_t param = 0;
    auto op = GetLayerOps(net, i); // , param);
    if (op > 0) {
        LOG(INFO) << "Layer: " << net.layer_names()[i] << ", type: " <<
            net.layers()[i]->type() << ", operations: " << op;
    }
/*
    if (param > 0) {
      LOG(INFO) << "Layer: " << net.layer_names()[i] << ", type: " << net.layers()[i]->type() << ", param: " << param;
    }
*/
    if (FLAGS_output != "") {
        ofile << net.layer_names()[i] << " " << op; // << " "<<param<<std::endl;
    }

    total_ops += op;
    // total_param+=param;
  }

  LOG(INFO) << "Total operations: " << total_ops;
  // LOG(INFO) << "Total params: " << total_param;
  if (FLAGS_output != "") {
      ofile << "Total " << total_ops; // <<" "<<total_param<<std::endl;
      ofile.close();
  }
  return 0;
}
