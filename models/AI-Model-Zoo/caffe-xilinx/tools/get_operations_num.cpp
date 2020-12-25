#include <vector>
#include <string>
#include <fstream>

#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"


DEFINE_string(output, "", "Output results to file");
DEFINE_int32(gpu, 0, "gpu device");

using namespace std;
using namespace caffe;

size_t GetOperationNum(const Net<float>& net, int layer_index, size_t &param) {
    const LayerParameter& layer_proto=net.layers()[layer_index]->layer_param();
    if (layer_proto.type()==string("Convolution") || 
        layer_proto.type()==string("ConvolutionFixed")||
        layer_proto.type()==string("ConvolutionBNFixed")) {
        const vector<Blob<float>*>& top_vecs=net.top_vecs()[layer_index];
        CHECK_EQ(top_vecs.size(), 1);
        const vector<caffe::shared_ptr<Blob<float>>>& blobs=net.layers()[layer_index]->blobs();
        CHECK_GE(blobs.size(), 1);
        CHECK_LE(blobs.size(), 6);
        size_t op_each_output=blobs[0]->count(1)*2;
        for(size_t blob_num=0; blob_num<blobs.size(); blob_num++) {
            param += blobs[blob_num]->count();
        }
        if(blobs.size()==2) {
            op_each_output+=1;
        }
        return op_each_output*top_vecs[0]->count(1);
    }
    else if(layer_proto.type()==string("BatchNorm")) {
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
    else if (layer_proto.type()==string("Deconvolution")) { 
        const vector<Blob<float>*>& top_vecs=net.top_vecs()[layer_index];
        CHECK_EQ(top_vecs.size(), 1);
        const vector<Blob<float>*>& bottom_vecs=net.bottom_vecs()[layer_index];
        CHECK_EQ(bottom_vecs.size(), 1);
        const vector<caffe::shared_ptr<Blob<float>>>& blobs=net.layers()[layer_index]->blobs();
        CHECK_GE(blobs.size(), 1);
        CHECK_LE(blobs.size(), 6);
        for(size_t blob_num=0; blob_num<blobs.size(); blob_num++) {
            param += blobs[blob_num]->count();
        }
        // get kernel_w, kernel_h
        int kernel_w=0, kernel_h=0;
        ConvolutionParameter conv_param = layer_proto.convolution_param();
        if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
            kernel_w = conv_param.kernel_h();
            kernel_h = conv_param.kernel_w();
        } else {
            const int num_kernel_dims = conv_param.kernel_size_size();
            CHECK(num_kernel_dims == 1 || num_kernel_dims == 2);
            if (num_kernel_dims==1) {
                kernel_h = conv_param.kernel_size(0);
                kernel_w = conv_param.kernel_size(0);
            }
            else {
                kernel_h = conv_param.kernel_size(0);
                kernel_w = conv_param.kernel_size(1);
            }
        }
        // op = 2*c*h*w  * kernel_h * kernel_w * n
        size_t op = 2*bottom_vecs[0]->count(1)*kernel_h*kernel_w*blobs[0]->shape()[0];
        if(blobs.size()==2) {
            // add bias
            op += top_vecs[0]->count(1);
        }
        return op;
    }
    else {
        return 0;
    }
}

int main(int argc, char** argv) {
    FLAGS_alsologtostderr=1;
    ::google::InitGoogleLogging(argv[0]);
    ::google::SetUsageMessage("\nusage: get_operations_num prototxt");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    if(argc!=2) {
        ::google::ShowUsageWithFlagsRestrict(argv[0], "tools/get_operations_num");
        return -1;
    }
    Caffe::set_mode(Caffe::CPU);
    Caffe::SetDevice(0);

    ofstream ofile(FLAGS_output);
    Net<float> net(argv[1], TEST);
    size_t total_op=0, total_param=0;
    for(size_t i=0; i<net.layers().size(); i++) {
        size_t param=0;
        size_t op=GetOperationNum(net, i, param);
        if(op>0) {
            LOG(INFO) << "Layer: " << net.layer_names()[i] << ", type: " << net.layers()[i]->type() << ", operations: " << op;
        }
        if(param>0) {
            LOG(INFO) << "Layer: " << net.layer_names()[i] << ", type: " << net.layers()[i]->type() << ", param: " << param;
        }
        if(FLAGS_output != "") {
           ofile<<net.layer_names()[i]<< " "<<op<<" "<<param<<std::endl; 
        }
        total_op+=op;
        total_param+=param;
    }
    LOG(INFO) << "Total operations: " << total_op;
    LOG(INFO) << "Total params: " << total_param;
    if(FLAGS_output != "") {
        ofile<<"Total "<<total_op<<" "<<total_param<<std::endl; 
        ofile.close();
    }
    return 0;
}
