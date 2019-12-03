// Copyright 2019 Xilinx Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "interface.h"
string g_output_layer_name;


classifycpp::classifycpp(){
	cout<< PFX << "C++ inferce object created" <<endl;
}
classifycpp::classifycpp(string xclbin_t, string dataDir_t, string netCfgFile_t,
                 string quantCfgFile_t, string labelFile_t, int batch_sz_t, string dir_path_t, vector<string> image_path_t){
    //# These mean values fixed both the classification networks,googlenet_v1 & resnet50
    mean[0]=104.007f;
	mean[1]=116.669f;
	mean[2]=122.679f;
    //# Classification neworks trained on Imagenet dataset,it has thousand classes.
	numClasses=1000;

    xclbin=xclbin_t;
    netCfgFile=netCfgFile_t;
    quantCfgFile=quantCfgFile_t;
    dataDir=dataDir_t;
	batch_sz=batch_sz_t;
	labelFile= labelFile_t;
	dir_path=dir_path_t;
	for(int i=0;i<batch_sz;i++){
        image_path.push_back(image_path_t[i]);
    }
}
classifycpp::~classifycpp(){
    //# Release Memory
    delete [] input;
    delete [] output;
    //# cleanup
    xblasDestroy(handle);
}
//# Pre-processing of input data
int prepareInputData(cv::Mat &in_frame, int img_h, int img_w, int img_depth, float *data_ptr,float *mean)
{
    cv::Mat resize_frame;

    int height = in_frame.rows;
    int width = in_frame.cols;
    int channels = in_frame.channels();

    cv::resize(in_frame, resize_frame, cv::Size(img_h, img_w));

    float *dst1 = &data_ptr[0];
    float *dst2 = &data_ptr[img_h*img_w];
    float *dst3 = &data_ptr[(channels-1)*img_h*img_w];

    uchar *img_data = resize_frame.data;

    int idx = 0, frame_cntr = 0;

    // The googlenet_v1 & resnet50 models trained based on below mean values
    //const float mean[3] = {104.007f,116.669f,122.679f};

    for(int l_rows = 0; l_rows < img_h; l_rows++)
    {
        for(int l_cols = 0; l_cols < img_w; l_cols++)
        {
            dst1[idx] = (float)img_data[frame_cntr++] - mean[0];
            dst2[idx] = (float)img_data[frame_cntr++] - mean[1];
            dst3[idx] = (float)img_data[frame_cntr++] - mean[2];

            idx++;
        } //l_cols
    } //l_rows
    return 0;
}
// prepareInputData


void softmax(vector<float> &input)
{
    float m = numeric_limits<float>::min();
    for (size_t i = 0; i < input.size(); i++)
        if (input[i] > m)
            m = input[i];

    float sum = 0.0;
    for (size_t i = 0; i < input.size(); i++)
        sum += expf(input[i] - m);

    for (size_t i = 0; i < input.size(); i++)
        input[i] = expf(input[i] - m) / sum;
}
// Read Imagenet dataset classes label file
vector<string> getLabels(string fname)
{
    ifstream f(fname.c_str());
    assert(f.good());

    vector<string> labels;
    for (string line; getline(f, line); )
        labels.push_back(line);

    return labels;
}
void update_app_args(vector<string> &app_args){
    app_args.push_back("--xclbin");
    app_args.push_back("--datadir");
    app_args.push_back("--netcfg");
    app_args.push_back("--quantizecfg");
    app_args.push_back("--labels");
    app_args.push_back("--images");
    app_args.push_back("--batch_sz");
    app_args.push_back("--image");
}
void check_arg_list(map<string ,string> &arg_map,vector<string> &app_args,vector<int> &miss_list){

    for (int i=0;i<app_args.size(); i++){
        map<string ,string>::iterator it=arg_map.find(app_args[i]);
        if (it == arg_map.end())
        {
            if(app_args[i].compare("--image")==0){
                it=arg_map.find("--images");
                if(it == arg_map.end())
            	    miss_list.push_back(i);
            }else if(app_args[i].compare("--images")==0){
                it=arg_map.find("--image");
                if(it == arg_map.end())
                    miss_list.push_back(i);
            }else{
                    miss_list.push_back(i);
            }
        }
        else{
                cout<<PFX <<" app_args="<< app_args[i] << " eq value="<<it->second<<endl;
        }
    }
}

void arg_list_help()
{
        cout<<"Expected arguments:"<<endl;
        cout<<" --xclbin=<MLsuite ROOT PATH>/overlaybins/<platform>/overlay_2.xclbin --data_dir=<MLsuite ROOT PATH>/apps/yolo/work/yolov2.caffemodel_data --cmd_json=<MLsuite ROOT PATH>/apps/yolo/work/yolo.cmds.json --quant_json=<MLsuite ROOT PATH>/apps/yolo/work/yolo_deploy_608_8b.json --labelfile=<MLsuite ROOT PATH>/examples/classification/coco_names.txt --in_img/--in_img_dir=<MLsuite ROOT PATH>/vai/dpuv1/tools/quantize/calibration_directory/4788821373_441cd29c9f_z.jpg " <<endl;
}

void ProcessArgs(int argc, char** argv, string &xclbin, string &dataDir, string &netCfgFile,
                 string &quantCfgFile, string &labelFile,int &batch_sz, string &dir_path, vector<string> &image_path
                )
{
    map<string ,string> arg_map;
    vector<string> app_args;

    update_app_args(app_args);
    for(int i=1;i<argc;i=i+2){
        vector<string> result;
        result.push_back(argv[i]);
        if(result[0].compare("--help")==0){
            arg_list_help();
            exit(0);
        }
        result.push_back(argv[i+1]);
        vector<string>::iterator arg_it = find(app_args.begin(), app_args.end(), result[0]);
        if (arg_it != app_args.end())
        {
            int v_indx = arg_it - app_args.begin();
            arg_map[app_args[v_indx]]=result[1];
        }
    }
    vector<int>miss_list;
    check_arg_list(arg_map,app_args,miss_list);

    if(miss_list.size()>0){
        cout <<PFX << " List of arguments are missing from command line\n" ;
        for(int i=0;i<miss_list.size();i++){
                cout << PFX << app_args[miss_list[i]] << " = argument not found"<<endl;
        }
        cout<<PFX <<" Please check the missing arguments"<<endl;
        arg_list_help();
        exit(0);
    }

    xclbin=arg_map[app_args[0]];
    dataDir=arg_map[app_args[1]];
    netCfgFile=arg_map[app_args[2]];
    quantCfgFile=arg_map[app_args[3]];
    labelFile=arg_map[app_args[4]];
    dir_path = arg_map[app_args[5]];
    batch_sz=stoi(arg_map[app_args[6]]);
    map<string ,string>::iterator it=arg_map.find(app_args[7]);

    if (it!= arg_map.end())
    {
        image_path.push_back(arg_map[app_args[7]]);
    }
    for(int i=1;i<batch_sz;i++){
        image_path.push_back(arg_map[app_args[7]]);
    }

}

int classifycpp::xdnn_infer_preprocess(void)
{

    vector<cv::Mat> in_frame;

    //# Loop through batch size
    for(int bi = 0; bi < batch_sz; bi++)
    {
        //# Read input image
        cv::Mat in_frame_t = cv::imread(image_path[bi], 1);
        if(!in_frame_t.data)
        {
	        cout <<PFX << " Image read failed - " << image_path[bi] << endl;
            return -1;
        }
       	else
       	{
	        in_frame.push_back(in_frame_t);
	    }

        float *ptr = &input[bi*(in_dim.height*in_dim.width*in_dim.depth)];

		int status = prepareInputData(in_frame[bi], in_dim.height, in_dim.width, in_dim.depth, ptr,mean);
    }

    return 0;
}

int classifycpp::xdnn_infer_postprocess(void)
{
      //# FC Output buffer

	vector<float> fcWeight;
	vector<float> fcBias;
	vector<vector<float>> fc_layer_vect;
	int fc_insize = out_dim.depth;
	int fc_outsize = numClasses;
	vector<float> fcOut(fc_outsize * batch_sz);
    float *fcOutPtr = &(fcOut[0]);
    if(fcOutPtr == NULL)
    {
        cout <<PFX << " Failed to create FC output memory" << endl;
        return -1;
    }
	if(!fc_wb_map.empty())
    {
    	fc_layer_vect=fc_wb_map[0];
	    fcWeight=fc_layer_vect[0];
	    fcBias=fc_layer_vect[1];
    }
    for(int bi = 0; bi < batch_sz; bi++)
    {
        float *in_ptr = &output[bi*fc_insize];
        float *out_ptr = &fcOutPtr[bi*fc_outsize];

        computeFC(&(fcWeight[0]), &(fcBias[0]), in_ptr, 1, fc_outsize, fc_insize, out_ptr);

        vector<float> sx_ptr(out_ptr, out_ptr+fc_outsize);
        softmax(sx_ptr);

        // print top-5 index and its score value
        for(int it_t=0;it_t<5;it_t++){
            auto maxIt = max_element(sx_ptr.begin(), sx_ptr.end());
            int maxIdx = maxIt - sx_ptr.begin();
            cout << PFX << " Prediction class name: " << labels[maxIdx];
            cout <<" , Score :" << sx_ptr[maxIdx] << endl;
            sx_ptr[maxIdx]=0.0;
        }
	}

    return 0;
}
int input_dim_read(boost::property_tree::ptree const& pt,x_blob_dim_t &in_dim){
    using boost::property_tree::ptree;
    ptree::const_iterator end = pt.end();
	vector<int> v_in_dim;
    int i=0;
    for (ptree::const_iterator it = pt.begin(); it != end; ++it){
        v_in_dim.push_back(stoi(it->second.get_value<string>()));
        i++;
    }
	if(v_in_dim.size()==4)
	{
		in_dim.batch = v_in_dim[0];
		in_dim.depth = v_in_dim[1];
		in_dim.height = v_in_dim[2];
		in_dim.width = v_in_dim[3];

	}else{
		cout<< PFX << "compiler input dim read failed "<< endl;
		return -1;
	}
	return 0;
}
int output_dim_read(boost::property_tree::ptree const& pt,x_blob_dim_t &out_dim){
    using boost::property_tree::ptree;
    ptree::const_iterator end = pt.end();
    int i=0;
	vector<int> v_out_dim;
    for (ptree::const_iterator it = pt.begin(); it != end; ++it){
        v_out_dim.push_back(stoi(it->second.get_value<string>()));
        i++;
    }
	if(v_out_dim.size()==4)
	{
		out_dim.batch = v_out_dim[0];
		out_dim.depth = v_out_dim[1];
		out_dim.height = v_out_dim[2];
		out_dim.width = v_out_dim[3];

	}else{
		cout<< PFX << "compiler input dim read failed "<< endl;
		return -1;
	}
	return 0;
}

int classifycpp::json_search(boost::property_tree::ptree const& pt)
{
    static int net_flag=0;
    static int out_flag=0;
    using boost::property_tree::ptree;
    ptree::const_iterator end = pt.end();
    ptree::iterator it_pre;
    static int ret=-1;
    for (ptree::const_iterator it = pt.begin(); it != end; ++it) {

        if(it->first.compare("network")==0){
            net_flag=1;
        }
        if((net_flag==1) && (it->first.compare("outputshapes")==0))
        {
            net_flag=2;
            ret=input_dim_read(it->second,in_dim);
        }
        string layer_name = it->second.get_value<string>();
        if(layer_name.compare(output_layer_name)==0){
            out_flag=1;
        }
        if((out_flag==1) && (it->first.compare("outputshapes")==0)){
            out_flag=2;
            ret=output_dim_read(it->second,out_dim);
        }
        if((net_flag == 2) && (out_flag == 2))
            ret=0;
        else{
            json_search(it->second);
        }
    }
    return ret;
}


//# Create FPGA handle and initialize script executor
int classifycpp::xdnn_infer_init(void)
{
    string kernelName = "kernelSxdnn_0";
    int ret = xblasCreate(handle, xclbin.c_str(), kernelName.c_str(),0,0,"");
  //  assert(ret == 0);

    ptree pt;

    ifstream jsonFile(netCfgFile);

    // Read compiler json and build trees structure
    read_json(jsonFile, pt);

    for ( const auto & layer : pt.get_child("inputs"))
    {
		string name = layer.second.get<string>("input_name", "");
		input_layer_name=name;
    }

    for ( const auto & layer : pt.get_child("outputs"))
    {
        string name = layer.second.get<string>("output_name", "");
		output_layer_name=name;
    }
    ret=json_search(pt);

    //# Allocate input/output buffers
	input = new float[in_dim.height*in_dim.width*in_dim.depth*batch_sz*sizeof(float)];
    output = new float[out_dim.depth * batch_sz *sizeof(float)];
    if((input == NULL) || (output == NULL))
    {
        ret=-1;
        cout << PFX <<" Failed to create memory" << endl;
        assert(ret==0);
    }
    int layercnt = 0;

    vector<XBLASHandle*> handles;
    handles.push_back(handle);
    char *wgt_path = new char[dataDir.size()+1];
    char *cnetCfgFile=new char [netCfgFile.size() + 1];
    char *cquantCfgFile=new char [quantCfgFile.size() + 1];
    strcpy(wgt_path, dataDir.c_str());
    strcpy(cnetCfgFile, netCfgFile.c_str());
    strcpy(cquantCfgFile, quantCfgFile.c_str());

    // Load weights and get the executor handler for launching acceletor function
    executor = (XDNNScriptExecutor<float>*)XDNNMakeScriptExecutorAndLoadWeights(&handle,
    handles.size(),wgt_path,cnetCfgFile,cquantCfgFile,0);

    //# Load FC Data
	XDNNLoadFCWeights(fc_wb_map, wgt_path);

    // get labels
    labels = getLabels(labelFile);

    delete [] wgt_path;
    delete [] cnetCfgFile;
    delete [] cquantCfgFile;
	return 0;
}


int classifycpp::xdnn_infer_Execute(void)
{
    for(int bi = 0; bi < batch_sz; bi++)
    {
        float *ptr = &input[bi*(in_dim.height*in_dim.width*in_dim.depth)];
        vector<const float*> hw_in;
        hw_in.push_back(ptr);

        float *out_ptr = &output[bi*out_dim.depth];
        vector<float*> hw_out;
        hw_out.push_back(out_ptr);

        if((ptr == NULL) || (out_ptr == NULL))
        {
            cout <<PFX << " Failed to load IO pointers" << endl;
            return 1;
        }

        input_ptrs[input_layer_name]=hw_in;
        output_ptrs[output_layer_name]=hw_out;

        // gettimeofday(&start, 0);
        executor->execute(input_ptrs, output_ptrs, 0);
		// gettimeofday(&end, 0);
    }
    return 0;
}
