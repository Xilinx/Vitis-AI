/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <zconf.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "experimental/xrt_xclbin.h"
#include "common.h"
#include "utils.h"
#include "vart/runner_ext.hpp"
#include "pp_wrapper.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;

#define NMS_THRESHOLD 0.25f


AcceleratorHandle* handle = nullptr;


/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images)
{
	images.clear();
	struct dirent *entry;

	/*Check if path is a valid directory path. */
	struct stat s;
	lstat(path.c_str(), &s);
	if (!S_ISDIR(s.st_mode))
	{
		fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
		exit(1);
	}

	DIR *dir = opendir(path.c_str());
	if (dir == nullptr)
	{
		fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
		exit(1);
	}

	while ((entry = readdir(dir)) != nullptr)
	{
		if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN)
		{
			string name = entry->d_name;
			string ext = name.substr(name.find_last_of(".") + 1);
			if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
					(ext == "jpg") || (ext == "PNG") || (ext == "png"))
			{
				images.push_back(name);
			}
		}
	}

	closedir(dir);
}

GraphInfo shapes;
/**
 * @brief Feed input frame into DPU for process
 *
 * @param task - pointer to DPU Task for YOLO-v3 network
 * @param frame - pointer to input frame
 * @param mean - mean value for YOLO-v3 network
 *
 * @return none
 */
int runTotal;



void setInputImageForYOLO(uint64_t data_in,Mat &inImage, float input_scale,int sw_pp_flag,int no_zcpy)
{
	/* mean values for YOLO-v3 */
	float mean[3] = {0.0f, 0.0f, 0.0f};
	int width = shapes.inTensorList[0].width;
	int height = shapes.inTensorList[0].height;
	int size = shapes.inTensorList[0].size;
	int channel = 3;

	if (!sw_pp_flag)
	{
		preprocess(handle, (unsigned char *)inImage.data, inImage.rows,inImage.cols, height, width, data_in,no_zcpy);
	}
	else
	{
		int8_t *out_data = reinterpret_cast<int8_t *>(data_in);

		/// Get input and output data shapes
		int inChannel = inImage.channels();
		int inHeight  = inImage.rows;
		int inWidth   = inImage.cols;

		/// Resize the image to Network Shape (LetterBox)
		cv::Mat outImage = cv::Mat(height,width, CV_32FC3);
		/// Crop Letter-Box
		letterBoxImage(inImage,height, width, outImage);


		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				for (int c = 0; c < 3; c++) {
					out_data[(h*width*3) + (w*3) + c]
						= static_cast<int8_t>(outImage.at<cv::Vec3f>(h,w)[c]*input_scale);
				}
			}
		}
	}
}


/**
 * @brief Post process after the running of DPU for YOLO-v3 network
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param frame
 * @param sWidth
 * @param sHeight
 *
 * @return none
 */
void postProcess(vart::Runner *runner, Mat &frame, vector<int8_t *> results,
		vector<float> output_scale, int sWidth, int sHeight,ofstream& myfile, int num, vector<string>images, int verbose, int disp_img, int perf_test, float conf_thres)
{
	const string classes[3] = {"car", "person", "cycle"};
	/* four output nodes of YOLO-v3 */
	// const string outputs_node[4] = {"layer81_conv", "layer93_conv",
	//                                    "layer105_conv", "layer117_conv"};
	string image_name=images[num].substr(0,images[num].find_last_of(".") );
	vector<vector<float>> boxes;

	for (int ii = 0; ii < 4; ii++)
	{
		int width = shapes.outTensorList[ii].width;
		int height = shapes.outTensorList[ii].height;
		int channel = shapes.outTensorList[ii].channel;
		int sizeOut = channel * width * height;
		vector<float> result(sizeOut);
		boxes.reserve(sizeOut);

		//std::cout << "output_scale: " << output_scale << "\n";
		/* Store every output node results */
		get_output(results[ii], output_scale[ii], sizeOut, channel, height, width, result);

		/* Store the object detection frames as coordinate information  */
		detect(boxes, result, channel, height, width, ii, sHeight, sWidth, conf_thres);
	}

	/* Restore the correct coordinate frame of the original image */
	correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth,
			sHeight);

	/* Apply the computation for NMS */
	vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD, conf_thres);

	float h = frame.rows;
	float w = frame.cols;

	if(!perf_test)
	{
		for (size_t i = 0; i < res.size(); ++i)
		{
			float xmin = (res[i][0] - res[i][2] / 2.0) * w + 1.0;
			float ymin = (res[i][1] - res[i][3] / 2.0) * h + 1.0;
			float xmax = (res[i][0] + res[i][2] / 2.0) * w + 1.0;
			float ymax = (res[i][1] + res[i][3] / 2.0) * h + 1.0;   

			if (res[i][res[i][4] + 6] > conf_thres)
			{
				int type = res[i][4];
				string classname = classes[type];

				if(disp_img)
				{
					if (type == 0)
					{
						rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
								Scalar(0, 0, 255), 1, 1, 0);
					}
					else if (type == 1)
					{
						rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
								Scalar(255, 0, 0), 1, 1, 0);
					}
					else
					{
						rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
								Scalar(0, 255, 255), 1, 1, 0);
					}

				}

				if(verbose)
				{
					cout<<"image name: "<<image_name<<endl;
					cout<<"  "<<"xmin, "<<"ymin, "<<"xmax, "<<"ymax :"<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax<<endl;
				}

				myfile << image_name <<" "<<classname <<" "<<res[i][5] <<" "<< xmin <<" "<<ymin<<" "<< xmax <<" "<<ymax<<endl;
			}

		}

		if(disp_img)
		{
			cv::imwrite("./output/" + images[num], frame);
		}
	}

}

/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 *
 * @return none
 */
void runYOLO(vart::Runner *runner, string baseImagePath, vector<string>images, int sw_pp_flag, int no_zcpy, int verbose, int disp_img, int perf_test, float conf_thres)
{
	auto dpu_runner_ext = dynamic_cast<vart::RunnerExt *>(runner);

	auto input_tensor_buffers = dpu_runner_ext->get_inputs();
	auto output_tensor_buffers = dpu_runner_ext->get_outputs();

	auto input_tensor = input_tensor_buffers[0]->get_tensor();
	auto output_tensor = output_tensor_buffers[0]->get_tensor();

	auto in_dims = input_tensor->get_shape();
	auto batch = input_tensor->get_shape().at(0);
	auto height = input_tensor->get_shape().at(1);
	auto width = input_tensor->get_shape().at(2);

	auto input_scale = vart::get_input_scale(dpu_runner_ext->get_input_tensors());
	auto output_scale = vart::get_output_scale(dpu_runner_ext->get_output_tensors());

	vector<uint64_t> dpu_input_phy_addr(batch, 0u);
	vector<uint64_t> data_in_addr(batch, 0u);
	uint64_t dpu_input_size = 0u;

	const char *xclbinPath = std::getenv("XLNX_VART_FIRMWARE");
	auto xclbin_obj=xrt::xclbin(xclbinPath);
	string name=xclbin_obj.get_xsa_name();
	string platform_name=name.substr(7,4);

	if (platform_name == "vck5")
	   platform_name = "vck5000";

	for (auto batch_idx = 0; batch_idx < batch; ++batch_idx) 
	{
		std::tie(data_in_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data({batch_idx, 0, 0, 0});
		std::tie(dpu_input_phy_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data_phy({batch_idx, 0, 0, 0});		
	}

	vector<vector<int8_t *>>result(batch);

	auto out_dims = output_tensor->get_shape();

	for (auto batch_idx = 0; batch_idx < batch; ++batch_idx)
	{
		auto idx = std::vector<int32_t>(out_dims.size());
		idx[0] = batch_idx;

        if(platform_name == "u200" || platform_name == "u280" || platform_name == "u50_" ||  platform_name == "VCK5000")
		{
			for(int i = 0; i<4; i++)
			{
				auto out_ptr = output_tensor_buffers[i]->data(idx);
				int8_t *result0 = (int8_t *)out_ptr.first;
				result[batch_idx].push_back(result0);  
			}
		}

        else
		{
			auto out_ptr0 = output_tensor_buffers[2]->data(idx);
			int8_t *result0 = (int8_t *)out_ptr0.first;
			result[batch_idx].push_back(result0);

			auto out_ptr1 = output_tensor_buffers[3]->data(idx);
			int8_t *result1 = (int8_t *)out_ptr1.first;
			result[batch_idx].push_back(result1);

			auto out_ptr2 = output_tensor_buffers[0]->data(idx);
			int8_t *result2 = (int8_t *)out_ptr2.first;
			result[batch_idx].push_back(result2);

			auto out_ptr3 = output_tensor_buffers[1]->data(idx);
			int8_t *result3 = (int8_t *)out_ptr3.first;
			result[batch_idx].push_back(result3);	
		}

	}

	/* Load xclbin for hardware pre-processor */
	if (!sw_pp_flag)
	{  
		handle = pp_kernel_init(input_scale[0], height, width,no_zcpy);
	}

	std::chrono::steady_clock::time_point start,t1,pre_t1,pre_t2,exec_t1,exec_t2,post_t2,end;
	long imread_time = 0, pre_time = 0, exec_time = 0, post_time = 0, e2e_time=0;

	vector<Mat>imageList;

	int total_img = 0;

	ofstream myfile;
	myfile.open ("result.txt"); 

	start = std::chrono::steady_clock::now();

	for (unsigned int n = 0; n < images.size(); n += batch)
	{
		unsigned int runSize = (images.size() < (n + batch)) ? (images.size() - n) : batch;
		Mat img;
		
		for (unsigned int i = 0; i < runSize; i++)
		{
			auto t1 = std::chrono::steady_clock::now();

			img = cv::imread(baseImagePath +"/"+ images[n+i]);

			if(img.rows > 1080 && img.cols > 1980)
			{
				cout<<"The image file "<<baseImagePath+"/"+ images[n + i]<<" exceeds maximum resolution supported"<<" (SKIPPING)"<<endl;
				continue;
			}
            
			pre_t1 = std::chrono::steady_clock::now();
			auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(pre_t1 - t1);
			imread_time += value_t1.count();

			if(no_zcpy)   
				setInputImageForYOLO(data_in_addr[i], img, input_scale[0], sw_pp_flag, no_zcpy);
			else 
				setInputImageForYOLO(dpu_input_phy_addr[i], img, input_scale[0], sw_pp_flag, no_zcpy);

			auto pre_t2 = std::chrono::steady_clock::now();
			auto prevalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(pre_t2 - pre_t1);
			pre_time += prevalue_t1.count();

			imageList.push_back(img);     
		}

		total_img += imageList.size(); 

		exec_t1 = std::chrono::steady_clock::now();    

		if(no_zcpy)
		{
			for (auto& input : input_tensor_buffers) 
			{
				input->sync_for_write(0, input->get_tensor()->get_data_size() / input->get_tensor()->get_shape()[0]);
			}
		}
		

		auto job_id = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
		runner->wait((int)job_id.first, -1);

		for (auto& output : output_tensor_buffers) 
		{
			output->sync_for_read(0, output->get_tensor()->get_data_size() / output->get_tensor()->get_shape()[0]);
		}
		

		exec_t2 = std::chrono::steady_clock::now();
		auto execvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(exec_t2 - exec_t1);
		exec_time += execvalue_t1.count();

		for (unsigned int i = 0; i < imageList.size(); i++)
		{
			postProcess(runner, imageList[i], result[i], output_scale, width, height, myfile, n+i, images, verbose, disp_img, perf_test, conf_thres);
		}

		post_t2 = std::chrono::steady_clock::now();
		auto postvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(post_t2 - exec_t2);
		post_time += postvalue_t1.count();

		imageList.clear();
	}
	myfile.close();

	end = std::chrono::steady_clock::now();
	auto e2e_value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	e2e_time = e2e_value_t1.count();

	if(perf_test)
	{ 
		if(sw_pp_flag)
			cout<<"Profiling result with software preprocessing: "<<endl;
		else if(no_zcpy)
			cout<<"Profiling result with hardware preprocessing without zero copy: "<<endl;
		else
			cout<<"Profiling result with hardware preprocessing with zero copy: "<<endl;

		std::cout << "   E2E Performance: " << 1000000.0 / ((float)((e2e_time - imread_time) / total_img )) << " fps\n";
		std::cout << "   Pre-process Latency: " << (float)(pre_time / total_img) / 1000 << " ms\n";
		std::cout << "   Execution Latency: " << (float)(exec_time / total_img) / 1000 << " ms\n";
		std::cout << "   Post-process Latency: " << (float)(post_time / total_img) / 1000 << " ms\n";
	}

}

/**
 * @brief Entry for running YOLO-v3 neural network for ADAS object detection
 *
 */
int main(const int argc, const char **argv)
{
	if (argc != 8)
	{
		cout << "Usage of ADAS detection: ./adas_detection <model>  <image directory> <use_sw_pre_proc (1 for sw pre / 0 for hw pre)> <no_zero_copy (1 for no zero copy / 1 for zero copy)> <verbose (1 for printing the detection coordinates else 0)> <perf_test (1 for testing the performance of the Application else 0)> " << endl;
		return -1;
	}

	vector<string> images;
	auto graph = xir::Graph::deserialize(argv[1]);
	auto attrs = xir::Attrs::create();
	string baseImagePath = argv[2];
	int sw_pp_flag = atoi(argv[3]);
	int no_zcpy = atoi(argv[4]);
	int verbose = atoi(argv[5]);
	int disp_img = atoi(argv[6]);
	int perf_test = atoi(argv[7]);

	if(sw_pp_flag)
		no_zcpy = 1;
    
	float conf_thres;

	if(perf_test)
        conf_thres = 0.5;
	else 
	    conf_thres = 0.005;  	

	if(verbose || disp_img)
		perf_test = 0;
    
	cout << "\nThe Confidence Threshold used in this demo is " << conf_thres << endl;

	/* Load all image names.*/
	ListImages(baseImagePath, images);

	if (images.size() == 0)
	{
		cerr << "\nError: No images existing under " << baseImagePath << endl;
		return -1;
	}

	runTotal = images.size();
	cout<<"Total number of images in the dataset is "<<runTotal<<endl;

	auto subgraph = get_dpu_subgraph(graph.get());
	CHECK_EQ(subgraph.size(), 1u)
		<< "yolov3 should have one and only one dpu subgraph.";

    if(!no_zcpy)
		attrs->set_attr<bool>("zero_copy",true);

	std::unique_ptr<vart::RunnerExt> runner = vart::RunnerExt::create_runner(subgraph[0], attrs.get());
	// get in/out tenosrs
	auto inputTensors = runner->get_input_tensors();
	auto outputTensors = runner->get_output_tensors();

	int inputCnt = inputTensors.size();
	int outputCnt = outputTensors.size();
	// init the shape info
	TensorShape inshapes[inputCnt];
	TensorShape outshapes[outputCnt];
	shapes.inTensorList = inshapes;
	shapes.outTensorList = outshapes;

	getTensorShape(runner.get(), &shapes, inputCnt,
			{"layer81", "layer93", "layer105", "layer117"});

	runYOLO(runner.get(), baseImagePath, images, sw_pp_flag, no_zcpy, verbose, disp_img, perf_test, conf_thres);
	return 0;
}
