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

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include "experimental/xrt_xclbin.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <filesystem>
#include "common.h"
#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>
#include <adf.h>
#include <config.h>
#include "xrt/xrt_graph.h"

// C++ Header
#include "pp_wrapper.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;

int correct_classification = 0;
int total_images = 0;
const string wordsPath = "./";

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, std::vector<std::string> &images)
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

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const &path, std::vector<std::string> &kinds)
{
	kinds.clear();
	ifstream fkinds(path);
	if (fkinds.fail())
	{
		fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
		exit(1);
	}
	string kind;
	while (getline(fkinds, kind))
	{
		kinds.push_back(kind);
	}

	fkinds.close();
}

void LoadLabels(string const &path, std::vector<std::string> &images,std::vector<int> &labels)
{
	images.clear();
	labels.clear();
	ifstream file(path);
	if (file.fail())
	{
		fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
		exit(1);
	}
	string img_name;
	int label;
	while (file >> img_name >> label)
	{
		images.push_back(img_name);
		labels.push_back(label);
	}

	file.close();
}

/**
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(signed char *data, size_t size, float *result, float scale)
{
	assert(data && result);
	double sum = 0.0f;
	for (size_t i = 0; i < size; i++)
	{
		result[i] = exp((float)data[i] * scale);
		sum += result[i];
	}
	for (size_t i = 0; i < size; i++)
		result[i] /= sum;
}

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float *d, int size, int k, std::vector<std::string> &vkinds,std::vector<std::string> &images,std::vector<int> &labels,int idx,int verbose)
{
	assert(d && size > 0 && k > 0);
	priority_queue<pair<float, int>> q;

	for (auto i = 0; i < size; ++i)
		q.push(pair<float, int>(d[i], i));

	if(labels.size()!=0)
	{
		pair<float,int>most_probable = q.top();	
		if(labels[idx] == most_probable.second)
			correct_classification++;
		
	}
	if(verbose==1)
	{
		for (auto i = 0; i < k; ++i)
		{
			pair<float, int> ki = q.top();
			printf("top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
					vkinds[ki.second].c_str());
			q.pop();
		}
	}
}

/**
 * @brief Run DPU Task for ResNet50
 *
 * @param taskResnet50 - pointer to ResNet50 Task
 *
 * @return none
 */
void runResnet50(vart::RunnerExt *runner, const xir::Subgraph *subgraph, int sw_pp_flag, int no_zcpy,const string baseImagePath,string label_path,int verbose)
{
	/* Mean value for ResNet50 specified in Caffe prototxt */
	std::vector<std::string> kinds, images;
	std::vector<int>labels;

	/* Load all image names.*/

	if(label_path != "")
	{
		LoadLabels(label_path,images,labels);
		cout<<"Number of images in the label file is: "<<images.size()<<endl;
		
		if (labels.size() == 0)
		{
			cerr << "\nError: No labels existing under " << label_path << endl;
			return;
		}

	}
	else
	{ 
		ListImages(baseImagePath, images);
		
		cout<<"Number of images in the image directory is: "<<images.size()<<endl;
	}
	
	if (images.size() == 0)
	{
		cerr << "\nError: No images existing under " << baseImagePath << endl;
		return;
	}

	/* Load all kinds words.*/
	LoadWords(wordsPath + "words.txt", kinds);
	
	if (kinds.size() == 0)
	{
		cerr << "\nError: No words exist in file words.txt." << endl;
		return;
	}

	float mean[3] = {104, 107, 123};

	auto input_tensor_buffers = runner->get_inputs();
	auto output_tensor_buffers = runner->get_outputs();
	
	CHECK_EQ(input_tensor_buffers.size(), 1u) << "only support resnet50 model";
	CHECK_EQ(output_tensor_buffers.size(), 1u) << "only support resnet50 model";

	auto input_tensor = input_tensor_buffers[0]->get_tensor();
	auto output_tensor = output_tensor_buffers[0]->get_tensor();

	auto batch = input_tensor->get_shape().at(0);
	auto height = input_tensor->get_shape().at(1);
	auto width = input_tensor->get_shape().at(2);
	auto channels = input_tensor->get_shape().at(3);
	auto inSize = height * width * channels;
	std::vector<Mat> imageList;
	
	int outSize;
	
	auto dim_num = output_tensor->get_shape().size();
	if (dim_num == 2)
		outSize = output_tensor->get_shape().at(1);
	else
		outSize = output_tensor->get_shape().at(3);
    
	auto input_scale = vart::get_input_scale(input_tensor);
	auto output_scale = vart::get_output_scale(output_tensor);
	float *softmax = new float[outSize];
	
	long imread_time = 0, pre_time = 0, exec_time = 0, post_time = 0;

	std::vector<uint64_t> dpu_input_phy_addr(batch, 0u);
	std::vector<uint64_t> data_in_addr(batch, 0u);
	uint64_t dpu_input_size = 0u;

	auto in_dims = input_tensor->get_shape();
	for (auto batch_idx = 0; batch_idx < batch; ++batch_idx) 
	{
		std::tie(data_in_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data({batch_idx, 0, 0, 0});
		std::tie(dpu_input_phy_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data_phy({batch_idx, 0, 0, 0});
	}

	std::vector<int8_t *> outptr_v;
	auto out_dims = output_tensor->get_shape();
	for (auto batch_idx = 0; batch_idx < batch; ++batch_idx)
	{
		auto idx = std::vector<int32_t>(out_dims.size());
		idx[0] = batch_idx;
		auto data = output_tensor_buffers[0]->data(idx);
		int8_t *data_out = (int8_t *)data.first;
		outptr_v.push_back(data_out);
	}

	AcceleratorHandle* preprocessor = nullptr;
	if (!sw_pp_flag)
		preprocessor = pp_kernel_init(height, width, no_zcpy);
	
	int count = images.size();

	auto start = std::chrono::system_clock::now();
	
	/*run with batch*/
	unsigned int i;
	Mat image;
	for (unsigned int n = 0; n < images.size(); n += batch)
	{
		unsigned int runSize = (images.size() < (n + batch)) ? (images.size() - n) : batch;
		int last_batch_flag = (images.size() <= (n + batch)) ? 1 : 0;
		
		for (i = 0; i < runSize; i++)
		{
			auto t1 = std::chrono::system_clock::now();

			if(labels.size() != 0&& !filesystem::exists(baseImagePath + "/" + images[n + i]))
			{
				cout<<"The image file "<<baseImagePath + "/" + images[n + i]<<" doesnot exist in the image directory "<<baseImagePath<<" (SKIPPING)"<<endl;  
				continue;
			} 
			
			image = imread(baseImagePath + "/" + images[n + i]);
			
			if(image.rows != 1080 || image.cols != 1920)
			{
				cout<<"The image file "<<baseImagePath + "/" + images[n + i]<<" is not a FHD image"<<" (SKIPPING)"<<endl;
				continue;
			}

			auto t2 = std::chrono::system_clock::now();
			auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
			imread_time += value_t1.count();

			auto pre_t1 = std::chrono::system_clock::now();
			
			if (!sw_pp_flag)	
			{
				//# Hardware preproces
				if(!no_zcpy)					
					preprocess(preprocessor,image, image.rows, image.cols, height, width, dpu_input_phy_addr[i], no_zcpy);
				else
					preprocess(preprocessor,image, image.rows, image.cols, height, width, data_in_addr[i], no_zcpy);	

			}
			else
			{
				//# Software Preprocess
				Mat image2 = cv::Mat(height, width, CV_8SC3);
				resize(image, image2, Size(height,width), 0, 0); //, INTER_NEAREST);
				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						for (int c = 0; c < 3; c++)
						{
							float img_data = (float)image2.at<Vec3b>(h, w)[c];
							auto var = (img_data - mean[c]) * input_scale;
							int8_t *data_ptr = (int8_t *)data_in_addr[i];
							data_ptr[h * width * 3 + w * 3 + c] = (signed char)var;
						}
					}
				}
				
			} 

			auto pre_t2 = std::chrono::system_clock::now();
			auto prevalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(pre_t2 - pre_t1);
			pre_time += prevalue_t1.count();
			imageList.push_back(image);
		}
		
		bool run_once = true;
		while(run_once)
		{
			total_images += imageList.size();
			auto exec_t1 = std::chrono::system_clock::now();

			if (no_zcpy) 
				for (auto &input : input_tensor_buffers)
					input->sync_for_write(0, input->get_tensor()->get_data_size() /
							input->get_tensor()->get_shape()[0]);

			/*run*/
			auto job_id = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
			
			runner->wait(job_id.first, -1);
			
			for (auto output : output_tensor_buffers)
				output->sync_for_read(0, output->get_tensor()->get_data_size() /
						output->get_tensor()->get_shape()[0]);

			auto exec_t2 = std::chrono::system_clock::now();
			auto execvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(exec_t2 - exec_t1);
			exec_time += execvalue_t1.count();

			for (unsigned int i = 0; i < imageList.size(); i++)
			{
				/* Calculate softmax on CPU and display TOP-5 classification results */
				CPUCalcSoftmax(outptr_v[i], outSize, softmax, output_scale);

				TopK(softmax, outSize, 5, kinds,images,labels,n+i,verbose);
			}
			auto post_t2 = std::chrono::system_clock::now();
			auto postvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(post_t2 - exec_t2);
			post_time += postvalue_t1.count();

			imageList.clear();
			run_once = (last_batch_flag==1) && (sw_pp_flag==0);
			if (run_once)
			{
				last_batch_flag=0;
				auto pre_t1 = std::chrono::system_clock::now();
				if(!no_zcpy)	
					preprocess(preprocessor,image, image.rows, image.cols, height, width, dpu_input_phy_addr[0], no_zcpy);
				else
					preprocess(preprocessor,image, image.rows, image.cols, height, width, data_in_addr[0], no_zcpy);
				auto pre_t2 = std::chrono::system_clock::now();
				auto prevalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(pre_t2 - pre_t1);
				pre_time += prevalue_t1.count();
				imageList.push_back(image);

			}

		}
			
	}

	if (!sw_pp_flag)
		pp_graph_close(preprocessor);

	auto end = std::chrono::system_clock::now();
	auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	long e2e_time = value_t1.count();

	if(!verbose)
	{
		if(sw_pp_flag)
			cout<< "Profiling result with software preprocessing: "<<endl;
		else if(no_zcpy)
			cout<< "Profiling result with hardware preprocessing without zero copy: "<<endl;
		else
			cout<< "Profiling result with hardware preprocessing with zero copy: "<<endl;
		std::cout << "   E2E Performance: " << 1000000.0 / ((float)((e2e_time - imread_time) / count)) << " fps\n";
		std::cout << "   Pre-process Latency: " << (float)(pre_time / count) / 1000 << " ms\n";
		std::cout << "   Execution Latency: " << (float)(exec_time / count) / 1000 << " ms\n";
		std::cout << "   Post-process Latency: " << (float)(post_time / count) / 1000 << " ms\n";
	}    
#if EN_BRKP
	std::cout << "imread latency: " << (float)(imread_time / count) / 1000 << "ms\n";
#endif
	
	if(labels.size()!=0)
	{
		if(total_images == 0)
		{
			cout<<"There are no images to calculate accuracy"<<endl;
		}  	
		else
		{
			cout<<correct_classification<< " out of "<<total_images<<" images have been classified correctly"<<endl;
			float accuracy = float(correct_classification)/float(total_images)*100;
			cout<< "The accuracy of the network is "<< accuracy<< " %"<<endl;
		}
	}
    //convgraph.end();
	delete[] softmax;
}

/**
 * @brief Entry for runing ResNet50 neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy ResNet50 on DPU platform.
 *
 */
int main(int argc, char *argv[])
{
	if (argc < 6 || argc > 7)
	{
		cout << "Usage of Resnet50: ./bin/resnet50_waa_aie.exe <model>  <image directory> <use_sw_pre_proc (1 for sw pre / 0 for hw pre)> <verbose (1 for printing the detection coordinates else 0)> <label_file <path> (OPTIONAL)>" << endl;
		return -1;
	}
    
	auto graph = xir::Graph::deserialize(argv[1]);
	string baseImagePath = argv[2];
	int sw_pp_flag = atoi(argv[3]);
	int no_zcpy = atoi(argv[4]);
	auto attrs = xir::Attrs::create();
	int verbose = atoi(argv[5]);
	
	string label_path = "";
	if(argv[6])
		label_path = argv[6];
    
	if(sw_pp_flag)
		no_zcpy = 1;
    
	auto subgraph = get_dpu_subgraph(graph.get());
	
	CHECK_EQ(subgraph.size(), 1u)
		<< "resnet50 should have one and only one dpu subgraph.";
	LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
	
	/*create runner*/
	std::unique_ptr<vart::RunnerExt> runner = vart::RunnerExt::create_runner(subgraph[0], attrs.get());
	/*run with batch*/
	runResnet50(runner.get(), subgraph[0], sw_pp_flag, no_zcpy, baseImagePath, label_path, verbose);
	return 0;
}
