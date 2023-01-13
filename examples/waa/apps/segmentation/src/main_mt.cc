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

// Author: Daniele Bagni, Xilinx Inc.
// date: 21 May 2021

// WARNING: this code assumes that the image stored in the HD have the same size
// and do not need any resize

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <vitis/ai/profiling.hpp>
#include "common.h"
#include "pp_wrapper.h"
#include "post_wrapper.h"

#include "vart/assistant/xrt_bo_tensor_buffer.hpp"
#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
#include "vart/zero_copy_helper.hpp"
#include "vitis/ai/collection_helper.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;

long pre_time = 0, exec_time = 0, post_time = 0;

int g_pre_type = 0;
int g_run_nums = 500;
int g_post_type = 0;
atomic<int> g_idx = 0;
atomic<bool> g_is_first = true;
string baseImagePath;  // they will get their values via argv[]
int i=0;
int num_threads = 0;
int num_of_images = 0;

uint8_t colorB[] = {128, 232, 70, 156, 153, 153, 30,  0,   35, 152,
	180, 60,  0,  142, 70,  100, 100, 230, 32};
uint8_t colorG[] = {64,  35, 70, 102, 153, 153, 170, 220, 142, 251,
	130, 20, 0,  0,   0,   60,  80,  0,   11};
uint8_t colorR[] = {128, 244, 70,  102, 190, 153, 250, 220, 107, 152,
	70,  220, 255, 0,   0,   0,   0,   0,   119};

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images_list - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images_list) {
	images_list.clear();
	struct dirent *entry;

	/*Check if path is a valid directory path. */
	struct stat s;
	lstat(path.c_str(), &s);
	if (!S_ISDIR(s.st_mode)) {
		fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
		exit(1);
	}

	DIR *dir = opendir(path.c_str());
	if (dir == nullptr) {
		fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
		exit(1);
	}

	while ((entry = readdir(dir)) != nullptr) {
		if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
			string name = entry->d_name;
			string ext = name.substr(name.find_last_of(".") + 1);
			if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
					(ext == "jpg") || (ext == "PNG") || (ext == "png")) {
				images_list.push_back(name);
			}
		}
	}

	closedir(dir);
}

/**
 * @brief Run DPU Task for CNN
 *
 * @param taskFCN8 - pointer to FCN8 Task
 *
 * @return none
 */
void runCNN(vart::RunnerExt *runner, const vector<Mat> &images) {

	auto input_tensor_buffers = runner->get_inputs();
	auto output_tensor_buffers = runner->get_outputs();
	CHECK_EQ(input_tensor_buffers.size(), 1u) << "only support 1 input";

	auto input_tensor = input_tensor_buffers[0]->get_tensor();
	auto batch = input_tensor->get_shape().at(0);

	int height = input_tensor->get_shape().at(1);
	int width = input_tensor->get_shape().at(2);
	auto channels = input_tensor->get_shape().at(3);
	auto input_scale = vart::get_input_scale(input_tensor);
	auto inSize = height * width * channels;
	vector<Mat> imageList;

	auto output_tensor = output_tensor_buffers[1]->get_tensor();
	auto out_height = output_tensor->get_shape().at(1);
	auto out_width = output_tensor->get_shape().at(2);
	auto output_scale = vart::get_output_scale(output_tensor);

	auto osize = out_height * out_width;
	vector<uint64_t> dpu_input_phy_addr(batch, 0u);
	uint64_t dpu_input_size = 0u;
	vector<int8_t *> inptr_v;
	auto in_dims = input_tensor->get_shape();

	vector<uint64_t> data_in_addr(batch, 0u);

	for (auto batch_idx = 0; batch_idx < batch; ++batch_idx) 
	{
		std::tie(data_in_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data({batch_idx, 0, 0, 0});
		std::tie(dpu_input_phy_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data_phy({batch_idx, 0, 0, 0});
	}


	vector<uint64_t> dpu_output_phy_addr(batch, 0u);
	uint64_t dpu_output_size = 0u;
	vector<int8_t *> outptr_v;

	auto dims = output_tensor->get_shape();
	for (auto batch_idx = 0; batch_idx < batch; ++batch_idx) 
	{
		auto idx = std::vector<int32_t>(dims.size());
		idx[0] = batch_idx;
		auto data = output_tensor_buffers[1]->data(idx);
		int8_t *data_out = (int8_t *)data.first;
		outptr_v.push_back(data_out);
		std::tie(dpu_output_phy_addr[batch_idx], dpu_output_size) = output_tensor_buffers[1]->data_phy({batch_idx, 0, 0, 0});

	}

	vector<float> mean{127.0f, 127.0f, 127.0f};
	vector<float> scale{1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
	vector<float> real_scale{scale[0] * input_scale, scale[1] * input_scale,
		scale[2] * input_scale};
	float norm_fact = 127.0f;
	float shift_fact = 1.0f;
	float scale_fact = 64.0f;
	float out_scale_fact = 1.0f;

	AcceleratorHandle *preprocessor = nullptr;
	PostHandle *posthandle = nullptr;
	if (g_pre_type == 0) 
		preprocessor = pp_kernel_init("/run/media/mmcblk0p1/dpu.xclbin", mean, real_scale[0], height, width, 0);

	if(g_post_type == 0)
		posthandle = post_kernel_init("/run/media/mmcblk0p1/dpu.xclbin", out_height, out_width); 
	// run loop
	while (g_idx < g_run_nums) {

		__TIC__(setinput);
		for (auto idx = 0u; idx < batch; idx++) 
		{
			Mat image = cv::Mat(height, width, CV_8UC3);
			image = images[(g_idx + idx) % images.size()];
			int8_t *data = (int8_t *)data_in_addr[idx];

			if (g_pre_type == 0)  // Hardware implementation of pre-process using zero copy

				preprocess(preprocessor, image.data, image.rows, image.cols, height,width, dpu_input_phy_addr[idx], 0);    

			else  // CPU implementation of pre-process 
			{  
				resize(image, image, Size(1920, 832));
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						for (int c = 0; c < 3; c++) {

							float img_data = (float) image.at<Vec3b>(y, x)[c];
							data[3 * (y * width + x) + c] = (int8_t)((img_data - mean[c]) * real_scale[c]);
						}
					}
				}
			}
		}
		__TOC__(setinput);
		__TIC__(dpu);
		// run dpu

		if (g_pre_type != 0)
			for (auto &input : input_tensor_buffers)
				input->sync_for_write(0, input->get_tensor()->get_data_size() /
						input->get_tensor()->get_shape()[0]);

		/*run*/
		auto job_id =
			runner->execute_async(input_tensor_buffers, output_tensor_buffers);
		runner->wait(job_id.first, -1);

		if (g_post_type != 0)
			for (auto output : output_tensor_buffers)
				output->sync_for_read(0, output->get_tensor()->get_data_size() /
						output->get_tensor()->get_shape()[0]);

		__TOC__(dpu);

		__TIC__(post);

		int8_t *out_idx_data = new int8_t(out_width * out_height);

		for (auto idx = 0u; idx < batch; idx++) 
		{    
			cv::Mat segMat(out_height, out_width, CV_8UC3);

			if(g_post_type) // CPU implementation of post-process
			{
				auto *OutData = (int8_t *)outptr_v[idx];

				for (int row = 0; row < out_height; row++) 
				{
					for (int col = 0; col < out_width; col++) 
					{

						int ii = row * out_width * 12 + col * 12; 
						auto max_ind = max_element(OutData + ii, OutData + ii + 12);
						int posit = distance(OutData + ii, max_ind);

						segMat.at<Vec3b>(row, col) = Vec3b(colorB[posit], colorG[posit], colorR[posit]);

					}

				}             
			}
			else   // Hardware implementation of post-process using zero copy
			{        
				postprocess(posthandle, out_idx_data, dpu_output_phy_addr[idx] , scale_fact, out_height, out_width);

				uint8_t *data_idx = (uint8_t*)posthandle->out_idx_m;

				for (int row = 0; row < out_height; row++) 
				{
					for (int col = 0; col < out_width; col++) 
					{
						int hw_posit = int(data_idx[row * out_width + col]);

						segMat.at<Vec3b>(row, col) =  Vec3b(colorB[hw_posit],   colorG[hw_posit],  colorR[hw_posit]);   
					}
				}

			}

			auto image = images[(g_idx + idx) % images.size()];
			Mat small_img;

			cv::Mat showMat(out_height, out_width, CV_8UC3);
			cv::resize(image, small_img, Size(out_width, out_height), 0, 0);

			for (int ii = 0; ii < showMat.rows * showMat.cols * 3; ii++) 
			{
				showMat.data[ii] = small_img.data[ii] * 0.4 + segMat.data[ii] *0.6;
			}

			if (g_idx <= batch ) {
				Mat dst;
				cv::hconcat(small_img, segMat, dst);  // horizontal
				cv::imwrite(format("out_%03d.png", int(g_idx + idx)), dst);
			}
		}
		__TOC__(post);

		g_idx += batch;
	}

}

/**
 * @brief Entry for running FCN8 neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy FCN8 on DPU platform.
 *
 */
int main(int argc, char *argv[]) {

	steady_clock::time_point t_start, t_end;
	steady_clock::time_point t_start2, t_end2, r_end2;
	steady_clock::time_point t_start3, t_end3;

	// Check args
	if (argc != 6) {
		cout << "Usage: run_cnn xmodel_path test_images_path thread_num (from 1 "
			"to "
			"6) post_type(1:cpu, 0:hw) pre_type(0:hw, 1:cpu) "
			<< endl;
		return -1;
	}
	baseImagePath =
		std::string(argv[2]);  // path name of the folder with test images
	num_threads = atoi(argv[3]);
	assert((num_threads <= 6) & (num_threads >= 1));
	g_post_type = atoi(argv[4]);
	g_pre_type = atoi(argv[5]);

	/////////////////////////////////////////////////////////////////////////////////////////////
	// PREPARE DPU STUFF
	auto attrs = xir::Attrs::create();
	auto graph = xir::Graph::deserialize(argv[1]);
	auto subgraph = get_dpu_subgraph(graph.get());
	CHECK_EQ(subgraph.size(), 1u)
		<< "CNN should have one and only one dpu subgraph.";
	LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

	// create runners
	vector<std::unique_ptr<vart::RunnerExt>> runner;

	for (auto i = 0; i < num_threads; i++)
	{
		runner.push_back(vart::RunnerExt::create_runner(subgraph[0], attrs.get()));
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////////
	// MEMORY ALLOCATION

	// Load all image filenames
	vector<string> image_filename;
	ListImages(baseImagePath, image_filename);
	if (image_filename.size() == 0) {
		cerr << "\nError: No images existing under " << baseImagePath << endl;
		exit(-1);
	} else {
		num_of_images = image_filename.size();
	}

	// memory allocation
	vector<Mat> imagesList;

	/////////////////////////////////////////////////////////////////////////////////////////////
	// PREPROCESSING ALL IMAGES
	t_start2 = steady_clock::now();
	// preprocess all images at once
	for (unsigned int n = 0; n < num_of_images; n++) {
		auto image = imread(baseImagePath + image_filename[n]);
		imagesList.push_back(image);
	}
	t_end2 = steady_clock::now();
	auto duration2 = (duration_cast<microseconds>(t_end2 - t_start2)).count();
	cout << "\n" << endl;
	cout << "[READ  Time ] " << duration2 << "us" << endl;
	cout << "[READ  FPS  ] " << num_of_images * 1000000.0 / duration2 << endl;
	cout << "\n" << endl;

	std::cout<<"Running for "<<g_run_nums<<" iterations"<<std::endl;

	// MULTITHREADING DPU EXECUTION WITH BATCH

	thread workers[num_threads];

	t_start = steady_clock::now();

	for (auto i = 0; i < num_threads; i++) 
	{
		 workers[i] = thread(runCNN, runner[i].get(), imagesList);			
	}
	// Release thread resources.
	for (auto &w : workers) {
		if (w.joinable()) w.join();
	}

	t_end = steady_clock::now();
	auto duration = (duration_cast<microseconds>(t_end - t_start)).count();
	cout << "\n" << endl;

	cout<< "Profiling result with ";  
	if(g_pre_type ==0)
		cout<< "hardware preprocess and ";
	else
		cout<< "software preprocess and ";

	if(g_post_type == 0)
		cout<< "hardware postprocess:"<<endl;
	else
		cout<< "software postprocess: "<<endl;         

	cout << "[e2e      Time ] " << duration / g_run_nums << "us" << endl;
	cout << "[e2e      FPS  ] " << g_run_nums * 1000000.0 / duration << endl;

	cout << "\n" << endl;

	// delete[] softmax;
	cout << "deleting imagesList  memory" << endl;
	imagesList.clear();

	return 0;
}
