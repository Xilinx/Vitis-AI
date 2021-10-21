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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "common.h"
#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>
#include "vart/assistant/xrt_bo_tensor_buffer.hpp"
#include "vart/zero_copy_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "pp_wrapper.h"
PPHandle *pphandle;

using namespace std;
using namespace cv;
using namespace std::chrono;

const string wordsPath = "./";

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

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const &path, vector<string> &kinds)
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
void TopK(const float *d, int size, int k, vector<string> &vkinds)
{
	assert(d && size > 0 && k > 0);
	priority_queue<pair<float, int>> q;

	for (auto i = 0; i < size; ++i)
		q.push(pair<float, int>(d[i], i));
	
	for (auto i = 0; i < k; ++i)
	{
		pair<float, int> ki = q.top();
		printf("top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
			   vkinds[ki.second].c_str());
		q.pop();
	}
}

/**
 * @brief Run DPU Task for ResNet50
 *
 * @param taskResnet50 - pointer to ResNet50 Task
 *
 * @return none
 */
void runResnet50(vart::RunnerExt *runner, const xir::Subgraph *subgraph, int hw_pp_flag, int en_profile, const string baseImagePath)
{
	/* Mean value for ResNet50 specified in Caffe prototxt */
	vector<string> kinds, images;

	/* Load all image names.*/
	ListImages(baseImagePath, images);
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
	auto batch = input_tensor->get_shape().at(0);
	auto height = input_tensor->get_shape().at(1);
	auto width = input_tensor->get_shape().at(2);
	auto channels = input_tensor->get_shape().at(3);
	auto input_scale = vart::get_input_scale(input_tensor);
	auto inSize = height * width * channels;

	vector<Mat> imageList;

	auto output_tensor = output_tensor_buffers[0]->get_tensor();
	auto outSize = output_tensor->get_shape().at(1);
	auto output_scale = vart::get_output_scale(output_tensor);

	float *softmax = new float[outSize];
	long imread_time = 0, pre_time = 0, exec_time = 0, post_time = 0;

	vector<uint64_t> dpu_input_phy_addr(batch, 0u);
	uint64_t dpu_input_size = 0u;

	vector<int8_t *> inptr_v;
	auto in_dims = input_tensor->get_shape();
	for (auto batch_idx = 0; batch_idx < in_dims[0]; ++batch_idx) 
	{
		auto data = input_tensor_buffers[0]->data({batch_idx, 0, 0, 0});
		int8_t *data_out = (int8_t *)data.first;
		inptr_v.push_back(data_out);
		std::tie(dpu_input_phy_addr[batch_idx], dpu_input_size) = input_tensor_buffers[0]->data_phy({batch_idx, 0, 0, 0});
	}

	vector<int8_t *> outptr_v;
	auto dims = output_tensor->get_shape();
	for (auto batch_idx = 0; batch_idx < dims[0]; ++batch_idx)
	{
		auto idx = std::vector<int32_t>(dims.size());
		idx[0] = batch_idx;
		auto data = output_tensor_buffers[0]->data(idx);
		int8_t *data_out = (int8_t *)data.first;
		outptr_v.push_back(data_out);
	}
	
	if (hw_pp_flag)
		pp_kernel_init(pphandle, "/usr/lib/dpu.xclbin", mean, input_scale, height, width);

	std::cout << "number of images: " << images.size() << "\n";
	int count = images.size();

	auto start = std::chrono::system_clock::now();

	/*run with batch*/
	for (unsigned int n = 0; n < images.size(); n += batch)
	{

		unsigned int runSize = (images.size() < (n + batch)) ? (images.size() - n) : batch;
		for (unsigned int i = 0; i < runSize; i++)
		{
			auto t1 = std::chrono::system_clock::now();
			Mat image = imread(baseImagePath + "/" + images[n + i]);
			auto t2 = std::chrono::system_clock::now();
			auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
			imread_time += value_t1.count();

#if EN_BRKP
			auto pre_t1 = std::chrono::system_clock::now();
#endif
			if (hw_pp_flag)	{
				//# Hardware preprocess
				preprocess(pphandle, image.data, dpu_input_phy_addr[i], image.rows, image.cols);
			}
			else
			{
				//# Software Preprocess
				Mat image2 = cv::Mat(height, width, CV_8SC3);
				resize(image, image2, Size(height, width), 0, 0); //, INTER_NEAREST);

				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						for (int c = 0; c < 3; c++)
						{
							float img_data = (float)image2.at<Vec3b>(h, w)[c];
							auto var = (img_data - mean[c]) * input_scale;
							int8_t *data_in = (int8_t *)inptr_v[i];
							data_in[h * width * 3 + w * 3 + c] = (signed char)var;
						}
					}
				}
			} // hw pp flag
#if EN_BRKP
			auto pre_t2 = std::chrono::system_clock::now();
			auto prevalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(pre_t2 - pre_t1);
			pre_time += prevalue_t1.count();
#endif
			imageList.push_back(image);
		}
#if EN_BRKP
		auto exec_t1 = std::chrono::system_clock::now();
#endif

		if (!hw_pp_flag)
			for (auto &input : input_tensor_buffers)
				input->sync_for_write(0, input->get_tensor()->get_data_size() /
											 input->get_tensor()->get_shape()[0]);

		/*run*/
		auto job_id = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
		runner->wait(job_id.first, -1);

		for (auto output : output_tensor_buffers)
			output->sync_for_read(0, output->get_tensor()->get_data_size() /
										 output->get_tensor()->get_shape()[0]);
#if EN_BRKP
		auto exec_t2 = std::chrono::system_clock::now();
		auto execvalue_t1 = std::chrono::duration_cast<std::chrono::microseconds>(exec_t2 - exec_t1);
		exec_time += execvalue_t1.count();
#endif
		//auto post_t1 = std::chrono::system_clock::now();
		for (unsigned int i = 0; i < runSize; i++)
		{
			//cout << "\nImage : ./img/" << images[n + i] << endl;
			/* Calculate softmax on CPU and display TOP-5 classification results */
			CPUCalcSoftmax(outptr_v[i], outSize, softmax, output_scale);

			if (!en_profile)
				TopK(softmax, outSize, 5, kinds);

			/* Display the impage */
			//cv::imshow("Classification of ResNet50", imageList[i]);
			//cv::waitKey(10000);
		}
		imageList.clear();
	}

	auto end = std::chrono::system_clock::now();
	auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	long e2e_time = value_t1.count();

	if (en_profile)
	{
		std::cout << "E2E Performance: " << 1000000.0 / ((float)((e2e_time - imread_time) / count)) << " fps\n";
#if EN_BRKP
		std::cout << "imread latency: " << (float)(imread_time / count) / 1000 << "ms\n";
		std::cout << "pre latency: " << (float)(pre_time / count) / 1000 << "ms\n";
		std::cout << "exec latency: " << (float)(exec_time / count) / 1000 << "ms\n";
#endif
	}

	delete[] softmax;

	if (hw_pp_flag)
		releaseBO(pphandle);
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
	// Check args
	if (argc != 5)
	{
		cout << "Usage of resnet50 demo: ./resnet50 [model_file] [img dir] [pp_flag] [profile]" << endl;
		return -1;
	}
	string baseImagePath = argv[2];
	auto hw_pp_flag = atoi(argv[3]);
	auto en_profile = atoi(argv[4]);
	auto attrs = xir::Attrs::create();
	auto graph = xir::Graph::deserialize(argv[1]);
	auto subgraph = get_dpu_subgraph(graph.get());

	CHECK_EQ(subgraph.size(), 1u)
		<< "resnet50 should have one and only one dpu subgraph.";
	LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

	/*create runner*/
	std::unique_ptr<vart::RunnerExt> runner =
		vart::RunnerExt::create_runner(subgraph[0], attrs.get());

	/*run with batch*/
	runResnet50(runner.get(), subgraph[0], hw_pp_flag, en_profile, baseImagePath);
	return 0;
}
