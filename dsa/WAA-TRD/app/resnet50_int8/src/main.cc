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
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>
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
	{
		result[i] /= sum;
	}
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
	{
		q.push(pair<float, int>(d[i], i));
	}

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
void runResnet50(vart::Runner *runner, const xir::Subgraph *subgraph, const string baseImagePath, int hw_pp_flag, int en_profile)
{
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

	/* Mean value for ResNet50 specified in Caffe prototxt */
	float mean[3] = {104, 107, 123};

	/* get in/out tensors and dims*/
	auto outputTensors = runner->get_output_tensors();
	auto inputTensors = runner->get_input_tensors();
	auto out_dims = outputTensors[0]->get_shape();
	auto in_dims = inputTensors[0]->get_shape();

	std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

	auto batch = inputTensors[0]->get_shape().at(0);
	auto height = inputTensors[0]->get_shape().at(1);
	auto width = inputTensors[0]->get_shape().at(2);
	auto channels = inputTensors[0]->get_shape().at(3);
	auto inSize = height * width * channels;

	auto input_scale = get_input_scale(inputTensors[0]);
	auto output_scale = get_output_scale(outputTensors[0]);

	auto dim_num = outputTensors[0]->get_shape().size();

	int outSize;
	if (dim_num == 2)
		outSize = outputTensors[0]->get_shape().at(1);
	else
		outSize = outputTensors[0]->get_shape().at(3);

	long imread_time = 0, pre_time = 0, exec_time = 0, post_time = 0;

	vector<Mat> imageList;
	int8_t *imageInputs = new int8_t[inSize * batch];
	float *softmax = new float[outSize];
	int8_t *FCResult = new int8_t[batch * outSize];

	std::vector<vart::TensorBuffer *> inputsPtr, outputsPtr;
	std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

	if (hw_pp_flag)
		pp_kernel_init(pphandle, mean, input_scale, height, width);

	std::cout << "number of images: " << images.size() << "\n";
	int count = images.size();

	auto start = std::chrono::system_clock::now();

	/*run with batch*/
	for (unsigned int n = 0; n < images.size(); n += batch)
	{

		unsigned int runSize = (images.size() < (n + batch)) ? (images.size() - n) : batch;
		for (unsigned int i = 0; i < runSize; i++)
		{
			if (!en_profile)
                		std::cout << "input image: " <<  baseImagePath + "/" + images[n + i] << std::endl;

			auto t1 = std::chrono::system_clock::now();
			Mat image = imread(baseImagePath + "/" + images[n + i]);
			auto t2 = std::chrono::system_clock::now();
			auto value_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
			imread_time += value_t1.count();

			if (hw_pp_flag)
			{
				//# Hardware preproces
				int8_t *data_ptr = (int8_t *)imageInputs + (i * inSize);
				preprocess(pphandle, image.data, image.rows, image.cols, height, width, data_ptr);
			}
			else
			{
				//# Software Preprocess
				Mat image2 = cv::Mat(height, width, CV_8SC3);
				resize(image, image2, Size(height, width), 0, 0, INTER_LINEAR);

				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						for (int c = 0; c < 3; c++)
						{
							float img_data = (float)image2.at<Vec3b>(h, w)[c];
							auto var = (img_data - mean[c]) * input_scale;
							imageInputs[i * inSize + h * width * 3 + w * 3 + c] = (signed char)var;
						}
					}
				}
			} // hw pp flag
			imageList.push_back(image);
		}

		/* in/out tensor refactory for batch inout/output */
		batchTensors.push_back(std::shared_ptr<xir::Tensor>(
			xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
								xir::DataType{xir::DataType::XINT, 8u})));
		inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
			imageInputs, batchTensors.back().get()));
		batchTensors.push_back(std::shared_ptr<xir::Tensor>(
			xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
								xir::DataType{xir::DataType::XINT, 8u})));
		outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
			FCResult, batchTensors.back().get()));

		/*tensor buffer input/output */
		inputsPtr.clear();
		outputsPtr.clear();
		inputsPtr.push_back(inputs[0].get());
		outputsPtr.push_back(outputs[0].get());

		auto job_id = runner->execute_async(inputsPtr, outputsPtr);
		runner->wait(job_id.first, -1);

		for (unsigned int i = 0; i < runSize; i++)
		{
			//cout << "\nImage : ./img/" << images[n + i] << endl;
			/* Calculate softmax on CPU and display TOP-5 classification results */
			CPUCalcSoftmax(&FCResult[i * outSize], outSize, softmax, output_scale);

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
		std::cout << "E2E Performance: " << 1000000.0 / ((float)((e2e_time-imread_time) / count)) << " fps\n";

 	delete[] FCResult;
	delete[] imageInputs;
  	delete[] softmax;

	//# Release xrt bo
	if (hw_pp_flag)
		release_pp(pphandle);
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
		cout << "Usage of resnet50 demo: ./resnet50 [model_file] [img dir] [pp_flag] [en_profile]" << endl;
		return -1;
	}
	auto img_path = argv[2];
	auto hw_pp_flag = atoi(argv[3]);
	auto en_profile = atoi(argv[4]);
	auto attrs = xir::Attrs::create();
	auto graph = xir::Graph::deserialize(argv[1]);
	auto subgraph = get_dpu_subgraph(graph.get());
	CHECK_EQ(subgraph.size(), 1u)
		<< "resnet50 should have one and only one dpu subgraph.";
	LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

	auto runner = vart::Runner::create_runner(subgraph[0], "run");

	/*run with batch*/
	runResnet50(runner.get(), subgraph[0], img_path, hw_pp_flag, en_profile);
	return 0;
}
