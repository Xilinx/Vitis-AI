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

#include <vitis/ai/profiling.hpp>
#include "common.h"
#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>
#include "post_wrapper.h"
#include "pp_wrapper.h"
#include "pre_wrapper.h"
#include "vart/assistant/xrt_bo_tensor_buffer.hpp"
#include "vart/zero_copy_helper.hpp"
#include "vitis/ai/collection_helper.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;

GraphInfo shapes;
int g_pre_type = 0;
int g_run_nums = 1200;
bool g_is_use_post = false;
atomic<int> g_idx = 0;
atomic<bool> g_is_first = true;
// const string baseImagePath = "./src/img_test/";
string baseImagePath;  // they will get their values via argv[]

int num_threads = 0;
int is_running_0 = 1;
int num_of_images = 0;
int num_images_x_thread = 0;

int NUM_TEST_IMAGES = 50;
unsigned long long int max_addr_imags;
unsigned long long int max_addr_FCres;

uint8_t colorB[] = {128, 232, 70, 156, 153, 153, 30,  0,   35, 152,
                    180, 60,  0,  142, 70,  100, 100, 230, 32};
uint8_t colorG[] = {64,  35, 70, 102, 153, 153, 170, 220, 142, 251,
                    130, 20, 0,  0,   0,   60,  80,  0,   11};
uint8_t colorR[] = {128, 244, 70,  102, 190, 153, 250, 220, 107, 152,
                    70,  220, 255, 0,   0,   0,   0,   0,   119};

static int8_t op_move(uint8_t i) {
  int8_t ret = int8_t(i - 128);
  return ret >> 1;
}

static uint64_t get_physical_address(const xclDeviceHandle &handle,
                                     const unsigned int bo) {
  xclBOProperties p;
  auto error_code = xclGetBOProperties(handle, bo, &p);
  uint64_t phy = 0u;
  if (error_code != 0) {
    LOG(INFO) << "cannot xclGetBOProperties !";
  }
  phy = error_code == 0 ? p.paddr : -1;
  return phy;
}

vector<int8_t> random_vector_char(size_t sz) {
  static std::mt19937 rng(100);
  static std::uniform_int_distribution<char> dist;
  auto ret = vector<int8_t>(sz);
  for (auto i = 0u; i < ret.size(); ++i) {
    ret[i] = dist(rng);
  }
  return ret;
}

void argmax_c(const int8_t *input, unsigned int cls, unsigned int group,
              uint8_t *output) {
  for (unsigned int i = 0; i < group; ++i) {
    auto max_ind = max_element(input + i * cls, input + (i + 1) * cls);
    int8_t posit = (int8_t)distance(input + i * cls, max_ind);
    // if (i < 100) cout << " posit= " << posit - 0 << endl;  //
    output[i] = posit;
  }
}

void compare(int cls, int group, signed char *input, uint8_t *output1,
             uint8_t *output2) {
  for (auto g = 0; g < group; ++g) {
    auto idx = g;
    auto diff = output1[idx] - output2[idx];
    // if ((diff != 0.0 && std::abs(diff) > 0.001)) {
    if (1) {
      if (g < 10)
        cout << " g=" << g << " "  //
             << input[idx * cls] - 0 << " " << input[idx * cls + 1] - 0 << " "
             << ": " << output1[idx] - 0 << " " << output2[idx] - 0 << " "
             << diff << endl;
    }
  }
}

void transform_bgr(int w, int h, unsigned char *src, signed char *dst,
                   float var_shift_B, float var_scale_B, float var_shift_G,
                   float var_scale_G, float var_shift_R, float var_scale_R) {
  float32x4_t shiftB = vdupq_n_f32(var_shift_B);
  float32x4_t shiftG = vdupq_n_f32(var_shift_G);
  float32x4_t shiftR = vdupq_n_f32(var_shift_R);

  float32x4_t scaleB = vdupq_n_f32(var_scale_B);
  float32x4_t scaleG = vdupq_n_f32(var_scale_G);
  float32x4_t scaleR = vdupq_n_f32(var_scale_R);

  for (int i = 0; i < h; i++) {
    int idx_base = i * w * 3;
    for (int j = 0; j < w; j += 8) {
      int idx = idx_base + j * 3;

      // init
      uint8x8x3_t sbgr_u8;
      uint16x8x3_t sbgr_u16;
      sbgr_u8 = vld3_u8(src + idx);
      sbgr_u16.val[0] = vmovl_u8(sbgr_u8.val[0]);
      sbgr_u16.val[1] = vmovl_u8(sbgr_u8.val[1]);
      sbgr_u16.val[2] = vmovl_u8(sbgr_u8.val[2]);

      // get low part u32
      uint32x4_t sb_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[0]));
      uint32x4_t sg_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[1]));
      uint32x4_t sr_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[2]));

      // get high part u32
      uint32x4_t sb_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[0]));
      uint32x4_t sg_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[1]));
      uint32x4_t sr_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[2]));

      // get low part float
      float32x4_t sb_low_f32 = vcvtq_f32_u32(sb_low_u32);
      float32x4_t sg_low_f32 = vcvtq_f32_u32(sg_low_u32);
      float32x4_t sr_low_f32 = vcvtq_f32_u32(sr_low_u32);

      // get high part float
      float32x4_t sb_high_f32 = vcvtq_f32_u32(sb_high_u32);
      float32x4_t sg_high_f32 = vcvtq_f32_u32(sg_high_u32);
      float32x4_t sr_high_f32 = vcvtq_f32_u32(sr_high_u32);

      // calculate low part float
      sb_low_f32 = vmulq_f32(vsubq_f32(sb_low_f32, shiftB), scaleB);
      sg_low_f32 = vmulq_f32(vsubq_f32(sg_low_f32, shiftG), scaleG);
      sr_low_f32 = vmulq_f32(vsubq_f32(sr_low_f32, shiftR), scaleR);

      // calculate low part float
      sb_high_f32 = vmulq_f32(vsubq_f32(sb_high_f32, shiftB), scaleB);
      sg_high_f32 = vmulq_f32(vsubq_f32(sg_high_f32, shiftG), scaleG);
      sr_high_f32 = vmulq_f32(vsubq_f32(sr_high_f32, shiftR), scaleR);

      // get the result low part int32
      int32x4_t db_low_s32 = vcvtq_s32_f32(sb_low_f32);
      int32x4_t dg_low_s32 = vcvtq_s32_f32(sg_low_f32);
      int32x4_t dr_low_s32 = vcvtq_s32_f32(sr_low_f32);

      // get the result high part int32
      int32x4_t db_high_s32 = vcvtq_s32_f32(sb_high_f32);
      int32x4_t dg_high_s32 = vcvtq_s32_f32(sg_high_f32);
      int32x4_t dr_high_s32 = vcvtq_s32_f32(sr_high_f32);

      // get the result low part int16
      int16x4_t db_low_s16 = vmovn_s32(db_low_s32);
      int16x4_t dg_low_s16 = vmovn_s32(dg_low_s32);
      int16x4_t dr_low_s16 = vmovn_s32(dr_low_s32);

      // get the result high part int16
      int16x4_t db_high_s16 = vmovn_s32(db_high_s32);
      int16x4_t dg_high_s16 = vmovn_s32(dg_high_s32);
      int16x4_t dr_high_s16 = vmovn_s32(dr_high_s32);

      // combine low and high into int16x8
      int16x8_t db_s16 = vcombine_s16(db_low_s16, db_high_s16);
      int16x8_t dg_s16 = vcombine_s16(dg_low_s16, dg_high_s16);
      int16x8_t dr_s16 = vcombine_s16(dr_low_s16, dr_high_s16);

      // combine low and high into int16x8
      int8x8x3_t dbgr;
      dbgr.val[0] = vmovn_s16(db_s16);
      dbgr.val[1] = vmovn_s16(dg_s16);
      dbgr.val[2] = vmovn_s16(dr_s16);

      // store...
      vst3_s8(dst + idx, dbgr);
    }
  }
}

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
  // CHECK_EQ(output_tensor_buffers.size(), 1u) << "only support resnet50
  // model";

  auto input_tensor = input_tensor_buffers[0]->get_tensor();
  auto batch = input_tensor->get_shape().at(0);
  // int16_t height = (int16_t)input_tensor->get_shape().at(1);
  // int16_t width = (int16_t)input_tensor->get_shape().at(2);
  int height = input_tensor->get_shape().at(1);
  int width = input_tensor->get_shape().at(2);
  auto channels = input_tensor->get_shape().at(3);
  auto input_scale = vart::get_input_scale(input_tensor);
  auto inSize = height * width * channels;
  vector<Mat> imageList;

  auto output_tensor = output_tensor_buffers[0]->get_tensor();
  auto out_height = output_tensor->get_shape().at(1);
  auto out_width = output_tensor->get_shape().at(2);
  auto output_scale = vart::get_output_scale(output_tensor);

  auto osize = out_height * out_width;
  auto random_input = random_vector_char(osize * 28);
  vector<uint64_t> dpu_input_phy_addr(batch, 0u);
  uint64_t dpu_input_size = 0u;
  vector<int8_t *> inptr_v;
  auto in_dims = input_tensor->get_shape();
  for (auto batch_idx = 0; batch_idx < in_dims[0]; ++batch_idx) {
    auto data = input_tensor_buffers[0]->data({batch_idx, 0, 0, 0});
    int8_t *data_out = (int8_t *)data.first;
    inptr_v.push_back(data_out);
    std::tie(dpu_input_phy_addr[batch_idx], dpu_input_size) =
        input_tensor_buffers[0]->data_phy({batch_idx, 0, 0, 0});
  }

  vector<uint64_t> dpu_output_phy_addr(batch, 0u);
  uint64_t dpu_output_size = 0u;
  vector<int8_t *> outptr_v;
  auto dims = output_tensor->get_shape();
  for (auto batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
    auto idx = std::vector<int32_t>(dims.size());
    idx[0] = batch_idx;
    auto data = output_tensor_buffers[0]->data(idx);
    int8_t *data_out = (int8_t *)data.first;
    outptr_v.push_back(data_out);
    std::tie(dpu_output_phy_addr[batch_idx], dpu_output_size) =
        output_tensor_buffers[0]->data_phy({batch_idx, 0, 0, 0});
  }

  vector<float> mean{127.0f, 127.0f, 127.0f};
  vector<float> scale{1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
  vector<float> real_scale{scale[0] * input_scale, scale[1] * input_scale,
                           scale[2] * input_scale};
  float norm_fact = 127.0f;
  float shift_fact = 1.0f;
  float scale_fact = 64.0f;
  float out_scale_fact = 1.0f;
  // cout << "scale: " << scale[0] << " fix: " << input_scale
  //     << " real scale: " << real_scale[0] << endl;
  // PreHandle *prehandle;
  PPHandle *pphandle;
  PostHandle *posthandle;
  if (g_pre_type == 0) {
    // pre_kernel_init(prehandle, "/usr/lib/dpu.xclbin", norm_fact, shift_fact,
    //               scale_fact, height, width);
    pp_kernel_init(pphandle, "/usr/lib/dpu.xclbin", mean, real_scale, height,
                   width);
    post_kernel_init(posthandle, "/usr/lib/dpu.xclbin", out_scale_fact,
                     out_height, out_width);
    // exit(0);
  }
  auto handle = posthandle->handle;
  unsigned randomBo = xclAllocBO(handle, osize * 28, 0, 0);
  signed char *out_idxptr = (signed char *)xclMapBO(handle, randomBo, true);
  memcpy(out_idxptr, &random_input[0], osize * 28);
  xclSyncBO(handle, randomBo, XCL_BO_SYNC_BO_TO_DEVICE, osize * 28, 0u);
  uint64_t random_output_phy_addr = get_physical_address(handle, randomBo);
  if ((random_output_phy_addr == (uint64_t)(-1))) cout << "error" << endl;

  // std::cout << " random_output_phy_addr " << std::hex <<
  // random_output_phy_addr
  //          << std::dec << std::endl;
  // run loop
  while (g_idx < g_run_nums) {
    g_idx += batch;
    __TIC__(setinput);
    for (auto idx = 0u; idx < batch; idx++) {
      auto image = images[(g_idx + idx) % images.size()];
      auto *data = (int8_t *)inptr_v[idx];
      if (g_pre_type == 0) {
        // cout << "befor hw pre" << endl;
        // for (int i = 0; i < 10; ++i) {
        //  cout << "idx: " << i << " " << image.data[i] - 0 << " "
        //       << pphandle->img_inptr[i] - 0 << endl;
        //}
        // preprocess(prehandle, image.data, random_output_phy_addr, height,
        // ppprocess(pphandle, image.data, random_output_phy_addr, image.rows,
        ppprocess(pphandle, image.data, dpu_input_phy_addr[idx], image.rows,
                  image.cols);
        // preprocess(prehandle, image.data, dpu_input_phy_addr[idx], height,
        //           width);
        // xclSyncBO(handle, randomBo, XCL_BO_SYNC_BO_FROM_DEVICE,
        //          width * height * 3, 0u);
        // cout << "after hw pre, w: " << width << " h: " << height << endl;
        // for (int i = 0; i < 10; ++i) {
        //  cout << "idx: " << i << " " << image.data[i] - 0 << " " << data[i] -
        //  0
        //       << " " << pphandle->img_inptr[i] - 0
        //       << " golden: " << (image.data[i] - 128) / 2 << endl;
        //}

        // compare(28, osize, &random_input[0], out_idx_data, golden_idx_out);
      }
      if (g_pre_type == 1) {  // Original treatment
        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
              // imageInputs[n*inSize + 3*(y*width+x) + 2-c] =
              // ((float)image2.at<Vec3b>(y,x)[c])/127.5 -1.0; //RGB
              // conversion
              cout << x << " " << y << endl;
              data[3 * (y * width + x) + c] =
                  ((int8_t)((image.at<Vec3b>(x, y)[c]) - mean[c]) *
                   real_scale[c]);  // BGR format
            }
          }
        }
      }
      // The following code can improve prepost performance, for reference
      if (g_pre_type == 2)  // Sequential storage
        transform(image.data, image.data + inSize, data, op_move);
      if (g_pre_type == 3) {  // neon accelerate
        for (auto i = 0; i < height; ++i) {
          transform_bgr(width, 1,
                        const_cast<uint8_t *>(image.data) + i * 3 * width,
                        data + i * width * 3, mean[0], real_scale[0], mean[1],
                        real_scale[1], mean[2], real_scale[2]);
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

    if (g_pre_type != 0)
      for (auto output : output_tensor_buffers)
        output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                     output->get_tensor()->get_shape()[0]);
    __TOC__(dpu);
    if (g_is_use_post) {
      __TIC__(post);
      uint8_t *out_idx_data = new uint8_t(out_width * out_height);
      for (auto idx = 0u; idx < batch; idx++) {
        // postprocess(posthandle, out_idx_data, dpu_output_phy_addr[idx]);
        postprocess(posthandle, out_idx_data, random_output_phy_addr);
        unsigned char *data_idx = posthandle->out_idxptr;
        if (1) {
          // auto *data = (int8_t *)outptr_v[idx];
          uint8_t *golden_idx_out = new uint8_t[osize];
          argmax_c(&random_input[0], 28, osize, golden_idx_out);
          compare(28, osize, &random_input[0], data_idx, golden_idx_out);
          // auto image = images[(g_idx + idx) % images.size()];
          // Mat small_img;
          // Mat showMat(out_height, out_width, CV_8UC3);
          // Mat segMat(out_height, out_width, CV_8UC3);
          // cv::resize(image, small_img, Size(out_width, out_height), 0, 0,
          //           INTER_AREA);
          // for (int row = 0; row < out_height; row++) {
          //  for (int col = 0; col < out_width; col++) {
          //    int posit = int(golden_idx_out[row * out_width + col]);
          //    segMat.at<Vec3b>(row, col) =
          //        Vec3b(colorB[posit], colorG[posit], colorR[posit]);
          //  }
          //}
          // for (int ii = 0; ii < showMat.rows * showMat.cols * 3; ii++) {
          //  showMat.data[ii] = small_img.data[ii] * 0.4 + segMat.data[ii] *
          //  0.6;
          //}
          // if (g_idx <= 5) {
          //  Mat dst;
          //  cv::hconcat(small_img, segMat, dst);  // horizontal
          //  cv::imwrite(format("out_%03d.png", int(g_idx + idx)), dst);
          //}
        }
      }
      __TOC__(post);
    }
  }
  // release
  if (g_pre_type == 0 && g_is_first) {
    g_is_first = false;
    // releaseBO(prehandle);
    releaseBO(pphandle);
    // releaseBO(posthandle);
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
  // using std::chrono::system_clock;
  system_clock::time_point t_start, t_end;
  system_clock::time_point t_start2, t_end2, r_end2;
  system_clock::time_point t_start3, t_end3;

  // Check args
  if (argc != 6) {
    cout << "Usage: run_cnn xmodel_path test_images_path thread_num (from 1 to "
            "6) use_post_proc(1:yes, 0:no) pre_type(0:hw, 1:cpu, 2:neon)"
         << endl;
    return -1;
  }
  baseImagePath =
      std::string(argv[2]);  // path name of the folder with test images
  num_threads = atoi(argv[3]);
  assert((num_threads <= 6) & (num_threads >= 1));
  g_is_use_post = bool(argv[4]);
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
  auto runner = vart::RunnerExt::create_runner(subgraph[0], attrs.get());
  auto runner1 = vart::RunnerExt::create_runner(subgraph[0], attrs.get());
  auto runner2 = vart::RunnerExt::create_runner(subgraph[0], attrs.get());
  auto runner3 = vart::RunnerExt::create_runner(subgraph[0], attrs.get());
  auto runner4 = vart::RunnerExt::create_runner(subgraph[0], attrs.get());
  auto runner5 = vart::RunnerExt::create_runner(subgraph[0], attrs.get());

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

  if (num_of_images > NUM_TEST_IMAGES) num_of_images = NUM_TEST_IMAGES;
  cout << "\n max num of images to read " << num_of_images << endl;

  // memory allocation
  vector<Mat> imagesList;

  /////////////////////////////////////////////////////////////////////////////////////////////
  // PREPROCESSING ALL IMAGES
  t_start2 = system_clock::now();
  // preprocess all images at once
  for (unsigned int n = 0; n < num_of_images; n++) {
    auto image = imread(baseImagePath + image_filename[n]);
    resize(image, image, Size(1920, 832));
    imagesList.push_back(image);
  }
  t_end2 = system_clock::now();
  auto duration2 = (duration_cast<microseconds>(t_end2 - t_start2)).count();
  cout << "\n" << endl;
  cout << "[READ  Time ] " << duration2 << "us" << endl;
  cout << "[READ  FPS  ] " << num_of_images * 1000000.0 / duration2 << endl;
  cout << "\n" << endl;

  // MULTITHREADING DPU EXECUTION WITH BATCH
  thread workers[num_threads];

  t_start = system_clock::now();

  for (auto i = 0; i < num_threads; i++) {
    if (i == 0) workers[i] = thread(runCNN, runner.get(), imagesList);
    if (i == 1) workers[i] = thread(runCNN, runner1.get(), imagesList);
    if (i == 2) workers[i] = thread(runCNN, runner2.get(), imagesList);
    if (i == 3) workers[i] = thread(runCNN, runner3.get(), imagesList);
    if (i == 4) workers[i] = thread(runCNN, runner4.get(), imagesList);
    if (i == 5) workers[i] = thread(runCNN, runner5.get(), imagesList);
  }
  // Release thread resources.
  for (auto &w : workers) {
    if (w.joinable()) w.join();
  }

  t_end = system_clock::now();
  auto duration = (duration_cast<microseconds>(t_end - t_start)).count();
  cout << "\n" << endl;
  cout << "[e2e      Time ] " << duration / g_run_nums << "us" << endl;
  cout << "[e2e      FPS  ] " << g_run_nums * 1000000.0 / duration << endl;
  cout << "\n" << endl;

  // delete[] softmax;
  cout << "deleting imagesList  memory" << endl;
  imagesList.clear();

  return 0;
}
