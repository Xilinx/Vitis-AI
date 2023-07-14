/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <thread>
#include <vitis/ai/bevdet.hpp>
#include <vitis/ai/env_config.hpp>

using namespace std;
using namespace cv;

std::string out_fname;
std::string db_name;
vector<string> names;
std::atomic<int> g_count = 0;

void accuracy_thread(vitis::ai::BEVdet* det, int i, int t_n,
                     std::string f_name) {
  std::ofstream out_fs(f_name, std::ofstream::out);
  if (!out_fs) {
    std::cout << "Can't open the file result!";
    abort();
  }

  // pic: _cam0.jpg, ..._cam5.jpg
  // bins: translation rotation points_z  points

  std::vector<std::string> bin_name_ext{"_translation", "_rotation",
                                        "_points_z", "_points"};
  std::vector<cv::Mat> images;
  std::vector<std::vector<char>> bins;

  // if (i == 0) out_fs << "[\n";

  int all_size = (int)names.size();
  int split = all_size / t_n;
  int start = i * split;
  int end = ((i != t_n - 1) ? split : all_size - i * split);
  auto start_time = std::chrono::system_clock::now();
  std::cout << "begin progress " << (int)names.size() << std::endl;
  std::queue<std::pair<int, std::chrono::_V2::system_clock::time_point>>
      time_queue;
  time_queue.push(std::make_pair(int(g_count), start_time));
  for (int ii = start; ii < start + end; ii++) {
    g_count++;
    if (i == 0) {
      auto pr = float(g_count) / float(all_size);
      auto now = std::chrono::system_clock::now();
      time_queue.push(std::make_pair(int(g_count), now));
      int barWidth = 70;
      std::cout << "[";
      int pos = barWidth * pr;
      char sp[4] = {'/', '-', '\\', '-'};
      for (int j = 0; j < barWidth; ++j) {
        if (j < pos)
          std::cout << "=";
        else if (j == pos)
          std::cout << sp[g_count % 4];
        else
          std::cout << " ";
      }
      std::cout << "]";
      std::cout << std::setw(3) << int(pr * 100.0) << "%," << std::setw(5)
                << ((now - start_time).count()) / 1000000000 << "s/"
                << std::setw(5)
                << (all_size - g_count) *
                       (time_queue.back().second - time_queue.front().second)
                           .count() /
                       1000000000 /
                       (time_queue.back().first - time_queue.front().first)
                << "s\r" << std::flush;
      if (time_queue.size() > 100) time_queue.pop();
    }
    images.clear();
    bins.clear();
    std::vector<std::string> bin_names;
    std::string fn_base(db_name);
    fn_base.append("/");
    fn_base.append(names[ii]);
    for (int j = 0; j < 6; j++) {
      std::string pic_name(fn_base);
      pic_name.append("_cam");
      pic_name.append(std::to_string(j));
      pic_name.append(".jpg");
      images.emplace_back(cv::imread(pic_name));
    }
    bins.clear();
    for (int j = 0; j < 4; j++) {
      std::string bin_name(fn_base);
      bin_name.append(bin_name_ext[j]);
      bin_name.append(".bin");
      auto infile = std::ifstream(bin_name, std::ios_base::binary);
      bins.emplace_back(
          std::vector<char>(std::istreambuf_iterator<char>(infile),
                            std::istreambuf_iterator<char>()));
    }
    auto res = det->run(images, bins);

    out_fs << "{\"pts_bbox\": { \"token\":\"" << names[ii] << "\",";
    out_fs << " \"boxes_3d\": [ ";
    for (int j = 0; j < (int)res.size(); j++) {
      out_fs << " [ ";
      for (int k = 0; k < 9; k++) {
        out_fs << res[j].bbox[k] << " ";
        if (k < 9 - 1) out_fs << ",";
      }
      out_fs << " ] ";
      if (j < (int)res.size() - 1) out_fs << ",";
    }
    out_fs << "     ], \"scores_3d\": [ ";
    for (int j = 0; j < (int)res.size(); j++) {
      out_fs << res[j].score << " ";
      if (j < (int)res.size() - 1) out_fs << ",";
    }
    out_fs << " ], \"labels_3d\": [ ";
    for (int j = 0; j < (int)res.size(); j++) {
      out_fs << res[j].label;
      if (j < (int)res.size() - 1) out_fs << ",";
    }
    out_fs << " ] } } ";
    if (ii < (int)names.size() - 1) {
      out_fs << "\n";
    }
  }
  // if (i == t_n - 1) out_fs << "\n]";
  out_fs.close();
  std::cout << std::endl;
}

void LoadListNames(const std::string& filename,
                   std::vector<std::string>& vlist) {
  ifstream Tin;
  Tin.open(filename, ios_base::in);
  std::string str;
  if (!Tin) {
    std::cout << "Can't open the file " << filename << "\n";
    exit(-1);
  }
  while (getline(Tin, str)) {
    vlist.emplace_back(str);
  }
  Tin.close();
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "usage : " << argv[0]
              << "<model_name0> <model_name1> <model_name2> <list_file> "
                 "<dataset_dir> <output_file> "
                 "[thread_num]\n"
              << "         <list_file> hold the item name list \n"
              << "         <dataset_dir> is the dir in which all the pic file "
                 "and bin file is placed\n"
              << "         [thread_num] is multi thread number if use multi "
                 "thread test. default is 1. \n"
              << std::endl;
    abort();
  }

  out_fname = std::string(argv[6]);
  LoadListNames(argv[4], names);
  db_name = std::string(argv[5]);

  int t_n = 1;
  if (argc == 6) {
    t_n = atoi(argv[5]);
  }

  std::vector<std::thread> vth;
  std::vector<std::unique_ptr<vitis::ai::BEVdet>> vdet;
  std::vector<std::string> out_fname_t(t_n);
  for (int i = 0; i < t_n; i++) {
    auto det = vitis::ai::BEVdet::create(argv[1], argv[2], argv[3]);
    if (!det) { // supress coverity complain
      std::cerr <<"create error\n";
      abort();
    }  
    vdet.emplace_back(std::move(det));

    out_fname_t[i] = out_fname + "_" + std::to_string(i);
    // std::cout <<" out_fname :" << out_fname_t[i]  << "\n";
    vth.emplace_back(
        std::thread(&accuracy_thread, vdet[i].get(), i, t_n, out_fname_t[i]));
  }
  for (int i = 0; i < t_n; i++) {
    vth[i].join();
  }

  // combine output files;
  std::string cmd("cat ");
  for (int i = 0; i < t_n; i++) {
    cmd.append(out_fname_t[i]);
    cmd.append(" ");
  }
  cmd.append(" > ");
  cmd.append(out_fname);
  auto ret = system(cmd.c_str());
  cmd = "rm -f ";
  cmd.append(out_fname);
  cmd.append("_*");
  ret = system(cmd.c_str());
  (void)ret;

  return 0;
}
