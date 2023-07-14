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
#include <memory>
#include <thread>
#include <iostream>
#include <mutex>
#include <google/protobuf/text_format.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
#include <xir/graph/graph.hpp>

#include "./pointpillars_imp.hpp"
#include "./postprocess/anchor.hpp"

#include "second/protos/pipeline.pb.h"

using namespace std;
namespace vitis {
namespace ai {

std::mutex mtx_init;
bool binit = false;

::second::protos::TrainEvalPipelineConfig cfg;
std::vector<int> g_grid_size;
G_ANCHOR g_anchor;

void get_cfg(const std::string&);
void get_grid_size();

DEF_ENV_PARAM(ENABLE_PP_DEBUG, "0");

PointPillarsImp::PointPillarsImp(const std::string &model_name, const std::string& model_name1 )
    : m0_(ConfigurableDpuTask::create(model_name, false)),
      m1_(ConfigurableDpuTask::create(model_name1, false)) {
  mtx_init.lock();
  if (binit == false) {
    binit = true;
    std::string model_namex(model_name);

    if (model_name.size() > 7 && model_name.substr( model_name.size()-7, 7) == ".xmodel") {
       size_t pos = 0;
       if ((pos = model_name.rfind("/")) != std::string::npos) {
          model_namex = model_name.substr(pos+1, model_name.size()-7-(pos+1) );
       } else {
          model_namex = model_name.substr(0, model_name.size()-7);
       }
    }

    std::string cfgpath = std::string(m0_->get_graph()->get_attr<std::string>("dirname")) + "/" + model_namex + "_officialcfg.prototxt";
  
    get_cfg( cfgpath);
    get_grid_size();
    __TIC__(create_all_anchors)
    anchor_stride::create_all_anchors();
    __TOC__(create_all_anchors)
  }
  mtx_init.unlock();
  batchnum = get_input_batch();

  std::vector<int8_t*> in_addr1;
  std::vector<int8_t*> out_addr1;
  std::vector<int8_t*> in_addr2;

  for(int i=0; i<batchnum; i++) {
       in_addr1.emplace_back( (int8_t*)m0_->getInputTensor()[0][0].get_data(i));
       out_addr1.emplace_back( (int8_t*)m0_->getOutputTensor()[0][0].get_data(i));
       in_addr2.emplace_back( (int8_t*)m1_->getInputTensor()[0][0].get_data(i));
  }

  pre_ = std::make_unique<PointPillarsPre>( 
        in_addr1,
        tensor_scale(m0_->getInputTensor()[0][0]),
        m0_->getInputTensor()[0][0].width,
        m0_->getInputTensor()[0][0].height,
        m0_->getInputTensor()[0][0].channel,
          out_addr1,
          tensor_scale(m0_->getOutputTensor()[0][0]),
          m0_->getOutputTensor()[0][0].width,
          m0_->getOutputTensor()[0][0].height,
          m0_->getOutputTensor()[0][0].channel,
        in_addr2,
        tensor_scale(m1_->getInputTensor()[0][0]),
        m1_->getInputTensor()[0][0].width,
        m1_->getInputTensor()[0][0].height,
        m1_->getInputTensor()[0][0].channel,
        batchnum, realbatchnum
  );
  post_ = vitis::ai::PointPillarsPost::create(
            m1_->getInputTensor()[0],
            m1_->getOutputTensor()[0],
            &g_grid_size,
            &g_anchor,
            &cfg,
            batchnum,
            realbatchnum
  );
}

PointPillarsImp::~PointPillarsImp() {
}

std::vector<PointPillarsResult> PointPillarsImp::run(const std::vector<const float*>& v_f, const vector<int>& vlen) {
  realbatchnum = std::min(batchnum, int(v_f.size()));

  __TIC__(PP_total)
  __TIC__(PP_pre)
  for(int i=0; i<realbatchnum; i++) {
     pre_->process_net0(v_f[i], vlen[i], i);
  }
  __TOC__(PP_pre)

  __TIC__(anchor_and_dpu)  // about 19ms, so it's shorter than the dpu0+dpu1+pp_middle time.
  thread anchor_mask_t(&PointPillarsPost::get_anchors_mask, post_.get() , std::cref(pre_->pre_dict_));
   //  anchor_mask_t.join();  // move fronter for easy debug

  thread process_net1_cleanmem_t(&PointPillarsPre::process_net1_cleanmem , pre_.get() );

  __TIC__(PP_dpu0)
  m0_->run(0);
  __TOC__(PP_dpu0)

  process_net1_cleanmem_t.join();
  __TIC__(PP_middle)
  for(int i=0;i<realbatchnum;i++) {
     pre_->process_net1(i);
  }
  __TOC__(PP_middle)

  __TIC__(PP_dpu1)
  m1_->run(0);
  __TOC__(PP_dpu1)

   anchor_mask_t.join();
  __TOC__(anchor_and_dpu)

  __TIC__(PP_post)

  std::vector<PointPillarsResult> results;
  for(int i=0;i<realbatchnum;i++) {
    results.emplace_back(post_->post_process( i));
  }
  __TOC__(PP_post)
  __TOC__(PP_total)
  return results;

}

std::vector<PointPillarsResult> PointPillarsImp::run(const V2F& points) {
  realbatchnum = std::min(batchnum, int(points.size()));
  std::vector<const float*> v_f;
  std::vector<int> vlen;
  for (int i=0; i<realbatchnum; i++) {
    v_f.emplace_back( (const float*)points[i].data());
    vlen.emplace_back(int(points[i].size()));
  }
  return run( v_f, vlen);
}

PointPillarsResult PointPillarsImp::run(const V1F& points) {
  realbatchnum = 1;
  return run(points.data(), points.size());
}

PointPillarsResult PointPillarsImp::run( const  float* p_f, int len ) {
  if (ENV_PARAM(ENABLE_PP_DEBUG) == 1) {
    {
      void* addr2 = m0_->getInputTensor()[0][0].get_data(0);   (void)addr2; printf("add-0-in: %p   ", addr2);
      std::vector<vitis::ai::library::InputTensor> inputs = m0_->getInputTensor()[0];
      const auto& layer_data = inputs[0];
      int sWidth = layer_data.width;
      int sHeight= layer_data.height;
      float scale =  tensor_scale(layer_data);
      auto channels = layer_data.channel;
      std::cout <<"net0in: sWidth heiht scale channel:  " << sWidth << "  " << sHeight << "  " << scale << "  " << channels << "\n";  //  100  12000  128  4  
    }
    {
      void* addr2 = m0_->getOutputTensor()[0][0].get_data(0);   (void)addr2;    printf("add-0-out: %p   ", addr2);
      std::vector<vitis::ai::library::OutputTensor> outputs = m0_->getOutputTensor()[0];
      const auto& layer_datao = outputs[0];
      int sWidth = layer_datao.width;
      int sHeight= layer_datao.height;
      float scale =  tensor_scale(layer_datao);
      auto channels = layer_datao.channel;
      std::cout <<"net0out: sWidth heiht scale channel:  " << sWidth << "  " << sHeight << "  " << scale << "  " << channels << "\n";  //   1  12000  0.0625  64
    }
    {
      void* addr2 = m1_->getInputTensor()[0][0].get_data(0);   (void)addr2;  printf("add-1-in: %p   ", addr2);
      std::vector<vitis::ai::library::InputTensor> inputs2 = m1_->getInputTensor()[0];
      const auto& layer_data2 = inputs2[0];
      int sWidth2 = layer_data2.width;
      int sHeight2= layer_data2.height;
      auto scale2 =  tensor_scale(layer_data2);
      auto channel2 = layer_data2.channel;
      std::cout <<"net1in: sWidth heiht scale channel:  " << sWidth2 << "  " << sHeight2 << "  " << scale2 << " " <<  channel2 << "\n";   //  432  496  16
    }
    {
      void* addr2 = m1_->getOutputTensor()[0][0].get_data(0);   (void)addr2;  printf("add-1-out: %p   ", addr2);
      std::vector<vitis::ai::library::OutputTensor> inputs2 = m1_->getOutputTensor()[0];
      const auto& layer_data2 = inputs2[0];
      int sWidth2 = layer_data2.width;
      int sHeight2= layer_data2.height;
      auto scale2 =  tensor_scale(layer_data2);
      auto channel2 = layer_data2.channel;
      std::cout <<"net1out: sWidth heiht scale channel:  " << sWidth2 << "  " << sHeight2 << "  " << scale2 << " " <<  channel2 << "\n";   //  432  496  16
    }
  }

  realbatchnum = 1;
  __TIC__(PP_total)
  __TIC__(PP_pre)
  pre_->process_net0(p_f, len, 0);
  __TOC__(PP_pre)

  __TIC__(anchor_and_dpu)  // about 19ms, so it's shorter than the dpu0+dpu1+pp_middle time.
  thread anchor_mask_t(&PointPillarsPost::get_anchors_mask , post_.get() , std::cref(pre_->pre_dict_));
   //  anchor_mask_t.join();  // move fronter for easy debug

  thread process_net1_cleanmem_t(&PointPillarsPre::process_net1_cleanmem , pre_.get() );

  __TIC__(PP_dpu0)
  m0_->run(0);
  __TOC__(PP_dpu0)

  process_net1_cleanmem_t.join();
  __TIC__(PP_middle)
  pre_->process_net1(0);
  __TOC__(PP_middle)

  __TIC__(PP_dpu1)
  m1_->run(0);
  __TOC__(PP_dpu1)

   anchor_mask_t.join();
  __TOC__(anchor_and_dpu)

  __TIC__(PP_post)
  auto results = post_->post_process(0);
  __TOC__(PP_post)
  __TOC__(PP_total)
  return results;

}

void PointPillarsImp::do_pointpillar_display(PointPillarsResult& res, int flag, DISPLAY_PARAM& g_test,
            cv::Mat& rgb_map, cv::Mat& bev_map, int imgwidth, int imgheight, ANNORET& annoret) {
  return post_->do_pointpillar_display(res, flag, g_test, rgb_map, bev_map ,imgwidth, imgheight, annoret);
}

void get_cfg(const std::string& confPath)
{
  auto text = slurp(confPath.c_str());
  auto ok = google::protobuf::TextFormat::ParseFromString(text, &cfg);
  if (!ok) {
    std::cerr << "parse error for tensorflow offical config file: " << confPath;
    exit(-1);
  }
}

void get_grid_size()
{
  for( int i=0; i<3; i++){
    g_grid_size.emplace_back( int((cfg.model().second().voxel_generator().point_cloud_range()[i+3]
                               - cfg.model().second().voxel_generator().point_cloud_range()[i]  )
                                /cfg.model().second().voxel_generator().voxel_size()[i]));
  }
}

}}

