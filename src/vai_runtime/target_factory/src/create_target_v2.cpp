/*
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <inttypes.h>

#include <array>

#include "vitis/ai/target_factory.hpp"

namespace vitis {
namespace ai {

const Target create_target_v2(const std::uint64_t fingerprint) {
  Target target;
  target.set_name("DPUCZDX8G_CUSTOMIZED");
  target.set_type("DPUCZDX8G");
  target.set_isa_version((fingerprint & 0x00ff000000000000) >> 48);
  target.set_feature_code(fingerprint & 0x0000ffffffffffff);

  constexpr std::array<uint64_t, 11> PP_MAP{4, 4, 8, 4, 8, 8, 8, 8, 4, 2, 2};
  constexpr std::array<uint64_t, 11> ICP_MAP{8,  10, 8, 12, 10, 12,
                                             14, 16, 4, 6,  8};
  constexpr std::array<uint64_t, 11> OCP_MAP{8,  10, 8, 12, 10, 12,
                                             14, 16, 4, 6,  8};

  auto ARCH = fingerprint & 0xf;
  auto PP = PP_MAP[ARCH];
  auto ICP = ICP_MAP[ARCH];
  auto OCP = OCP_MAP[ARCH];
  auto PEAK = PP * ICP * OCP * 2;

  auto PL_PP = (fingerprint & 0xf00) >> 8;
  auto DW_PP = (fingerprint & 0xf000) >> 12;
  auto EL_PP = (fingerprint & 0xf0000) >> 16;
  auto IMG_BG = (fingerprint & 0x3000000) >> 24;
  auto LD_AUGM = (fingerprint & 0x4000000) >> 26;
  // auto LD_MEAN = (fingerprint & 0x8000000) >> 27;
  auto CV_LKRL = (fingerprint & 0x10000000) >> 28;
  auto CV_RL6 = (fingerprint & 0x20000000) >> 29;
  auto PL_AVG = (fingerprint & 0x40000000) >> 30;
  auto DW_RL6 = (fingerprint & 0x80000000) >> 31;
  auto EL_MULT = (fingerprint & 0x1000000000) >> 36;
  auto PL_DSM = (fingerprint & 0x2000000000) >> 37;
  auto ISA = (fingerprint & 0x00ff000000000000) >> 48;

  std::string NAME = "DPUCZDX8G_ISAx_Bx_x";
  char finger_hex[17];
  sprintf(finger_hex, "%016" PRIX64, fingerprint);
  NAME.replace(18, 1, finger_hex);
  NAME.replace(16, 1, std::to_string(PEAK));
  NAME.replace(13, 1, std::to_string(ISA));
  target.set_name(NAME);

  auto base_id = 0U;
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    auto img_bank_group = target.add_bank_group();
    img_bank_group->set_name("VB" + std::to_string(idx));
    img_bank_group->set_type("Virtual");
    img_bank_group->set_base_id(base_id);
    img_bank_group->set_bank_num(PP);
    img_bank_group->set_bank_width(ICP);
    img_bank_group->set_bank_depth(2048);
    base_id += PP;
  }
  auto convw_bank_group = target.add_bank_group();
  convw_bank_group->set_name("CONVW");
  convw_bank_group->set_type("Param");
  convw_bank_group->set_base_id(base_id);
  convw_bank_group->set_bank_num(OCP);
  convw_bank_group->set_bank_width(ICP);
  convw_bank_group->set_bank_depth(2048);
  base_id += OCP;
  if (DW_PP) {
    auto dwconvw_bank_group = target.add_bank_group();
    dwconvw_bank_group->set_name("DWCONVW");
    dwconvw_bank_group->set_type("Param");
    dwconvw_bank_group->set_base_id(base_id);
    dwconvw_bank_group->set_bank_num(1);
    dwconvw_bank_group->set_bank_width(ICP);
    dwconvw_bank_group->set_bank_depth(2048);
    base_id += 1;
  }
  auto bias_bank_group = target.add_bank_group();
  bias_bank_group->set_name("BIAS");
  bias_bank_group->set_type("Param");
  bias_bank_group->set_base_id(base_id);
  bias_bank_group->set_bank_num(1);
  bias_bank_group->set_bank_width(ICP);
  bias_bank_group->set_bank_depth(2048);
  base_id += 1;

  auto load_engine = target.mutable_load_engine();
  load_engine->set_channel_parallel(ICP);
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    load_engine->add_output_bank("VB" + std::to_string(idx));
  }

  auto save_engine = target.mutable_save_engine();
  save_engine->set_channel_parallel(ICP);
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    save_engine->add_input_bank("VB" + std::to_string(idx));
  }

  auto conv_engine = target.mutable_conv_engine();
  conv_engine->set_input_channel_parallel(ICP);
  conv_engine->set_output_channel_parallel(OCP);
  conv_engine->set_pixel_parallel(PP);
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    conv_engine->add_input_bank("VB" + std::to_string(idx));
    conv_engine->add_output_bank("VB" + std::to_string(idx));
  }
  conv_engine->set_weight_bank("CONVW");
  conv_engine->set_bias_bank("BIAS");
  if (LD_AUGM) {
    conv_engine->mutable_channel_augmentation()->set_channel_num(ICP * 2);
  }
  auto conv_nonlinear = conv_engine->mutable_nonlinear();
  conv_nonlinear->add_nonlinear_type(Target::Nonlinear::relu);
  if (CV_LKRL) {
    conv_nonlinear->add_nonlinear_type(Target::Nonlinear::leaky_relu);
  }
  if (CV_RL6) {
    conv_nonlinear->add_nonlinear_type(Target::Nonlinear::relu_six);
  }
  auto conv_limit = conv_engine->mutable_conv_limit();
  conv_limit->set_kernel_size("1-16");
  conv_limit->set_stride("1-8");

  auto eltwise_engine = target.mutable_eltwise_engine();
  eltwise_engine->set_channel_parallel(ICP);
  eltwise_engine->set_pixel_parallel(EL_PP);
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    eltwise_engine->add_input_bank("VB" + std::to_string(idx));
    eltwise_engine->add_output_bank("VB" + std::to_string(idx));
  }
  auto eltwise_nonlinear = eltwise_engine->mutable_nonlinear();
  eltwise_nonlinear->add_nonlinear_type(Target::Nonlinear::relu);
  eltwise_engine->add_elew_type(Target::Eltwise::add);
  if (EL_MULT) {
    eltwise_engine->add_elew_type(Target::Eltwise::mult);
  }

  auto pool_engine = target.mutable_pool_engine();
  pool_engine->set_channel_parallel(ICP);
  pool_engine->set_pixel_parallel(PL_PP);
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    pool_engine->add_input_bank("VB" + std::to_string(idx));
    pool_engine->add_output_bank("VB" + std::to_string(idx));
  }
  auto pool_nonlinear = pool_engine->mutable_nonlinear();
  pool_nonlinear->add_nonlinear_type(Target::Nonlinear::relu);
  pool_engine->add_pool_type(Target::Pool::max);
  auto max_limit = pool_engine->mutable_max_limit();
  max_limit->set_kernel_size("1-8");
  max_limit->set_stride("1-8");
  if (PL_AVG) {
    pool_engine->add_pool_type(Target::Pool::avg);
    auto avg_limit = pool_engine->mutable_avg_limit();
    avg_limit->set_kernel_size("2-8");
    avg_limit->set_stride("1-8");
  }
  if (PL_DSM) {
    pool_engine->add_pool_type(Target::Pool::max_reduce);
  }

  if (DW_PP) {
    auto dwconv_engine = target.mutable_dwconv_engine();
    dwconv_engine->set_channel_parallel(ICP);
    dwconv_engine->set_pixel_parallel(DW_PP);
    for (auto idx = 0U; idx < IMG_BG; idx++) {
      dwconv_engine->add_input_bank("VB" + std::to_string(idx));
      dwconv_engine->add_output_bank("VB" + std::to_string(idx));
    }
    dwconv_engine->set_weight_bank("DWCONVW");
    dwconv_engine->set_bias_bank("BIAS");
    auto dwconv_nonlinear = dwconv_engine->mutable_nonlinear();
    dwconv_nonlinear->add_nonlinear_type(Target::Nonlinear::relu);
    if (DW_RL6) {
      dwconv_nonlinear->add_nonlinear_type(Target::Nonlinear::relu_six);
    }
    auto dwconv_limit = dwconv_engine->mutable_dwconv_limit();
    dwconv_limit->set_kernel_size("1-16");
    dwconv_limit->set_stride("1-8");
  }

  return target;
}

}  // namespace ai
}  // namespace vitis
