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

const Target create_target_DPUCVDX8G_ISA2(const std::uint64_t fingerprint) {
  Target target;
  target.set_type("DPUCVDX8G");
  target.set_isa_version((fingerprint & 0x00ff000000000000) >> 48);
  target.set_feature_code(fingerprint & 0x0000ffffffffffff);

  constexpr std::array<uint64_t, 3> PP_MAP{8, 8, 8};
  constexpr std::array<uint64_t, 3> ICP_MAP{16, 16, 16};
  constexpr std::array<uint64_t, 3> OCP_MAP{16, 32, 64};
  std::array<std::string, 3> ALU_MAP{"NONE", "ALL", "3D"};
  constexpr std::array<uint64_t, 7> IMG_RD_MAP{13, 13, 12, 13, 12, 14, 14};
  constexpr std::array<uint64_t, 7> WGT_RD_MAP{14, 13, 13, 12, 12, 14, 13};
  constexpr std::array<uint64_t, 7> BIAS_RD_MAP{11, 11, 11, 11, 11, 11, 11};

  auto ARCH = fingerprint & 0xf;
  auto PP = PP_MAP[ARCH];
  auto ICP = ICP_MAP[ARCH];
  auto OCP = OCP_MAP[ARCH];
  auto WGT_BANK_NUM = 16;

  auto BATCH = (fingerprint & 0xf0) >> 4;
  auto ALU = (fingerprint & 0xf00) >> 8;
  auto ALU_MODE = ALU_MAP[ALU];
  auto ALU_PP = (fingerprint & 0xf000) >> 12;
  auto EL_PP = (fingerprint & 0xf0000) >> 16;
  auto IMG_BG = (fingerprint & 0x3000000) >> 24;
  auto LD_AUGM = (fingerprint & 0x4000000) >> 26;
  // auto LD_MEAN = (fingerprint & 0x8000000) >> 27;
  auto CV_LKRL = (fingerprint & 0x10000000) >> 28;
  auto CV_RL6 = (fingerprint & 0x20000000) >> 29;
  auto RDEPTH = (fingerprint & 0xf00000000) >> 32;
  auto IMG_RD = IMG_RD_MAP[RDEPTH];
  auto WGT_RD = WGT_RD_MAP[RDEPTH];
  auto BIAS_RD = BIAS_RD_MAP[RDEPTH];
  auto EL_MULT = (fingerprint & 0x1000000000) >> 36;
  auto ISA = (fingerprint & 0x00ff000000000000) >> 48;

  std::string NAME = "DPUCVDX8G_ISAx_CxBx_x";
  char finger_hex[17];
  sprintf(finger_hex, "%016" PRIX64, fingerprint);
  NAME.replace(20, 1, finger_hex);
  NAME.replace(18, 1, std::to_string(BATCH));
  NAME.replace(16, 1, std::to_string(OCP));
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
    img_bank_group->set_bank_depth(1 << IMG_RD);
    base_id += PP;
  }
  auto convw_bank_group = target.add_bank_group();
  convw_bank_group->set_name("CONVW");
  convw_bank_group->set_type("Param");
  convw_bank_group->set_base_id(base_id);
  convw_bank_group->set_bank_num(WGT_BANK_NUM);
  convw_bank_group->set_bank_width(ICP);
  convw_bank_group->set_bank_depth(1 << WGT_RD);
  base_id += WGT_BANK_NUM;
  if (ALU_MODE == "ALL" || ALU_MODE == "3D") {
    auto dwconvw_bank_group = target.add_bank_group();
    dwconvw_bank_group->set_name("DWCONVW");
    dwconvw_bank_group->set_type("Param");
    dwconvw_bank_group->set_base_id(base_id);
    dwconvw_bank_group->set_bank_num(1);
    dwconvw_bank_group->set_bank_width(ICP);
    dwconvw_bank_group->set_bank_depth(1 << WGT_RD);
    base_id += 1;
  }
  auto bias_bank_group = target.add_bank_group();
  bias_bank_group->set_name("BIAS");
  bias_bank_group->set_type("Param");
  bias_bank_group->set_base_id(base_id);
  bias_bank_group->set_bank_num(1);
  bias_bank_group->set_bank_width(WGT_BANK_NUM);
  bias_bank_group->set_bank_depth(1 << BIAS_RD);
  base_id += 1;
  if (ALU_MODE == "ALL" || ALU_MODE == "3D") {
    auto dwbias_bank_group = target.add_bank_group();
    dwbias_bank_group->set_name("DWCVBIAS");
    dwbias_bank_group->set_type("Param");
    dwbias_bank_group->set_base_id(base_id);
    dwbias_bank_group->set_bank_num(1);
    dwbias_bank_group->set_bank_width(ICP);
    dwbias_bank_group->set_bank_depth(1 << BIAS_RD);
    base_id += 1;
  }

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
    conv_nonlinear->add_nonlinear_type(Target::Nonlinear::hsigmoid);
    conv_nonlinear->add_nonlinear_type(Target::Nonlinear::hswish);
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

  auto alu_engine = target.mutable_alu_engine();
  alu_engine->set_channel_parallel(ICP);
  alu_engine->set_pixel_parallel(ALU_PP);
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    alu_engine->add_input_bank("VB" + std::to_string(idx));
    alu_engine->add_output_bank("VB" + std::to_string(idx));
  }
  if (ALU_MODE == "ALL" || ALU_MODE == "3D") {
    alu_engine->set_weight_bank("DWCONVW");
    alu_engine->set_bias_bank("DWCVBIAS");
  }
  auto alu_nonlinear = alu_engine->mutable_nonlinear();
  alu_nonlinear->add_nonlinear_type(Target::Nonlinear::relu);
  if (ALU_MODE == "ALL" || ALU_MODE == "3D") {
    alu_nonlinear->add_nonlinear_type(Target::Nonlinear::relu_six);
  }
  if (ALU_MODE == "ALL" || ALU_MODE == "3D") {
    alu_engine->add_alu_type(Target::Alu::dwconv);
    alu_engine->add_alu_type(Target::Alu::prelu);
    alu_engine->add_alu_type(Target::Alu::avg_pool);
    alu_engine->add_alu_type(Target::Alu::max_pool);
    alu_engine->add_alu_type(Target::Alu::leaky_relu);
    alu_engine->add_alu_type(Target::Alu::max_reduce);
    alu_engine->add_alu_type(Target::Alu::dwconv_no_bias);
    alu_engine->add_alu_type(Target::Alu::hsigmoid);
  }
  if (ALU_MODE == "3D") {
    alu_engine->add_alu_type(Target::Alu::w16b0);
  }
  auto alu_limit = alu_engine->mutable_alu_limit();
  alu_limit->set_kernel_size("1-256");
  if (ALU_MODE == "ALL") {
    alu_limit->set_stride("1-8");
  } else if (ALU_MODE == "3D") {
    alu_limit->set_stride("1-256");
  }
  auto pad_limit = alu_engine->mutable_pad_limit();
  pad_limit->set_pad_left("0-15");
  pad_limit->set_pad_top("0-15");
  pad_limit->set_pad_right("0-255");
  pad_limit->set_pad_bottom("0-255");

  target.set_batch(BATCH);

  return target;
}

}  // namespace ai
}  // namespace vitis
