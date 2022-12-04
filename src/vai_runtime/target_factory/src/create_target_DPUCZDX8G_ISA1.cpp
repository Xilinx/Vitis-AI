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
#include <math.h>
#include "vitis/ai/target_factory.hpp"
#include <UniLog/UniLog.hpp>
#include <iomanip>
#include <iostream>

namespace vitis {
namespace ai {

const Target create_target_DPUCZDX8G_ISA1(const std::uint64_t fingerprint) {
  Target target;
  target.set_type("DPUCZDX8G");
  target.set_isa_version((fingerprint & 0x00ff000000000000) >> 48);
  target.set_feature_code(fingerprint & 0x0000ffffffffffff);

  constexpr std::array<uint64_t, 12> PP_MAP{ 4,  4,  8, 4,  8,  8,
                                             8,  8,  4, 2,  2,  2};
  constexpr std::array<uint64_t, 12> ICP_MAP{8,  10, 8, 12, 10, 12,
                                             14, 16, 4, 6,  8,  4};
  constexpr std::array<uint64_t, 12> OCP_MAP{8,  10, 8, 12, 10, 12,
                                             14, 16, 4, 6,  8,  4};

  auto ARCH = fingerprint & 0xf;
  auto PP = PP_MAP[ARCH];
  auto ICP = ICP_MAP[ARCH];
  auto OCP = OCP_MAP[ARCH];
  auto PEAK = PP * ICP * OCP * 2;

  auto ALU_PP = (fingerprint & 0xf00) >> 8;
  auto IMG_BG = (fingerprint & 0x3000000) >> 24;
  auto LD_AUGM = (fingerprint & 0x4000000) >> 26;
  // auto LD_MEAN = (fingerprint & 0x8000000) >> 27;
  auto CV_LKRL = (fingerprint & 0x10000000) >> 28;
  auto DW_LKRL = (fingerprint & 0x20000000) >> 29;
  auto SV_AM = (fingerprint & 0x40000000) >> 30;
  auto CV_WP = (uint64_t)pow(2,(fingerprint & 0x300000000) >> 32);
  auto ALU_WP = (uint64_t)pow(2,(fingerprint & 0xc00000000) >> 34);
  auto ISA = (fingerprint & 0x00ff000000000000) >> 48;

  std::string NAME = "DPUCZDX8G_ISAx_Bx_x";
  char finger_hex[17];
  sprintf(finger_hex, "%016" PRIX64, fingerprint);
  NAME.replace(18, 1, finger_hex);
  NAME.replace(16, 1, std::to_string(PEAK));
  NAME.replace(13, 1, std::to_string(ISA));
  target.set_name(NAME);

  if(ARCH >= PP_MAP.size()) {
    UNI_LOG_FATAL(TARGET_FACTORY_INVALID_ARCH)
        << "Unregistered fingerprint=0x" << std::hex
        << std::setfill('0') << std::setw(16) << fingerprint
        << " with invalid ARCH=0x" << std::hex << ARCH
        << ", please check the HW configuration, or contact us.";
  }

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
  auto dwconvw_bank_group = target.add_bank_group();
  dwconvw_bank_group->set_name("DWCONVW");
  dwconvw_bank_group->set_type("Param");
  dwconvw_bank_group->set_base_id(base_id);
  dwconvw_bank_group->set_bank_num(1);
  dwconvw_bank_group->set_bank_width(ICP);
  dwconvw_bank_group->set_bank_depth(2048);
  base_id += 1;
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
  save_engine->set_argmax(SV_AM==1);

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
  conv_nonlinear->add_nonlinear_type(Target::Nonlinear::relu_six);
  auto conv_limit = conv_engine->mutable_conv_limit();
  conv_limit->set_kernel_size("1-16");
  conv_limit->set_stride("1-8");
  conv_limit->set_stride_out_h("1-" + std::to_string((CV_WP==1||PP/CV_WP>=4)?(4):(PP/CV_WP)));

  auto eltwise_engine = target.mutable_eltwise_engine();
  eltwise_engine->set_channel_parallel(ICP);
  eltwise_engine->set_pixel_parallel(ALU_PP);
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    eltwise_engine->add_input_bank("VB" + std::to_string(idx));
    eltwise_engine->add_output_bank("VB" + std::to_string(idx));
  }
  auto eltwise_nonlinear = eltwise_engine->mutable_nonlinear();
  eltwise_nonlinear->add_nonlinear_type(Target::Nonlinear::relu);
  eltwise_engine->add_elew_type(Target::Eltwise::add);
  eltwise_engine->add_elew_type(Target::Eltwise::mult);

  auto alu_engine = target.mutable_alu_engine();
  alu_engine->set_channel_parallel(ICP);
  alu_engine->set_pixel_parallel(ALU_PP);
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    alu_engine->add_input_bank("VB" + std::to_string(idx));
    alu_engine->add_output_bank("VB" + std::to_string(idx));
  }
  alu_engine->set_weight_bank("DWCONVW");
  alu_engine->set_bias_bank("BIAS");
  auto alu_nonlinear = alu_engine->mutable_nonlinear();
  alu_nonlinear->add_nonlinear_type(Target::Nonlinear::relu);
  if (DW_LKRL) {
    alu_nonlinear->add_nonlinear_type(Target::Nonlinear::leaky_relu);
  }
  alu_nonlinear->add_nonlinear_type(Target::Nonlinear::relu_six);
  alu_engine->add_alu_type(Target::Alu::dwconv);
  alu_engine->add_alu_type(Target::Alu::prelu);
  alu_engine->add_alu_type(Target::Alu::avg_pool);
  alu_engine->add_alu_type(Target::Alu::max_pool);
  alu_engine->add_alu_type(Target::Alu::leaky_relu);
  alu_engine->add_alu_type(Target::Alu::max_reduce);
  alu_engine->add_alu_type(Target::Alu::dwconv_no_bias);
  alu_engine->add_alu_type(Target::Alu::hsigmoid);
  alu_engine->add_alu_type(Target::Alu::w16b0);
  auto alu_limit = alu_engine->mutable_alu_limit();
  alu_limit->set_kernel_size("1-256");
  alu_limit->set_stride("1-256");
  alu_limit->set_stride_out_h("1-" + std::to_string((ALU_WP==1||PP/ALU_WP>=4)?(4):(PP/ALU_WP)));
  auto pad_limit = alu_engine->mutable_pad_limit();
  pad_limit->set_pad_left("0-15");
  pad_limit->set_pad_top("0-15");
  pad_limit->set_pad_right("0-255");
  pad_limit->set_pad_bottom("0-255");

  return target;
}

}  // namespace ai
}  // namespace vitis
