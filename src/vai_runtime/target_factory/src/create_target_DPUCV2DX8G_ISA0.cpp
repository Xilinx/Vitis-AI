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

const Target create_target_DPUCV2DX8G_ISA0(const std::uint64_t fingerprint) {
  Target target;
  target.set_type("DPUCV2DX8G");
  target.set_isa_version((fingerprint & 0x00ff000000000000) >> 48);
  target.set_feature_code(fingerprint & 0x0000ffffffffffff);

  constexpr std::array<uint64_t, 5>  PP_MAP{ 1,  1,  1,  2,  4};
  constexpr std::array<uint64_t, 5> ICP_MAP{16, 16, 32, 32, 32};
  constexpr std::array<uint64_t, 5> OCP_MAP{16, 32, 32, 32, 32};

  auto CV_ARCH = fingerprint & 0xf;
  auto ALU_CORE_N = (fingerprint & 0xf0) >> 4;
  auto AIE_CORE_N = (1 << CV_ARCH)+ALU_CORE_N;
  auto PP = PP_MAP[CV_ARCH];
  auto ICP = ICP_MAP[CV_ARCH];
  auto OCP = OCP_MAP[CV_ARCH];
  auto ALU_PP = ALU_CORE_N/2;
  auto IMG_BG = 1U;
  auto IMG_BANK_NUM = 1;
  auto WGT_BANK_NUM = 4;
  auto IMG_RD_DEPTH = 65528;
  auto WGT_RD = 15;

  auto BATCH = (fingerprint & 0xff00) >> 8;
  auto LD_AUGM = (fingerprint & 0x4000000) >> 26;
  // auto LD_MEAN = (fingerprint & 0x8000000) >> 27;
  auto SV_AM = (fingerprint & 0x40000000) >> 30;
  auto ISA = (fingerprint & 0x00ff000000000000) >> 48;

  std::string NAME = "DPUCV2DX8G_ISAx_CxBx_x";
  char finger_hex[17];
  sprintf(finger_hex, "%016" PRIX64, fingerprint);
  NAME.replace(21, 1, finger_hex);
  NAME.replace(19, 1, std::to_string(BATCH));
  NAME.replace(17, 1, std::to_string(AIE_CORE_N));
  NAME.replace(14, 1, std::to_string(ISA));
  target.set_name(NAME);

  if(CV_ARCH >= PP_MAP.size()) {
    UNI_LOG_FATAL(TARGET_FACTORY_INVALID_ARCH)
        << "Unregistered fingerprint=0x" << std::hex
        << std::setfill('0') << std::setw(16) << fingerprint
        << " with invalid ARCH=0x" << std::hex << CV_ARCH
        << ", please check the HW configuration, or contact us.";
  }

  auto base_id = 0U;
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    auto img_bank_group = target.add_bank_group();
    img_bank_group->set_name("VB" + std::to_string(idx));
    img_bank_group->set_type("Virtual");
    img_bank_group->set_base_id(base_id);
    img_bank_group->set_bank_num(IMG_BANK_NUM);
    img_bank_group->set_bank_width(ICP);
    img_bank_group->set_bank_depth(IMG_RD_DEPTH);
    base_id += IMG_BANK_NUM;
  }
  auto convw_bank_group = target.add_bank_group();
  convw_bank_group->set_name("CONVW");
  convw_bank_group->set_type("Param");
  convw_bank_group->set_base_id(base_id);
  convw_bank_group->set_bank_num(WGT_BANK_NUM);
  convw_bank_group->set_bank_width(ICP/2);
  convw_bank_group->set_bank_depth(1 << WGT_RD);
  base_id += WGT_BANK_NUM;

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
  conv_engine->set_bias_bank("CONVW");
  if (LD_AUGM) {
    conv_engine->mutable_channel_augmentation()->set_channel_num(ICP * 2);
  }
  auto conv_nonlinear = conv_engine->mutable_nonlinear();
  conv_nonlinear->add_nonlinear_type(Target::Nonlinear::relu);
  conv_nonlinear->add_nonlinear_type(Target::Nonlinear::leaky_relu);
  conv_nonlinear->add_nonlinear_type(Target::Nonlinear::hsigmoid);
  conv_nonlinear->add_nonlinear_type(Target::Nonlinear::hswish);
  conv_nonlinear->add_nonlinear_type(Target::Nonlinear::relu_six);
  auto conv_limit = conv_engine->mutable_conv_limit();
  conv_limit->set_kernel_size("1-16");
  conv_limit->set_stride("1-16");
  //conv_limit->set_stride_out_h("1-8");

  auto alu_engine = target.mutable_alu_engine();
  alu_engine->set_channel_parallel(ICP);
  alu_engine->set_pixel_parallel(ALU_PP);
  for (auto idx = 0U; idx < IMG_BG; idx++) {
    alu_engine->add_input_bank("VB" + std::to_string(idx));
    alu_engine->add_output_bank("VB" + std::to_string(idx));
  }
  alu_engine->set_weight_bank("CONVW");
  alu_engine->set_bias_bank("CONVW");
  auto alu_nonlinear = alu_engine->mutable_nonlinear();
  alu_nonlinear->add_nonlinear_type(Target::Nonlinear::relu);
  alu_nonlinear->add_nonlinear_type(Target::Nonlinear::leaky_relu);
  alu_nonlinear->add_nonlinear_type(Target::Nonlinear::hsigmoid);
  alu_nonlinear->add_nonlinear_type(Target::Nonlinear::hswish);
  alu_nonlinear->add_nonlinear_type(Target::Nonlinear::relu_six);
  alu_engine->add_alu_type(Target::Alu::macc_temp);
  alu_engine->add_alu_type(Target::Alu::comp_temp);
  alu_engine->add_alu_type(Target::Alu::elew_add);
  alu_engine->add_alu_type(Target::Alu::elew_mult);
  //alu_engine->add_alu_type(Target::Alu::softmax);
  auto alu_limit = alu_engine->mutable_alu_limit();
  alu_limit->set_kernel_size("1-256");
  alu_limit->set_stride("1-256");
  //alu_limit->set_stride_out_h("1-" + std::to_string((ALU_WP==1||PP/ALU_WP>=4)?(4):(PP/ALU_WP)));
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
