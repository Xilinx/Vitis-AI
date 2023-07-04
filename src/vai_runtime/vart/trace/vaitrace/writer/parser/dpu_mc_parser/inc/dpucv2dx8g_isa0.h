/**
 * Copyright 2022-2023 Advanced Micro Devices Inc..
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

struct Load {
  uint32_t mt_addr : 17, mt_dst : 3, dpdby : 4, dpdon : 4, opcode : 4;
  uint32_t pad_idx : 5, pad_end : 5, pad_start : 5, broadcast : 4,
      block_num : 10, redvered_1 : 3;
  uint32_t jump_read : 16, const_value : 8, reg_id : 8;
  uint32_t channel : 15, length : 10, redvered_2 : 5, mode_avg : 2;
  uint32_t ddr_addr : 29, redvered_3 : 3;
  uint32_t jump_write_endl : 18, jump_write : 12, redvered_4 : 2;
};

struct Save {
  uint32_t mt_addr : 17, const_en : 1, redvered_1 : 2, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t jump_write : 16, const_value : 8, reg_id : 8;
  uint32_t channel : 12, length : 10, jump_read : 8, redvered_2 : 2;
  uint32_t ddr_addr : 29, redvered_3 : 3;
};

struct ConvInit {
  uint32_t stride_w : 4, stride_h : 4, kernel_w : 4, kernel_h : 4, act_type : 3,
      redvered_1 : 1, dpdby : 4, dpdon : 4, opcode : 4;
  uint32_t tile_owg : 3, tile_ohg : 6, tile_ocg : 6, tile_icg : 6,
      total_in : 11;
  uint32_t hsigmoid_in : 4, shift_hsigmoid : 5, shift_hswish : 4, ow_offset : 6,
      ow_iter : 5, ic_iter : 6, redvered_2 : 2;
  uint32_t jump_read_endl : 15, jump_read : 15, redvered_3 : 2;
  uint32_t jump_write_endl : 15, jump_write : 15, redvered_4 : 2;
  uint32_t jump_read_weights_endl : 11, jump_read_weights : 10, tile_en : 1,
      conv_num : 10;
  uint32_t one_line : 10, one_height : 10, shift_cut : 6, shift_bias : 6;
  uint32_t total_oc_in : 17, total_tile : 9, redvered_5 : 6;
  uint32_t one_size : 20, one_cube : 12;
};

struct ConvAddr {
  uint32_t h_num : 6, direction : 1, last : 1, invalid : 1, redvered_1 : 11,
      dpdby : 4, dpdon : 4, opcode : 4;
  uint32_t mt_word_addr : 17, redvered_2 : 15;
};

struct Conv {
  uint32_t pad_bottom : 4, pad_right : 4, redvered_1 : 4, ocg_offset : 4,
      icg_offset : 4, dpdby : 4, dpdon : 4, opcode : 4;
  uint32_t mt_word_addr_weights : 17, macc_cont : 2, reuse : 1, redvered_2 : 4,
      pad_top : 4, pad_left : 4;
  uint32_t mt_word_addr_bias : 17, redvered_3 : 15;
};

struct AluInit {
  uint32_t kernel_w : 8, kernel_h : 8, exec_mode : 4, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t tile_ohg : 6, tile_owg : 3, share_channel_group : 1,
      share_kernel : 1, num : 2, b_mode : 3, stride_w : 8, stride_h : 8;
  uint32_t hsigmoid_in : 4, shift_hsigmoid : 5, shift_hswish : 4, ow_offset : 6,
      ow_iter : 5, tile_cg : 6, macc_cont : 2;
  uint32_t alu_num : 12, share_pp_0 : 1, share_pp_1 : 1, share_pp_2 : 1,
      share_pp_3 : 1, shift_cut : 7, shift_bias : 6, act_type : 3;
  uint32_t shift_read_0 : 4, shift_read_1 : 4, shift_read_2 : 4,
      shift_read_3 : 4, redvered_1 : 1, weights_lines : 15;
  uint32_t one_line : 10, one_height : 10, total_tile : 9, redvered_2 : 3;
  uint32_t incAO3 : 24, redvered_3 : 8;
  uint32_t incAO2 : 18, redvered_4 : 14;
};

struct AluAddr {
  uint32_t h_num : 6, direction : 1, last : 1, invalid : 1, redvered_1 : 11,
      dpdby : 4, dpdon : 4, opcode : 4;
  uint32_t mt_addr : 17, redvered_2 : 9, macc_dim : 2, redvered_3 : 4;
  uint32_t jump_endl : 15, jump : 15, id : 2;
};

struct Alu {
  uint32_t pad_bottom : 4, pad_right : 4, pad_top : 4, pad_left : 4,
      macc_cont : 2, reuse : 1, redvered_1 : 1, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t mt_addr_weights : 17, redvered_2 : 15;
  uint32_t mt_addr_bias : 17, redvered_3 : 15;
};

struct End {
  uint32_t redvered_1 : 20, dpdby : 4, dpdon : 4, opcode : 4;
};

struct DumpBank {
  uint32_t save_name : 20, dpdby : 4, dpdon : 4, opcode : 4;
  uint32_t redvered_1 : 10, bank_num : 8, bank_start : 8, save_fmt : 6;
};

struct DumpDDR {
  uint32_t save_name : 20, dpdby : 4, dpdon : 4, opcode : 4;
  uint32_t redvered_1 : 20, reg_id : 6, save_fmt : 6;
  uint32_t ddr_start : 32;
  uint32_t ddr_size : 32;
};

struct DumpDDRSlice {
  uint32_t save_name : 20, dpdby : 4, dpdon : 4, opcode : 4;
  uint32_t redvered_1 : 20, reg_id : 6, save_fmt : 6;
  uint32_t ddr_start : 32;
  uint32_t height_stride : 16, height : 16;
  uint32_t width_stride : 16, width : 16;
  uint32_t channel_stride : 16, channel : 16;
};

std::vector<class inst_desc> inst_table = {
    create_inst_desc(LOAD, 0x00, 6),
    create_inst_desc(SAVE, 0x04, 4),
    create_inst_desc(CONVINIT, 0x09, 9),
    create_inst_desc(CONVADDR, 0x05, 2),
    create_inst_desc(CONV, 0x08, 3),
    create_inst_desc(ALUINIT, 0x01, 8),
    create_inst_desc(ALUADDR, 0x03, 3),
    create_inst_desc(ALU, 0x02, 3),
    create_inst_desc(END, 0x07, 1),
    create_inst_desc(DUMPBANK, 0xFF, 2),
    create_inst_desc(DUMPDDR, 0xFE, 4),
    create_inst_desc(DUMPDDRSLICE, 0xFD, 6)};
