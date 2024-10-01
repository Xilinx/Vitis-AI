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
  uint32_t bank_addr : 12, redvered_1 : 2, bank_id : 6, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t jump_write : 10, jump_read : 16, redvered_2 : 6;
  uint32_t length : 10, channel : 14, redvered_3 : 3, pad_idx : 5;
  uint32_t reg_id : 3, redvered_4 : 5, const_value : 8, broadcast : 4,
      mode_avg : 2, pad_end : 5, pad_start : 5;
  uint32_t ddr_addr : 29, redvered_5 : 3;
};

struct Save {
  uint32_t bank_addr : 12, redvered_1 : 2, bank_id : 6, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t jump_read : 10, jump_write : 16, redvered_2 : 3, hp_id : 2,
      redvered_3 : 1;
  uint32_t length : 10, channel : 12, redvered_4 : 10;
  uint32_t reg_id : 3, redvered_5 : 5, const_value : 8, const_en : 1,
      redvered_6 : 15;
  uint32_t ddr_addr : 29, redvered_7 : 3;
};

struct ConvInit {
  uint32_t jump_read : 10, shift_cut : 6, act_type : 4, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t jump_read_endl : 10, shift_bias : 6, redvered_1 : 12, stride_out : 4;
  uint32_t jump_write : 10, stride_offset_in : 3, valid_pixel_parallel : 3,
      redvered_2 : 8, kernel_h : 4, stride_h : 4;
  uint32_t jump_write_endl : 10, stride_offset_out : 3, redvered_3 : 11,
      kernel_w : 4, stride_w : 4;
};

struct Conv {
  uint32_t bank_addr_in : 12, pad_top : 4, pad_left : 4, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t bank_addr_out : 12, pad_bottom : 4, pad_right : 4, redvered_1 : 4,
      channel_group : 8;
  uint32_t bank_addr_weights : 12, bank_id_in : 6, redvered_2 : 4, length : 10;
  uint32_t bank_addr_bias : 12, bank_id_out : 6, bank_addr_in_1 : 12,
      redvered_3 : 2;
  uint32_t bank_addr_in_2 : 12, bank_addr_in_3 : 12, channel_offset : 5,
      redvered_4 : 3;
};

struct PoolInit {
  uint32_t jump_read : 10, kernel_h : 3, kernel_w : 3, shift_cut : 4, dpdby : 4,
      dpdon : 4, opcode : 4;
  uint32_t pool_type : 2, stride_h : 3, stride_w : 3, stride_offset_in : 3,
      valid_pixel_parallel : 3, stride_offset_out : 3, stride_out : 4,
      jump_write : 10, redvered_1 : 1;
};

struct Pool {
  uint32_t bank_addr_in : 12, channel_group : 8, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t bank_id_in : 6, pad_bottom : 3, pad_right : 3, pad_top : 3,
      pad_left : 3, jump_write_endl : 12, redvered_1 : 2;
  uint32_t bank_id_out : 6, length : 10, jump_read_endl : 12, redvered_2 : 4;
  uint32_t bank_addr_out : 12, bank_addr_in_1 : 12, downsample_kernel_w : 8;
  uint32_t bank_addr_in_2 : 12, bank_addr_in_3 : 12, downsample_kernel_h : 8;
};

struct AluInit {
  uint32_t kernel_w : 8, kernel_h : 8, exec_mode : 4, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t stride_offset_out : 8, stride_offset_in : 8, stride_w : 8,
      stride_h : 8;
  uint32_t length : 11, channel_offset : 5, channel_group : 8, stride_out : 8;
  uint32_t shift_prelu_n : 6, shift_cut : 6, shift_bias : 6, act_type : 4,
      shift_prelu_p : 6, redvered_1 : 4;
  uint32_t jump_read_endl : 13, jump_read : 10, multi_factor : 8,
      redvered_2 : 1;
  uint32_t jump_write_endl : 13, jump_write : 10, jump_read_weights : 9;
};

struct Alu {
  uint32_t bank_addr_out : 14, bank_id_out : 6, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t bank_addr_in : 14, bank_id_in : 6, valid_pixel_parallel : 3,
      pad_bottom : 8, redvered_1 : 1;
  uint32_t bank_addr_in_1 : 14, pad_right : 8, pad_left : 4, pad_top : 4,
      redvered_2 : 2;
  uint32_t bank_addr_in_2 : 14, bank_addr_weights : 14, kernel_d : 4;
  uint32_t bank_addr_in_3 : 14, bank_addr_bias : 14, redvered_3 : 4;
};

struct DWInit {
  uint32_t jump_read_endl : 10, jump_read : 10, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t jump_write_endl : 10, kernel_h : 4, kernel_w : 4, stride_h : 4,
      stride_w : 4, valid_pixel_parallel : 3, stride_offset_in : 3;
  uint32_t shift_cut : 6, shift_bias : 6, stride_offset_out : 3, stride_out : 4,
      jump_write : 10, redvered_1 : 3;
};

struct DptWise {
  uint32_t bank_addr_in : 12, channel_group : 8, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t bank_addr_out : 12, redvered_1 : 4, pad_bottom : 4, pad_top : 4,
      pad_right : 4, pad_left : 4;
  uint32_t bank_addr_weights : 12, bank_id_in : 6, length : 10, act_type : 4;
  uint32_t bank_addr_bias : 12, bank_id_out : 6, bank_addr_in_1 : 12,
      redvered_2 : 2;
  uint32_t bank_addr_in_2 : 12, bank_addr_in_3 : 12, channel_offset : 5,
      redvered_3 : 3;
};

struct ElewInit {
  uint32_t redvered_1 : 13, share_pp : 1, bank_id_in : 6, dpdby : 4, dpdon : 4,
      opcode : 4;
  uint32_t bank_addr_in : 14, jump_read_endl : 13, id : 2, jump_bank : 3;
  uint32_t jump_read : 12, shift_read : 4, redvered_2 : 16;
};

struct Elew {
  uint32_t length : 11, act_type : 1, elew_type : 2, bank_id_out : 6, dpdby : 4,
      dpdon : 4, opcode : 4;
  uint32_t bank_addr_out : 14, jump_write_endl : 13, num : 2,
      valid_pixel_parallel : 3;
  uint32_t jump_write : 12, shift_write : 5, channel_group : 8, redvered_1 : 7;
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
    create_inst_desc(LOAD, 0x00, 5),
    create_inst_desc(SAVE, 0x04, 5),
    create_inst_desc(CONVINIT, 0x09, 4),
    create_inst_desc(CONV, 0x08, 5),
    create_inst_desc(POOLINIT, 0x06, 2),
    create_inst_desc(POOL, 0x0C, 5),
    create_inst_desc(ALUINIT, 0x01, 6),
    create_inst_desc(ALU, 0x02, 5),
    create_inst_desc(DWINIT, 0x0B, 3),
    create_inst_desc(DPTWISE, 0x0A, 5),
    create_inst_desc(ELEWINIT, 0x0D, 3),
    create_inst_desc(ELEW, 0x0E, 3),
    create_inst_desc(END, 0x07, 1),
    create_inst_desc(DUMPBANK, 0xFF, 2),
    create_inst_desc(DUMPDDR, 0xFE, 4),
    create_inst_desc(DUMPDDRSLICE, 0xFD, 6)};
