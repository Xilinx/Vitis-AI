/**
 * Copyright 2022 Xilinx Inc.
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
  uint32_t bank_addr : 12, bank_id : 6, hp_id : 2, dpby : 4, dpdon : 4,
      opcode : 4;
  uint32_t jump_read : 16, pad_idx : 5, pad_end : 5, pad_start : 5, r0 : 1;
  uint32_t channel : 12, mode_avg : 2, length : 10, jump_write : 8;
  uint32_t ddr_addr : 29, reg_id : 3;
};
struct Save {
  uint32_t bank_addr : 12, bank_id : 6, hp_id : 2, dpby : 4, dpdon : 4,
      opcode : 4;
  uint32_t jump_write : 16, r0 : 16;
  uint32_t channel : 12, r1 : 2, length : 10, jump_read : 8;
  uint32_t ddr_addr : 29, reg_id : 3;
};

std::vector<class inst_desc> inst_table = {
    create_inst_desc(LOAD, 0b0000, 4),
    create_inst_desc(SAVE, 0b0100, 4),
    create_inst_desc(CONV, 0b1000, 5),
    create_inst_desc(CONVINIT, 0b1001, 4),
    create_inst_desc(DPTWISE, 0b1010, 5),
    create_inst_desc(DWINIT, 0b1011, 3),
    create_inst_desc(POOLINIT, 0b0110, 2),
    create_inst_desc(POOL, 0b1100, 5),
    create_inst_desc(ELEWINIT, 0b1101, 2),
    create_inst_desc(ELEW, 0b1110, 3),
    create_inst_desc(END, 0b0111, 1)};
