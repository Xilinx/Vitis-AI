/**
 * Copyright 2021 Xilinx Inc.
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

#include "dpuMcParser.hpp"

struct Load
{
  uint32_t bank_addr : 14, bank_id : 6, dpby : 4, dpdon : 4, opcode : 4;
  uint32_t jump_read : 16, pad_idx : 5, pad_end : 5, pad_start : 5, r0 : 1;
  uint32_t channel : 12, mode_avg : 2, length : 10, jump_write : 8;
  uint32_t ddr_addr : 29, reg_id : 3;
  uint32_t jump_write_endl : 14, block_num : 10, r1 : 8;
};

struct Save
{
  uint32_t bank_addr : 14, bank_id : 6, dpby : 4, dpdon : 4, opcode : 4;
  uint32_t jump_write : 16, r0 : 16;
  uint32_t channel : 12, r1 : 2, length : 10, jump_read : 8;
  uint32_t ddr_addr : 29, reg_id : 3;
};

std::vector<class inst_desc> inst_table = {
    create_inst_desc (LOAD, 0b0000, 5),
    create_inst_desc (SAVE, 0b0100, 4),
    create_inst_desc (CONV, 0b1000, 5),
    create_inst_desc (CONVINIT, 0b1001, 6),
    create_inst_desc (DPTWISE, 0b1010, 5),
    create_inst_desc (DWINIT, 0b1011, 4),
    create_inst_desc (POOLINIT, 0b0110, 2),
    create_inst_desc (POOL, 0b1100, 5),
    create_inst_desc (ELEWINIT, 0b1101, 2),
    create_inst_desc (ELEW, 0b1110, 3),
    create_inst_desc (ALUINIT, 0b0001, 6),
    create_inst_desc (ALU, 0b0010, 5),
    create_inst_desc (END, 0b0111, 1)
};

inline void
process_inst (enum inst_type type, const uint8_t *inst, bool debug, uint32_t *load_img_size,
              uint32_t *load_para_size, uint32_t *save_size)
{
  switch (type)
    {
    case LOAD:
      {
        struct Load *l = (struct Load *)inst;
        auto length = l->length + 1;
        auto chan = l->channel + 1;
        auto block_num = l->block_num + 1;
        auto bank_id = l->bank_id;
        auto jump_read = l->jump_read + 1;

        if (bank_id < 16)
          {
            auto l_size = 0u;
            if (jump_read >= chan)
              l_size = length * chan;
            else
              l_size = jump_read * length + chan;
            *load_img_size += l_size;
          }
        else
          {
            *load_para_size += length * chan * block_num;
          }
        // printf("LOAD,%d,%d", length, chan);
        break;
      }
    case SAVE:
      {
        struct Save *s = (struct Save *)inst;

        auto length = s->length + 1;
        auto chan = s->channel + 1;
        *save_size += (length * chan);
        // printf("SAVE %d %d", length, chan);
        break;
      }
    default:
      break;
    }
}

extern "C" void
process (const uint8_t *mc, uint32_t mc_len, bool debug, uint32_t *load_img_size,
         uint32_t *load_para_size, uint32_t *save_size)
{
  process_common (mc, mc_len, debug, load_img_size, load_para_size, save_size, inst_table,
                  process_inst);
};
