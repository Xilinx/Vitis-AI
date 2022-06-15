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
  uint32_t bank_addr : 12, r0 : 2, bank_id : 6, dpby : 4, dpdon : 4, opcode : 4;
  uint32_t jump_write : 10, jump_read : 16, r1 : 6;
  uint32_t length : 10, channel : 14, r2 : 3, pad_idx : 5;
  uint32_t reg_id : 3, r3 : 5, const_value : 8, broadcast : 4, mode_avg : 2, pad_end : 5, pad_start : 5;
  uint32_t ddr_addr : 29, r4 : 3;
};

struct Save
{
  uint32_t bank_addr : 12, r0 : 2, bank_id : 6, dpby : 4, dpdon : 4, opcode : 4;
  uint32_t jump_read : 10, jump_write : 16, r2 : 3, hp_id : 2, r1 : 1;
  uint32_t length : 10, channel : 12, r3 : 10;
  uint32_t reg_id : 3, r5 : 5, const_value : 8, const_en : 1, r4 : 15;
  uint32_t ddr_addr : 29, r6 : 3;
};

std::vector<class inst_desc> inst_table = {
    create_inst_desc (LOAD, 0b0000, 5),
    create_inst_desc (SAVE, 0b0100, 5),
    create_inst_desc (CONV, 0b1000, 5),
    create_inst_desc (CONVINIT, 0b1001, 4),
    create_inst_desc (DPTWISE, 0b1010, 5),
    create_inst_desc (DWINIT, 0b1011, 3),
    create_inst_desc (POOLINIT, 0b0110, 2),
    create_inst_desc (POOL, 0b1100, 5),
    create_inst_desc (ELEWINIT, 0b1101, 3),
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
        auto bank_id = l->bank_id;

        if (bank_id < 16)
          *load_img_size += length * chan;
        else
          *load_para_size += length * chan;
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
  //	printf("debug: %d\n", debug);
  //	printf("mc@%p\n", mc);
  //	printf("mc_len: %d\n", mc_len);
  //
  //	printf("load_img_size@%p\n", load_img_size);
  //	printf("load_para_size@%p\n", load_para_size);
  //	printf("save_size@%p\n", save_size);

  *load_img_size = 0;
  *load_para_size = 0;
  *save_size = 0;

  struct inst_head *h;
  bool inst_matched = false;

  for (uint32_t pos = 0; pos < mc_len;)
    {
      h = (struct inst_head *)(mc + pos);
      for (auto &inst : inst_table)
        {
          if (h->opcode == inst.opcode)
            {
              process_inst (inst.type, mc + pos, debug, load_img_size, load_para_size, save_size);
              pos += inst.length_byte;
              inst_matched = true;
              continue;
            }
        }

      if (inst_matched == false)
        printf ("error\n");
    }
};
