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

#include "inc/dpuMcParser.hpp"
#include "inc/dpucv2dx8g_isa1.h"

inline void process_inst(enum inst_type type, const uint8_t* inst, bool debug,
                         uint32_t* load_img_size, uint32_t* load_para_size,
                         uint32_t* save_size) {
  switch (type) {
    case LOAD: {
      struct Load* l = (struct Load*)inst;
      auto mt_dst = l->mt_dst;

      auto length = l->length + 1;
      auto chan = l->channel + 1;
      auto block_num = l->block_num + 1;
      auto jump_read = l->jump_read + 1;

      // constexpr uint8_t bank_width = 16;
      // constexpr uint8_t bank_num = 4;
      // printf("Load_img: jump_read: %d, chan: %d, length: %d, ddr_addr:
      // 0x%08x\n", jump_read, chan, length, ddr_addr); printf("Load_IMG: len
      // %d, chan %d, blkn %d\n", length, chan, block_num); printf("Load_IMG:
      // total_len:%x, ddr_addr:0x%08x\n", length * chan * block_num, ddr_addr);

      switch (mt_dst) {
        case 0:
          /* load_img */
          if (jump_read >= chan) {
            *load_img_size += length * chan;
          } else {
            *load_img_size += length * jump_read + chan;
          }

          break;
        case 1:
          /* load_weights_conv */
          *load_para_size += length * chan * block_num;
          break;
        case 2:
          /* load_bias_conv */
          *load_para_size += length * chan * block_num;
          break;
        case 3:
          /* load_weights_misc */
          *load_para_size += length * chan * block_num;
          break;
        case 4:
          /* load_bias_misc */
          *load_para_size += length * chan * block_num;
          break;
        default:
          break;
      };
      break;
    }
    case SAVE: {
      struct Save* s = (struct Save*)inst;

      auto length = s->length + 1;
      auto chan = s->channel + 1;
      *save_size += (length * chan);
      break;
    }
    default:
      break;
  }
}

extern "C" void process(const uint8_t* mc, uint32_t mc_len, bool debug,
                        uint32_t* load_img_size, uint32_t* load_para_size,
                        uint32_t* save_size) {
  process_common(mc, mc_len, debug, load_img_size, load_para_size, save_size,
                 inst_table, process_inst);
};
