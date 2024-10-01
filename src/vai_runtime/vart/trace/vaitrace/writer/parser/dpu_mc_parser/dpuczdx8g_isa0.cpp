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
#include "inc/dpuczdx8g_isa0.h"

inline void process_inst(enum inst_type type, const uint8_t* inst, bool debug,
                         uint32_t* load_img_size, uint32_t* load_para_size,
                         uint32_t* save_size) {
  switch (type) {
    case LOAD: {
      struct Load* l = (struct Load*)inst;
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
    case SAVE: {
      struct Save* s = (struct Save*)inst;

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

extern "C" void process(const uint8_t* mc, uint32_t mc_len, bool debug,
                        uint32_t* load_img_size, uint32_t* load_para_size,
                        uint32_t* save_size) {
  process_common(mc, mc_len, debug, load_img_size, load_para_size, save_size,
                 inst_table, process_inst);
};
