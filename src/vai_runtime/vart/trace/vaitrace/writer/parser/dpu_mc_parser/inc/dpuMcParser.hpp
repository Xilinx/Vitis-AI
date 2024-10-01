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

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <functional>
#include <vector>

struct inst_head {
  uint32_t resverd : 20, dpby : 4, dpdon : 4, opcode : 4;
};

enum inst_type {
  LOAD,
  SAVE,
  CONV,
  CONVINIT,
  CONVADDR,
  DPTWISE,
  DWINIT,
  POOLINIT,
  POOL,
  ELEWINIT,
  ELEW,
  ALUINIT,
  ALUADDR,
  ALU,
  END,
  DUMPBANK,
  DUMPDDR,
  DUMPDDRSLICE,

  INST_MAX
};

struct process_inst_result {
  uint32_t load_img_size;
  uint32_t load_para_size;
  uint32_t save_size;
  uint32_t inst_counter[INST_MAX];
};

#define create_inst_desc(inst, op, lw) inst_desc(#inst, inst, op, lw)
class inst_desc {
 public:
  inst_desc(const char* _name, enum inst_type _it, uint8_t _opcode,
            uint32_t _length_w) {
    name = _name;
    type = _it;
    opcode = _opcode;
    length_w = _length_w;
    length_byte = _length_w * 4;
  };

  const char* name;
  uint8_t opcode;
  enum inst_type type;
  uint32_t length_w;
  uint32_t length_byte;
};

using inst_proc_f = void(enum inst_type, const uint8_t*, bool, uint32_t*,
                         uint32_t*, uint32_t*);

inline void process_common(const uint8_t* mc, uint32_t mc_len, bool debug,
                           uint32_t* load_img_size, uint32_t* load_para_size,
                           uint32_t* save_size,
                           std::vector<class inst_desc>& inst_table,
                           std::function<inst_proc_f> process_inst) {
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

  struct inst_head* h;
  bool inst_matched;
  process_inst_result res = {0};

  for (uint32_t pos = 0; pos < mc_len;) {
    h = (struct inst_head*)(mc + pos);
    inst_matched = false;

    for (const auto& inst : inst_table) {
      if (h->opcode == inst.opcode) {
        process_inst(inst.type, mc + pos, debug, load_img_size, load_para_size,
                     save_size);
        res.inst_counter[inst.type] += 1;
        pos += inst.length_byte;
        inst_matched = true;
        continue;
      }
    }

    if (inst_matched == false) {
      printf("Error: Invalid opcode %d\n", h->opcode);
      return;
    }
  }
};
