

/*
* Copyright 2019 Xilinx Inc.
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


#ifndef BIN_GENERATOR_H
#define BIN_GENERATOR_H

#include <string>
#include <bitset>
#include <stdint.h>
using namespace std;



namespace bin{





  typedef struct{
    uint32_t layer_num;
    uint32_t kernel_num;
    uint32_t layer_i;
    uint32_t init_instr_num;
    uint32_t loop_instr_num;
    uint32_t* load_src_reg0;
    uint32_t* load_src_reg1;
    uint32_t* load_dest_bank0;
    uint32_t* load_dest_bank1;
    uint32_t* load_dest_bank_addr0;
    uint32_t* load_dest_bank_addr1;
    uint32_t* save_dest_reg0;
    uint32_t end_instr_addr;
    uint32_t end_instr_len;
  } regs;
  /*
   *\calculate number of strings (split by space or tab) in one line
   */
  uint32_t cal_strNum(string str);

  /*
   *\give the value for dpd string
   */
  uint8_t dpd_str2val(char *str);
  /*
   *\parse instructions in the text instruction file.
   *\write binary data into ddr_bin
   */
  void parse_instr(uint32_t layer_num, uint32_t kernel_num, uint32_t* line_num, 
                   const char* f_name, const char* f_bin);
  
  void pack_regs(regs info, uint32_t start_addr, char* res);
  uint32_t cal_regs_size(regs info);
  uint32_t cal_end_instr_addr(uint32_t start_addr, uint32_t layer_num, uint32_t kernel_num, uint32_t* instr_line_num);

  bitset<128> parse_load(string line);
  bitset<128> parse_save(string line);
  bitset<128> parse_mmul(string line);
  bitset<128> parse_add(string line);
  bitset<128> parse_actv(string line);
  bitset<128> parse_emul(string line);
  bitset<128> parse_end(string line);
  /*
   *\based on the ddr_init file to generate binary file
   *
   */
  void ddr2bin(const char* f_ddr, const char* f_bin);
  void create_bin(uint32_t layer_num, uint32_t kernel_num, uint32_t* line_num, 
                  const char* f_ddr, const char* f_instr, const char* f_bin);
                  

}//namespace bin








#endif


