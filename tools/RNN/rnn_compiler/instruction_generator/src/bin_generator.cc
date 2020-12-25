

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


#include <string.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include "bin_generator.h"
using namespace std;




namespace bin{
  
  uint32_t DDR_BW = 16; 
  uint32_t INSTR_BASEADDR = 0x7000000;
  uint32_t VECTOR_BANK_ID[4] = {32, 33, 34, 35};
  uint32_t sent_vec_bank_reg = 2;
  uint32_t oie_vec_bank_reg[8] = {0xe, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14};
  uint32_t io_hbm3[3] = {0x70000000, 0x90000000, 0xb0000000};
  uint32_t io_hbm4[4] = {0x80000000, 0xa0000000, 0xc0000000, 0xe0000000};

  /////////////////////////////////cal_strNum///////////////////////////////
  uint32_t cal_strNum(string str){
  
    uint32_t num = 0;
    stringstream ss(str);

    string s;
    while(ss >> s){
      num++;
    }

    return num;

  }//cal_strNum


  /////////////////////////////dpd_str2val///////////////////////////////////
  uint8_t dpd_str2val(char *str){
    uint8_t dpd_val = 0;

    char *tok = strtok(str, "+");
    while(tok != NULL){
      if(strstr("none load save mmul add actv emul end", tok) == NULL){
        cout << "ERROR: wrong dependon string " << tok << endl;
        return -1;
      }
      
      if(strcmp(tok, "load") == 0){
        dpd_val |= 1;
      }

      if(strcmp(tok, "save") == 0){
        dpd_val |= (1 << 1);
      }

      if(strcmp(tok, "mmul") == 0){
        dpd_val |= (1 << 2);
      }

      if(strcmp(tok, "add") == 0){
        dpd_val |= (1 << 3);
      }

      if(strcmp(tok, "actv") == 0){
        dpd_val |= (1 << 4);
      }

      if(strcmp(tok, "emul") == 0){
        dpd_val |= (1 << 5);
      }

      if(strcmp(tok, "end") == 0){
        dpd_val |= (1 << 6);
      }

      tok = strtok(NULL, "+");
    }//while
    return dpd_val;

  }//dpd_str2val




  /////////////////////////////////parse_load///////////////////////////////
  bitset<128> parse_load(string line){
    char dpdon_str[100];
    char dpdby_str[100];
    uint32_t ddr_id;
    uint32_t bank_id;
    uint32_t bank_addr;
    uint32_t len;
    uint32_t ddr_addr;
    uint32_t type;

    
    
    
    if(cal_strNum(line) != 17){
      printf("ERROR: words length: %d not match in [LOAD]\n", cal_strNum(line));
      return bitset<128>(string(128, '0'));
    }

    sscanf(line.c_str(), "LOAD %*s %s %*s %s %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d\n", 
                         dpdon_str, dpdby_str, &ddr_id, &bank_id, &bank_addr, &len, &ddr_addr, &type);
    
    
    bitset<8> opcode_bit(0x01);
    bitset<8> dpdon_bit(dpd_str2val(dpdon_str));
    bitset<8> dpdby_bit(dpd_str2val(dpdby_str));
    bitset<16> ddr_id_bit(ddr_id);
    bitset<8> bank_id_bit(bank_id);
    bitset<16> bank_addr_bit(bank_addr);
    bitset<16> len_bit(len);
    bitset<32> ddr_addr_bit(ddr_addr);
    bitset<2> type_bit(type);

    string instr_str = opcode_bit.to_string();
    instr_str += dpdon_bit.to_string();
    instr_str += dpdby_bit.to_string();
    instr_str += ddr_id_bit.to_string();
    instr_str += bank_id_bit.to_string();
    instr_str += bank_addr_bit.to_string();
    instr_str += len_bit.to_string();
    instr_str += ddr_addr_bit.to_string();
    instr_str += string(14, '0');
    instr_str += type_bit.to_string();

    return bitset<128>(instr_str);

  }//parse_load



  ///////////////////////////parse_save/////////////////////////////
  bitset<128> parse_save(string line){
    char dpdon_str[100];
    char dpdby_str[100];
    uint32_t ddr_id;
    uint32_t ddr_addr;
    uint32_t len;
    uint32_t bank_id;
    uint32_t bank_addr;
    uint32_t type;

    if(cal_strNum(line) != 17){
      printf("ERROR: words length: %d not match in [SAVE]\n", cal_strNum(line));
      return bitset<128>(string(128, '0'));
    }

    sscanf(line.c_str(), "SAVE %*s %s %*s %s %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d", 
                          dpdon_str, dpdby_str, &ddr_id, &ddr_addr, &len, &bank_id, &bank_addr, &type);

    bitset<8> opcode_bit(0x02);
    bitset<8> dpdon_bit(dpd_str2val(dpdon_str));
    bitset<8> dpdby_bit(dpd_str2val(dpdby_str));

    bitset<16> ddr_id_bit(ddr_id);
    bitset<32> ddr_addr_bit(ddr_addr);
    bitset<16> len_bit(len);
    bitset<8> bank_id_bit(bank_id);
    bitset<16> bank_addr_bit(bank_addr);
    bitset<2> type_bit(type);
        

    string instr_str = opcode_bit.to_string();
    instr_str += dpdon_bit.to_string();
    instr_str += dpdby_bit.to_string();
    instr_str += ddr_id_bit.to_string();
    instr_str += ddr_addr_bit.to_string();
    instr_str += len_bit.to_string();
    instr_str += bank_id_bit.to_string();
    instr_str += bank_addr_bit.to_string();        
    instr_str += string(14, '0');
    instr_str += type_bit.to_string();

    return bitset<128>(instr_str);
 

  }//parse_save



  ///////////////////////////parse_mmul///////////////////////////
  bitset<128> parse_mmul(string line){
    char dpdon_str[100];
    char dpdby_str[100];

    uint32_t vec_len;
    uint32_t vec_id;
    uint32_t vec_addr;
    uint32_t wgt_id;
    uint32_t wgt_addr;
    uint32_t wgt_row;
    uint32_t wgt_size;
    uint32_t bias_addr;
    uint32_t res_id;
    uint32_t res_addr;
    int trunc;
    uint32_t relu;
    uint32_t type;


    if(cal_strNum(line) != 29){
      printf("ERROR: words length: %d not match in [MMUL]\n", cal_strNum(line));
      return bitset<128>(string(128, '0'));
      
    }

    sscanf(line.c_str(), "MMUL %*s %s %*s %s %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d\n", 
        dpdon_str, dpdby_str, &vec_len, &vec_id, &vec_addr, &wgt_id, &wgt_addr, &wgt_row, &wgt_size, &res_id, &res_addr, &trunc, &relu, &type);
  


    bitset<8> opcode_bit(0x04);
    bitset<8> dpdon_bit(dpd_str2val(dpdon_str));
    bitset<8> dpdby_bit(dpd_str2val(dpdby_str));

    bitset<12>  vec_len_bit(vec_len);
    bitset<4>  vec_id_bit(vec_id);
    bitset<12>  vec_addr_bit(vec_addr);
    bitset<4>  wgt_id_bit(wgt_id);
    bitset<12>  wgt_addr_bit(wgt_addr);
    bitset<12>  wgt_row_bit(wgt_row);
    bitset<12>  wgt_size_bit(wgt_size);    
    bitset<4>  res_id_bit(res_id);
    bitset<12>  res_addr_bit(res_addr);
    bitset<8>  trunc_bit(trunc);
    bitset<1>  relu_bit(relu);    
    bitset<2> type_bit(type);

    string instr_str = opcode_bit.to_string();
    instr_str += dpdon_bit.to_string();
    instr_str += dpdby_bit.to_string();
    instr_str += vec_len_bit.to_string();
    instr_str += vec_id_bit.to_string();
    instr_str += vec_addr_bit.to_string();
    instr_str += wgt_id_bit.to_string();
    instr_str += wgt_addr_bit.to_string();
    instr_str += wgt_row_bit.to_string();
    instr_str += wgt_size_bit.to_string();
    instr_str += string(4, '0');    
    instr_str += res_id_bit.to_string();
    instr_str += res_addr_bit.to_string();
    instr_str += trunc_bit.to_string();
    instr_str += string(4, '0');
    instr_str += relu_bit.to_string();    
    instr_str += string(1, '0');
    instr_str += type_bit.to_string();
    

    return bitset<128>(instr_str);

  }//parse_mmul


  

  ////////////////////////////parse_add/////////////////////////////
  bitset<128> parse_add(string line){
    char dpdon_str[100];
    char dpdby_str[100];

    uint32_t add_num;
    int trunc;
    uint32_t paral_num;
    uint32_t sign;
    uint32_t len;
    uint32_t a_id;
    uint32_t a_addr;
    uint32_t b_id;
    uint32_t b_addr;
    uint32_t c_id;
    uint32_t c_addr;
    uint32_t out_type;
    uint32_t relu;
    uint32_t res_id;
    uint32_t res_addr;


    

  
    if(cal_strNum(line) != 35){
      printf("ERROR: words length: %d not match in [ADD]\n", cal_strNum(line));
      return bitset<128>(string(128, '0'));
    }

    sscanf(line.c_str(), "ADD %*s %s %*s %s %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d\n", 
    dpdon_str, dpdby_str, &add_num, &trunc, &paral_num, &sign, &len, &a_id, &a_addr, &b_id, &b_addr, &c_id, &c_addr, &out_type, &relu, &res_id, &res_addr);

    bitset<8> opcode_bit(0x08);
    bitset<8> dpdon_bit(dpd_str2val(dpdon_str));
    bitset<8> dpdby_bit(dpd_str2val(dpdby_str));


    bitset<2> add_num_bit(add_num);
    bitset<6> trunc_bit(trunc);
    bitset<2> paral_num_bit(paral_num);
    bitset<2> sign_bit(sign);
    bitset<12> len_bit(len);
    bitset<6> a_id_bit(a_id);
    bitset<12> a_addr_bit(a_addr);
    bitset<6> b_id_bit(b_id);
    bitset<12> b_addr_bit(b_addr);
    bitset<6> c_id_bit(c_id);
    bitset<12> c_addr_bit(c_addr);
    bitset<1> out_type_bit(out_type);
    bitset<1> relu_bit(relu);
    bitset<6> res_id_bit(res_id);
    bitset<12> res_addr_bit(res_addr);


    string instr_str = opcode_bit.to_string();
    instr_str += dpdon_bit.to_string();
    instr_str += dpdby_bit.to_string();
    instr_str += add_num_bit.to_string();
    instr_str += trunc_bit.to_string();     
    instr_str += paral_num_bit.to_string();
    instr_str += sign_bit.to_string();
    instr_str += len_bit.to_string();
    instr_str += a_id_bit.to_string();
    instr_str += a_addr_bit.to_string();
    instr_str += b_id_bit.to_string();
    instr_str += b_addr_bit.to_string();
    instr_str += c_id_bit.to_string();
    instr_str += c_addr_bit.to_string();
    instr_str += out_type_bit.to_string();    
    instr_str += relu_bit.to_string();
    instr_str += string(6, '0');
    instr_str += res_id_bit.to_string();
    instr_str += res_addr_bit.to_string();


    return bitset<128>(instr_str);



  }//parse_add
  
  

  ///////////////////////////////parse_actv////////////////////////
  bitset<128> parse_actv(string line){
    char dpdon_str[100];
    char dpdby_str[100];

    uint32_t actv_type;
    uint32_t actv_id;
    uint32_t len;
    uint32_t in_id;
    uint32_t in_addr;
    uint32_t res_id;
    uint32_t res_addr;
    int trunc;
    uint32_t init;


    if(cal_strNum(line) != 23){
      printf("ERROR: words length: %d not match in [ACTV]\n", cal_strNum(line));
      return bitset<128>(string(128, '0'));
    }

	  sscanf(line.c_str(), "ACTV %*s %s %*s %s %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d\n", 
                            dpdon_str, dpdby_str, &actv_type, &actv_id, &len, &in_id, &in_addr, &res_id, &res_addr, 
                            &trunc, &init);



    bitset<8> opcode_bit(0x10);
    bitset<8> dpdon_bit(dpd_str2val(dpdon_str));
    bitset<8> dpdby_bit(dpd_str2val(dpdby_str));

    bitset<2> actv_type_bit(actv_type);
    bitset<4> actv_id_bit(actv_id);
    bitset<12> len_bit(len);
    bitset<4> in_id_bit(in_id);
    bitset<12> in_addr_bit(in_addr);
    bitset<4> res_id_bit(res_id);
    bitset<12> res_addr_bit(res_addr);
    bitset<8> trunc_bit(trunc);
    bitset<1> init_bit(init);


    string instr_str = opcode_bit.to_string();
    instr_str += dpdon_bit.to_string();
    instr_str += dpdby_bit.to_string();
    instr_str += actv_type_bit.to_string();
    instr_str += string(2, '0');
    instr_str += actv_id_bit.to_string();
    instr_str += string(4, '0');
    instr_str += len_bit.to_string();
    instr_str += in_id_bit.to_string();
    instr_str += in_addr_bit.to_string();
    instr_str += res_id_bit.to_string();
    instr_str += res_addr_bit.to_string();
    instr_str += trunc_bit.to_string();
    instr_str += init_bit.to_string();
    instr_str += string(39, '0');

    return bitset<128>(instr_str);


  }//parse_actv


  ////////////////////////////parse_emul//////////////////////////
  bitset<128> parse_emul(string line){
    char dpdon_str[100];
    char dpdby_str[100];

    int trunc;
    uint32_t emul_id;
    uint32_t len;
    uint32_t v1_id;
    uint32_t v1_addr;
    uint32_t v2_id;
    uint32_t v2_addr;
    uint32_t res_id;
    uint32_t res_addr;

    
    if(cal_strNum(line) != 23){
      printf("ERROR: words length: %d not match in [EMUL]\n", cal_strNum(line));
      return bitset<128>(string(128, '0'));
    }

	  sscanf(line.c_str(), "EMUL %*s %s %*s %s %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d %*s %d\n", 
                            dpdon_str, dpdby_str, &trunc, &emul_id, &len, &v1_id, &v1_addr, &v2_id, &v2_addr, 
                            &res_id, &res_addr);



    bitset<8> opcode_bit(0x20);
    bitset<8> dpdon_bit(dpd_str2val(dpdon_str));
    bitset<8> dpdby_bit(dpd_str2val(dpdby_str));

    bitset<8> trunc_bit(trunc);
    bitset<4> emul_id_bit(emul_id);
    bitset<12> len_bit(len);
    bitset<4> v1_id_bit(v1_id);
    bitset<12> v1_addr_bit(v1_addr);
    bitset<4> v2_id_bit(v2_id);
    bitset<12> v2_addr_bit(v2_addr);
    bitset<4> res_id_bit(res_id);
    bitset<12> res_addr_bit(res_addr);

    string instr_str = opcode_bit.to_string();
    instr_str += dpdon_bit.to_string();
    instr_str += dpdby_bit.to_string();
    instr_str += trunc_bit.to_string();
    instr_str += emul_id_bit.to_string();
    instr_str += len_bit.to_string();
    instr_str += v1_id_bit.to_string();
    instr_str += v1_addr_bit.to_string();
    instr_str += v2_id_bit.to_string();
    instr_str += v2_addr_bit.to_string();
    instr_str += res_id_bit.to_string();
    instr_str += res_addr_bit.to_string();
    instr_str += string(32, '0');

    return bitset<128>(instr_str);


  }//parse_emul



  ///////////////////////////parse_end////////////////////////////
  bitset<128> parse_end(string line){
    char dpdon_str[100];
    char dpdby_str[100];


    if(cal_strNum(line) != 5){
      printf("ERROR: words length: %d not match in [END]\n", cal_strNum(line));
      return bitset<128>(string(128, '0'));
    }

    sscanf(line.c_str(), "END %*s %s %*s %s\n", dpdon_str, dpdby_str);

    bitset<8> opcode_bit(0x40);
    bitset<8> dpdon_bit(dpd_str2val(dpdon_str));
    bitset<8> dpdby_bit(dpd_str2val(dpdby_str));
    
    string instr_str = opcode_bit.to_string();
    instr_str += dpdon_bit.to_string();
    instr_str += dpdby_bit.to_string();
    instr_str += string(104, '0');
    

    return bitset<128>(instr_str);

  }//parse_end




    uint32_t cal_end_instr_addr(uint32_t start_addr, uint32_t layer_num, uint32_t kernel_num, uint32_t* instr_line_num){
        uint32_t end_instr_addr;
        uint32_t line_num = 0;
        uint32_t regs_line_num = 0;
        for(int l=0;l<layer_num;l++){
            regs_line_num = (l==0) ? kernel_num*3+3 : kernel_num*3+2;
            line_num += regs_line_num;
            line_num += instr_line_num[2*l] + instr_line_num[2*l+1];

        }//for l

        end_instr_addr = start_addr + 16*line_num;

        return end_instr_addr;
    
    }

    


    uint32_t cal_regs_size(regs info){
        uint32_t regs_size;

        regs_size = (info.layer_i==0) ? info.kernel_num*3+3 : info.kernel_num*3+2;
        return regs_size;
    
    }



    void pack_regs(regs info, uint32_t start_addr, char* res){
        char* line;
        line = new char[16];
        memset(line, 0, 16);

        uint32_t zero = 0;
        uint32_t line_num = 0;
        uint32_t head_len = cal_regs_size(info)-1;
        
        memcpy(line, &head_len, 4);
        memcpy(res, line, 16);
        line_num += 16;

        
        
        if(info.layer_i==0){
            memcpy(line, &info.layer_num, 4);             
            memcpy(line+4, &info.end_instr_addr, 4);            
            memcpy(line+8, &info.end_instr_len, 4);
            memcpy(line+12, &zero, 4);
            memcpy(res+line_num, line, 16);
            line_num += 16;

        }

        uint32_t init_instr_addr = start_addr + (head_len+1)*16;        
        uint32_t loop_instr_addr = init_instr_addr + info.init_instr_num*16;    
        memcpy(line, &init_instr_addr, 4);
        memcpy(line+4, &info.init_instr_num, 4);
        memcpy(line+8, &loop_instr_addr, 4);
        memcpy(line+12, &info.loop_instr_num, 4);
        memcpy(res+line_num, line, 16);
        line_num += 16;


        for(int k=0;k<info.kernel_num;k++){
            memcpy(line, &info.load_src_reg0[k], 4);
            memcpy(line+4, &info.load_src_reg1[k], 4);
            memcpy(line+8, &zero, 4);
            memcpy(line+12, &zero, 4);
            memcpy(res+line_num, line, 16);
            line_num += 16;
            
            uint32_t load_dest_reg0 = (info.load_dest_bank0[k]<<16) + info.load_dest_bank_addr0[k];
            uint32_t load_dest_reg1 = (info.load_dest_bank1[k]<<16) + info.load_dest_bank_addr1[k];
            memcpy(line, &load_dest_reg0, 4);
            memcpy(line+4, &load_dest_reg1, 4);
            memcpy(line+8, &zero, 4);
            memcpy(line+12, &zero, 4);
            memcpy(res+line_num, line, 16);
            line_num += 16;

            memcpy(line, &info.save_dest_reg0[k], 4);
            memcpy(line+4, &zero, 4);
            memcpy(line+8, &zero, 4);
            memcpy(line+12, &zero, 4);
            memcpy(res+line_num, line, 16);
            line_num += 16;
        }


        delete[] line;

    }



  ///////////////////////////////parse_instr////////////////////////
  void parse_instr(uint32_t layer_num, uint32_t kernel_num, uint32_t* line_num, 
                   const char* f_name, const char* f_bin){
    

        uint32_t* load_src_reg0;
        uint32_t* load_src_reg1;
        uint32_t* load_dest_bank0;
        uint32_t* load_dest_bank1;
        uint32_t* load_dest_bank_addr0;
        uint32_t* load_dest_bank_addr1;
        uint32_t* save_dest_reg0;
                
        regs info;

        load_src_reg0 = new uint32_t[kernel_num];
        load_src_reg1 = new uint32_t[kernel_num];
        load_dest_bank0 = new uint32_t[kernel_num];
        load_dest_bank1 = new uint32_t[kernel_num];
        load_dest_bank_addr0 = new uint32_t[kernel_num];
        load_dest_bank_addr1 = new uint32_t[kernel_num];
        save_dest_reg0 = new uint32_t[kernel_num];

        ifstream str_fp(f_name, ios::in);
        if(!str_fp.is_open()){
            cout << "FATAL: fail to open file: " << f_name << endl;
            return;
        }
  

        fstream bin_fp(f_bin, ios::out | ios::binary | ios::in);
        if(!bin_fp.is_open()){
            cout << "FATAL: fail to open file: " << f_bin << endl;
            return;
        }

    
        uint32_t bin_addr = INSTR_BASEADDR;
        string line = "";
        bitset<128> bin_instr;
        char* out_instr;
        out_instr = new char[16];
    
        for(int l=0;l<layer_num;l++){

            info.layer_num = layer_num;
            info.kernel_num = kernel_num;
            info.layer_i = l;
            info.init_instr_num = line_num[2*l];
            info.loop_instr_num = line_num[2*l+1];
            for(int k=0;k<kernel_num;k++){
                load_dest_bank0[k] = VECTOR_BANK_ID[k];
                load_dest_bank_addr0[k] = 0;
                load_dest_bank1[k] = VECTOR_BANK_ID[k];
                load_dest_bank_addr1[k] = (layer_num==8) ? oie_vec_bank_reg[l] : sent_vec_bank_reg;

                if(kernel_num==3){
                    load_src_reg0[k] = (l%2==0) ? io_hbm3[k] + 0x3000000 : io_hbm3[k] + 0x6000000;
                    load_src_reg1[k] = (l%2==0) ? io_hbm3[k] + 0x6000000 : io_hbm3[k] + 0x3000000;
                    save_dest_reg0[k] = (l%2==0) ? io_hbm3[k] + 0x6000000 : io_hbm3[k] + 0x3000000;
                } else if(kernel_num==4){
                    load_src_reg0[k] = (l%2==0) ? io_hbm4[k] + 0x3000000 : io_hbm4[k] + 0x6000000;
                    load_src_reg1[k] = (l%2==0) ? io_hbm4[k] + 0x6000000 : io_hbm4[k] + 0x3000000;
                    save_dest_reg0[k] = (l%2==0) ? io_hbm4[k] + 0x6000000 : io_hbm4[k] + 0x3000000;
                
                } else{
                    load_src_reg0[k] = (l%2==0) ? io_hbm4[k] + 0x3000000 : io_hbm4[k] + 0x6000000;
                    load_src_reg1[k] = (l%2==0) ? io_hbm4[k] + 0x6000000 : io_hbm4[k] + 0x3000000;
                    save_dest_reg0[k] = (l%2==0) ? io_hbm4[k] + 0x6000000 : io_hbm4[k] + 0x3000000;
                
                }

                                                    ;

            }//for k


            info.load_dest_bank0 = load_dest_bank0;
            info.load_dest_bank1 = load_dest_bank1;
            info.load_dest_bank_addr0 = load_dest_bank_addr0;
            info.load_dest_bank_addr1 = load_dest_bank_addr1;
            info.load_src_reg0 = load_src_reg0;
            info.load_src_reg1 = load_src_reg1;
            info.save_dest_reg0  = save_dest_reg0;
            info.end_instr_addr = cal_end_instr_addr(INSTR_BASEADDR, layer_num, kernel_num, line_num);
            info.end_instr_len = line_num[2*layer_num];
            

            uint32_t reg_size = cal_regs_size(info);
            char* reg_line;
            reg_line = new char[reg_size*16];
            pack_regs(info, bin_addr, reg_line);
            bin_fp.seekp(bin_addr, ios::beg);
            bin_fp.write(reg_line, reg_size*16);
            bin_addr += reg_size*16;

            delete[] reg_line;


            for(int instr_i=0;instr_i<line_num[2*l]+line_num[2*l+1];instr_i++){
                getline(str_fp, line);

                if(line.find("LOAD") != string::npos){
                    bin_instr = parse_load(line);
                }
                else if(line.find("SAVE") != string::npos){
                    bin_instr = parse_save(line);
                }
                else if(line.find("MMUL") != string::npos){
                    bin_instr = parse_mmul(line);
                }
                else if(line.find("ADD") != string::npos){
                    bin_instr = parse_add(line);
                }
                else if(line.find("ACTV") != string::npos){
                    bin_instr = parse_actv(line);
                }
                else if(line.find("EMUL") != string::npos){
                    bin_instr = parse_emul(line);
                }
                else if(line.find("END") != string::npos){
                    bin_instr = parse_end(line);
                }

                for(int i=0;i<16;i++){
                    out_instr[i] = 0;
                    for(int j=0;j<8;j++){
                        if(bin_instr[8*i+j])
                            out_instr[i] |= (1<<j);

                    }//for j
                }//for i

                bin_fp.seekp(bin_addr, ios::beg);
                bin_fp.write(out_instr, 16);
                bin_addr += 16;
            
            
            }



        }//for l
         
      
        //the end instructions
        for(int instr_i=0;instr_i<line_num[2*layer_num];instr_i++){
            getline(str_fp, line);

            if(line.find("LOAD") != string::npos){
                bin_instr = parse_load(line);
            }
            else if(line.find("SAVE") != string::npos){
                bin_instr = parse_save(line);
            }
            else if(line.find("MMUL") != string::npos){
                bin_instr = parse_mmul(line);
            }
            else if(line.find("ADD") != string::npos){
                bin_instr = parse_add(line);
            }
            else if(line.find("ACTV") != string::npos){
                bin_instr = parse_actv(line);
            }
            else if(line.find("EMUL") != string::npos){
                bin_instr = parse_emul(line);
            }
            else if(line.find("END") != string::npos){
                bin_instr = parse_end(line);
            }

            for(int i=0;i<16;i++){
                out_instr[i] = 0;
                for(int j=0;j<8;j++){
                    if(bin_instr[8*i+j])
                        out_instr[i] |= (1<<j);

                }//for j
            }//for i

            bin_fp.seekp(bin_addr, ios::beg);
            bin_fp.write(out_instr, 16);
            bin_addr += 16;
        
        
        }//for instr_i



      
        delete[] load_src_reg0;
        delete[] load_src_reg1;
        delete[] load_dest_bank0;
        delete[] load_dest_bank1;
        delete[] load_dest_bank_addr0;
        delete[] load_dest_bank_addr1;
        delete[] save_dest_reg0;

  
        delete[] out_instr;


  }//parse_instr



  ////////////////////////////////ddr2bin/////////////////////////////
  void ddr2bin(const char* f_ddr, const char* f_bin){  
   
    char *buffer;
    string line;
    uint32_t rd_addr;
    char rd_str[200];
    
    ifstream ifile(f_ddr, ios::in);
    if(!ifile.is_open()){
      cout << "FATAL: fail to open file "<< f_ddr << endl;
      return;
    }
    
    ofstream ofile(f_bin, ios::out | ios::binary | ios::trunc);
    if(!ofile.is_open()){
      cout << "fail to open file "<< f_bin << endl;
      return;
    }
    
    buffer=new char[DDR_BW];
    while(getline(ifile, line)){
      int str_pos;
      sscanf(line.c_str(), "%x : %s\n", &rd_addr, rd_str);
      

      str_pos=strlen(rd_str)-2;
      for(uint32_t i=0;i<DDR_BW;i++){
        char data;

        if(rd_str[str_pos]<='9' && rd_str[str_pos]>='0'){
          data=rd_str[str_pos]-'0';
        }
        else if(rd_str[str_pos]<='F' && rd_str[str_pos]>='A'){
          data=rd_str[str_pos]-'A'+10;
        }
        else if(rd_str[str_pos]<='f' && rd_str[str_pos]>='a'){
          data=rd_str[str_pos]-'a'+10;
        }
        
        if(rd_str[str_pos+1]<='9' && rd_str[str_pos+1]>='0'){
          data=data*16+rd_str[str_pos+1]-'0';
        }
        else if(rd_str[str_pos+1]<='F' && rd_str[str_pos+1]>='A'){
          data=data*16+rd_str[str_pos+1]-'A'+10;
        }
        else if(rd_str[str_pos+1]<='f' && rd_str[str_pos+1]>='a'){
          data=data*16+rd_str[str_pos+1]-'a'+10;
        }
        
        buffer[i]=data;
        str_pos -= 2;
        
      }
      ofile.seekp(rd_addr, ios::beg);
      ofile.write(buffer, DDR_BW);
    }
    
    ifile.close();
    ofile.close();
    
    delete[] buffer;



  
  }//ddr2bin



  

  void create_bin(uint32_t layer_num, uint32_t kernel_num, uint32_t* line_num, 
                  const char* f_ddr, const char* f_instr, const char* f_bin){

    ddr2bin(f_ddr, f_bin);
    parse_instr(layer_num, kernel_num, line_num, f_instr, f_bin);

    
  }//create_bin






}//namespace bin



/*
 *define extern C for used in python using ctypes
 *
 */
extern "C"{
    void create_bin(uint32_t layer_num, uint32_t kernel_num, uint32_t* line_num, 
                  const char* f_ddr, const char* f_instr, const char* f_bin){   
        bin::create_bin(layer_num, kernel_num, line_num, f_ddr, f_instr, f_bin);
    }


}
