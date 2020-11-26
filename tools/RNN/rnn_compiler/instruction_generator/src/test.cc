/*
 *file_name: test.cc
 *creator: pingbo an
 *date: 2020-3-25
 *description: test bin_generator
 */


#include <string>
#include "bin_generator.h"
using namespace std;
using namespace bin;


int main(){
  const char* f_ddr = "../output/ddr_init_orign.txt";
  const char* f_instr = "../output/instr_ac_3.txt";
  const char* f_bin = "../output/ddr_bin";
  uint32_t line_num[3] = {334, 66, 4};
  bin::create_bin(1, 3, line_num, f_ddr, f_instr, f_bin);
  
  return 1;

}
