#the file list:
  test --- executation file
  test.cpp --- c++ test host program
  test.py --- python test host program
  src/* --- xvrnn lib source files
  include/* --- xvrnn lib header files
  python/* --- xvrnn lib python APIs
  make.sh --- compile script
  CMakeLists.txt --- cmake config file
  32input --- input data for functional test
  32frm --- decoder freame number corresponding to "32input"
  model/rnnt.bin --- model and instruction data
  xclbin/xvrnn.hw.xclbin --- hardware bitstream
  golden_inout/gold_result_of_32input_32frm --- golden result of "32input"

#compile:
  sh make.sh  //some python and pyband11 lib are needed, you should change the correspond settings in CMakeLists.txt

#execute in c++ mode:
  ./test [command] [options]
    Command and option summary:
    help: [-h]
    profiler download: [-p download_size]  // the profiler data will be wrote into the file prof/prof_txt

#execute in python mode:
  python test.py

#More details for using this program:
  1. the input files(32input, 32frm, rnnt.bin, xvrnn.hw.xclbin) used by this program are listed upon, but the demo can use other input files if the python functions are called in other path.
  2. xclbin file:     must be named as xvrnn.hw.xclbin and located in the path ./xclbin/
  3. ddr_bin file:      must be named as rnnt.bin and located in the path ./model/
  4. the vector input size of the c++ test host program is defined in the test.cpp ( INPUT_SIZE ).
  5. the result output size of the c++ test host program is defined in the test.cpp ( OUTPUT_SIZE ).
  6. the result output:  For the c++ program, it's rslt/rslt_cpp_bin. For the python program, it's rslt/rslt_bin.
  7. the encoder fream_num of the input in is defined in the test.cpp:L48 ( int frame_num ) and in the test.py:L4( frame_num ).
  8. for 32 batches, before we run the hardware, the encoder and decoder frame number are needed to be configed. If we have (10, 20, 30, ..., 90)  valid frames for each batch, then the encoder input frame number should be configured as the max value 90 by axi_lite interface, more details, please refer to item 7.  For decoder frame number it will be half of the valid frame number(5, 10, 15, ..., 45). those number will be written into ddr with address 0x7c4_0000 for 32 continuous 32bits data.
License
Copyright 2019 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
