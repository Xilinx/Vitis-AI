# /*******************************************************************************
# /*                                                                         
# * Copyright 2019 Xilinx Inc.                                               
# *                                                                          
# * Licensed under the Apache License, Version 2.0 (the "License");          
# * you may not use this file except in compliance with the License.         
# * You may obtain a copy of the License at                                  
# *                                                                          
# *    http://www.apache.org/licenses/LICENSE-2.0                            
# *                                                                          
# * Unless required by applicable law or agreed to in writing, software      
# * distributed under the License is distributed on an "AS IS" BASIS,        
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# * See the License for the specific language governing permissions and      
# * limitations under the License.                                           
# */
# *******************************************************************************/

import sys
import os

Batch_N    = int(sys.argv[1])
LOAD_I_P   = int(sys.argv[2])
CPB        = int(sys.argv[3])
BAT_SHRWGT = int(sys.argv[4])
LOAD_W_P   = int(sys.argv[5])


IMG_ports  = Batch_N*LOAD_I_P
WGTBC_N    = int((Batch_N+BAT_SHRWGT-1)/BAT_SHRWGT)

if CPB==16:
    ifm_number = 8*Batch_N 
    ofm_number = 8*Batch_N
    wgt_number = 4*WGTBC_N
elif CPB==32:
    ifm_number = 8*Batch_N    
    ofm_number = 16*Batch_N
    wgt_number = 8*WGTBC_N
elif CPB==64:
    ifm_number = 8*Batch_N
    ofm_number = 32*Batch_N
    wgt_number = 16*WGTBC_N
else:
    pass
    

def IMG_AXI(ports_num):
    char = ""
    for i in range (ports_num):
        char +=  r'	  <port name="M' + str(i).rjust(2,'0') + r'_IMG_AXI"   mode="master" range="0x7FFFFFFF" dataWidth="128" portType="addressable" base="0x0"/>' + "\n"
    return char

def WGT_AXI(ports_num):
    char = ""
    for i in range (ports_num):
        char +=  r'	  <port name="M' + str(i).rjust(2,'0') + r'_WGT_AXI"   mode="master" range="0x7FFFFFFF" dataWidth="512" portType="addressable" base="0x0"/>' + "\n"
    return char
 
def IFM_AXIS(ports_num):
    char = ""
    for i in range (ports_num):
        char += r'	  <port name="M' + str(i).rjust(2,'0') + r'_IFM_AXIS"  mode="write_only" dataWidth="128" portType="stream"/>' + "\n"
    return char
    
def WGT_AXIS(ports_num):
    char = ""
    for i in range (ports_num):
        char += r'	  <port name="M' + str(i).rjust(2,'0') + r'_WGT_AXIS"  mode="write_only" dataWidth="128" portType="stream"/>' + "\n"
    return char
  
def OFM_AXIS(ports_num):
    char = ""
    for i in range (ports_num):
        char += r'	  <port name="S' + str(i).rjust(2,'0') + r'_OFM_AXIS"  mode="read_only"  dataWidth="64" portType="stream"/>' + "\n"
    return char


first_char =  (r'''<?xml version="1.0" encoding="UTF-8"?>
<root versionMajor="1" versionMinor="0">
  <kernel name="DPUCVDX8G" language="ip" vlnv="xilinx.com:ip:DPUCVDX8G:0.0" attributes="" preferredWorkGroupSizeMultiple="0" workGroupSize="1" debug="true" compileOptions=" -g" profileType="none">
    <ports>
      <port name="S_AXI_CONTROL" mode="slave"  range="0x00000FFF" dataWidth="32"  portType="addressable" base="0x80000000"/>
	  <port name="M00_INSTR_AXI" mode="master" range="0x7FFFFFFF" dataWidth="32"  portType="addressable" base="0x0"/>
      <port name="M00_BIAS_AXI"  mode="master" range="0x7FFFFFFF" dataWidth="128" portType="addressable" base="0x0"/>
''')

args_1st_char = (r'''    </ports>
    <args>            
	  <arg name="dpu_sys_reg"       addressQualifier="0" id="0"   port="S_AXI_CONTROL" size="0x4" offset="0x20"  hostOffset="0x0" hostSize="0x4" type="int*"/> 	
	  <arg name="dpu_freq"          addressQualifier="0" id="1"   port="S_AXI_CONTROL" size="0x4" offset="0x28"  hostOffset="0x0" hostSize="0x4" type="int*"/> 
	  <arg name="dpu_doneclr"       addressQualifier="0" id="2"   port="S_AXI_CONTROL" size="0x4" offset="0x40"  hostOffset="0x0" hostSize="0x4" type="int*"/> 	
	  <arg name="dpu_prof_en"       addressQualifier="0" id="3"   port="S_AXI_CONTROL" size="0x4" offset="0x44"  hostOffset="0x0" hostSize="0x4" type="int*"/> 
	  <arg name="dpu_cmd"           addressQualifier="0" id="4"   port="S_AXI_CONTROL" size="0x4" offset="0x48"  hostOffset="0x0" hostSize="0x4" type="int*"/>
	  <arg name="dpu_instr_addr"    addressQualifier="1" id="5"   port="M00_INSTR_AXI" size="0x8" offset="0x50"  hostOffset="0x0" hostSize="0x8" type="int*"/>
	  <arg name="dpu_prof_addr"     addressQualifier="1" id="6"   port="M00_INSTR_AXI" size="0x8" offset="0x58"  hostOffset="0x0" hostSize="0x8" type="int*"/>
      <arg name="dpu_batch0_addr0"  addressQualifier="1" id="7"   port="M00_BIAS_AXI"  size="0x8" offset="0x200" hostOffset="0x0" hostSize="0x8" type="int*"/>
	  <arg name="dpu_batch0_addr1"  addressQualifier="1" id="8"   port="M00_BIAS_AXI"  size="0x8" offset="0x208" hostOffset="0x0" hostSize="0x8" type="int*"/>      
	  <arg name="dpu_batch0_addr2"  addressQualifier="1" id="9"   port="M00_BIAS_AXI"  size="0x8" offset="0x210" hostOffset="0x0" hostSize="0x8" type="int*"/>
	  <arg name="dpu_batch0_addr3"  addressQualifier="1" id="10"  port="M00_BIAS_AXI"  size="0x8" offset="0x218" hostOffset="0x0" hostSize="0x8" type="int*"/>	
 	  <arg name="dpu_batch0_addr4"  addressQualifier="1" id="11"  port="M00_BIAS_AXI"  size="0x8" offset="0x220" hostOffset="0x0" hostSize="0x8" type="int*"/>
	  <arg name="dpu_batch0_addr5"  addressQualifier="1" id="12"  port="M00_BIAS_AXI"  size="0x8" offset="0x228" hostOffset="0x0" hostSize="0x8" type="int*"/>      
	  <arg name="dpu_batch0_addr6"  addressQualifier="1" id="13"  port="M00_BIAS_AXI"  size="0x8" offset="0x230" hostOffset="0x0" hostSize="0x8" type="int*"/>
	  <arg name="dpu_batch0_addr7"  addressQualifier="1" id="14"  port="M00_BIAS_AXI"  size="0x8" offset="0x238" hostOffset="0x0" hostSize="0x8" type="int*"/>	
''')

global arg_id
arg_id = 15

def WGT_ARGS(ports_num):
    char = ""
    global arg_id
    for i in range (ports_num):
        offset_Batch0 = 0x200
        for x in range (8):
            char   +=  r'	  <arg name="dpu_batch0_addr' + str(x) + r'"  addressQualifier="1" id="' + str(arg_id) + r'"  port="M' + str(i).rjust(2,'0')  + r'_WGT_AXI"   size="0x8" offset="' + str(hex(offset_Batch0)) + r'" hostOffset="0x0" hostSize="0x8" type="int*"/>' + "\n"
            arg_id  = arg_id + 1
            offset_Batch0 = offset_Batch0 + 8 
    return char 
    
def IMG_ARGS(batch, img_port):
    char = ""
    global arg_id
    for i in range (batch):
        offset_batch = i*64
        offset_No = 0x200 + offset_batch  
        IMG_P     = img_port * i
        offset_Batch0 = 0x200
        for x in range (img_port):
            total_IMG_P = IMG_P + x
            offset_perBatch = offset_No
            for y in range (8):
                char   +=  r'	  <arg name="dpu_batch' + str(i) + r'_addr' + str(y) + r'"  addressQualifier="1" id="' + str(arg_id) + r'"  port="M' + str(total_IMG_P).rjust(2,'0') + r'_IMG_AXI"   size="0x8" offset="' + str(hex(offset_perBatch)) + r'" hostOffset="0x0" hostSize="0x8" type="int*"/>' + "\n"
                arg_id  = arg_id + 1
                offset_perBatch = offset_perBatch + 8 
    return char    

global AXIS_No

def IFM_AXIS_ARGS(ports_num):
    char = ""    
    AXIS_offset = 0x600
    global arg_id
    global AXIS_No    
    for i in range (ports_num):
        AXIS_No = AXIS_offset + i*8
        char += r'      <arg name="M' + str(i).rjust(2,'0') + r'_IFM_AXIS"      addressQualifier="4" id="' + str(arg_id) + r'"  port="M' + str(i).rjust(2,'0') + r'_IFM_AXIS"  size="0x4" offset="' + str(hex(AXIS_No)) + r'" hostOffset="0x0" hostSize="0x4" type="stream&lt;qdma_axis&lt;128, 0, 0, 0>>&amp;"/>' + "\n"
        arg_id = arg_id + 1
    return char

def WGT_AXIS_ARGS(ports_num):
    char = ""
    global arg_id
    global AXIS_No
    for i in range (ports_num):
        AXIS_No = AXIS_No + 8
        char += r'      <arg name="M' + str(i).rjust(2,'0') + r'_WGT_AXIS"      addressQualifier="4" id="' + str(arg_id) + r'"  port="M' + str(i).rjust(2,'0') + r'_WGT_AXIS"  size="0x4" offset="' + str(hex(AXIS_No)) + r'" hostOffset="0x0" hostSize="0x4" type="stream&lt;qdma_axis&lt;128, 0, 0, 0>>&amp;"/>' + "\n"
        arg_id = arg_id + 1
    return char

def OFM_AXIS_ARGS(ports_num):
    char = ""
    global arg_id
    global AXIS_No
    for i in range (ports_num):
        AXIS_No = AXIS_No + 8
        char += r'      <arg name="S' + str(i).rjust(2,'0') + r'_OFM_AXIS"      addressQualifier="4" id="' + str(arg_id) + r'"  port="S' + str(i).rjust(2,'0') + r'_OFM_AXIS"  size="0x4" offset="' + str(hex(AXIS_No)) + r'" hostOffset="0x0" hostSize="0x4" type="stream&lt;qdma_axis&lt;64, 0, 0, 0>>&amp;"/>' + "\n"
        arg_id = arg_id + 1
    return char

result = first_char    
result += IMG_AXI(IMG_ports)
result += WGT_AXI(LOAD_W_P)
result += IFM_AXIS(ifm_number)
result += WGT_AXIS(wgt_number)
result += OFM_AXIS(ofm_number)
result += args_1st_char
result += WGT_ARGS(LOAD_W_P)
result += IMG_ARGS(Batch_N, LOAD_I_P)
result += IFM_AXIS_ARGS(ifm_number)
result += WGT_AXIS_ARGS(wgt_number)
result += OFM_AXIS_ARGS(ofm_number)

result += (r'''    </args>
  </kernel>
</root>
''')

file_name ="kernel" + ".xml"
new_file  = open(file_name, "w+")
new_file.write(result)

new_file.close()



