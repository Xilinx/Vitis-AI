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

IMG_ports  = Batch_N*LOAD_I_P
WGTBC_N    = int((Batch_N+BAT_SHRWGT-1)/BAT_SHRWGT)

result=[]

result.append(r'''<?xml version="1.0" encoding="UTF-8"?>
<root versionMajor="1" versionMinor="0">
  <kernel name="DPUCVDX8G" language="ip" vlnv="xilinx.com:ip:DPUCVDX8G:0.0" attributes="" preferredWorkGroupSizeMultiple="0" workGroupSize="1" debug="true" compileOptions=" -g" profileType="none">
    <ports>
      <port name="S_AXI_CONTROL" mode="slave"  range="0x0000FFFF" dataWidth="32"  portType="addressable" base="0x80000000"/>
	  <port name="M00_INSTR_AXI" mode="master" range="0x7FFFFFFF" dataWidth="32"  portType="addressable" base="0x0"/>
''')

#PORTS section: IMG_AXI 
for i in range (IMG_ports): 
    number_img = str(i).rjust(2,'0')
    Char1 = r'	  <port name="M'
    Char2 = r'_IMG_AXI"   mode="master" range="0x7FFFFFFF" dataWidth="128" portType="addressable" base="0x0"/>'
    list_buf=Char1 + number_img + Char2 +  "\n"
    str_buf="".join(list_buf) 
    result.append(str_buf)    

result.append(r'''	  <port name="M00_WGT_AXI"   mode="master" range="0x7FFFFFFF" dataWidth="512" portType="addressable" base="0x0"/>
	  <port name="M01_WGT_AXI"   mode="master" range="0x7FFFFFFF" dataWidth="512" portType="addressable" base="0x0"/>
	  <port name="M02_WGT_AXI"   mode="master" range="0x7FFFFFFF" dataWidth="512" portType="addressable" base="0x0"/>
	  <port name="M03_WGT_AXI"   mode="master" range="0x7FFFFFFF" dataWidth="512" portType="addressable" base="0x0"/>
      <port name="M00_BIAS_AXI"  mode="master" range="0x7FFFFFFF" dataWidth="128" portType="addressable" base="0x0"/> 
''')

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

#PORTS section: AXIS
  
#ifm 
for i in range (ifm_number):
    number_ifm = str(i).rjust(2,'0')
    Char1 = r'	  <port name="M'
    Char2 = r'_IFM_AXIS"  mode="write_only" dataWidth="128" portType="stream"/>'
    list_buf =Char1 + number_ifm + Char2 +  "\n"
    str_buf  ="".join(list_buf) 
    result.append(str_buf)   

#wgt 
for i in range (wgt_number):
    number_wgt = str(i).rjust(2,'0')         
    Char1 = r'	  <port name="M'
    Char2 = r'_WGT_AXIS"  mode="write_only" dataWidth="128" portType="stream"/>'
    list_buf =Char1 + number_wgt + Char2 +  "\n"
    str_buf  ="".join(list_buf) 
    result.append(str_buf)  
	 
#ofm 
for i in range (ofm_number):
    number_ofm = str(i).rjust(2,'0')     
    Char1 = r'	  <port name="S'
    Char2 = r'_OFM_AXIS"  mode="read_only"  dataWidth="64" portType="stream"/>'
    list_buf = Char1 + number_ofm + Char2 +  "\n"
    str_buf ="".join(list_buf) 
    result.append(str_buf)  
    
##############################################################################################################################################################    
#First part of arg section    
result.append(r'''    </ports>
    <args>            
	  <arg name="dpu_sys_reg"       addressQualifier="0" id="0"   port="S_AXI_CONTROL" size="0x4" offset="0x20"  hostOffset="0x0" hostSize="0x4" type="int*"/> 	
	  <arg name="dpu_freq"          addressQualifier="0" id="1"   port="S_AXI_CONTROL" size="0x4" offset="0x28"  hostOffset="0x0" hostSize="0x4" type="int*"/> 
	  <arg name="dpu_doneclr"       addressQualifier="0" id="2"   port="S_AXI_CONTROL" size="0x4" offset="0x40"  hostOffset="0x0" hostSize="0x4" type="int*"/> 	
	  <arg name="dpu_prof_en"       addressQualifier="0" id="3"   port="S_AXI_CONTROL" size="0x4" offset="0x44"  hostOffset="0x0" hostSize="0x4" type="int*"/> 
	  <arg name="dpu_cmd"           addressQualifier="0" id="4"   port="S_AXI_CONTROL" size="0x4" offset="0x48"  hostOffset="0x0" hostSize="0x4" type="int*"/>
	  <arg name="dpu_instr_addr"    addressQualifier="1" id="5"   port="M00_INSTR_AXI" size="0x8" offset="0x50"  hostOffset="0x0" hostSize="0x8" type="int*"/>
	  <arg name="dpu_prof_addr"     addressQualifier="1" id="6"   port="M00_INSTR_AXI" size="0x8" offset="0x58"  hostOffset="0x0" hostSize="0x8" type="int*"/>
      <arg name="dpu_batch0_addr0"  addressQualifier="1" id="7"   port="M00_WGT_AXI"   size="0x8" offset="0x200" hostOffset="0x0" hostSize="0x8" type="int*"/>	
      <arg name="dpu_batch0_addr0"  addressQualifier="1" id="8"   port="M01_WGT_AXI"   size="0x8" offset="0x200" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr0"  addressQualifier="1" id="9"   port="M02_WGT_AXI"   size="0x8" offset="0x200" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr0"  addressQualifier="1" id="10"  port="M03_WGT_AXI"   size="0x8" offset="0x200" hostOffset="0x0" hostSize="0x8" type="int*"/>	  
	  <arg name="dpu_batch0_addr0"  addressQualifier="1" id="11"  port="M00_BIAS_AXI"  size="0x8" offset="0x200" hostOffset="0x0" hostSize="0x8" type="int*"/>
      <arg name="dpu_batch0_addr1"  addressQualifier="1" id="12"  port="M00_WGT_AXI"   size="0x8" offset="0x208" hostOffset="0x0" hostSize="0x8" type="int*"/>	
      <arg name="dpu_batch0_addr1"  addressQualifier="1" id="13"  port="M01_WGT_AXI"   size="0x8" offset="0x208" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr1"  addressQualifier="1" id="14"  port="M02_WGT_AXI"   size="0x8" offset="0x208" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr1"  addressQualifier="1" id="15"  port="M03_WGT_AXI"   size="0x8" offset="0x208" hostOffset="0x0" hostSize="0x8" type="int*"/>	  
	  <arg name="dpu_batch0_addr1"  addressQualifier="1" id="16"  port="M00_BIAS_AXI"  size="0x8" offset="0x208" hostOffset="0x0" hostSize="0x8" type="int*"/>      
      <arg name="dpu_batch0_addr2"  addressQualifier="1" id="17"  port="M00_WGT_AXI"   size="0x8" offset="0x210" hostOffset="0x0" hostSize="0x8" type="int*"/>	
      <arg name="dpu_batch0_addr2"  addressQualifier="1" id="18"  port="M01_WGT_AXI"   size="0x8" offset="0x210" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr2"  addressQualifier="1" id="19"  port="M02_WGT_AXI"   size="0x8" offset="0x210" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr2"  addressQualifier="1" id="20"  port="M03_WGT_AXI"   size="0x8" offset="0x210" hostOffset="0x0" hostSize="0x8" type="int*"/>	  
	  <arg name="dpu_batch0_addr2"  addressQualifier="1" id="21"  port="M00_BIAS_AXI"  size="0x8" offset="0x210" hostOffset="0x0" hostSize="0x8" type="int*"/>
      <arg name="dpu_batch0_addr3"  addressQualifier="1" id="22"  port="M00_WGT_AXI"   size="0x8" offset="0x218" hostOffset="0x0" hostSize="0x8" type="int*"/>	
      <arg name="dpu_batch0_addr3"  addressQualifier="1" id="23"  port="M01_WGT_AXI"   size="0x8" offset="0x218" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr3"  addressQualifier="1" id="24"  port="M02_WGT_AXI"   size="0x8" offset="0x218" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr3"  addressQualifier="1" id="25"  port="M03_WGT_AXI"   size="0x8" offset="0x218" hostOffset="0x0" hostSize="0x8" type="int*"/>	  
	  <arg name="dpu_batch0_addr3"  addressQualifier="1" id="26"  port="M00_BIAS_AXI"  size="0x8" offset="0x218" hostOffset="0x0" hostSize="0x8" type="int*"/>	
      <arg name="dpu_batch0_addr4"  addressQualifier="1" id="27"  port="M00_WGT_AXI"   size="0x8" offset="0x220" hostOffset="0x0" hostSize="0x8" type="int*"/>	
      <arg name="dpu_batch0_addr4"  addressQualifier="1" id="28"  port="M01_WGT_AXI"   size="0x8" offset="0x220" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr4"  addressQualifier="1" id="29"  port="M02_WGT_AXI"   size="0x8" offset="0x220" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr4"  addressQualifier="1" id="30"  port="M03_WGT_AXI"   size="0x8" offset="0x220" hostOffset="0x0" hostSize="0x8" type="int*"/>	  
	  <arg name="dpu_batch0_addr4"  addressQualifier="1" id="31"  port="M00_BIAS_AXI"  size="0x8" offset="0x220" hostOffset="0x0" hostSize="0x8" type="int*"/>
      <arg name="dpu_batch0_addr5"  addressQualifier="1" id="32"  port="M00_WGT_AXI"   size="0x8" offset="0x228" hostOffset="0x0" hostSize="0x8" type="int*"/>	
      <arg name="dpu_batch0_addr5"  addressQualifier="1" id="33"  port="M01_WGT_AXI"   size="0x8" offset="0x228" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr5"  addressQualifier="1" id="34"  port="M02_WGT_AXI"   size="0x8" offset="0x228" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr5"  addressQualifier="1" id="35"  port="M03_WGT_AXI"   size="0x8" offset="0x228" hostOffset="0x0" hostSize="0x8" type="int*"/>	  
	  <arg name="dpu_batch0_addr5"  addressQualifier="1" id="36"  port="M00_BIAS_AXI"  size="0x8" offset="0x228" hostOffset="0x0" hostSize="0x8" type="int*"/>      
      <arg name="dpu_batch0_addr6"  addressQualifier="1" id="37"  port="M00_WGT_AXI"   size="0x8" offset="0x230" hostOffset="0x0" hostSize="0x8" type="int*"/>	
      <arg name="dpu_batch0_addr6"  addressQualifier="1" id="38"  port="M01_WGT_AXI"   size="0x8" offset="0x230" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr6"  addressQualifier="1" id="39"  port="M02_WGT_AXI"   size="0x8" offset="0x230" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr6"  addressQualifier="1" id="40"  port="M03_WGT_AXI"   size="0x8" offset="0x230" hostOffset="0x0" hostSize="0x8" type="int*"/>	  
	  <arg name="dpu_batch0_addr6"  addressQualifier="1" id="41"  port="M00_BIAS_AXI"  size="0x8" offset="0x230" hostOffset="0x0" hostSize="0x8" type="int*"/>
      <arg name="dpu_batch0_addr7"  addressQualifier="1" id="42"  port="M00_WGT_AXI"   size="0x8" offset="0x238" hostOffset="0x0" hostSize="0x8" type="int*"/>	
      <arg name="dpu_batch0_addr7"  addressQualifier="1" id="43"  port="M01_WGT_AXI"   size="0x8" offset="0x238" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr7"  addressQualifier="1" id="44"  port="M02_WGT_AXI"   size="0x8" offset="0x238" hostOffset="0x0" hostSize="0x8" type="int*"/>	
	  <arg name="dpu_batch0_addr7"  addressQualifier="1" id="45"  port="M03_WGT_AXI"   size="0x8" offset="0x238" hostOffset="0x0" hostSize="0x8" type="int*"/>	  
	  <arg name="dpu_batch0_addr7"  addressQualifier="1" id="46"  port="M00_BIAS_AXI"  size="0x8" offset="0x238" hostOffset="0x0" hostSize="0x8" type="int*"/>	
''')

arg_id = 47


#arg IMG ports
for b in range (Batch_N):
    number_b = str(b) 
    offset_batch = b*64
    offset_No = 0x200 + offset_batch  
    IMG_P = LOAD_I_P * b
    for i in range (LOAD_I_P):
        total_IMG_P = IMG_P + i
        number_arg = str(total_IMG_P).rjust(2,'0')            
        Char1 = r'	  <arg name="dpu_batch'
        Char2 = r'_addr'
        Char3 = r'"  addressQualifier="1" id="'
        Char4 = r'"  port="M'
        Char5 = r'_IMG_AXI"   size="0x8" offset="'
        Char6 = r'" hostOffset="0x0" hostSize="0x8" type="int*"/>' 
        offset_perBatch =offset_No
        for x in range (8):
            number_addr = str(x)      
            number_arg_id = str(arg_id)    
            number_offset = str (hex(offset_perBatch))
            list_buf =Char1 + number_b + Char2 + number_addr + Char3 + number_arg_id + Char4 + number_arg + Char5 + number_offset  + Char6 +"\n"
            arg_id = arg_id + 1
            offset_perBatch = offset_perBatch + 8                  
            str_buf ="".join(list_buf) 
            result.append(str_buf)  

#ARGs section: AXIS  
AXIS_offset = 0x600
#ifm 
for i in range (ifm_number):
    number_ifm = str(i).rjust(2,'0')  
    number_arg_id = str(arg_id)  
    AXIS_No = AXIS_offset + i*8
    number_AXIS_No = str(hex(AXIS_No))
    Char1 = r'      <arg name="M'
    Char2 = r'_IFM_AXIS"      addressQualifier="4" id="'
    Char3 = r'"  port="M'
    Char4 = r'_IFM_AXIS"  size="0x4" offset="'
    Char5 = r'" hostOffset="0x0" hostSize="0x4" type="stream&lt;qdma_axis&lt;128, 0, 0, 0>>&amp;"/>'
    list_buf =Char1 + number_ifm + Char2 + number_arg_id + Char3 + number_ifm + Char4 + number_AXIS_No + Char5 + "\n"
    arg_id = arg_id + 1
    str_buf ="".join(list_buf) 
    result.append(str_buf)   

#wgt 
for i in range (wgt_number):
    number_wgt = str(i).rjust(2,'0') 
    number_arg_id = str(arg_id)  
    AXIS_No = AXIS_No + 8
    number_AXIS_No = str(hex(AXIS_No))
    Char1 = r'      <arg name="M'
    Char2 = r'_WGT_AXIS"      addressQualifier="4" id="'
    Char3 = r'"  port="M'
    Char4 = r'_WGT_AXIS"  size="0x4" offset="'
    Char5 = r'" hostOffset="0x0" hostSize="0x4" type="stream&lt;qdma_axis&lt;128, 0, 0, 0>>&amp;"/>'
    list_buf =Char1 + number_wgt + Char2 + number_arg_id + Char3 + number_wgt + Char4 + number_AXIS_No + Char5 + "\n"
    arg_id = arg_id + 1
    str_buf ="".join(list_buf)
    result.append(str_buf)   
 
#ofm 
for i in range (ofm_number):
    ofm_number = str(i).rjust(2,'0')
    number_arg_id = str(arg_id)  
    AXIS_No = AXIS_No + 8
    number_AXIS_No = str(hex(AXIS_No))
    Char1 = r'      <arg name="S'
    Char2 = r'_OFM_AXIS"      addressQualifier="4" id="'
    Char3 = r'"  port="S'
    Char4 = r'_OFM_AXIS"  size="0x4" offset="'
    Char5 = r'" hostOffset="0x0" hostSize="0x4" type="stream&lt;qdma_axis&lt;64, 0, 0, 0>>&amp;"/>'
    list_buf =Char1 + ofm_number + Char2 + number_arg_id + Char3 + ofm_number + Char4 + number_AXIS_No + Char5 + "\n"
    arg_id = arg_id + 1
    str_buf ="".join(list_buf) 
    result.append(str_buf)   

#End of arg section    
result.append(r'''    </args>
  </kernel>
</root>
''')


result_str="".join(result) 
file_name ="kernel" + ".xml"
new_file  = open(file_name, "w+")
new_file.write(result_str)

new_file.close()


