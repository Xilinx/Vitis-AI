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

LOAD_P        = int(sys.argv[1])

if LOAD_P==1:
	load_port = 0
	port_name = 0 
elif LOAD_P==2:
	load_port = 1
	port_name = 2
else:
    pass
    
def LOAD_FUNC(ports_num):
    char = ""
    for i in range (ports_num):
        char +=  r'	  <port name="M_AXI_HP' + str(2)+'" mode="master" range="0xFFFFFFFF" dataWidth="128" portType="addressable" base="0x0"/>' + "\n"
    return char

  

first_char =  (r'''<?xml version="1.0" encoding="UTF-8"?>
<root versionMajor="1" versionMinor="0">
  <kernel name="DPUCZDX8G" language="ip" vlnv="xilinx.com:RTLKernel:DPUCZDX8G:1.0" attributes="" preferredWorkGroupSizeMultiple="0" workGroupSize="1" debug="true" compileOptions=" -g" profileType="none">
    <ports>
      <port name="S_AXI_CONTROL" mode="slave" range="0x00001000" dataWidth="32" portType="addressable" base="0x0"/>
	  <port name="M_AXI_GP0" mode="master" range="0xFFFFFFFF" dataWidth="32" portType="addressable" base="0x0"/>
	  <port name="M_AXI_HP0" mode="master" range="0xFFFFFFFF" dataWidth="128" portType="addressable" base="0x0"/>
''')

args_1st_char = (r'''    </ports>
    <args>      
	<arg name="dpu_doneclr"     addressQualifier="0" id="0"  port="S_AXI_CONTROL" size="0x4" offset="0x40" hostOffset="0x0" hostSize="0x4" type="int*"/>
	<arg name="dpu_prof_en"     addressQualifier="0" id="1"  port="S_AXI_CONTROL" size="0x4" offset="0x44" hostOffset="0x0" hostSize="0x4" type="int*"/>
	<arg name="dpu_cmd"         addressQualifier="0" id="2"  port="S_AXI_CONTROL" size="0x4" offset="0x48" hostOffset="0x0" hostSize="0x4" type="int*"/>
	<arg name="dpu_instr_addr"  addressQualifier="1" id="3"  port="M_AXI_GP0"     size="0x8" offset="0x50" hostOffset="0x0" hostSize="0x8" type="int*"/>
	<arg name="dpu_prof_addr"   addressQualifier="1" id="4"  port="M_AXI_GP0"     size="0x8" offset="0x58" hostOffset="0x0" hostSize="0x8" type="int*"/>
	<arg name="dpu_base0_addr"  addressQualifier="1" id="5"  port="M_AXI_HP0"     size="0x8" offset="0x60" hostOffset="0x0" hostSize="0x8" type="int*"/>
	<arg name="dpu_base1_addr"  addressQualifier="1" id="6"  port="M_AXI_HP0"     size="0x8" offset="0x68" hostOffset="0x0" hostSize="0x8" type="int*"/>
	<arg name="dpu_base2_addr"  addressQualifier="1" id="7"  port="M_AXI_HP0"     size="0x8" offset="0x70" hostOffset="0x0" hostSize="0x8" type="int*"/>
	<arg name="dpu_base3_addr"  addressQualifier="1" id="8"  port="M_AXI_HP0"     size="0x8" offset="0x78" hostOffset="0x0" hostSize="0x8" type="int*"/>
''')

def PORT_ARGS(ports_num):
	char = ""
	char   +=  r'	<arg name="dpu_base4_addr"  addressQualifier="1" id="9"  port="M_AXI_HP' + str(ports_num)+'"     size="0x8" offset="0x80" hostOffset="0x0" hostSize="0x8" type="int*"/>' + "\n"
	char   +=  r'	<arg name="dpu_base5_addr"  addressQualifier="1" id="10"  port="M_AXI_HP' + str(ports_num)+'"    size="0x8" offset="0x88" hostOffset="0x0" hostSize="0x8" type="int*"/>' + "\n"
	char   +=  r'	<arg name="dpu_base6_addr"  addressQualifier="1" id="11"  port="M_AXI_HP' + str(ports_num)+'"    size="0x8" offset="0x90" hostOffset="0x0" hostSize="0x8" type="int*"/>' + "\n"
	char   +=  r'	<arg name="dpu_base7_addr"  addressQualifier="1" id="12"  port="M_AXI_HP' + str(ports_num)+'"    size="0x8" offset="0x98" hostOffset="0x0" hostSize="0x8" type="int*"/>' + "\n"
	return char



result = first_char   
result += LOAD_FUNC(load_port)  
result += args_1st_char
result += PORT_ARGS(port_name) 
result += (r'''    </args>
  </kernel>
</root>
''')

file_name ="./kernel_xml/dpu/kernel" + ".xml"
new_file  = open(file_name, "w+")
new_file.write(result)

new_file.close()



