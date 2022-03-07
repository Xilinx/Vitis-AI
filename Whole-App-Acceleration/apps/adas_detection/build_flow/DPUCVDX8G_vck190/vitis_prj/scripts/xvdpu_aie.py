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
CPB        = int(sys.argv[2])
BAT_SHRWGT = int(sys.argv[3])

WGTBC_N    = int((Batch_N+BAT_SHRWGT-1)/BAT_SHRWGT)

result=[]
Char1 = "[connectivity]\n"
Char2 = "nk=DPUCVDX8G:1:DPUCVDX8G\n"
result.append(Char1)
result.append(Char2)

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

###########AXI-stream connection between XVDPU and AIE#################################
#ofm connection
for i in range (ofm_number): 
    number_o = str(i).rjust(2,'0')
    list_buf ="stream_connect=ai_engine_0.M" + number_o + "_AXIS:DPUCVDX8G.S" + number_o + "_OFM_AXIS\n"
    str_buf  ="".join(list_buf) 
    result.append(str_buf)      

#ifm connection
for i in range (ifm_number):
    number_i = str(i).rjust(2,'0')
    list_buf ="stream_connect=DPUCVDX8G.M" + number_i + "_IFM_AXIS:ai_engine_0.S" + number_i + "_AXIS\n"
    str_buf  ="".join(list_buf) 
    result.append(str_buf)  

#wgt connection
for i in range (wgt_number):
    number_w = str(i).rjust(2,'0')
    wgt_s = i+ifm_number
    number_ws = str(wgt_s).rjust(2,'0')
    list_buf ="stream_connect=DPUCVDX8G.M" + number_w + "_WGT_AXIS:ai_engine_0.S" + number_ws + "_AXIS\n"
    str_buf  ="".join(list_buf)
    result.append(str_buf)
    
result_str="".join(result) 
file_name="xvdpu_aie" + ".cfg"
new_file = open(file_name, "w+")
new_file.write(result_str)

new_file.close()
