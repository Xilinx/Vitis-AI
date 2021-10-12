"""
Copyright 2019 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import xrnn_py
import numpy as np

frame_num = 20
a = xrnn_py.xrnn("rnnt")
input_num = np.fromfile("32input", dtype=np.int8)
input_size = input_num.size # number of input data

# just for demo the usage of function xrnn::rnnt_update_ddr and xrnn::rnnt_download_ddr
# it can be used to update the ddr data 
offset_in_ddr_bin = 0x7c40000
decoder_frm_num = np.fromfile("32frm", dtype=np.int32)
frm_num_size = decoder_frm_num.size
a.rnnt_update_ddr(decoder_frm_num.flatten(), frm_num_size, offset_in_ddr_bin)
#
a.rnnt_reflash_ddr()
output_size = 0x400000 # 0x400000 * 2 bytes 
output_num = np.zeros(output_size, dtype=np.int16)
a.lstm_run(input_num.flatten(), input_size, output_num, output_size, frame_num)
output_num.tofile("./rslt/rslt_bin")


