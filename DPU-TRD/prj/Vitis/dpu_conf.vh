/*
 * * Copyright 2019 Xilinx Inc.
 * *
 * * Licensed under the Apache License, Version 2.0 (the "License");
 * * you may not use this file except in compliance with the License.
 * * You may obtain a copy of the License at
 * *
 * *    http://www.apache.org/licenses/LICENSE-2.0
 * *
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS,
 * * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * * See the License for the specific language governing permissions and
 * * limitations under the License.
 * */


//Setting the arch of DPU, For more details, Please read the PG338 


/*====== Architecture Options ======*/
// |------------------------------------------------------|
// | Support 8 DPU size
// +------------------------------------------------------+
// | `define B512               
// +------------------------------------------------------+
// | `define B800                 
// +------------------------------------------------------+
// | `define B1024                 
// +------------------------------------------------------+
// | `define B1152                 
// +------------------------------------------------------+
// | `define B1600                 
// +------------------------------------------------------+
// | `define B2304                 
// +------------------------------------------------------+
// | `define B3136                 
// +------------------------------------------------------+
// | `define B4096                 
// |------------------------------------------------------|

`define B4096 

// |------------------------------------------------------|
// |If the FPGA has Uram. You can define URAM_EN parameter               
// +------------------------------------------------------+
// | for zcu104 : `define URAM_ENABLE               
// +------------------------------------------------------+
// | for zcu102 : `define URAM_DISABLE                 
// |------------------------------------------------------|

`define URAM_DISABLE 

//config URAM
`ifdef URAM_ENABLE
    `define def_UBANK_IMG_N          5
    `define def_UBANK_WGT_N          17
    `define def_UBANK_BIAS           1
`elsif URAM_DISABLE
    `define def_UBANK_IMG_N          0
    `define def_UBANK_WGT_N          0
    `define def_UBANK_BIAS           0
`endif

// |------------------------------------------------------|
// | RAM Usage Configuration              
// +------------------------------------------------------+
// | RAM Usage High : `define RAM_USAGE_HIGH               
// +------------------------------------------------------+
// | RAM Usage Low  : `define RAM_USAGE_LOW                 
// |------------------------------------------------------|

`define RAM_USAGE_LOW

// |------------------------------------------------------|
// | Channel Augmentation Configuration
// +------------------------------------------------------+
// | Enable  : `define CHANNEL_AUGMENTATION_ENABLE              
// +------------------------------------------------------+
// | Disable : `define CHANNEL_AUGMENTATION_DISABLE                
// |------------------------------------------------------|

`define CHANNEL_AUGMENTATION_ENABLE

// |------------------------------------------------------|
// | DepthWiseConv Configuration
// +------------------------------------------------------+
// | Enable  : `define DWCV_ENABLE              
// +------------------------------------------------------+
// | Disable : `define DWCV_DISABLE               
// |------------------------------------------------------|

`define DWCV_ENABLE

// |------------------------------------------------------|
// | Pool Average Configuration
// +------------------------------------------------------+
// | Enable  : `define POOL_AVG_ENABLE              
// +------------------------------------------------------+
// | Disable : `define POOL_AVG_DISABLE                
// |------------------------------------------------------|

`define POOL_AVG_ENABLE

// |------------------------------------------------------|
// | RELU Type Configuration
// +------------------------------------------------------+
// | `define RELU_RELU6
// +------------------------------------------------------+
// | `define RELU_LEAKYRELU_RELU6
// |------------------------------------------------------|

`define RELU_LEAKYRELU_RELU6

// |------------------------------------------------------|
// | DSP48 Usage Configuration
// +------------------------------------------------------+
// | `define DSP48_USAGE_HIGH              
// +------------------------------------------------------+
// | `define DSP48_USAGE_LOW                
// |------------------------------------------------------|

`define DSP48_USAGE_LOW  




 
